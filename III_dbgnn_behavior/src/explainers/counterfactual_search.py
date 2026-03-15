from __future__ import annotations

"""Notebook-friendly counterfactual *search* utilities.

The benchmarking explainer in :mod:`explainers.counterfactual` learns
an edge *ranking* (deletion priority). In notebook 02 we also want a concrete
counterfactual: a **small set of edges** whose deletion flips the node
prediction.

This module provides a thin wrapper around:

  - :class:`explainers.counterfactual.CounterfactualEdgeDeletionExplainer`
  - :func:`eval.metrics.min_k_to_flip`

to produce a minimal (greedy) deletion set and a compact result dictionary.

Compared to the benchmark runner, this adds notebook-oriented **sanity prints**
(when ``verbose=True``) to help debug cases where no flip is found.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from eval.metrics import default_candidate_mask_from_triples, min_k_to_flip, ranked_edge_indices
from explainers.counterfactual import CounterfactualConfig, CounterfactualEdgeDeletionExplainer
from explainers.statistical_counterfactual import (
    StatisticalCounterfactualConfig,
    StatisticalCounterfactualEdgeDeletionExplainer,
)
from pathpy_utils import idx_to_node_list


@dataclass(frozen=True)
class CounterfactualSearchResult:
    """Result returned by :func:`find_min_ho_edge_deletions_to_flip`."""

    # Query
    target_node_idx: int
    orig_pred: int

    # Outcome
    success: bool
    n_removed: int
    new_pred: Optional[int]

    # Deletions (global indices into the higher-order edge list)
    removed_edge_indices: List[int]
    ranked_edge_indices: List[int]

    # Optional, human-readable representation of removed edges
    removed_edges_as_node_ids: Optional[List[tuple[Any, Any]]] = None
    removed_triples_as_node_ids: Optional[List[tuple[Any, Any, Any]]] = None

    # Diagnostics
    p0: Optional[List[float]] = None
    p_after: Optional[List[float]] = None
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@torch.no_grad()
def _class_probs(logits_row: torch.Tensor) -> List[float]:
    p = torch.softmax(logits_row, dim=-1).detach().cpu().numpy()
    return [float(x) for x in p.tolist()]


@torch.no_grad()
def _logit_margin(row: torch.Tensor, cls: int) -> float:
    """Margin z_cls - max_{c!=cls} z_c (robust to probability saturation)."""
    cls = int(cls)
    if row.numel() <= 1:
        return float(row[cls].item())
    other = torch.cat([row[:cls], row[cls + 1 :]], dim=0)
    return float((row[cls] - other.max()).item())


def _maybe_node_id(assets, node_idx: int) -> Any:
    """Best-effort map PyG node index -> original node id (netzschleuder can be non-contiguous)."""
    if assets is None:
        return None
    g = getattr(assets, 'g', None)
    if g is None:
        return None
    try:
        idx_to_id = idx_to_node_list(g)
        if 0 <= int(node_idx) < len(idx_to_id):
            return idx_to_id[int(node_idx)]
    except Exception:
        return None
    return None


def _default_k_schedule(cap: int) -> List[int]:
    """Return a small, logarithmic-ish schedule of k values up to ``cap``.

    The counterfactual objective (class after dropping top-k edges) is not
    guaranteed to be monotone in k, so an *exact* minimal-k search requires a
    linear scan. In the notebook workflow we usually prefer speed and a compact
    summary over an exhaustive scan.

    This schedule is designed to:
      - be exact for very small k (1..10)
      - cover a range of larger k with a few probes
      - always include ``cap``
    """

    cap = int(max(0, cap))
    if cap <= 0:
        return []

    base: List[int] = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        15,
        20,
        30,
        40,
        50,
        75,
        100,
        150,
        200,
        300,
        500,
        750,
        1000,
        1500,
        2000,
        3000,
        5000,
        7500,
        10000,
    ]
    ks = [k for k in base if 1 <= k <= cap]
    if cap not in ks:
        ks.append(cap)
    return sorted(set(ks))


def _first_flip_search_schedule(
    *,
    adapter,
    data,
    node_idx: int,
    orig_class: int,
    edge_index_full: torch.Tensor,
    edge_weight_full: Optional[torch.Tensor],
    ranked_edge_idx: torch.Tensor,
    cap: int,
    refine_max_linear: int = 200,
    schedule: Optional[Sequence[int]] = None,
    drop_mode: str = "remove",
) -> Tuple[Optional[int], Optional[int], List[Tuple[int, int, float]]]:
    """Fast(ish) search for a flip using a k-schedule + local refinement.

    Returns:
        (k_flip, new_pred, probe)
        where probe is a list of (k, pred_k, margin_wrt_orig).

    Notes:
        This is *not* a substitute for an exhaustive linear scan in adversarial
        cases. It is meant for notebook usage where runtime matters.
    """

    E = int(edge_index_full.size(1))
    drop_mode = str(drop_mode).lower()
    if drop_mode not in {"remove", "zero"}:
        raise ValueError("drop_mode must be 'remove' or 'zero'")
    if drop_mode == "zero" and edge_weight_full is None:
        raise ValueError("drop_mode='zero' requires edge weights in the explain space")

    ranked_edge_idx = ranked_edge_idx.to(device=edge_index_full.device, dtype=torch.long).view(-1)
    cap = int(max(0, min(int(cap), int(ranked_edge_idx.numel()))))
    if cap == 0 or E == 0:
        return None, None, []

    # Cache evaluated k -> (pred, margin)
    cache: Dict[int, Tuple[int, float]] = {}

    def _pred_margin(k: int) -> Tuple[int, float]:
        k = int(k)
        if k in cache:
            return cache[k]

        if drop_mode == "remove":
            keep_mask = torch.ones(E, dtype=torch.bool, device=edge_index_full.device)
            keep_mask[ranked_edge_idx[:k]] = False
            ei = edge_index_full[:, keep_mask]
            ew = edge_weight_full[keep_mask] if edge_weight_full is not None else None
            data2 = adapter.clone_with_perturbed_edges(data, ei, new_edge_weight=ew)
        else:
            ew = edge_weight_full.detach().clone()
            ew[ranked_edge_idx[:k]] = 0.0
            data2 = adapter.clone_with_perturbed_edges(data, edge_index_full, new_edge_weight=ew)
        logits = adapter.predict_logits(data2)
        row = logits[int(node_idx)]
        pred = int(row.argmax().item())
        margin = _logit_margin(row, int(orig_class))
        cache[k] = (pred, float(margin))
        return cache[k]

    ks = list(schedule) if schedule is not None else _default_k_schedule(cap)
    ks = sorted({int(k) for k in ks if 1 <= int(k) <= cap})
    if cap not in ks:
        ks.append(cap)
    ks = sorted(set(ks))

    probe: List[Tuple[int, int, float]] = []
    prev_k = 0
    prev_pred = int(orig_class)

    for k in ks:
        pred_k, margin_k = _pred_margin(k)
        probe.append((int(k), int(pred_k), float(margin_k)))
        if int(pred_k) != int(orig_class):
            # Found a flip at this probed k.
            k_hi = int(k)
            k_lo = int(prev_k) if int(prev_pred) == int(orig_class) else 0

            # Local refinement: if the bracket is small, do an exact linear scan.
            if k_hi - k_lo <= int(refine_max_linear):
                for kk in range(int(k_lo) + 1, int(k_hi) + 1):
                    pred_kk, _ = _pred_margin(kk)
                    if int(pred_kk) != int(orig_class):
                        return int(kk), int(pred_kk), probe
                # Should not happen, but fall back.
                return int(k_hi), int(pred_k), probe

            # Otherwise do a monotone-style bisection to shrink the bracket,
            # then linear scan the final window.
            lo, hi = int(k_lo), int(k_hi)
            while hi - lo > int(refine_max_linear):
                mid = (lo + hi) // 2
                pred_mid, _ = _pred_margin(mid)
                if int(pred_mid) != int(orig_class):
                    hi = mid
                else:
                    lo = mid

            for kk in range(int(lo) + 1, int(hi) + 1):
                pred_kk, _ = _pred_margin(kk)
                if int(pred_kk) != int(orig_class):
                    return int(kk), int(pred_kk), probe

            return int(k_hi), int(pred_k), probe

        prev_k, prev_pred = int(k), int(pred_k)

    return None, None, probe


def _removed_edges_as_ids(
    *,
    edge_index: torch.Tensor,
    removed_edge_idx: List[int],
    g2,
) -> tuple[Optional[List[tuple[Any, Any]]], Optional[List[tuple[Any, Any, Any]]]]:
    """Map removed edge indices to human-readable HO node IDs + triples."""

    if g2 is None:
        return None, None

    try:
        idx_to_ho = idx_to_node_list(g2)
    except Exception:
        return None, None

    edges: List[tuple[Any, Any]] = []
    triples: List[tuple[Any, Any, Any]] = []

    for e in removed_edge_idx:
        if e < 0 or e >= int(edge_index.size(1)):
            continue
        src_idx = int(edge_index[0, e])
        dst_idx = int(edge_index[1, e])
        try:
            src = idx_to_ho[src_idx]
            dst = idx_to_ho[dst_idx]
        except Exception:
            continue
        edges.append((src, dst))

        # Best-effort triple for order-2 De Bruijn: (u,v)->(v,w) => (u,v,w)
        try:
            if isinstance(src, (tuple, list)) and len(src) >= 2 and isinstance(dst, (tuple, list)) and len(dst) >= 1:
                u = src[-2]
                v = src[-1]
                w = dst[-1]
                triples.append((u, v, w))
        except Exception:
            pass

    return edges or None, triples or None


def find_min_ho_edge_deletions_to_flip(
    *,
    adapter,
    data,
    assets=None,
    target_node_idx: int,
    cfg: Optional[CounterfactualConfig] = None,
    stat_cfg: Optional[StatisticalCounterfactualConfig] = None,
    seed: int = 0,
    max_k: int = 200,
    optimizer: str = 'counterfactual',
    drop_mode: str = "remove",
    # Search
    search_mode: str = 'schedule',
    refine_max_linear: int = 200,
    schedule: Optional[Sequence[int]] = None,
    # When no flip occurs, return a small "best effort" deletion set anyway.
    return_best_effort: bool = True,
    best_effort_max_k: int = 200,
    verbose: bool = True,
) -> CounterfactualSearchResult:
    """Find a small set of higher-order edge deletions that flips a node prediction.

    The workflow is:
      1) get the original prediction
      2) obtain an edge ranking (counterfactual mask-optimization or random)
      3) greedily delete edges in that order and return the first k that flips

    Args:
        adapter: model adapter (e.g. :class:`models.dbgnn.DBGNNAdapter`).
        data: full PyG Data.
        assets: optional dataset assets (used only for mapping to readable IDs).
        target_node_idx: node index in the PyG Data.
        cfg: counterfactual explainer configuration.
        stat_cfg: statistical explainer configuration (used when optimizer="statistical").
        seed: RNG seed for the explainer / random ranking.
        max_k: maximum number of deletions to consider.
        optimizer: "counterfactual" (recommended), "statistical" (shuffle-null ranking), or "random" (sanity check).
        drop_mode: "remove" to drop edges from edge_index, "zero" to zero their edge weights.
        search_mode: "schedule" (fast notebook default) or "linear" (exact but potentially slow).
        refine_max_linear: in schedule mode, do an exact linear refinement when the bracket is <= this size.
        schedule: optional custom schedule of k values to probe in schedule mode.
        return_best_effort: if True and no flip occurs, return a small set of deletions that maximally reduces
            the original class margin (useful for plotting even when flipping is impossible).
        best_effort_max_k: cap the size of this best-effort set.
        verbose: whether to print debugging info.

    Returns:
        CounterfactualSearchResult
    """

    v = int(target_node_idx)

    # --- Baseline prediction ---
    logits0 = adapter.predict_logits(data)
    row0 = logits0[v]
    orig_pred = int(row0.argmax().item())
    p0 = _class_probs(row0)
    margin0 = _logit_margin(row0, orig_pred)

    # True label (if available)
    y_true = None
    if hasattr(data, 'y') and isinstance(getattr(data, 'y'), torch.Tensor):
        y = getattr(data, 'y')
        if 0 <= v < int(y.numel()):
            try:
                y_true = int(y[v].item())
            except Exception:
                y_true = None

    node_id = _maybe_node_id(assets, v)

    expl_cfg = cfg or CounterfactualConfig()

    # --- Ranking ---
    scores = None
    cand_mask = None

    if optimizer == 'counterfactual':
        explainer = CounterfactualEdgeDeletionExplainer(cfg=expl_cfg, seed=int(seed))
        exp = explainer.explain_node(adapter=adapter, data=data, node_idx=v, target_class=orig_pred)
        scores = exp.edge_score
        cand_mask = exp.candidate_mask
        edge_index_full = exp.edge_index
        ranked_edge_idx = ranked_edge_indices(scores, candidate_mask=cand_mask, descending=True)

    elif optimizer == 'statistical':
        stat_expl_cfg = stat_cfg or StatisticalCounterfactualConfig()
        explainer = StatisticalCounterfactualEdgeDeletionExplainer(cfg=stat_expl_cfg)
        exp = explainer.explain_node(adapter=adapter, data=data, node_idx=v, target_class=orig_pred)
        scores = exp.edge_score
        cand_mask = exp.candidate_mask
        edge_index_full = exp.edge_index
        ranked_edge_idx = ranked_edge_indices(scores, candidate_mask=cand_mask, descending=True)

    elif optimizer == 'random':
        # Cheap sanity-check baseline: random ranking within a best-effort candidate set.
        space = adapter.explain_space()
        edge_index_full = getattr(data, space.edge_index_attr)
        E = int(edge_index_full.size(1))
        cand_mask = default_candidate_mask_from_triples(
            data,
            E=E,
            node_idx=v,
            triples_attr=expl_cfg.triples_attr,
        )
        if cand_mask is None:
            cand_mask = torch.ones(E, dtype=torch.bool, device=edge_index_full.device)

        rng = np.random.default_rng(int(seed))
        order = np.arange(E)
        cand = cand_mask.detach().cpu().numpy().astype(bool)
        cand_idx = order[cand]
        rng.shuffle(cand_idx)
        non_idx = order[~cand]
        ranked = np.concatenate([cand_idx, non_idx], axis=0)
        ranked_edge_idx = torch.tensor(ranked, dtype=torch.long, device=edge_index_full.device)

    else:
        raise ValueError("optimizer must be 'counterfactual' or 'random'")

    # Edge weights
    space = adapter.explain_space()
    edge_weight_full = None
    if space.edge_weight_attr is not None and hasattr(data, space.edge_weight_attr):
        edge_weight_full = getattr(data, space.edge_weight_attr)

    drop_mode = str(drop_mode).lower()

    E_full = int(edge_index_full.size(1))
    C = int(cand_mask.sum().item()) if (cand_mask is not None and cand_mask.numel() == E_full) else E_full

    meta: Dict[str, Any] = {
        'optimizer': optimizer,
        'search_mode': str(search_mode),
        'drop_mode': str(drop_mode),
        'node_idx': v,
        'node_id': node_id,
        'y_true': y_true,
        'orig_pred': orig_pred,
        'margin0': float(margin0),
        'E_ho_edges': E_full,
        'C_candidates': C,
    }

    if verbose:
        nid_str = f" id={node_id!r}" if node_id is not None else ''
        yt_str = f" y={y_true}" if y_true is not None else ''
        print(f"[counterfactual] node idx={v}{nid_str}{yt_str}")
        print(f"  baseline pred={orig_pred}  margin={margin0:.4f}  p_max={max(p0):.4f}")
        print(f"  explain-space: E={E_full} HO edges, candidates={C} ({(C / max(E_full,1)):.1%}), optimizer={optimizer}")

        # Sanity: does removing ALL HO edges change logits at all?
        try:
            empty_ei = edge_index_full[:, :0]
            if edge_weight_full is not None:
                empty_ew = edge_weight_full[:0]
            else:
                empty_ew = torch.empty((0,), dtype=torch.float32, device=edge_index_full.device)
            data_empty = adapter.clone_with_perturbed_edges(data, empty_ei, new_edge_weight=empty_ew)
            logits_empty = adapter.predict_logits(data_empty)
            d = (row0 - logits_empty[v]).abs()
            print(f"  sanity: drop ALL HO edges -> max|Δlogit|={float(d.max().item()):.3e}, mean|Δlogit|={float(d.mean().item()):.3e}")
            meta['empty_graph_logit_diff_max'] = float(d.max().item())
            meta['empty_graph_logit_diff_mean'] = float(d.mean().item())
        except Exception as e:
            print(f"  sanity: could not compute empty-graph diff: {type(e).__name__}: {e}")

        # Soft-mask sanity: does the *learned soft mask* already flip?
        if optimizer == 'counterfactual' and scores is not None and edge_weight_full is not None:
            try:
                keep_prob = torch.ones(E_full, dtype=edge_weight_full.dtype, device=edge_weight_full.device)
                if cand_mask is not None:
                    keep_prob[cand_mask] = (1.0 - scores[cand_mask]).clamp(0.0, 1.0).to(dtype=edge_weight_full.dtype)
                else:
                    keep_prob = (1.0 - scores).clamp(0.0, 1.0).to(dtype=edge_weight_full.dtype)

                w_soft = edge_weight_full * keep_prob
                data_soft = adapter.clone_with_perturbed_edges(data, edge_index_full, new_edge_weight=w_soft)
                logits_soft = adapter.predict_logits(data_soft)
                pred_soft = int(logits_soft[v].argmax().item())
                margin_soft = _logit_margin(logits_soft[v], orig_pred)
                print(f"  sanity: optimizer soft-mask pred={pred_soft} (margin wrt orig={margin_soft:.4f})")
                meta['soft_mask_pred'] = pred_soft
                meta['soft_mask_margin_wrt_orig'] = float(margin_soft)
            except Exception as e:
                print(f"  sanity: could not evaluate soft-mask graph: {type(e).__name__}: {e}")

    # --- Flip search ---
    cap = int(min(int(max_k), int(ranked_edge_idx.numel())))
    meta['k_cap'] = int(cap)

    probe: List[Tuple[int, int, float]] = []
    if str(search_mode).lower() == 'linear':
        # Exact but potentially slow for large cap.
        k_flip, new_pred_at = min_k_to_flip(
            adapter=adapter,
            data=data,
            node_idx=v,
            orig_class=orig_pred,
            edge_index_full=edge_index_full,
            edge_weight_full=edge_weight_full,
            ranked_edge_idx=ranked_edge_idx,
            max_k=int(cap),
            drop_mode=str(drop_mode),
        )
    else:
        # Fast notebook mode: probe a schedule + local refinement.
        k_flip, new_pred_at, probe = _first_flip_search_schedule(
            adapter=adapter,
            data=data,
            node_idx=v,
            orig_class=orig_pred,
            edge_index_full=edge_index_full,
            edge_weight_full=edge_weight_full,
            ranked_edge_idx=ranked_edge_idx,
            cap=int(cap),
            refine_max_linear=int(refine_max_linear),
            schedule=schedule,
            drop_mode=str(drop_mode),
        )

    success = k_flip is not None

    # If no flip: choose a compact "best effort" k that reduces the original margin.
    best_effort_k = 0
    best_effort_margin = float(margin0)
    best_effort_pred = int(orig_pred)
    if (not success) and bool(return_best_effort) and probe:
        cap_best = int(min(int(best_effort_max_k), int(cap)))
        eligible = [(k, p, m) for (k, p, m) in probe if int(k) <= cap_best]
        if not eligible:
            eligible = probe
        # Pick the k with smallest margin; ties -> smaller k.
        k_star, p_star, m_star = min(eligible, key=lambda t: (float(t[2]), int(t[0])))
        best_effort_k = int(k_star)
        best_effort_pred = int(p_star)
        best_effort_margin = float(m_star)
        meta['best_effort_k'] = int(best_effort_k)
        meta['best_effort_pred'] = int(best_effort_pred)
        meta['best_effort_margin_wrt_orig'] = float(best_effort_margin)

    n_removed = int(k_flip) if success else int(best_effort_k)
    new_pred = int(new_pred_at) if (success and new_pred_at is not None) else None

    removed_edge_indices = ranked_edge_idx[:n_removed].detach().cpu().tolist() if n_removed > 0 else []
    ranked_list = ranked_edge_idx.detach().cpu().tolist()

    # --- Probability after deletion (only for the returned set) ---
    p_after = None
    pred_after = int(orig_pred)
    margin_after = float(margin0)
    if n_removed > 0:
        if drop_mode == "remove":
            keep_mask = torch.ones(E_full, dtype=torch.bool, device=edge_index_full.device)
            keep_mask[ranked_edge_idx[:n_removed]] = False
            ei = edge_index_full[:, keep_mask]
            ew = edge_weight_full[keep_mask] if edge_weight_full is not None else None
            data2 = adapter.clone_with_perturbed_edges(data, ei, new_edge_weight=ew)
        else:
            ew = edge_weight_full.detach().clone()
            ew[ranked_edge_idx[:n_removed]] = 0.0
            data2 = adapter.clone_with_perturbed_edges(data, edge_index_full, new_edge_weight=ew)
        logits1 = adapter.predict_logits(data2)
        row1 = logits1[v]
        pred_after = int(row1.argmax().item())
        margin_after = _logit_margin(row1, orig_pred)
        p_after = _class_probs(row1)

    meta['pred_after'] = int(pred_after)
    meta['margin_after_wrt_orig'] = float(margin_after)

    # Fractions (requested by user)
    deleted_cand = int(min(int(n_removed), int(C)))
    frac_cand = float(deleted_cand / max(int(C), 1))
    frac_total = float(int(n_removed) / max(int(E_full), 1))
    meta['deleted_candidates'] = int(deleted_cand)
    meta['fraction_candidates_removed'] = float(frac_cand)
    meta['fraction_total_ho_edges_removed'] = float(frac_total)

    if probe:
        # Store a compact probe summary (k, pred, margin) for inspection.
        meta['probe'] = [(int(k), int(p), float(m)) for (k, p, m) in probe]

    # --- Map removed edges to HO node IDs for plotting ---
    g2 = getattr(assets, 'g2', None) if assets is not None else None
    removed_edges_as_node_ids, removed_triples_as_node_ids = _removed_edges_as_ids(
        edge_index=edge_index_full.detach().cpu(),
        removed_edge_idx=[int(i) for i in removed_edge_indices],
        g2=g2,
    )

    # --- Extra debugging + requested fractions ---
    if verbose:
        del_line = (
            f"  deletions: {int(n_removed)}/{int(C)} candidates ({100*frac_cand:.2f}%), "
            f"{int(n_removed)}/{int(E_full)} total HO edges ({100*frac_total:.2f}%)"
        )
        if success:
            print(f"  result: FLIP at k={int(n_removed)}: {orig_pred} -> {new_pred}")
            print(del_line)
        else:
            if str(search_mode).lower() == 'linear':
                print(f"  result: NO FLIP up to k={int(cap)} (linear scan)")
            else:
                print(f"  result: NO FLIP observed up to k={int(cap)} (schedule mode; tested {len(probe)} ks)")
            print(del_line)
            if n_removed > 0:
                print(
                    f"  best-effort: k={int(n_removed)} keeps pred={int(pred_after)} "
                    f"but reduces margin wrt orig to {float(margin_after):.4f} (from {float(margin0):.4f})"
                )

            # Print a small probe table
            if probe:
                show = probe
                # If the probe list is long, only show a compact subset.
                if len(show) > 12:
                    show = show[:8] + [show[-1]]
                print("  probe preds:")
                for k, p, m in show:
                    print(f"    k={int(k):>4}: pred={int(p)}  margin_wrt_orig={float(m):.4f}")

    meta.update({
        'success': bool(success),
        'n_removed': int(n_removed),
        'new_pred': (int(new_pred) if success and new_pred is not None else None),
    })

    return CounterfactualSearchResult(
        target_node_idx=v,
        orig_pred=orig_pred,
        success=bool(success),
        n_removed=n_removed,
        new_pred=new_pred if success else None,
        removed_edge_indices=[int(i) for i in removed_edge_indices],
        ranked_edge_indices=[int(i) for i in ranked_list],
        removed_edges_as_node_ids=removed_edges_as_node_ids,
        removed_triples_as_node_ids=removed_triples_as_node_ids,
        p0=p0,
        p_after=p_after,
        meta=meta,
    )
