from __future__ import annotations

from typing import Optional

import torch


def default_candidate_mask_from_triples(
    data,
    *,
    E: int,
    node_idx: int,
    triples_attr: str = "ho_triples",
) -> Optional[torch.Tensor]:
    """Best-effort default candidate set for higher-order (triple) explanations.

    Many explainers return edge scores for *all* higher-order edges but are
    intended to be used as *node-focused* explainers. For the temporal-clusters
    tutorial, the natural explanation atoms are higher-order transitions
    (u,v)->(v,w), represented as triples (u,v,w).

    If `data.<triples_attr>` exists and is aligned with the explain-space edge
    list (shape [E,3]), we restrict the candidate set for a first-order node
    `node_idx` to triples whose middle node equals `node_idx`.

    Returns:
        A boolean Tensor of shape [E] on the same device as `data.<triples_attr>`,
        or None if the required attribute is missing or mis-shaped.
    """

    if not hasattr(data, triples_attr):
        return None

    triples = getattr(data, triples_attr)
    if triples is None or not isinstance(triples, torch.Tensor):
        return None

    if triples.ndim != 2 or triples.size(0) != int(E) or triples.size(1) != 3:
        return None

    return (triples[:, 1] == int(node_idx)).to(torch.bool)


def topk_mask(
    scores: torch.Tensor,
    frac: float,
    *,
    candidate_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return a boolean mask for the top-k fraction of scores.

    If `candidate_mask` is provided, the top-k selection is performed *within*
    the candidate set and the returned mask is still of shape [E] (full edge
    set), with True only for selected candidate edges.

    This is important for node-focused explainers that only assign meaningful
    scores to a subset of edges; using global-E top-k can otherwise force the
    selection of irrelevant edges when k exceeds the number of candidates.
    """
    if scores.dim() != 1:
        raise ValueError(f"scores must be 1D, got shape {tuple(scores.shape)}")

    E = scores.numel()
    if E == 0:
        return torch.zeros(0, dtype=torch.bool, device=scores.device)

    # Default: select among all edges.
    if candidate_mask is None:
        k = int(max(1, round(float(frac) * E)))
        k = min(k, E)
        idx = torch.topk(scores, k=k).indices
        mask = torch.zeros(E, dtype=torch.bool, device=scores.device)
        mask[idx] = True
        return mask

    # Candidate-restricted selection
    cand = candidate_mask
    if cand.dim() != 1 or cand.numel() != E:
        raise ValueError(
            f"candidate_mask must be 1D of shape [E]={E}, got shape {tuple(cand.shape)}"
        )
    if cand.dtype != torch.bool:
        cand = cand != 0
    cand = cand.to(device=scores.device)

    idx_cand = cand.nonzero(as_tuple=False).view(-1)
    C = int(idx_cand.numel())
    if C == 0:
        return torch.zeros(E, dtype=torch.bool, device=scores.device)

    k = int(max(1, round(float(frac) * C)))
    k = min(k, C)

    cand_scores = scores[idx_cand]
    top_local = torch.topk(cand_scores, k=k).indices
    top_global = idx_cand[top_local]

    mask = torch.zeros(E, dtype=torch.bool, device=scores.device)
    mask[top_global] = True
    return mask


def filter_edges(edge_index: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    """Filter an edge_index [2, E] by a boolean mask of shape [E]."""
    if edge_index.size(1) != keep_mask.numel():
        raise ValueError(
            f"edge_index has {edge_index.size(1)} edges but keep_mask has {keep_mask.numel()}"
        )
    return edge_index[:, keep_mask]


def filter_edge_weights(edge_weight: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    if edge_weight.numel() != keep_mask.numel():
        raise ValueError(
            f"edge_weight has {edge_weight.numel()} entries but keep_mask has {keep_mask.numel()}"
        )
    return edge_weight[keep_mask]


def target_prob(logits: torch.Tensor, node_idx: int, target_class: int) -> float:
    """Softmax probability assigned to target_class for node_idx."""
    p = torch.softmax(logits[int(node_idx)], dim=0)[int(target_class)].item()
    return float(p)


def target_logit(logits: torch.Tensor, node_idx: int, target_class: int) -> float:
    """Raw (pre-softmax) logit for target_class at node_idx."""
    return float(logits[int(node_idx), int(target_class)].item())


def target_margin(logits: torch.Tensor, node_idx: int, target_class: int) -> float:
    """Logit margin: z_target - max_{c!=target} z_c.

    Why this is useful:
      - When p0 is near 1.0, probability-based metrics saturate and different
        perturbations look identical.
      - The margin can change substantially even when the probability remains
        close to 1.
    """

    row = logits[int(node_idx)]
    t = int(target_class)
    if row.numel() <= 1:
        return float(0.0)
    z_t = row[t]
    # max over all other classes
    z_others = torch.cat([row[:t], row[t + 1 :]])
    z_max_other = torch.max(z_others)
    return float((z_t - z_max_other).item())


def target_log_prob(logits: torch.Tensor, node_idx: int, target_class: int) -> float:
    """Log-softmax for numerical stability (log probability of target class)."""
    lp = torch.log_softmax(logits[int(node_idx)], dim=0)[int(target_class)].item()
    return float(lp)


def keep_drop_masks(
    scores: torch.Tensor,
    frac: float,
    *,
    candidate_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (keep_sel, keep_mask, drop_mask) for fidelity evaluation.

    Definitions (matching `eval.runner.run_benchmark`):

    - keep_sel: selected *candidate* edges (top-k within candidate set if provided)
    - keep_mask: edges kept for the sufficiency graph
      - if candidate_mask is provided: keep all non-candidate edges + selected candidate edges
      - else: keep only selected edges
    - drop_mask: edges kept for the comprehensiveness graph
      - always `~keep_sel` (i.e., drop only the selected edges)

    This helper keeps evaluation logic consistent across notebooks and plotting.
    """

    keep_sel = topk_mask(scores, frac, candidate_mask=candidate_mask)

    if candidate_mask is None:
        keep_mask = keep_sel
        drop_mask = ~keep_sel
        return keep_sel, keep_mask, drop_mask

    cand = candidate_mask
    if cand.dtype != torch.bool:
        cand = cand != 0
    cand = cand.to(device=scores.device)

    keep_mask = (~cand) | keep_sel
    drop_mask = ~keep_sel
    return keep_sel, keep_mask, drop_mask


def ranked_edge_indices(
    scores: torch.Tensor,
    *,
    candidate_mask: Optional[torch.Tensor] = None,
    descending: bool = True,
) -> torch.Tensor:
    """Return edge indices ranked by score.

    If `candidate_mask` is provided, the ranking is restricted to candidates and
    returned indices are *global* indices into the full edge list.
    """
    if scores.dim() != 1:
        raise ValueError(f"scores must be 1D, got shape {tuple(scores.shape)}")

    E = scores.numel()
    if E == 0:
        return torch.zeros(0, dtype=torch.long, device=scores.device)

    if candidate_mask is None:
        return torch.argsort(scores, descending=bool(descending))

    cand = candidate_mask
    if cand.dim() != 1 or cand.numel() != E:
        raise ValueError(
            f"candidate_mask must be 1D of shape [E]={E}, got shape {tuple(cand.shape)}"
        )
    if cand.dtype != torch.bool:
        cand = cand != 0
    cand = cand.to(device=scores.device)

    idx_cand = cand.nonzero(as_tuple=False).view(-1)
    if idx_cand.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=scores.device)

    s_cand = scores[idx_cand]
    order_local = torch.argsort(s_cand, descending=bool(descending))
    return idx_cand[order_local]


@torch.no_grad()
def min_k_to_flip(
    *,
    adapter,
    data,
    node_idx: int,
    orig_class: int,
    edge_index_full: torch.Tensor,
    edge_weight_full: Optional[torch.Tensor],
    ranked_edge_idx: torch.Tensor,
    max_k: Optional[int] = None,
    drop_mode: str = "remove",
) -> tuple[Optional[int], Optional[int]]:
    """Return the minimal k such that dropping top-k ranked edges flips the prediction.

    This is the metric you want for counterfactual explainers:
      - take an explainer's ranking
      - greedily drop edges in that order
      - find the first k at which the predicted class changes

    NOTE on correctness vs speed:
        The predicted class as a function of k (number of dropped edges) is NOT
        guaranteed to be monotone. In particular, it can flip and then flip back
        as more edges are removed.

        Therefore, this function uses a **linear scan** over k=1..cap to find
        the *first* flip, which is the correct counterfactual size.

    Args:
        adapter: model adapter (must provide clone_with_perturbed_edges + predict_logits)
        data: full graph data
        node_idx: node to evaluate
        orig_class: original predicted class
        edge_index_full: original explain-space edge_index
        edge_weight_full: original explain-space edge weights (or None)
        ranked_edge_idx: global edge indices sorted by deletion priority (descending)
        max_k: optional cap on k (defaults to len(ranked_edge_idx))
        drop_mode: "remove" to drop edges from edge_index, "zero" to zero their edge weights.

    Returns:
        (k_flip, new_pred_at_k_flip)
        - k_flip is None if no flip occurs up to max_k.
    """

    E = int(edge_index_full.size(1))
    if E == 0:
        return None, None

    drop_mode = str(drop_mode).lower()
    if drop_mode not in {"remove", "zero"}:
        raise ValueError("drop_mode must be 'remove' or 'zero'")
    if drop_mode == "zero" and edge_weight_full is None:
        raise ValueError("drop_mode='zero' requires edge weights in the explain space")

    ranked_edge_idx = ranked_edge_idx.to(device=edge_index_full.device, dtype=torch.long).view(-1)
    if ranked_edge_idx.numel() == 0:
        return None, None

    cap = int(max_k) if max_k is not None else int(ranked_edge_idx.numel())
    cap = max(0, min(cap, int(ranked_edge_idx.numel())))
    if cap == 0:
        return None, None

    # Cache predictions for tested k to avoid duplicate forward passes.
    pred_cache: dict[int, int] = {}

    def _pred_after_drop(k: int) -> int:
        k = int(k)
        if k in pred_cache:
            return pred_cache[k]

        if drop_mode == "remove":
            keep_mask = torch.ones(E, dtype=torch.bool, device=edge_index_full.device)
            drop_idx = ranked_edge_idx[:k]
            keep_mask[drop_idx] = False

            ei = edge_index_full[:, keep_mask]
            ew = edge_weight_full[keep_mask] if edge_weight_full is not None else None
            data2 = adapter.clone_with_perturbed_edges(data, ei, new_edge_weight=ew)
        else:
            ew = edge_weight_full.detach().clone()
            drop_idx = ranked_edge_idx[:k]
            ew[drop_idx] = 0.0
            data2 = adapter.clone_with_perturbed_edges(data, edge_index_full, new_edge_weight=ew)
        logits = adapter.predict_logits(data2)
        pred = int(logits[int(node_idx)].argmax().item())
        pred_cache[k] = pred
        return pred

    # Linear scan for the *first* flip.
    for k in range(1, cap + 1):
        pred_k = _pred_after_drop(k)
        if pred_k != int(orig_class):
            return int(k), int(pred_k)

    # No flip within cap.
    pred_cap = _pred_after_drop(cap)
    return None, int(pred_cap)
