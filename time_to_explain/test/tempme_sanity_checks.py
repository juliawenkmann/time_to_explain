# time_to_explain/test/tempme_sanity_checks.py
from __future__ import annotations

"""
Here are sanity checks I'd run in roughly this order, from "fastest smoke test"
-> "proves the wiring is correct" -> "behavior looks like TempME should".

These helpers implement the checks as small functions you can call from a notebook
or a script. They intentionally do not hide any heavy lifting behind a test runner.

TempME invariants covered:
  - motif sampling strict time order + connectivity
  - event anonymization h(e)
  - motif equivalence class code
  - IB term sanity
  - filtered neighbor finder identity and restoration
"""

from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
except Exception:  # pragma: no cover - optional dependency in some environments
    torch = None  # type: ignore

from submodules.explainer.tempme_tgn_impl.tempme import (
    AdjacencyListNeighborFinder,
    FilteredNeighborFinder,
    TemporalMotif,
    event_anonymization_h,
    ib_kl_term,
    motif_code,
    patch_neighbor_finder,
    tgn_predict_proba,
)


# -----------------------------------------------------------------------------
# 1) Smoke test: end-to-end explain() without touching TGN
# -----------------------------------------------------------------------------

def smoke_test_explain(
    *,
    explainer,
    neighbor_finder: AdjacencyListNeighborFinder,
    u: int,
    v: int,
    t: float,
    node_raw_features: np.ndarray,
    edge_raw_features: np.ndarray,
    top_k: int = 5,
) -> Any:
    exp = explainer.explain(
        neighbor_finder_for_sampling=neighbor_finder,
        u=int(u),
        v=int(v),
        t=float(t),
        node_raw_features=node_raw_features,
        edge_raw_features=edge_raw_features,
        selection_mode="topk",
        top_k=int(top_k),
    )

    has_hist = _has_history(neighbor_finder, u, t) or _has_history(neighbor_finder, v, t)

    if has_hist:
        assert len(exp.motifs) > 0
    if exp.motif_scores is not None:
        assert int(exp.motif_scores.shape[0]) == len(exp.motifs)

    if exp.selected_motif_mask is not None:
        sel = int(_to_scalar(exp.selected_motif_mask.sum()))
        expected = min(int(top_k), len(exp.motifs))
        if len(exp.motifs) > 0:
            assert sel == expected

    if has_hist:
        assert len(exp.edges) > 0

    return exp


# -----------------------------------------------------------------------------
# 2) Motif sampling invariants (Algorithm 1)
# -----------------------------------------------------------------------------

def assert_motif_ok(motif: TemporalMotif, u0: int, t0: float, l: int, n: int, allow_shorter: bool = False) -> None:
    if not allow_shorter:
        assert len(motif.events) == int(l)
    ts = [e.ts for e in motif.events]
    assert all(t < float(t0) for t in ts)
    assert all(ts[i] > ts[i + 1] for i in range(len(ts) - 1)), ts

    nodes: Set[int] = set()
    adj: Dict[int, Set[int]] = {}
    for e in motif.events:
        nodes.add(e.src)
        nodes.add(e.dst)
        adj.setdefault(e.src, set()).add(e.dst)
        adj.setdefault(e.dst, set()).add(e.src)

    assert len(nodes) <= int(n)

    # Connectivity check
    if nodes:
        start = next(iter(nodes))
        seen = {start}
        stack = [start]
        while stack:
            x = stack.pop()
            for y in adj.get(x, set()):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        assert seen == nodes


def check_motif_sampling_invariants(
    motifs: Sequence[TemporalMotif],
    *,
    u0: int,
    t0: float,
    motif_len: int,
    motif_max_nodes: int,
    allow_shorter: bool = False,
) -> None:
    for mot in motifs:
        assert_motif_ok(mot, u0=u0, t0=t0, l=motif_len, n=motif_max_nodes, allow_shorter=allow_shorter)


# -----------------------------------------------------------------------------
# 3) Event anonymization h(e) (Eq. 2)
# -----------------------------------------------------------------------------

def check_event_anonymization(motifs: Sequence[TemporalMotif], l: int) -> None:
    h1 = event_anonymization_h(motifs, num_events=int(l))
    h2 = _brute_h(motifs, int(l))
    keys = set(h1.keys()) | set(h2.keys())
    for k in keys:
        v1 = h1.get(k)
        v2 = h2.get(k)
        assert v1 is not None and v2 is not None
        assert np.allclose(v1, v2)


def _brute_h(motifs: Sequence[TemporalMotif], l: int) -> Dict[Tuple[int, int], np.ndarray]:
    out: Dict[Tuple[int, int], np.ndarray] = {}
    for mot in motifs:
        for j, e in enumerate(mot.events[:l]):
            pair = tuple(sorted((int(e.src), int(e.dst))))
            if pair not in out:
                out[pair] = np.zeros((l,), dtype=np.float32)
            out[pair][j] += 1.0
    return out


# -----------------------------------------------------------------------------
# 4) Motif equivalence class codes
# -----------------------------------------------------------------------------

def motif_code_diversity(motifs: Sequence[TemporalMotif], anchors: Sequence[int]) -> int:
    codes = [motif_code(m, anchor=int(a)) for m, a in zip(motifs, anchors)]
    return len(set(codes))


# -----------------------------------------------------------------------------
# 5) IB term sanity: zero in matching case
# -----------------------------------------------------------------------------

def check_ib_term_zero_case(
    class_ids: Sequence[str],
    *,
    p_prior: float = 0.2,
    tol: float = 1e-6,
    device: Optional[Any] = None,
) -> float:
    if torch is None:
        raise ImportError("torch is required for ib_kl_term sanity checks")
    if not class_ids:
        raise ValueError("class_ids is empty")

    p = torch.full((len(class_ids),), float(p_prior), device=device or "cpu")
    q = _class_histogram(class_ids)
    ib = ib_kl_term(class_ids, p, q, p_prior=float(p_prior))
    val = float(_to_scalar(ib))
    assert np.isfinite(val)
    assert val >= -1e-6
    assert abs(val) < float(tol)
    return val


def _class_histogram(class_ids: Sequence[str]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    for cid in class_ids:
        counts[cid] = counts.get(cid, 0) + 1
    total = float(sum(counts.values()))
    return {cid: cnt / total for cid, cnt in counts.items()}


# -----------------------------------------------------------------------------
# 6) Wiring test with official TGN: identity and restoration
# -----------------------------------------------------------------------------

def tgn_identity_test(
    *,
    model: Any,
    u: int,
    v: int,
    t: float,
    edge_idx: int,
    n_neighbors: int,
    all_edge_idxs: Iterable[int],
    tol: float = 1e-6,
    device: Optional[Any] = None,
) -> float:
    p_full = tgn_predict_proba(model, u, v, t, edge_idx, n_neighbors=n_neighbors, device=device)

    base_nf = _get_base_neighbor_finder(model)
    filtered_nf = FilteredNeighborFinder(
        base_nf,
        allowed_edge_idxs=set(int(e) for e in all_edge_idxs),
        uniform=getattr(base_nf, "uniform", True),
    )

    with _swap_neighbor_finder(model, filtered_nf):
        p_all = tgn_predict_proba(model, u, v, t, edge_idx, n_neighbors=n_neighbors, device=device)

    diff = float(abs(_to_scalar(p_full) - _to_scalar(p_all)))
    assert diff < float(tol)
    return diff


def tgn_restoration_test(model: Any, neighbor_finder) -> None:
    orig = _get_base_neighbor_finder(model)
    with _swap_neighbor_finder(model, neighbor_finder):
        assert _get_base_neighbor_finder(model) is neighbor_finder
    assert _get_base_neighbor_finder(model) is orig


# -----------------------------------------------------------------------------
# 7) Filtering test: allow none should usually change the score
# -----------------------------------------------------------------------------

def tgn_filtering_delta(
    *,
    model: Any,
    u: int,
    v: int,
    t: float,
    edge_idx: int,
    n_neighbors: int,
    device: Optional[Any] = None,
) -> float:
    p_full = tgn_predict_proba(model, u, v, t, edge_idx, n_neighbors=n_neighbors, device=device)

    base_nf = _get_base_neighbor_finder(model)
    filtered_nf = FilteredNeighborFinder(
        base_nf,
        allowed_edge_idxs=set(),
        uniform=getattr(base_nf, "uniform", True),
    )

    with _swap_neighbor_finder(model, filtered_nf):
        p_none = tgn_predict_proba(model, u, v, t, edge_idx, n_neighbors=n_neighbors, device=device)

    return float(abs(_to_scalar(p_full) - _to_scalar(p_none)))


# -----------------------------------------------------------------------------
# 8) Monotonicity-ish check: more edges => closer to full prediction
# -----------------------------------------------------------------------------

def monotonicity_diffs(
    *,
    model: Any,
    explainer,
    neighbor_finder: AdjacencyListNeighborFinder,
    u: int,
    v: int,
    t: float,
    edge_idx: int,
    node_raw_features: np.ndarray,
    edge_raw_features: np.ndarray,
    top_ks: Sequence[int],
    n_neighbors: int,
    device: Optional[Any] = None,
) -> List[Tuple[int, float]]:
    p_full = tgn_predict_proba(model, u, v, t, edge_idx, n_neighbors=n_neighbors, device=device)
    base_nf = _get_base_neighbor_finder(model)
    out: List[Tuple[int, float]] = []

    for k in top_ks:
        exp = explainer.explain(
            neighbor_finder_for_sampling=neighbor_finder,
            u=int(u),
            v=int(v),
            t=float(t),
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            selection_mode="topk",
            top_k=int(k),
        )
        allowed = set(int(e.edge_idx) for e in exp.edges)
        filtered_nf = FilteredNeighborFinder(
            base_nf,
            allowed_edge_idxs=allowed,
            uniform=getattr(base_nf, "uniform", True),
        )
        with _swap_neighbor_finder(model, filtered_nf):
            p_k = tgn_predict_proba(model, u, v, t, edge_idx, n_neighbors=n_neighbors, device=device)
        diff = float(abs(_to_scalar(p_full) - _to_scalar(p_k)))
        out.append((int(k), diff))
    return out


# -----------------------------------------------------------------------------
# 9) Training sanity checks (requires your own logging)
# -----------------------------------------------------------------------------

def check_training_sanity(
    *,
    loss: float,
    ib_value: float,
    p_mean: float,
    p_min: float,
    p_max: float,
    selected_count: int,
) -> None:
    assert np.isfinite(float(loss))
    assert np.isfinite(float(ib_value))
    assert 0.0 <= float(p_min) <= float(p_max) <= 1.0
    assert 0.0 <= float(p_mean) <= 1.0
    assert int(selected_count) >= 0


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _has_history(neighbor_finder: Any, node: int, t: float) -> bool:
    if not hasattr(neighbor_finder, "find_before"):
        return True
    _neigh, _eidx, ts = neighbor_finder.find_before(int(node), float(t))
    return len(ts) > 0


def _to_scalar(x: Any) -> float:
    if torch is not None and torch.is_tensor(x):
        if x.numel() == 0:
            return float("nan")
        return float(x.detach().cpu().reshape(-1)[0].item())
    return float(x)


def _get_base_neighbor_finder(model: Any):
    for attr in ("neighbor_finder", "ngh_finder"):
        if hasattr(model, attr):
            return getattr(model, attr)
    if hasattr(model, "embedding_module") and hasattr(model.embedding_module, "neighbor_finder"):
        return model.embedding_module.neighbor_finder
    raise AttributeError("Could not locate neighbor_finder on model")


@contextmanager
def _swap_neighbor_finder(model: Any, neighbor_finder: Any):
    if hasattr(model, "set_neighbor_finder") and callable(getattr(model, "set_neighbor_finder")):
        orig = _get_base_neighbor_finder(model)
        model.set_neighbor_finder(neighbor_finder)
        try:
            yield
        finally:
            model.set_neighbor_finder(orig)
    else:
        with patch_neighbor_finder(model, neighbor_finder):
            yield
