# time_to_explain/metrics/cohesiveness.py
from __future__ import annotations
from typing import Any, Mapping, Dict, List, Sequence, Optional, Tuple, Callable
import numpy as np

from time_to_explain.core.registry import register_metric
from time_to_explain.core.types import ExplanationContext, ExplanationResult, MetricResult
from time_to_explain.core.metrics import BaseMetric, MetricDirection


# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #

def _to_array(x: Sequence[float] | None) -> np.ndarray:
    if x is None:
        return np.empty((0,), dtype=float)
    return np.asarray(list(x), dtype=float)


def _rank_order(importance: np.ndarray, *, normalize: str = "minmax", by: str = "value") -> np.ndarray:
    """
    Return indices that would sort importance in DESC order, using the same ranking
    conventions used elsewhere (minmax-normalize for ranking, rank by |value| if requested).
    """
    x = importance.copy()
    if by == "abs":
        x = np.abs(x)
    if normalize == "minmax" and np.max(x) > np.min(x):
        x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-12)
    return np.argsort(-x)  # descending


def _ensure_2xn_or_nx2_edges(arr: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accepts edge index in one of these forms and returns (src, dst) as 1D arrays:
      - shape (2, E)  -> PyG style
      - shape (E, 2)  -> list-of-pairs style
    Raises ValueError if not parseable.
    """
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError("edge index array must be 2D")
    if a.shape[0] == 2:
        src, dst = a[0], a[1]
    elif a.shape[1] == 2:
        src, dst = a[:, 0], a[:, 1]
    else:
        raise ValueError("edge index must have shape (2, E) or (E, 2)")
    return np.asarray(src).astype(np.int64), np.asarray(dst).astype(np.int64)


def _resolve_times_and_endpoints(
    context: ExplanationContext,
    *,
    get_edge_times: Optional[Callable[[ExplanationContext, ExplanationResult], Sequence[float]]] = None,
    get_edge_endpoints: Optional[Callable[[ExplanationContext, ExplanationResult], Tuple[Sequence[int], Sequence[int]]]] = None,
    edge_time_key: Optional[str] = None,
    candidate_endpoints_key: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resolve arrays for candidate-edge times and endpoints, aligned to result.importance_edges:
      - times: shape (N,)
      - src:   shape (N,)
      - dst:   shape (N,)

    Priority:
      1) Provided callables (get_edge_times / get_edge_endpoints)
      2) Payload keys for candidate arrays (edge_time_key, candidate_endpoints_key)
      3) Derive from full graph arrays + candidate indices if available
    """
    payload = getattr(context.subgraph, "payload", {}) or {}

    # --- TIMES ---
    times_arr = None
    if callable(get_edge_times):
        try:
            times_arr = np.asarray(list(get_edge_times(context, None)), dtype=float)
        except TypeError:
            # If the user implemented with a single-arg signature
            times_arr = np.asarray(list(get_edge_times(context)), dtype=float)
    if times_arr is None and edge_time_key:
        if edge_time_key in payload:
            times_arr = np.asarray(payload[edge_time_key], dtype=float)

    if times_arr is None:
        # Try a set of common keys for candidate-edge times
        for key in ("candidate_edge_times", "candidate_edge_timestamps", "edge_times", "edge_timestamps", "timestamps", "t", "edge_ts"):
            if key in payload:
                times_arr = np.asarray(payload[key], dtype=float)
                break

    # --- ENDPOINTS (candidate) ---
    src_arr = None
    dst_arr = None
    if callable(get_edge_endpoints):
        try:
            s, d = get_edge_endpoints(context, None)
        except TypeError:
            s, d = get_edge_endpoints(context)
        src_arr = np.asarray(list(s), dtype=np.int64)
        dst_arr = np.asarray(list(d), dtype=np.int64)

    if src_arr is None or dst_arr is None:
        key = candidate_endpoints_key or "candidate_edge_index"
        if key in payload:
            src_arr, dst_arr = _ensure_2xn_or_nx2_edges(payload[key])

    # --- Fallback: derive candidate endpoints from full edge_index + candidate indices ---
    if (src_arr is None or dst_arr is None) and "candidate_eidx" in payload:
        cand_idx = np.asarray(payload["candidate_eidx"], dtype=np.int64)
        # Seek a global edge_index; slice it by candidate indices if it aligns
        for key in ("edge_index", "edges", "all_edge_index"):
            if key in payload:
                full_src, full_dst = _ensure_2xn_or_nx2_edges(payload[key])
                src_arr, dst_arr = full_src[cand_idx], full_dst[cand_idx]
                break

    # Final checks
    if times_arr is None or src_arr is None or dst_arr is None:
        raise KeyError("Could not resolve candidate-edge times and/or endpoints from context payload or config.")

    if not (len(times_arr) == len(src_arr) == len(dst_arr)):
        raise ValueError("Resolved times and endpoints have mismatched lengths.")

    return times_arr.astype(float), src_arr.astype(np.int64), dst_arr.astype(np.int64)


# ----------------------------------------------------------------------------- #
# Cohesiveness Metric
# ----------------------------------------------------------------------------- #

class CohesivenessMetric(BaseMetric):
    """
    Cohesiveness over sparsity levels s (keep top-s edges by importance):

        Coh(s) = [1 / (m^2 - m)] * sum_{i != j} cos(|t_i - t_j| / ΔT) * 1(e_i ~ e_j),
        where m = |G_e^exp(s)| is the number of kept edges at level s and "~" means
        "share at least one endpoint" (undirected adjacency).

    Config:
      - sparsity_levels: float | list[float]   # s ∈ (0,1], default [0.05, 0.1, 0.2, 0.3]
        OR
      - k / topk: int | list[int]             # absolute edge counts

      - delta_T: float | None                 # normalizer in cos(|Δt| / ΔT); if None, uses
                                              # max(candidate_times) - min(candidate_times)
      - normalize: "minmax" | "none"          # ranking normalization for importance (default "minmax")
      - by: "value" | "abs"                   # rank edges by raw or absolute importance (default "value")

      - edge_time_key: str | None             # payload key for candidate-edge times (if not using callable)
      - candidate_endpoints_key: str | None   # payload key for candidate edge index (2xN or Nx2)
      - get_edge_times: callable(context[, result]) -> Sequence[float]
      - get_edge_endpoints: callable(context[, result]) -> (Sequence[int], Sequence[int])

    Returns:
      MetricResult with `values` containing Cohesiveness at each configured @s or @k.
    """

    def __init__(self, name: str = "cohesiveness", config: Mapping[str, Any] | None = None):
        cfg = dict(config or {})
        super().__init__(
            name=name,
            direction=MetricDirection.HIGHER_IS_BETTER,  # more cohesive is better
            config=cfg,
        )

        # Ranking behaviour for selecting the explanation set
        self.normalize = str(cfg.get("normalize", "minmax"))
        self.rank_by = str(cfg.get("by", "value"))

        # Sparsity or K
        levels = cfg.get("sparsity_levels") or cfg.get("levels")
        self.use_levels: bool = levels is not None
        if self.use_levels:
            if isinstance(levels, (float, int)):
                levels = [levels]
            lvl_list: List[float] = []
            for lvl in levels:
                try:
                    f = float(lvl)
                except Exception:
                    continue
                f = max(0.0, min(1.0, f))
                lvl_list.append(f)
            self.sparsity_levels = sorted({v for v in lvl_list if v > 0.0})
            if not self.sparsity_levels:
                # sensible defaults
                self.sparsity_levels = [0.05, 0.1, 0.2, 0.3]
            self.output_keys = [f"@s={lvl:g}" for lvl in self.sparsity_levels]
            self.k_values = None
        else:
            k_values = cfg.get("k") or cfg.get("topk")
            if k_values is None:
                k_values = [6, 12, 18]
            if isinstance(k_values, int):
                k_values = [k_values]
            self.k_values = sorted({int(k) for k in k_values if int(k) > 0})
            self.output_keys = [f"@{k}" for k in self.k_values]
            self.sparsity_levels = None

        # Time normalization
        self.delta_T_cfg = cfg.get("delta_T", None)  # if None, we infer per sample

        # How to fetch times and endpoints
        self.edge_time_key = cfg.get("edge_time_key", None)
        self.candidate_endpoints_key = cfg.get("candidate_endpoints_key", None)
        self.get_edge_times = cfg.get("get_edge_times", None)
        self.get_edge_endpoints = cfg.get("get_edge_endpoints", None)

    # ---------------------------------------------------------------- interface #
    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        # Importance ranking (aligned to candidate edges)
        imp = _to_array(result.importance_edges)
        n_edges = int(imp.size)

        # Resolve times and endpoints for candidate edges
        try:
            times_all, src_all, dst_all = _resolve_times_and_endpoints(
                context,
                get_edge_times=self.get_edge_times,
                get_edge_endpoints=self.get_edge_endpoints,
                edge_time_key=self.edge_time_key,
                candidate_endpoints_key=self.candidate_endpoints_key,
            )
        except Exception as e:
            # Cannot compute without times and endpoints
            missing_vals = {key: float("nan") for key in getattr(self, "output_keys", [])}
            return MetricResult(
                name=self.name,
                values=missing_vals,
                direction=self.direction.value,
                run_id=context.run_id,
                explainer=result.explainer,
                context_fp=context.fingerprint(),
                extras={
                    "reason": f"missing_times_or_endpoints: {type(e).__name__}: {e}",
                    "n_importance": n_edges,
                },
            )

        if n_edges == 0:
            missing_vals = {key: float("nan") for key in getattr(self, "output_keys", [])}
            return MetricResult(
                name=self.name,
                values=missing_vals,
                direction=self.direction.value,
                run_id=context.run_id,
                explainer=result.explainer,
                context_fp=context.fingerprint(),
                extras={"reason": "no_candidate_edges", "n_importance": 0},
            )

        # Rank order once (shared across evaluation points)
        order = _rank_order(imp, normalize=self.normalize, by=self.rank_by)

        # ΔT normalizer (constant across all s for this sample unless provided)
        if self.delta_T_cfg is not None:
            delta_T = float(self.delta_T_cfg)
        else:
            tmin = float(np.min(times_all))
            tmax = float(np.max(times_all))
            delta_T = max(1e-12, tmax - tmin)  # avoid division by zero
        delta_T_used = float(delta_T)

        values: Dict[str, float] = {}
        points_meta: Dict[str, Dict[str, float]] = {}

        # Build evaluation grid
        eval_points: List[Tuple[str, int, float]] = []  # (key, keep_count, requested_level)
        if self.use_levels and self.sparsity_levels is not None:
            for s, key in zip(self.sparsity_levels, self.output_keys):
                keep = int(round(s * n_edges))
                if s > 0.0 and keep == 0 and n_edges > 0:
                    keep = 1
                keep = max(0, min(keep, n_edges))
                eval_points.append((key, keep, s))
        elif self.k_values is not None:
            for k, key in zip(self.k_values, self.output_keys):
                keep = max(0, min(int(k), n_edges))
                if k > 0 and keep == 0 and n_edges > 0:
                    keep = 1
                eval_points.append((key, keep, float("nan")))

        # Evaluate cohesiveness at each point
        for key, keep_count, requested_level in eval_points:
            if keep_count < 2:
                values[key] = float("nan")
                points_meta[key] = {
                    "requested_level": requested_level,
                    "kept_edges": keep_count,
                    "achieved_keep_frac": (keep_count / float(n_edges)) if n_edges > 0 else 0.0,
                    "delta_T": delta_T_used,
                    "pairs_total_ordered": float(keep_count * (keep_count - 1)),
                    "pairs_adj_ordered": 0.0,
                    "reason": "insufficient_edges_for_pairs",
                }
                continue

            sel = order[:keep_count]
            ts = times_all[sel].astype(float)
            src = src_all[sel]
            dst = dst_all[sel]
            m = int(len(ts))

            # Pairwise time diffs (ordered pairs; diagonals excluded later)
            dt = np.abs(ts[:, None] - ts[None, :]) / delta_T_used
            cos_mat = np.cos(dt)

            # Undirected adjacency: share any endpoint
            adj = (
                (src[:, None] == src[None, :]) |
                (src[:, None] == dst[None, :]) |
                (dst[:, None] == src[None, :]) |
                (dst[:, None] == dst[None, :])
            )

            # Exclude self-pairs (i == j)
            eye = np.eye(m, dtype=bool)
            mask = adj & (~eye)

            # Sum over ordered i != j
            num_adj_pairs = float(mask.sum())
            denom = float(m * (m - 1))  # ordered pairs
            numer = float((cos_mat * mask).sum())
            value = numer / denom if denom > 0 else float("nan")

            values[key] = value
            achieved_keep_frac = keep_count / float(n_edges) if n_edges > 0 else 0.0
            points_meta[key] = {
                "requested_level": requested_level,
                "kept_edges": keep_count,
                "achieved_keep_frac": achieved_keep_frac,
                "delta_T": delta_T_used,
                "pairs_total_ordered": denom,
                "pairs_adj_ordered": num_adj_pairs,
                "adjacency_mode": "share_node_undirected",
            }

        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=context.fingerprint(),
            extras={
                "normalize": self.normalize,
                "rank_by": self.rank_by,
                "n_importance": int(n_edges),
                "delta_T": delta_T_used,
                "sparsity_levels": getattr(self, "sparsity_levels", None),
                "k_values": getattr(self, "k_values", None),
                "points": points_meta,
            },
        )


# ---------- registry factory ----------
@register_metric("cohesiveness")
def build_cohesiveness(config: Mapping[str, Any] | None = None):
    return CohesivenessMetric(name="cohesiveness", config=config)
