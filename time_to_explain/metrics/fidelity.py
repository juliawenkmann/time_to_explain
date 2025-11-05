# time_to_explain/metrics/fidelity.py
from __future__ import annotations
from typing import Any, Mapping, Dict, List, Sequence, Optional
import math
import numpy as np

from time_to_explain.core.registry import register_metric
from time_to_explain.core.types import ExplanationContext, ExplanationResult, MetricResult
from time_to_explain.metrics.legacy.base import BaseMetric, MetricDirection


def _to_array(x: Sequence[float] | None) -> np.ndarray:
    if x is None:
        return np.empty((0,), dtype=float)
    return np.asarray(list(x), dtype=float)

def _to_scalar(pred: Any, *, result_as_logit: bool) -> float:
    """
    Convert model output (scalar, vector, torch/numpy) to a scalar.
    If result_as_logit=True, treat vectors as logits (softmax->max), scalars as logit (sigmoid).
    Otherwise, take max probability if vector, or float scalar.
    """
    try:
        import torch
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
    except Exception:
        pass

    arr = np.asarray(pred)
    if arr.ndim == 0:
        val = float(arr)
        if result_as_logit:
            # scalar logit -> prob
            return float(1.0 / (1.0 + math.exp(-val)))
        return val

    # vector
    if result_as_logit:
        # vector of logits -> softmax and take max class prob
        m = np.max(arr)
        ex = np.exp(arr - m)
        p = ex / np.sum(ex)
        return float(np.max(p))
    else:
        # assume already probabilities -> take max class prob
        return float(np.max(arr))


def _topk_mask(
    importance: np.ndarray,
    k: int,
    *,
    mode: str,                   # "drop" (fidelity_minus) or "keep" (sufficiency)
    normalize: str = "minmax",   # "minmax" | "none"
    by: str = "value"            # "value" | "abs"
) -> List[float]:
    """
    Build a 1=keep / 0=drop mask of same length as importance.
    - fidelity_minus ("drop"): zeros at top-k, ones elsewhere
    - fidelity_keep  ("keep"): ones at top-k, zeros elsewhere
    """
    n = importance.size
    if n == 0:
        return []

    x = importance.copy()
    if by == "abs":
        x = np.abs(x)
    if normalize == "minmax" and np.max(x) > np.min(x):
        x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-12)

    order = np.argsort(-x)  # descending
    k = max(0, min(k, n))
    idx_top = set(order[:k])

    if mode == "drop":
        # keep=1 everywhere except drop top-k
        mask = [0.0 if i in idx_top else 1.0 for i in range(n)]
    elif mode == "keep":
        mask = [1.0 if i in idx_top else 0.0 for i in range(n)]
    else:
        raise ValueError(f"Unknown mode={mode!r}")
    return mask


class FidelityAtKMetric(BaseMetric):
    """
    Fidelity-style metric at multiple K or sparsity levels:
      - mode="drop"  → fidelity_drop = |full_score - masked_score| after dropping top edges
      - mode="keep"  → fidelity_keep = |full_score - keep_only_score| after keeping the strongest edges

    Requirements:
      - context.subgraph.payload["candidate_eidx"] (ordering of candidate edges)
      - result.importance_edges aligned to the same order
      - model implements predict_proba(...) and predict_proba_with_mask(...),
        with edge_mask semantics: 1=keep, 0=drop (aligned to candidate_eidx)

    Config:
      topk: int | list[int]                     (drop/keep by absolute count)
      sparsity_levels: float | list[float]      (drop/keep by fraction 0–1)
      mode: "drop" | "keep"  (default "drop" → fidelity_drop)
      result_as_logit: bool  (default True; set False if model returns probabilities)
      normalize: "minmax" | "none"    (default "minmax" for ranking)
      by: "value" | "abs"             (default "value" for ranking)
    """
    def __init__(self, name: str, config: Mapping[str, Any] | None = None):
        cfg = dict(config or {})
        super().__init__(
            name=name,
            direction=MetricDirection.HIGHER_IS_BETTER,
            config=cfg,
        )
        self.use_levels = False
        self.sparsity_levels: Optional[List[float]] = None
        self.k_values: Optional[List[int]] = None

        levels = cfg.get("sparsity_levels") or cfg.get("levels")
        if levels is not None:
            if isinstance(levels, (float, int)):
                levels = [levels]
            lvl_list = []
            for lvl in levels:
                try:
                    lvl_f = float(lvl)
                except (TypeError, ValueError):
                    continue
                lvl_list.append(max(0.0, min(1.0, lvl_f)))
            self.sparsity_levels = sorted(set(lvl_list))
            self.use_levels = True
            self.output_keys = [f"@s={lvl:g}" for lvl in self.sparsity_levels]
        else:
            k_values = cfg.get("k") or cfg.get("topk") or [6, 12, 18]
            if isinstance(k_values, int):
                k_values = [k_values]
            self.k_values = sorted({int(k) for k in k_values if int(k) > 0})
            self.output_keys = [f"@{k}" for k in self.k_values]

        self.mode = cfg.get("mode", "drop")  # "drop" or "keep"
        self.result_as_logit = bool(cfg.get("result_as_logit", True))
        self.normalize = str(cfg.get("normalize", "minmax"))
        self.rank_by = str(cfg.get("by", "value"))

    # ---------------------------------------------------------------- interface #
    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        # 0) sanity & fetch
        imp = _to_array(result.importance_edges)
        cand = None
        if context.subgraph and context.subgraph.payload:
            cand = context.subgraph.payload.get("candidate_eidx")

        # No importances or no model API → cannot compute
        if imp.size == 0 or not hasattr(self.model, "predict_proba_with_mask"):
            return MetricResult(
                name=self.name,
                values={f"@{k}": float("nan") for k in self.k_values},
                direction=self.direction.value,
                run_id=context.run_id,
                explainer=result.explainer,
                context_fp=context.fingerprint(),
                extras={"reason": "missing_importance_or_model_api",
                        "has_importance": imp.size > 0,
                        "has_predict_proba_with_mask": hasattr(self.model, "predict_proba_with_mask")},
            )

        # 1) full score
        full = self.model.predict_proba(context.subgraph, context.target)
        full_s = _to_scalar(full, result_as_logit=self.result_as_logit)
        values: Dict[str, float] = {"prediction_full": full_s}

        # 2) compute deltas at each evaluation point (top-k or sparsity level)
        masked_scores: Dict[str, float] = {}
        delta_signed: Dict[str, float] = {}
        points_meta: Dict[str, Dict[str, float]] = {}

        n_edges = int(imp.size)
        eval_iter: List[tuple[str, int]] = []

        if self.use_levels and self.sparsity_levels is not None:
            for lvl, key_suffix in zip(self.sparsity_levels, self.output_keys):
                proportion = max(0.0, min(1.0, float(lvl)))
                raw_count = proportion * n_edges
                k_count = int(round(raw_count))
                if proportion > 0.0 and k_count == 0 and n_edges > 0:
                    k_count = 1
                k_count = max(0, min(k_count, n_edges))
                eval_iter.append((key_suffix, k_count))
                points_meta[key_suffix] = {
                    "kind": "sparsity",
                    "level": proportion,
                    "count": k_count,
                    "achieved_sparsity": float(k_count) / float(n_edges) if n_edges > 0 else 0.0,
                }
        elif self.k_values is not None:
            for k, key_suffix in zip(self.k_values, self.output_keys):
                k_count = max(0, min(int(k), n_edges))
                if k > 0 and k_count == 0 and n_edges > 0:
                    k_count = 1
                eval_iter.append((key_suffix, k_count))
                points_meta[key_suffix] = {
                    "kind": "topk",
                    "k": int(k),
                    "count": k_count,
                    "achieved_sparsity": float(k_count) / float(n_edges) if n_edges > 0 else 0.0,
                }
        else:
            eval_iter = []

        for key_suffix, k_count in eval_iter:
            mask = _topk_mask(
                imp, k_count,
                mode=self.mode,
                normalize=self.normalize,
                by=self.rank_by
            )
            masked = self.model.predict_proba_with_mask(
                context.subgraph, context.target, edge_mask=mask
            )
            masked_s = _to_scalar(masked, result_as_logit=self.result_as_logit)
            prediction_key = f"prediction_{self.mode}.{key_suffix}" if self.mode in ("drop", "keep") else f"prediction.{key_suffix}"
            masked_scores[key_suffix] = masked_s
            values[prediction_key] = masked_s
            signed = full_s - masked_s
            delta_signed[key_suffix] = signed
            values[key_suffix] = abs(signed)

        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=context.fingerprint(),
            extras={
                "mode": self.mode,
                "k_values": self.k_values,
                "sparsity_levels": self.sparsity_levels,
                "result_as_logit": self.result_as_logit,
                "normalize": self.normalize,
                "rank_by": self.rank_by,
                "n_importance": int(imp.size),
                "has_candidate_eidx": bool(cand is not None),
                "prediction_full": full_s,
                "masked_predictions": masked_scores,
                "delta_signed": delta_signed,
                "points": points_meta,
            },
        )


# ---------- registry factories (one style) ----------
@register_metric("fidelity_drop")
def build_fidelity_drop(config: Mapping[str, Any] | None = None):
    """
    Drop top-k edges (most important) and measure drop in score:
    fidelity_drop@k = |full_score - score_after_dropping_topk|
    """
    cfg = dict(config or {})
    cfg.setdefault("mode", "drop")
    return FidelityAtKMetric(name="fidelity_drop", config=cfg)


@register_metric("fidelity_minus")
def build_fidelity_minus(config: Mapping[str, Any] | None = None):
    """
    Backward-compatible alias for fidelity_drop.
    """
    return build_fidelity_drop(config)

@register_metric("fidelity_keep")
def build_fidelity_keep(config: Mapping[str, Any] | None = None):
    """
    Keep only top-k edges (sufficiency) and measure retained score:
    fidelity_keep@k = |full_score - score_with_only_topk_kept|
    """
    cfg = dict(config or {})
    cfg.setdefault("mode", "keep")
    return FidelityAtKMetric(name="fidelity_keep", config=cfg)



class FidelityBestMetric(FidelityAtKMetric):
    """
    Aggregate fidelity across multiple K by taking the *best* value
    according to the metric direction:

      - Regardless of mode, fidelity values are absolute differences,
        so larger is better. `fidelity_best` therefore selects the maximum
        across the requested @k values.

    The metric still reports the per-@k values (same as `fidelity_drop`
    / `fidelity_keep`) and adds:
      - `best`: the maximum absolute drop
      - `best.k`: which K achieved it

    Config is the same as `FidelityAtKMetric`.
    """

    def __init__(self, name: str = "fidelity_best", config: Mapping[str, Any] | None = None):
        super().__init__(name=name, config=config)

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        # First compute standard @k values using the parent implementation.
        base_res = super().compute(context, result)
        values = dict(base_res.values)

        # Choose best depending on direction
        best_val = None
        best_key = None
        points_meta = (base_res.extras or {}).get("points", {})

        for key in self.output_keys:
            v = values.get(key)
            # Skip NaNs or missing
            try:
                is_nan = v is None or (isinstance(v, float) and math.isnan(v))
            except Exception:
                is_nan = False
            if is_nan:
                continue

            if best_val is None or v > best_val:
                best_val, best_key = v, key

        if best_val is None:
            # no valid @k values
            values["best"] = float("nan")
            values["best.k"] = float("nan")
        else:
            values["best"] = float(best_val)
            info = points_meta.get(best_key, {})
            identifier = info.get("level")
            if identifier is None:
                identifier = info.get("k")
            values["best.k"] = float(identifier) if isinstance(identifier, (int, float)) else identifier

        # Reuse extras from base and annotate.
        extras = dict(getattr(base_res, "extras", {}) or {})
        extras.update({"best_key": best_key, "best_value": best_val})

        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=context.fingerprint(),
            extras=extras,
        )


@register_metric("fidelity_best")
def build_fidelity_best(config: Mapping[str, Any] | None = None):
    """
    Aggregate fidelity across multiple sparsity levels and return the maximum score,
    while still exposing the per-level values.
    """
    cfg = dict(config or {})
    # Keep parity with other fidelity metrics: default to 'drop'
    cfg.setdefault("mode", "drop")
    return FidelityBestMetric(name="fidelity_best", config=cfg)
