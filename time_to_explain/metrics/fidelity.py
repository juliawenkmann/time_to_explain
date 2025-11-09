# time_to_explain/metrics/fidelity.py
from __future__ import annotations
"""
Fidelity-style metrics for GNN explainability.

This module provides:
  - fidelity_drop (alias: fidelity_minus): drop top-k/top-s edges by importance and
    report |full - masked| (larger is better)
  - fidelity_keep: keep only the top-k/top-s edges and report |full - keep-only| (larger is better)
  - fidelity_best: aggregate across multiple k/sparsity levels by taking the maximum
  - fidelity_tempme: TEMP-ME-style fidelity across sparsity levels

Assumptions:
  - `result.importance_edges` aligns with `context.subgraph.payload["candidate_eidx"]`
  - Model exposes:
      * predict_proba(subgraph, target)
      * predict_proba_with_mask(subgraph, target, edge_mask=[0/1 floats aligned to candidate edges])
  - Edge mask semantics: 1 = keep, 0 = drop
"""

from typing import Any, Mapping, Dict, List, Sequence, Optional
import math
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


def _to_scalar(pred: Any, *, result_as_logit: bool) -> float:
    """
    Convert model output (scalar, vector, torch/numpy) to a scalar in [0,1] if `result_as_logit`
    (sigmoid/softmax), else pass through as probability.

    Rules:
      - 0D (scalar): if logits -> sigmoid; else -> float
      - 1D+ (vector): if logits -> softmax then take max prob; else -> take max prob
    """
    try:
        import torch  # noqa: F401
        if "torch" in str(type(pred)):
            pred = pred.detach().cpu().numpy()
    except Exception:
        pass

    arr = np.asarray(pred)
    if arr.ndim == 0:
        val = float(arr)
        if result_as_logit:
            return float(1.0 / (1.0 + math.exp(-val)))  # sigmoid
        return val

    # vector
    if result_as_logit:
        m = np.max(arr)
        ex = np.exp(arr - m)
        p = ex / (np.sum(ex) + 1e-12)
        return float(np.max(p))
    else:
        return float(np.max(arr))


def _topk_mask(
    importance: np.ndarray,
    k: int,
    *,
    mode: str,                   # "drop" or "keep"
    normalize: str = "minmax",   # "minmax" | "none"
    by: str = "value"            # "value" | "abs"
) -> List[float]:
    """
    Create a 1=keep / 0=drop mask of same length as `importance`.
      - mode="drop": zeros at top-k (drop them), ones elsewhere
      - mode="keep": ones at top-k (keep only those), zeros elsewhere
    """
    n = importance.size
    if n == 0:
        return []

    x = importance.copy()
    if by == "abs":
        x = np.abs(x)
    if normalize == "minmax" and np.max(x) > np.min(x):
        x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-12)

    order = np.argsort(-x)            # descending
    k = max(0, min(k, n))
    idx_top = set(order[:k])

    if mode == "drop":
        return [0.0 if i in idx_top else 1.0 for i in range(n)]
    if mode == "keep":
        return [1.0 if i in idx_top else 0.0 for i in range(n)]
    raise ValueError(f"Unknown mode={mode!r}")


# ----------------------------------------------------------------------------- #
# Fidelity @K / @S
# ----------------------------------------------------------------------------- #

class FidelityAtKMetric(BaseMetric):
    """
    Fidelity at multiple K or sparsity levels:

      - mode="drop": fidelity_drop = |full_score - score_after_dropping_topk|
      - mode="keep": fidelity_keep = |full_score - score_with_only_topk|

    Config:
      topk: int | list[int]
      sparsity_levels: float | list[float]        # each in [0,1]
      mode: "drop" | "keep"   (default "drop")
      result_as_logit: bool   (default True; set False if model returns probabilities)
      normalize: "minmax" | "none"  (default "minmax" for ranking)
      by: "value" | "abs"            (default "value" for ranking)

    Notes:
      - If `sparsity_levels` is provided, each level l is interpreted as a fraction of edges
        selected by ranking (top-l * |E|). For mode="drop" those are dropped; for "keep"
        only those are kept.
      - If `topk`/`k` is provided instead, we evaluate absolute counts.
    """

    def __init__(self, name: str, config: Mapping[str, Any] | None = None):
        cfg = dict(config or {})
        super().__init__(
            name=name,
            direction=MetricDirection.HIGHER_IS_BETTER,
            config=cfg,
        )
        self.use_levels: bool = False
        self.sparsity_levels: Optional[List[float]] = None
        self.k_values: Optional[List[int]] = None
        self.mode = str(cfg.get("mode", "drop"))  # "drop" or "keep"
        self.result_as_logit = bool(cfg.get("result_as_logit", True))
        self.normalize = str(cfg.get("normalize", "minmax"))
        self.rank_by = str(cfg.get("by", "value"))

        # Decide evaluation grid
        levels = cfg.get("sparsity_levels") or cfg.get("levels")
        if levels is not None:
            if isinstance(levels, (float, int)):
                levels = [levels]
            lvl_list: List[float] = []
            for lvl in levels:
                try:
                    f = float(lvl)
                except Exception:
                    continue
                lvl_list.append(max(0.0, min(1.0, f)))
            self.sparsity_levels = sorted(set(lvl_list))
            self.use_levels = True
            self.output_keys = [f"@s={lvl:g}" for lvl in self.sparsity_levels]
        else:
            k_values = cfg.get("k") or cfg.get("topk") or [6, 12, 18]
            if isinstance(k_values, int):
                k_values = [k_values]
            self.k_values = sorted({int(k) for k in k_values if int(k) > 0})
            self.output_keys = [f"@{k}" for k in self.k_values]

    # ---------------------------------------------------------------- interface #
    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        # 0) Sanity & fetch
        imp = _to_array(result.importance_edges)

        # No importances or no model API → cannot compute
        has_mask_api = hasattr(self.model, "predict_proba_with_mask")
        if imp.size == 0 or not has_mask_api:
            missing_keys = list(self.output_keys) if hasattr(self, "output_keys") else []
            values = {key: float("nan") for key in missing_keys}
            return MetricResult(
                name=self.name,
                values=values,
                direction=self.direction.value,
                run_id=context.run_id,
                explainer=result.explainer,
                context_fp=context.fingerprint(),
                extras={
                    "reason": "missing_importance_or_model_api",
                    "has_importance": imp.size > 0,
                    "has_predict_proba_with_mask": has_mask_api,
                    "n_importance": int(imp.size),
                    "mode": self.mode,
                    "sparsity_levels": self.sparsity_levels,
                    "k_values": self.k_values,
                },
            )

        # 1) Full score
        full = self.model.predict_proba(context.subgraph, context.target)
        full_s = _to_scalar(full, result_as_logit=self.result_as_logit)
        values: Dict[str, float] = {"prediction_full": full_s}

        # 2) Compute deltas at each evaluation point (top-k or sparsity level)
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

                # In "drop": sparsity means fraction removed; in "keep": sparsity means fraction removed = 1 - kept
                kept_frac = (k_count / float(n_edges)) if n_edges > 0 else 0.0
                sparsity_removed = (1.0 - kept_frac) if self.mode == "keep" else kept_frac

                points_meta[key_suffix] = {
                    "kind": "sparsity",
                    "level": proportion,                 # requested fraction of edges selected for ranking
                    "count": k_count,                    # number of edges selected (drop or keep)
                    "achieved_keep_frac": kept_frac if self.mode == "keep" else (1.0 - kept_frac),
                    "achieved_sparsity_removed": sparsity_removed,
                }
        elif self.k_values is not None:
            for k, key_suffix in zip(self.k_values, self.output_keys):
                k_count = max(0, min(int(k), n_edges))
                if k > 0 and k_count == 0 and n_edges > 0:
                    k_count = 1
                eval_iter.append((key_suffix, k_count))

                kept_frac = (k_count / float(n_edges)) if n_edges > 0 else 0.0
                sparsity_removed = (1.0 - kept_frac) if self.mode == "keep" else kept_frac

                points_meta[key_suffix] = {
                    "kind": "topk",
                    "k": int(k),
                    "count": k_count,
                    "achieved_keep_frac": kept_frac if self.mode == "keep" else (1.0 - kept_frac),
                    "achieved_sparsity_removed": sparsity_removed,
                }
        else:
            eval_iter = []

        for key_suffix, k_count in eval_iter:
            mask = _topk_mask(
                imp, k_count,
                mode=self.mode,
                normalize=self.normalize,
                by=self.rank_by,
            )
            masked = self.model.predict_proba_with_mask(
                context.subgraph, context.target, edge_mask=mask
            )
            masked_s = _to_scalar(masked, result_as_logit=self.result_as_logit)
            prediction_key = (
                f"prediction_{self.mode}.{key_suffix}" if self.mode in ("drop", "keep") else f"prediction.{key_suffix}"
            )
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
                "masked_predictions": masked_scores,
                "delta_signed": delta_signed,
                "points": points_meta,
            },
        )


# ---------- registry factories ----------
@register_metric("fidelity_drop")
def build_fidelity_drop(config: Mapping[str, Any] | None = None):
    """
    fidelity_drop@k/s = |full_score - score_after_dropping_topk_or_top_s|
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
    fidelity_keep@k/s = |full_score - score_with_only_topk_or_top_s_kept|
    """
    cfg = dict(config or {})
    cfg.setdefault("mode", "keep")
    return FidelityAtKMetric(name="fidelity_keep", config=cfg)


# ----------------------------------------------------------------------------- #
# Fidelity Best (aggregate)
# ----------------------------------------------------------------------------- #

class FidelityBestMetric(FidelityAtKMetric):
    """
    Aggregate fidelity across multiple k/sparsity levels by taking the maximum
    absolute difference (larger is better). Per-level values are still reported.
    """

    def __init__(self, name: str = "fidelity_best", config: Mapping[str, Any] | None = None):
        super().__init__(name=name, config=config)

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        base_res = super().compute(context, result)
        values = dict(base_res.values)

        best_val: Optional[float] = None
        best_key: Optional[str] = None
        points_meta = (base_res.extras or {}).get("points", {})

        for key in getattr(self, "output_keys", []):
            v = values.get(key)
            if v is None:
                continue
            if isinstance(v, float) and math.isnan(v):
                continue
            if best_val is None or float(v) > best_val:
                best_val, best_key = float(v), key

        if best_val is None:
            values["best"] = float("nan")
            values["best.k"] = float("nan")
        else:
            values["best"] = float(best_val)
            info = points_meta.get(best_key, {})
            identifier = info.get("level", info.get("k"))
            values["best.k"] = float(identifier) if isinstance(identifier, (int, float)) else identifier

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
    Aggregate over multiple k/sparsity levels and return the maximum fidelity value.
    """
    cfg = dict(config or {})
    cfg.setdefault("mode", "drop")  # default parity with other fidelity metrics
    return FidelityBestMetric(name="fidelity_best", config=cfg)


# ----------------------------------------------------------------------------- #
# TEMP-ME Fidelity
# ----------------------------------------------------------------------------- #

class FidelityTempmeMetric(BaseMetric):
    """
    TEMP-ME-style fidelity evaluated across sparsity levels s = |G_e_exp| / |G(e)|:

        Fid(G, G_e_exp) = 1(Y_f[e] = 1) * (f(G_e_exp)[e] - f(G)[e])
                        + 1(Y_f[e] = 0) * (f(G)[e] - f(G_e_exp)[e])

    For each sparsity s, we KEEP the top-s portion of edges (by importance ranking) and
    report the TEMP-ME fidelity value at that s.
    """

    def __init__(self, name: str = "fidelity_tempme", config: Mapping[str, Any] | None = None):
        cfg = dict(config or {})
        super().__init__(
            name=name,
            direction=MetricDirection.HIGHER_IS_BETTER,
            config=cfg,
        )

        levels = cfg.get("sparsity_levels") or cfg.get("levels")
        if levels is None:
            levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        lvl_list: List[float] = []
        for lvl in levels:
            try:
                f = float(lvl)
            except Exception:
                continue
            lvl_list.append(max(0.0, min(1.0, f)))
        self.sparsity_levels = sorted(set(lvl_list))
        if not self.sparsity_levels:
            raise ValueError("fidelity_tempme requires at least one sparsity level within [0, 1].")

        self.output_keys = [f"@s={lvl:g}" for lvl in self.sparsity_levels]
        self.normalize = str(cfg.get("normalize", "minmax"))
        self.rank_by = str(cfg.get("by", "value"))
        self.result_as_logit = bool(cfg.get("result_as_logit", True))
        self.label_threshold = float(cfg.get("label_threshold", 0.5))

    @staticmethod
    def _resolve_label(context: ExplanationContext, default_score: float, threshold: float) -> int:
        """
        Resolve a binary label (0/1) from the context if present; otherwise derive from
        the model score using `threshold`.
        """
        label_sources = [
            getattr(context, "label", None),
            context.target.get("label") if isinstance(context.target, dict) else None,
            (context.meta or {}).get("label") if context.meta else None,
        ]
        for candidate in label_sources:
            if candidate is None:
                continue
            try:
                return 1 if int(candidate) == 1 else 0
            except Exception:
                try:
                    return 1 if float(candidate) >= 0.5 else 0
                except Exception:
                    continue
        return 1 if default_score >= threshold else 0

    # ---------------------------------------------------------------- interface #
    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        imp = _to_array(result.importance_edges)
        n_edges = int(imp.size)
        has_api = hasattr(self.model, "predict_proba_with_mask")

        if n_edges == 0 or not has_api:
            values = {key: float("nan") for key in self.output_keys}
            return MetricResult(
                name=self.name,
                values=values,
                direction=self.direction.value,
                run_id=context.run_id,
                explainer=result.explainer,
                context_fp=context.fingerprint(),
                extras={
                    "reason": "missing_importance_or_model_api",
                    "n_importance": n_edges,
                    "has_predict_proba_with_mask": has_api,
                    "sparsity_levels": self.sparsity_levels,
                },
            )

        full_pred = self.model.predict_proba(context.subgraph, context.target)
        full_score = _to_scalar(full_pred, result_as_logit=self.result_as_logit)
        label = self._resolve_label(context, full_score, self.label_threshold)

        values: Dict[str, float] = {}
        masked_scores: Dict[str, float] = {}
        delta_signed: Dict[str, float] = {}
        points_meta: Dict[str, Dict[str, float]] = {}

        for lvl, key in zip(self.sparsity_levels, self.output_keys):
            s = float(max(0.0, min(1.0, lvl)))
            keep_count = int(round(s * n_edges))
            if s > 0.0 and keep_count == 0 and n_edges > 0:
                keep_count = 1
            keep_count = max(0, min(keep_count, n_edges))

            mask = _topk_mask(
                imp,
                keep_count,
                mode="keep",
                normalize=self.normalize,
                by=self.rank_by,
            )
            masked_pred = self.model.predict_proba_with_mask(
                context.subgraph,
                context.target,
                edge_mask=mask,
            )
            masked_score = _to_scalar(masked_pred, result_as_logit=self.result_as_logit)
            masked_scores[key] = masked_score
            delta_signed[key] = masked_score - full_score

            # TEMP-ME definition
            if label == 1:
                fidelity_value = masked_score - full_score
            else:
                fidelity_value = full_score - masked_score

            values[key] = float(fidelity_value)

            achieved_keep = keep_count / float(n_edges) if n_edges > 0 else 0.0
            points_meta[key] = {
                "requested_sparsity_keep_frac": s,
                "achieved_keep_frac": achieved_keep,
                "achieved_sparsity_removed": 1.0 - achieved_keep,
                "kept_edges": keep_count,
                "sparsity_percent": achieved_keep * 100.0,
            }

        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=context.fingerprint(),
            extras={
                "full_score": full_score,
                "full_label": label,
                "label_threshold": self.label_threshold,
                "normalize": self.normalize,
                "rank_by": self.rank_by,
                "result_as_logit": self.result_as_logit,
                "sparsity_levels": self.sparsity_levels,
                "masked_scores": masked_scores,
                "delta_signed": delta_signed,
                "points": points_meta,
            },
        )


@register_metric("fidelity_tempme")
def build_fidelity_tempme(config: Mapping[str, Any] | None = None):
    """TEMP-ME fidelity evaluated across multiple sparsity levels."""
    return FidelityTempmeMetric(name="fidelity_tempme", config=config)
