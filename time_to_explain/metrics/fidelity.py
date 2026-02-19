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
  - acc_auc: prediction match AUC over sparsity
  - prediction_profile: predictions vs sparsity
  - monotonicity: Spearman rank between importance and per-edge impact
  - temgx_fidelity_minus/plus + temgx_sparsity
  - singular_value: largest singular value vs sparsity

Assumptions:
  - `result.importance_edges` aligns with `context.subgraph.payload["candidate_eidx"]`
  - Model exposes:
      * predict_proba(subgraph, target)
      * predict_proba_with_mask(subgraph, target, edge_mask=[0/1 floats aligned to candidate edges])
  - Edge mask semantics: 1 = keep, 0 = drop
"""

from typing import Any, Mapping, Dict, List, Sequence, Optional, Tuple
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


def _parse_levels(levels: Any) -> List[float]:
    if levels is None:
        return []
    if isinstance(levels, (float, int)):
        levels = [levels]
    out: List[float] = []
    for lvl in levels:
        try:
            out.append(max(0.0, min(1.0, float(lvl))))
        except Exception:
            continue
    return sorted(set(out))


def _score_and_label(pred: Any, *, result_as_logit: bool, label_threshold: float) -> Tuple[float, Optional[int]]:
    arr = pred
    try:
        import torch  # noqa: F401
        if "torch" in str(type(arr)):
            arr = arr.detach().cpu().numpy()
    except Exception:
        pass
    arr = np.asarray(arr)

    if arr.size == 0:
        return float("nan"), None

    if arr.ndim == 0:
        val = float(arr)
        score = float(1.0 / (1.0 + math.exp(-val))) if result_as_logit else float(val)
        label = int(score >= label_threshold) if np.isfinite(score) else None
        return score, label

    arr = np.asarray(arr, dtype=float).reshape(-1)
    if result_as_logit:
        mx = float(np.max(arr))
        ex = np.exp(arr - mx)
        probs = ex / (float(np.sum(ex)) + 1e-12)
    else:
        probs = arr
        total = float(np.sum(probs))
        if not np.isclose(total, 1.0) and total > 0:
            probs = probs / (total + 1e-12)

    idx = int(np.argmax(probs))
    score = float(probs[idx])
    return score, idx


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
            diff = abs(signed)

            if self.mode == "drop":
                # dropping important edges should change prediction a lot
                values[key_suffix] = diff
            elif self.mode == "keep":
                # keeping only important edges should preserve prediction
                # assuming scores are in [0,1] (true if result_as_logit=True)
                values[key_suffix] = 1.0 - diff
            else:
                values[key_suffix] = diff


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


# ----------------------------------------------------------------------------- #
# ACC-AUC
# ----------------------------------------------------------------------------- #


class AccAucMetric(BaseMetric):
    """
    ACC-AUC: area under the curve of *prediction match rate* vs sparsity.
    """

    def __init__(self, name: str = "acc_auc", config: Mapping[str, Any] | None = None):
        cfg = dict(config or {})
        super().__init__(name=name, direction=MetricDirection.HIGHER_IS_BETTER, config=cfg)

        s_min = float(cfg.get("s_min", 0.0))
        s_max = float(cfg.get("s_max", 0.3))
        num_points = int(cfg.get("num_points", 16))
        if num_points <= 1:
            self.sparsity_levels = [s_min, s_max]
        else:
            self.sparsity_levels = [float(x) for x in np.linspace(s_min, s_max, num_points)]
        self.output_keys = [f"acc@s={lvl:g}" for lvl in self.sparsity_levels]

        self.result_as_logit = bool(cfg.get("result_as_logit", True))
        self.normalize = str(cfg.get("normalize", "minmax"))
        self.rank_by = str(cfg.get("by", "value"))
        self.label_threshold = float(cfg.get("label_threshold", 0.5))
        self.normalize_auc = bool(cfg.get("normalize_auc", True))

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        imp = _to_array(result.importance_edges)
        n_edges = int(imp.size)
        has_api = hasattr(self.model, "predict_proba_with_mask")

        if n_edges == 0 or not has_api:
            values = {key: float("nan") for key in self.output_keys + ["auc"]}
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
        full_score, full_label = _score_and_label(
            full_pred, result_as_logit=self.result_as_logit, label_threshold=self.label_threshold
        )

        values: Dict[str, float] = {}
        matches: List[float] = []
        levels: List[float] = []

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
            masked_pred = self.model.predict_proba_with_mask(context.subgraph, context.target, edge_mask=mask)
            _, masked_label = _score_and_label(
                masked_pred, result_as_logit=self.result_as_logit, label_threshold=self.label_threshold
            )
            match = float(masked_label == full_label) if masked_label is not None and full_label is not None else float("nan")
            values[key] = match
            if np.isfinite(match):
                matches.append(match)
                levels.append(s)

        if matches and len(levels) > 1:
            area = float(np.trapz(matches, levels))
            if self.normalize_auc:
                span = float(max(levels) - min(levels))
                area = area / span if span > 0 else float("nan")
        else:
            area = float("nan")
        values["auc"] = area

        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=context.fingerprint(),
            extras={
                "sparsity_levels": self.sparsity_levels,
                "full_label": full_label,
                "full_score": full_score,
                "matches": matches,
            },
        )


@register_metric("acc_auc")
def build_acc_auc(config: Mapping[str, Any] | None = None):
    cfg = dict(config or {})
    return AccAucMetric(name="acc_auc", config=cfg)


# ----------------------------------------------------------------------------- #
# Prediction profile
# ----------------------------------------------------------------------------- #


class PredictionProfileMetric(BaseMetric):
    """Track model predictions as edges are pruned by sparsity/Top-K."""

    def __init__(self, name: str = "prediction_profile", config: Mapping[str, Any] | None = None):
        cfg = dict(config or {})
        super().__init__(name=name, direction=MetricDirection.HIGHER_IS_BETTER, config=cfg)

        self.mode = str(cfg.get("mode", "keep"))  # "keep" retains top edges, "drop" removes them
        self.result_as_logit = bool(cfg.get("result_as_logit", True))
        self.normalize = str(cfg.get("normalize", "minmax"))
        self.rank_by = str(cfg.get("by", "value"))
        self.include_delta = bool(cfg.get("include_delta", True))
        self.include_abs_delta = bool(cfg.get("include_abs_delta", True))
        self.store_edge_details = bool(cfg.get("store_edge_details", True))
        edge_limit = cfg.get("edge_detail_limit")
        self.edge_detail_limit: Optional[int] = None if edge_limit in (None, "none") else int(edge_limit)
        if self.edge_detail_limit is not None and self.edge_detail_limit <= 0:
            self.edge_detail_limit = None
        self.store_masks = bool(cfg.get("store_masks", False))
        self.label_threshold = float(cfg.get("label_threshold", 0.5))
        self.emit_match_flags = bool(cfg.get("emit_match_flags", True))

        self.use_levels: bool = False
        self.sparsity_levels: Optional[List[float]] = None
        self.k_values: Optional[List[int]] = None
        self.output_keys: List[str] = []

        levels = cfg.get("sparsity_levels") or cfg.get("levels")
        if levels is not None:
            if isinstance(levels, (int, float)):
                levels = [levels]
            lvl_values: List[float] = []
            for lvl in levels:
                try:
                    lvl_values.append(max(0.0, min(1.0, float(lvl))))
                except Exception:
                    continue
            self.sparsity_levels = sorted(set(lvl_values)) if lvl_values else [0.1, 0.2, 0.3]
            self.use_levels = True
            self.output_keys = [f"@s={lvl:g}" for lvl in self.sparsity_levels]
        else:
            k_values = cfg.get("k") or cfg.get("topk") or [5, 10, 25]
            if isinstance(k_values, int):
                k_values = [k_values]
            self.k_values = sorted({int(k) for k in k_values if int(k) > 0}) or [5, 10, 25]
            self.output_keys = [f"@{k}" for k in self.k_values]

    def _eval_points(self, n_edges: int) -> List[Dict[str, Any]]:
        points: List[Dict[str, Any]] = []
        if n_edges <= 0:
            return points

        if self.use_levels and self.sparsity_levels is not None:
            for lvl, key in zip(self.sparsity_levels, self.output_keys):
                fraction = max(0.0, min(1.0, float(lvl)))
                raw = fraction * n_edges
                count = int(round(raw))
                if fraction > 0 and count == 0:
                    count = 1
                count = max(0, min(count, n_edges))
                kept_frac = (count / float(n_edges)) if n_edges > 0 else 0.0
                sparsity_removed = (1.0 - kept_frac) if self.mode == "keep" else kept_frac
                points.append({
                    "key": key,
                    "count": count,
                    "level": fraction,
                    "kind": "sparsity",
                    "achieved_keep_frac": kept_frac if self.mode == "keep" else (1.0 - kept_frac),
                    "achieved_sparsity_removed": sparsity_removed,
                })
        elif self.k_values is not None:
            for k, key in zip(self.k_values, self.output_keys):
                count = max(0, min(int(k), n_edges))
                if k > 0 and count == 0:
                    count = 1
                kept_frac = (count / float(n_edges)) if n_edges > 0 else 0.0
                sparsity_removed = (1.0 - kept_frac) if self.mode == "keep" else kept_frac
                points.append({
                    "key": key,
                    "count": count,
                    "k": int(k),
                    "kind": "topk",
                    "achieved_keep_frac": kept_frac if self.mode == "keep" else (1.0 - kept_frac),
                    "achieved_sparsity_removed": sparsity_removed,
                })
        return points

    def _select_indices(self, mask: Sequence[float], *, select: str) -> List[int]:
        if select == "kept":
            return [idx for idx, val in enumerate(mask) if val >= 0.5]
        return [idx for idx, val in enumerate(mask) if val < 0.5]

    def _edge_records(
        self,
        indices: Sequence[int],
        candidate_ids: Optional[Sequence[int]],
        candidate_edges: Optional[Sequence[Sequence[int]]],
        candidate_times: Optional[Sequence[float]],
        importances: Sequence[float],
    ) -> List[Dict[str, Any]]:
        if not indices:
            return []
        records: List[Dict[str, Any]] = []
        limit = self.edge_detail_limit
        for local_pos in indices:
            rec: Dict[str, Any] = {
                "position": int(local_pos),
                "importance": float(importances[local_pos]) if local_pos < len(importances) else float("nan"),
            }
            if candidate_ids is not None and local_pos < len(candidate_ids):
                rec["candidate_eidx"] = int(candidate_ids[local_pos])
            if candidate_edges is not None and local_pos < len(candidate_edges):
                rec["edge"] = candidate_edges[local_pos]
            if candidate_times is not None and local_pos < len(candidate_times):
                rec["timestamp"] = float(candidate_times[local_pos])
            records.append(rec)
            if limit is not None and len(records) >= limit:
                break
        return records

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        subgraph = context.subgraph
        if subgraph is None:
            return MetricResult(
                name=self.name,
                values={"prediction_full": float("nan")},
                direction=self.direction.value,
                run_id=context.run_id,
                explainer=result.explainer,
                context_fp=context.fingerprint(),
                extras={"reason": "missing_subgraph"},
            )

        imp = _to_array(result.importance_edges)
        values: Dict[str, float] = {}
        extras: Dict[str, Any] = {
            "mode": self.mode,
            "sparsity_levels": self.sparsity_levels,
            "k_values": self.k_values,
            "result_as_logit": self.result_as_logit,
        }

        full_pred = self.model.predict_proba(context.subgraph, context.target) if hasattr(self.model, "predict_proba") else float("nan")
        full_score, full_label = _score_and_label(
            full_pred, result_as_logit=self.result_as_logit, label_threshold=self.label_threshold
        )
        values["prediction_full"] = full_score
        extras["prediction_full"] = full_score
        extras["label_full"] = full_label
        if full_label is not None:
            values["label_full"] = full_label

        if imp.size == 0:
            extras.update({
                "reason": "missing_importance",
                "n_importance": 0,
            })
            return MetricResult(
                name=self.name,
                values=values,
                direction=self.direction.value,
                run_id=context.run_id,
                explainer=result.explainer,
                context_fp=context.fingerprint(),
                extras=extras,
            )

        candidate_ids = None
        candidate_edges = None
        candidate_times = None
        if subgraph.payload:
            candidate_ids = subgraph.payload.get("candidate_eidx")
            candidate_edges = subgraph.payload.get("candidate_edge_index") or subgraph.payload.get("edge_index")
            candidate_times = subgraph.payload.get("candidate_edge_times")

        for point in self._eval_points(int(imp.size)):
            mask = _topk_mask(
                imp,
                point["count"],
                mode=self.mode,
                normalize=self.normalize,
                by=self.rank_by,
            )
            masked_pred = self.model.predict_proba_with_mask(context.subgraph, context.target, edge_mask=mask)
            masked_score, masked_label = _score_and_label(
                masked_pred, result_as_logit=self.result_as_logit, label_threshold=self.label_threshold
            )

            key = point["key"]
            prediction_key = f"prediction_{self.mode}.{key}"
            values[prediction_key] = masked_score
            extras[prediction_key] = masked_score

            if self.include_delta:
                values[f"delta_{self.mode}.{key}"] = masked_score - full_score
            if self.include_abs_delta:
                values[f"delta_abs_{self.mode}.{key}"] = abs(masked_score - full_score)
            if self.emit_match_flags:
                match = float(masked_label == full_label) if masked_label is not None and full_label is not None else float("nan")
                values[f"match_{self.mode}.{key}"] = match

            details: Dict[str, Any] = {
                "requested_level": point.get("level"),
                "requested_k": point.get("k"),
                "achieved_keep_frac": point.get("achieved_keep_frac"),
                "achieved_sparsity_removed": point.get("achieved_sparsity_removed"),
                "prediction": masked_score,
                "label": masked_label,
            }
            if self.store_masks:
                details["edge_mask"] = list(mask)

            if self.store_edge_details:
                kept_idx = self._select_indices(mask, select="kept")
                dropped_idx = self._select_indices(mask, select="dropped")
                details["edges_kept"] = self._edge_records(
                    kept_idx, candidate_ids, candidate_edges, candidate_times, imp
                )
                details["edges_dropped"] = self._edge_records(
                    dropped_idx, candidate_ids, candidate_edges, candidate_times, imp
                )

            extras.setdefault("points", {})[key] = details

        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=context.fingerprint(),
            extras=extras,
        )


@register_metric("prediction_profile")
def build_prediction_profile(config: Mapping[str, Any] | None = None):
    cfg = dict(config or {})
    return PredictionProfileMetric(name="prediction_profile", config=cfg)


# ----------------------------------------------------------------------------- #
# Monotonicity
# ----------------------------------------------------------------------------- #


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)

    unique, idx_start, counts = np.unique(x, return_index=True, return_counts=True)
    for start, cnt in zip(idx_start, counts):
        if cnt > 1:
            idxs = order[start:start + cnt]
            avg = np.mean(ranks[idxs])
            ranks[idxs] = avg
    return ranks


def _spearmanr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    ra = _rankdata(a)
    rb = _rankdata(b)
    da = ra - ra.mean()
    db = rb - rb.mean()
    denom = np.sqrt(np.sum(da ** 2)) * np.sqrt(np.sum(db ** 2))
    if denom == 0:
        return float("nan")
    return float(np.sum(da * db) / denom)


class MonotonicityMetric(BaseMetric):
    def __init__(self, name: str, config: Mapping[str, Any] | None = None):
        cfg = dict(config or {})
        super().__init__(name=name, direction=MetricDirection.HIGHER_IS_BETTER, config=cfg)
        self.mode = str(cfg.get("mode", "drop"))  # "drop" | "keep"
        self.result_as_logit = bool(cfg.get("result_as_logit", True))

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        imp = np.asarray(result.importance_edges or [], dtype=float)
        candidate = None
        if context.subgraph and context.subgraph.payload:
            candidate = context.subgraph.payload.get("candidate_eidx")

        has_mask_api = hasattr(self.model, "predict_proba_with_mask")
        if imp.size == 0 or candidate is None or not has_mask_api:
            values = {"spearman_rho": float("nan")}
            extras = {
                "reason": "missing_importance_or_candidates_or_model_api",
                "has_importance": imp.size > 0,
                "has_candidates": candidate is not None,
                "has_predict_proba_with_mask": has_mask_api,
                "n_importance": int(imp.size),
                "n_candidates": len(candidate) if candidate is not None else 0,
                "mode": self.mode,
            }
            return MetricResult(
                name=self.name,
                values=values,
                direction=self.direction.value,
                run_id=context.run_id,
                explainer=result.explainer,
                context_fp=context.fingerprint(),
                extras=extras,
            )

        n = imp.size
        full = self.model.predict_proba(context.subgraph, context.target)
        full_s = _to_scalar(full, result_as_logit=self.result_as_logit)

        impacts: List[float] = []
        for idx in range(n):
            mask = [1.0] * n
            if self.mode == "drop":
                mask[idx] = 0.0
                masked = self.model.predict_proba_with_mask(
                    context.subgraph, context.target, edge_mask=mask
                )
                val = full_s - _to_scalar(masked, result_as_logit=self.result_as_logit)
            elif self.mode == "keep":
                mask = [0.0] * n
                mask[idx] = 1.0
                masked = self.model.predict_proba_with_mask(
                    context.subgraph, context.target, edge_mask=mask
                )
                val = _to_scalar(masked, result_as_logit=self.result_as_logit)
            else:
                raise ValueError(f"Unknown mode={self.mode!r}")
            impacts.append(val)

        impacts_arr = np.asarray(impacts, dtype=float)
        rho = _spearmanr(imp, impacts_arr)

        values = {"spearman_rho": rho}
        extras = {
            "mode": self.mode,
            "n_candidates": n,
            "full_score": full_s,
        }

        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=context.fingerprint(),
            extras=extras,
        )


@register_metric("monotonicity")
def build_monotonicity(config: Mapping[str, Any] | None = None):
    return MonotonicityMetric(name="monotonicity", config=config)


# ----------------------------------------------------------------------------- #
# TemGX fidelity + sparsity
# ----------------------------------------------------------------------------- #


def _aufsc(points: Sequence[Tuple[float, float]], *, max_sparsity: float = 1.0, n_grid: int = 101) -> float:
    pts = [(float(s), float(v)) for s, v in points if np.isfinite(s) and np.isfinite(v)]
    if not pts:
        return 0.0

    s = np.asarray([p[0] for p in pts], dtype=float)
    f = np.asarray([p[1] for p in pts], dtype=float)
    max_sparsity = float(max_sparsity)
    if max_sparsity <= 0:
        return 0.0
    s = np.clip(s, 0.0, max_sparsity)

    grid = np.linspace(0.0, max_sparsity, int(n_grid))
    y = np.zeros_like(grid)

    order = np.argsort(s)
    s_sorted = s[order]
    f_sorted = f[order]
    prefix_sum = np.cumsum(f_sorted)

    idx = 0
    for i, t in enumerate(grid):
        while idx < s_sorted.size and s_sorted[idx] <= t:
            idx += 1
        y[i] = 0.0 if idx == 0 else float(prefix_sum[idx - 1]) / float(idx)

    area = float(np.trapz(y, grid))
    return area / max_sparsity


class TemGXFidelityMetric(BaseMetric):
    """
    TemGX fidelity metrics over sparsity levels.
    """

    def __init__(self, name: str, config: Mapping[str, Any] | None = None):
        cfg = dict(config or {})
        super().__init__(name=name, direction=MetricDirection.HIGHER_IS_BETTER, config=cfg)

        self.mode = str(cfg.get("mode", "minus"))
        self.result_as_logit = bool(cfg.get("result_as_logit", True))
        self.normalize = str(cfg.get("normalize", "minmax"))
        self.rank_by = str(cfg.get("by", "value"))

        levels = _parse_levels(cfg.get("sparsity_levels") or cfg.get("levels"))
        self.use_levels = bool(levels)
        if self.use_levels:
            self.sparsity_levels = levels
            self.output_keys = [f"@s={lvl:g}" for lvl in self.sparsity_levels]
            self.k_values = None
        else:
            k_values = cfg.get("k") or cfg.get("topk") or [5, 10, 25]
            if isinstance(k_values, int):
                k_values = [k_values]
            self.k_values = sorted({int(k) for k in k_values if int(k) >= 0})
            self.output_keys = [f"@{k}" for k in self.k_values]
            self.sparsity_levels = None

    def _iter_points(self, n_edges: int) -> List[Dict[str, float]]:
        points: List[Dict[str, float]] = []
        if n_edges <= 0:
            return points
        if self.use_levels and self.sparsity_levels is not None:
            for lvl, key in zip(self.sparsity_levels, self.output_keys):
                fraction = max(0.0, min(1.0, float(lvl)))
                raw = fraction * n_edges
                count = int(round(raw))
                if fraction > 0.0 and count == 0:
                    count = 1
                count = max(0, min(count, n_edges))
                points.append({
                    "key": key,
                    "count": count,
                    "sparsity": (count / float(n_edges)) if n_edges else 0.0,
                })
        elif self.k_values is not None:
            for k, key in zip(self.k_values, self.output_keys):
                count = max(0, min(int(k), n_edges))
                if k > 0 and count == 0:
                    count = 1
                points.append({
                    "key": key,
                    "count": count,
                    "sparsity": (count / float(n_edges)) if n_edges else 0.0,
                })
        return points

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        imp = _to_array(result.importance_edges)
        n_edges = int(imp.size)
        has_api = hasattr(self.model, "predict_proba_with_mask")

        if n_edges == 0 or not has_api:
            values = {key: float("nan") for key in self.output_keys}
            values["aufsc"] = float("nan")
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
                    "k_values": self.k_values,
                },
            )

        full = self.model.predict_proba(context.subgraph, context.target)
        full_s = _to_scalar(full, result_as_logit=self.result_as_logit)

        values: Dict[str, float] = {"prediction_full": float(full_s)}
        points: List[Tuple[float, float]] = []

        mask_mode = "drop" if self.mode == "minus" else "keep"
        for point in self._iter_points(n_edges):
            mask = _topk_mask(
                imp,
                int(point["count"]),
                mode=mask_mode,
                normalize=self.normalize,
                by=self.rank_by,
            )
            masked_pred = self.model.predict_proba_with_mask(context.subgraph, context.target, edge_mask=mask)
            masked_s = _to_scalar(masked_pred, result_as_logit=self.result_as_logit)

            if self.mode == "minus":
                val = abs(full_s - masked_s)
            else:
                val = 1.0 - abs(masked_s - full_s)

            values[point["key"]] = float(val)
            points.append((float(point["sparsity"]), float(val)))

        values["aufsc"] = _aufsc(points)

        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=context.fingerprint(),
            extras={
                "mode": self.mode,
                "sparsity_levels": self.sparsity_levels,
                "k_values": self.k_values,
            },
        )


class TemGXSparsityMetric(BaseMetric):
    """TemGX sparsity: |E_expl| / |E_Lhop|."""

    def __init__(self, config: Mapping[str, Any] | None = None):
        cfg = dict(config or {})
        super().__init__(name="temgx_sparsity", direction=MetricDirection.LOWER_IS_BETTER, config=cfg)
        self.prefer_coalition = bool(cfg.get("prefer_coalition", True))
        self.fallback_to_levels = bool(cfg.get("fallback_to_levels", False))
        self.sparsity_levels = _parse_levels(cfg.get("sparsity_levels") or cfg.get("levels"))
        self.use_levels = bool(self.sparsity_levels)
        if self.use_levels:
            self.output_keys = [f"@s={lvl:g}" for lvl in self.sparsity_levels]
        else:
            self.output_keys = []

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        payload = context.subgraph.payload if context.subgraph else None
        candidate = None
        if payload and payload.get("candidate_eidx") is not None:
            candidate = payload.get("candidate_eidx")
        if candidate is None and result.extras:
            candidate = result.extras.get("candidate_eidx")
        n_edges = len(candidate) if candidate else len(_to_array(result.importance_edges))

        values: Dict[str, float] = {}
        coalition = None
        if self.prefer_coalition and result.extras:
            coalition = result.extras.get("coalition_eidx")

        if coalition is not None:
            ratio = len(coalition) / float(max(1, n_edges))
            values["ratio"] = float(ratio)
        elif self.fallback_to_levels and self.use_levels and n_edges > 0:
            for lvl, key in zip(self.sparsity_levels, self.output_keys):
                count = int(round(float(lvl) * n_edges))
                if lvl > 0.0 and count == 0:
                    count = 1
                count = max(0, min(count, n_edges))
                values[key] = float(count / float(n_edges))
        else:
            values["ratio"] = float("nan")

        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=context.fingerprint(),
            extras={
                "n_candidate": int(n_edges),
                "used_coalition": coalition is not None,
            },
        )


@register_metric("temgx_fidelity_minus")
def build_temgx_fidelity_minus(config: Mapping[str, Any] | None = None):
    cfg = dict(config or {})
    cfg["mode"] = "minus"
    return TemGXFidelityMetric(name="temgx_fidelity_minus", config=cfg)


@register_metric("temgx_fidelity_plus")
def build_temgx_fidelity_plus(config: Mapping[str, Any] | None = None):
    cfg = dict(config or {})
    cfg["mode"] = "plus"
    return TemGXFidelityMetric(name="temgx_fidelity_plus", config=cfg)


@register_metric("temgx_sparsity")
def build_temgx_sparsity(config: Mapping[str, Any] | None = None):
    return TemGXSparsityMetric(config)


# ----------------------------------------------------------------------------- #
# Singular value
# ----------------------------------------------------------------------------- #


def _normalize_importance(values: np.ndarray, *, normalize: str, by: str) -> np.ndarray:
    out = values.astype(float, copy=True)
    if by == "abs":
        out = np.abs(out)
    if normalize == "minmax":
        v_min = float(np.min(out)) if out.size else 0.0
        v_max = float(np.max(out)) if out.size else 0.0
        if v_max > v_min:
            out = (out - v_min) / (v_max - v_min + 1e-12)
    return out


def _extract_edges(context: ExplanationContext, result: ExplanationResult) -> List[Sequence[int]]:
    edges = None
    if context.subgraph is not None:
        if context.subgraph.payload:
            payload = context.subgraph.payload
            edges = payload.get("candidate_edge_index") or payload.get("edge_index")
        if edges is None and context.subgraph.edge_index:
            edges = context.subgraph.edge_index
    if edges is None and result.extras:
        edges = result.extras.get("candidate_edge_index") or result.extras.get("edge_index")
    return list(edges or [])


class SingularValueMetric(BaseMetric):
    """Largest singular value over sparsity levels for the explanation subgraph."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        cfg = dict(config or {})
        super().__init__(
            name="singular_value",
            direction=MetricDirection.LOWER_IS_BETTER,
            config=cfg,
        )
        self.sparsity_levels = _parse_levels(cfg.get("sparsity_levels") or cfg.get("levels"))
        self.output_keys = [f"@s={lvl:g}" for lvl in self.sparsity_levels]
        self.normalize = str(cfg.get("normalize", "minmax"))
        self.rank_by = str(cfg.get("by", "value"))
        self.weighted = bool(cfg.get("weighted", False))
        self.symmetrize = bool(cfg.get("symmetrize", True))
        self.summary = list(cfg.get("summary", []))

    def _iter_counts(self, n_edges: int) -> List[int]:
        counts = []
        for lvl in self.sparsity_levels:
            raw = float(lvl) * n_edges
            count = int(round(raw))
            if lvl > 0.0 and count == 0:
                count = 1
            counts.append(max(0, min(count, n_edges)))
        return counts

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        edges = _extract_edges(context, result)
        importance = _to_array(result.importance_edges)
        n_edges = min(len(edges), int(importance.size))

        if n_edges == 0:
            values = {key: float("nan") for key in self.output_keys}
            for key in self.summary:
                values[key] = float("nan")
            return MetricResult(
                name=self.name,
                values=values,
                direction=self.direction.value,
                run_id=context.run_id,
                explainer=result.explainer,
                context_fp=context.fingerprint(),
                extras={"reason": "missing_edges_or_importance"},
            )

        edges = edges[:n_edges]
        importance = importance[:n_edges]
        weights = _normalize_importance(importance, normalize=self.normalize, by=self.rank_by)

        nodes = sorted({int(u) for u, v in edges} | {int(v) for u, v in edges})
        node_index = {node: idx for idx, node in enumerate(nodes)}
        n_nodes = len(nodes)

        values: Dict[str, float] = {}
        for key, count in zip(self.output_keys, self._iter_counts(n_edges)):
            mask = _topk_mask(
                importance,
                int(count),
                mode="keep",
                normalize=self.normalize,
                by=self.rank_by,
            )
            adj = np.zeros((n_nodes, n_nodes), dtype=float)
            for idx, (u, v) in enumerate(edges):
                if idx >= len(mask) or not mask[idx]:
                    continue
                w = weights[idx] if self.weighted else 1.0
                ui = node_index[int(u)]
                vi = node_index[int(v)]
                adj[ui, vi] += float(w)
                if self.symmetrize:
                    adj[vi, ui] += float(w)

            if n_nodes == 0:
                values[key] = float("nan")
                continue
            svals = np.linalg.svd(adj, compute_uv=False)
            values[key] = float(svals[0]) if len(svals) else float("nan")

        finite_vals = [v for v in values.values() if np.isfinite(v)]
        if finite_vals:
            if "mean" in self.summary:
                values["mean"] = float(np.mean(finite_vals))
            if "max" in self.summary:
                values["max"] = float(np.max(finite_vals))
            if "min" in self.summary:
                values["min"] = float(np.min(finite_vals))
        else:
            for key in self.summary:
                values[key] = float("nan")

        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=context.fingerprint(),
            extras={
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "weighted": self.weighted,
                "symmetrize": self.symmetrize,
            },
        )


@register_metric("singular_value")
def build_singular_value(config: Mapping[str, Any] | None = None):
    return SingularValueMetric(config)
