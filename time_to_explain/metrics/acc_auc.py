# time_to_explain/metrics/acc_auc.py
from __future__ import annotations
from typing import Any, Dict, List, Mapping
import numpy as np

from time_to_explain.core.registry import register_metric
from time_to_explain.core.types import ExplanationContext, ExplanationResult, MetricResult
from time_to_explain.core.metrics import BaseMetric, MetricDirection
from time_to_explain.metrics.fidelity import FidelityTempmeMetric


class AccAucMetric(BaseMetric):
    """
    Fidelity AUC computed directly from the TEMP-ME curve.

    This metric reuses `FidelityTempmeMetric` to obtain fidelity values at multiple
    sparsity levels (|G_e^exp| / |G(e)|) and integrates that curve using the
    trapezoidal rule. It therefore mirrors the TEMP-ME definition exactly while
    providing an aggregated scalar (AUC) for ranking.

    Config:
      - sparsity_levels: list[float]       # explicit list in [0, s_max]
        OR
        - s_min: float = 0.0
        - s_max: float = 0.3
        - num_points: int = 16            # number of points between s_min..s_max inclusive
      - result_as_logit: bool = True      # interpret model output as logits
      - normalize: "minmax" | "none"      # ranking normalization (for importance)
      - by: "value" | "abs"               # rank by raw or |value|
      - normalize_auc: bool = False       # divide AUC by (s_max - s_min) to map to [0,1]
      - label_threshold: float = 0.5      # fallback threshold for determining Y_f[e]
    """

    def __init__(self, name: str = "acc_auc", config: Mapping[str, Any] | None = None):
        cfg = dict(config or {})
        super().__init__(
            name=name,
            direction=MetricDirection.HIGHER_IS_BETTER,
            config=cfg,
        )

        # Sparsity schedule
        levels = cfg.get("sparsity_levels")
        s_min = float(cfg.get("s_min", 0.0))
        s_max = float(cfg.get("s_max", 0.3))
        num_points = int(cfg.get("num_points", 16))

        if levels is not None:
            lev = []
            for v in levels:
                try:
                    f = float(v)
                except Exception:
                    continue
                # clamp to [0, s_max] (typical definition uses s_max=0.3)
                lev.append(max(0.0, min(s_max, f)))
            self.sparsity_levels: List[float] = sorted(set(lev))
        else:
            s_min = max(0.0, s_min)
            s_max = max(s_min, s_max)
            num_points = max(2, num_points)
            self.sparsity_levels = list(np.linspace(s_min, s_max, num_points))

        # Output key names for the fidelity curve (keep historical "acc" prefix for compatibility)
        self.output_keys = [f"acc@s={lvl:g}" for lvl in self.sparsity_levels]

        # Behavior
        self.result_as_logit = bool(cfg.get("result_as_logit", True))
        self.normalize = str(cfg.get("normalize", "minmax"))
        self.rank_by = str(cfg.get("by", "value"))
        self.normalize_auc = bool(cfg.get("normalize_auc", False))
        self.label_threshold = float(cfg.get("label_threshold", 0.5))

        tempme_cfg = {
            "sparsity_levels": self.sparsity_levels,
            "result_as_logit": self.result_as_logit,
            "normalize": self.normalize,
            "by": self.rank_by,
            "label_threshold": self.label_threshold,
        }
        self._tempme_metric = FidelityTempmeMetric(name=f"{name}_tempme_curve", config=tempme_cfg)

    def setup(self, model: Any, dataset: Any) -> None:
        super().setup(model=model, dataset=dataset)
        if hasattr(self._tempme_metric, "setup"):
            try:
                self._tempme_metric.setup(model=model, dataset=dataset)
            except TypeError:
                self._tempme_metric.setup(model, dataset)

    # ---------------------------------------------------------------- interface #
    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        tempme_result = self._tempme_metric.compute(context, result)

        curve_values: Dict[str, float] = {}
        for acc_key, temp_key in zip(self.output_keys, self._tempme_metric.output_keys):
            val = tempme_result.values.get(temp_key, float("nan"))
            try:
                curve_values[acc_key] = float(val)
            except (TypeError, ValueError):
                curve_values[acc_key] = float("nan")

        s_levels = np.asarray(self.sparsity_levels, dtype=float)
        fidelity_curve = np.asarray([curve_values[k] for k in self.output_keys], dtype=float)

        if fidelity_curve.size == 0:
            auc = float("nan")
        else:
            auc = float(np.trapz(fidelity_curve, s_levels))
            if self.normalize_auc:
                denom = float(max(s_levels) - min(s_levels)) if len(s_levels) > 1 else 1.0
                if denom > 0:
                    auc = auc / denom

        values: Dict[str, float] = {"auc": auc}
        values.update(curve_values)

        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=context.fingerprint(),
            extras={
                "result_as_logit": self.result_as_logit,
                "normalize": self.normalize,
                "rank_by": self.rank_by,
                "normalize_auc": self.normalize_auc,
                "sparsity_levels": self.sparsity_levels,
                "sparsity_levels_percent": [lvl * 100.0 for lvl in self.sparsity_levels],
                "source_metric": "fidelity_tempme",
                "tempme_values": tempme_result.values,
                "tempme_extras": tempme_result.extras,
            },
        )


# ---------- registry factory (matches EvaluationRunner factory style) ----------
@register_metric("acc_auc")
def build_acc_auc(config: Mapping[str, Any] | None = None):
    """
    Factory for fidelity AUC that EvaluationRunner will call as:
        factory(dict_config) -> BaseMetric
    """
    cfg = dict(config or {})
    # Default to 'keep' semantics, which is the common fidelity-ACC definition
    cfg.setdefault("mode", "keep")
    # Defaults for the standard range [0, 0.3]
    cfg.setdefault("s_min", 0.0)
    cfg.setdefault("s_max", 0.3)
    cfg.setdefault("num_points", 16)
    return AccAucMetric(name="acc_auc", config=cfg)
