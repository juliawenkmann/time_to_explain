from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from time_to_explain.core.registries import register_metric
from time_to_explain.core.types import ExplanationContext, ExplanationResult, MetricResult
from time_to_explain.metrics.base import BaseMetric, MetricDirection


def _to_float(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return _to_float(item)
            except (TypeError, ValueError):
                continue
        raise TypeError(f"Cannot convert {value!r} to float")
    if hasattr(value, "detach"):
        return float(value.detach().cpu().item())
    if hasattr(value, "item") and callable(value.item):
        return float(value.item())
    return float(value)


class FidelityMinusMetric(BaseMetric):
    def __init__(self, config: Mapping[str, Any] | None = None):
        cfg = dict(config or {})
        super().__init__(
            name="fidelity_minus",
            direction=MetricDirection.LOWER_IS_BETTER,
            config=cfg,
        )
        k_values = cfg.get("k") or cfg.get("topk") or [6, 12, 18]
        if isinstance(k_values, int):
            k_values = [k_values]
        self.k_values = sorted({int(k) for k in k_values if int(k) > 0})
        self.result_as_logit = bool(cfg.get("result_as_logit", True))

    # ----------------------------------------------------------------- helpers #
    def _extract_original_prediction(self, context: ExplanationContext, result: ExplanationResult) -> float:
        for mapping in (result.primary.metadata, result.statistics):
            for key in ("original_score", "original_prediction", "original_prob"):
                if key in mapping:
                    try:
                        return float(mapping[key])
                    except (TypeError, ValueError):
                        continue
        raw = result.raw
        if raw is not None:
            for key in ("original_score", "original_prediction"):
                if hasattr(raw, key):
                    try:
                        return float(getattr(raw, key))
                    except (TypeError, ValueError):
                        continue
        prediction = context.model.predict_event(context.event_id, result_as_logit=self.result_as_logit)
        return _to_float(prediction)

    def _compute_prediction(self, context: ExplanationContext, edges_to_drop: Iterable[int]) -> float:
        model_result = context.model.compute_edge_probabilities_for_subgraph(
            context.event_id,
            edges_to_drop=np.array(list(edges_to_drop), dtype=int),
            result_as_logit=self.result_as_logit,
        )
        return _to_float(model_result)

    # ---------------------------------------------------------------- interface #
    def compute(self, context: ExplanationContext, result: ExplanationResult):
        original_prediction = self._extract_original_prediction(context, result)
        edge_ids = list(map(int, result.primary.edge_ids))
        metric_results: list[MetricResult] = []

        for k in self.k_values:
            topk_edges = edge_ids[:k]
            if not topk_edges:
                continue
            masked_prediction = self._compute_prediction(context, topk_edges)
            fidelity_minus = original_prediction - masked_prediction
            metric_results.append(
                MetricResult(
                    name=f"{self.name}@{k}",
                    value=fidelity_minus,
                    event_id=context.event_id,
                    explainer_name=result.explainer_name,
                    details={
                        "k": k,
                        "original_prediction": original_prediction,
                        "masked_prediction": masked_prediction,
                    },
                )
            )
        if not metric_results:
            metric_results.append(
                MetricResult(
                    name=self.name,
                    value=float("nan"),
                    event_id=context.event_id,
                    explainer_name=result.explainer_name,
                    details={"reason": "no_explanation_edges"},
                )
            )
        return metric_results


@register_metric("fidelity_minus")
def build_fidelity_minus(cfg: Mapping[str, Any] | None = None):
    return FidelityMinusMetric(cfg)
