from __future__ import annotations

"""Minimal built-in metrics used by the evaluation runner.

Active metrics:
- `sparsity`       = |E_expl| / |E_candidates|
- `fidelity_minus` = |f(G) - f(G \\ E_expl)|
- `fidelity_plus`  = 1 - |f(G) - f(E_expl)|
- `aufsc`          = area under cumulative mean fidelity vs sparsity
"""

from collections.abc import Iterable, Sequence
from typing import Callable

import numpy as np

from ..core.registry import register_metric
from ..core.types import ExplanationContext, ExplanationResult, MetricResult
from .base import BaseMetric, MetricDirection
from .selection import (
    importance_for_candidates,
    rank_candidate_positions,
    select_candidate_eidx,
)


def fidelity_minus(z_full: float, z_removed: float) -> float:
    """Fidelity- = |f(G) - f(G \\ E_expl)|."""
    return abs(float(z_full) - float(z_removed))


def fidelity_plus(z_full: float, z_expl: float) -> float:
    """Fidelity+ = 1 - |f(G) - f(E_expl)|."""
    return 1.0 - abs(float(z_full) - float(z_expl))


def sparsity(num_edges_expl: int, num_edges_candidates: int) -> float:
    """Sparsity ratio = |E_expl| / |E_candidates|."""
    return float(num_edges_expl) / float(max(1, num_edges_candidates))


def aufsc(
    points: Iterable[tuple[float, float]],
    *,
    max_sparsity: float = 1.0,
    n_grid: int = 101,
) -> float:
    """
    AUFSC: area under cumulative mean fidelity vs sparsity.
    y(t) = mean(fid_i for s_i <= t), then integrate y on [0, max_sparsity].
    """
    valid_points = [(float(s), float(v)) for s, v in points if np.isfinite(s) and np.isfinite(v)]
    if not valid_points or max_sparsity <= 0:
        return 0.0

    s = np.asarray([point[0] for point in valid_points], dtype=float)
    f = np.asarray([point[1] for point in valid_points], dtype=float)
    s = np.clip(s, 0.0, float(max_sparsity))

    order = np.argsort(s)
    s_sorted = s[order]
    f_prefix = np.cumsum(f[order])

    grid = np.linspace(0.0, float(max_sparsity), int(n_grid))
    y = np.zeros_like(grid, dtype=float)

    seen = 0
    for i, threshold in enumerate(grid):
        while seen < s_sorted.size and s_sorted[seen] <= threshold:
            seen += 1
        y[i] = 0.0 if seen == 0 else float(f_prefix[seen - 1]) / float(seen)

    area = float(np.trapz(y, grid))
    return area / float(max_sparsity)


def candidate_eidx_from(context: ExplanationContext, result: ExplanationResult) -> list[int]:
    payload = context.subgraph.payload if context.subgraph else None
    if isinstance(payload, dict) and "candidate_eidx" in payload:
        return [int(eidx) for eidx in payload["candidate_eidx"]]
    return [int(eidx) for eidx in result.extras.get("candidate_eidx", [])]


def explanation_eidx_from(
    result: ExplanationResult,
    candidate_eidx: Sequence[int],
    *,
    k: int | None,
    context: ExplanationContext | None = None,
    model: object | None = None,
    config: dict | None = None,
) -> list[int]:
    n = len(candidate_eidx)
    if n <= 0:
        return []

    importance = importance_for_candidates(result.importance_edges, n)
    order, _ = rank_candidate_positions(
        model=model if hasattr(model, "predict_proba_with_mask") else None,
        context=context,
        result=result,
        candidate_eidx=candidate_eidx,
        importance=importance,
        config=config,
    )
    return select_candidate_eidx(
        result=result,
        candidate_eidx=candidate_eidx,
        order=order,
        importance=importance,
        k=k,
        config=config,
    )


def edge_mask_keep(candidate_eidx: Sequence[int], keep_eidx: Sequence[int]) -> list[float]:
    keep_set = {int(eidx) for eidx in keep_eidx}
    return [1.0 if int(eidx) in keep_set else 0.0 for eidx in candidate_eidx]


def edge_mask_drop(candidate_eidx: Sequence[int], drop_eidx: Sequence[int]) -> list[float]:
    drop_set = {int(eidx) for eidx in drop_eidx}
    return [0.0 if int(eidx) in drop_set else 1.0 for eidx in candidate_eidx]


def _resolve_k(config: dict, n_candidates: int) -> int:
    if config.get("k") is not None:
        return int(config["k"])
    if config.get("topk") is not None:
        return int(config["topk"])
    if config.get("sparsity") is not None:
        return int(round(float(config["sparsity"]) * float(n_candidates)))
    return int(n_candidates)


def _resolve_k_max(config: dict, n_candidates: int) -> int:
    if config.get("k_max") is not None:
        return int(config["k_max"])
    return int(n_candidates)


def _resolve_sparsity_levels(config: dict) -> list[float]:
    levels = config.get("sparsity_levels", config.get("levels"))
    if levels is None:
        return [0.0, 0.25, 0.5, 0.75, 1.0]
    if isinstance(levels, (int, float)):
        return [float(levels)]
    if len(levels) == 0:
        return [0.0, 0.25, 0.5, 0.75, 1.0]
    return [float(level) for level in levels]


def _k_from_level(level: float, *, k_max: int, n_candidates: int, ensure_min_one: bool = False) -> int:
    k = int(round(float(level) * float(k_max)))
    if ensure_min_one and level > 0.0 and k == 0 and n_candidates > 0:
        return 1
    return max(0, min(k, n_candidates))


def _predict_with_selection(
    metric: BaseMetric,
    context: ExplanationContext,
    candidate_eidx: Sequence[int],
    selected_eidx: Sequence[int],
    *,
    keep_only_selected: bool,
) -> float:
    edge_mask = (
        edge_mask_keep(candidate_eidx, selected_eidx)
        if keep_only_selected
        else edge_mask_drop(candidate_eidx, selected_eidx)
    )
    prediction = metric.model.predict_proba_with_mask(
        context.subgraph,
        context.target,
        edge_mask=edge_mask,
    )
    return float(prediction)


def _compute_fidelity_values(
    metric: BaseMetric,
    context: ExplanationContext,
    result: ExplanationResult,
    *,
    keep_only_selected: bool,
    score_fn: Callable[[float, float], float],
) -> dict[str, float]:
    candidate_eidx = candidate_eidx_from(context, result)
    n_candidates = len(candidate_eidx)
    has_levels = metric.config.get("sparsity_levels") is not None or metric.config.get("levels") is not None
    has_explicit_value_budget = any(metric.config.get(key) is not None for key in ("k", "topk", "sparsity"))
    value_sparsity = metric.config.get("value_sparsity")

    z_full = float(metric.model.predict_proba(context.subgraph, context.target))
    values: dict[str, float] = {}

    if (not has_levels) or has_explicit_value_budget:
        k = _resolve_k(metric.config, n_candidates)
        selected_eidx = explanation_eidx_from(
            result,
            candidate_eidx,
            k=k,
            context=context,
            model=metric.model,
            config=metric.config,
        )
        z_masked = _predict_with_selection(
            metric,
            context,
            candidate_eidx,
            selected_eidx,
            keep_only_selected=keep_only_selected,
        )
        values["value"] = float(score_fn(z_full, z_masked))
    elif value_sparsity is not None:
        k_max_value = _resolve_k_max(metric.config, n_candidates)
        k_value = _k_from_level(
            float(value_sparsity),
            k_max=k_max_value,
            n_candidates=n_candidates,
            ensure_min_one=True,
        )
        selected_value = explanation_eidx_from(
            result,
            candidate_eidx,
            k=k_value,
            context=context,
            model=metric.model,
            config=metric.config,
        )
        z_value = _predict_with_selection(
            metric,
            context,
            candidate_eidx,
            selected_value,
            keep_only_selected=keep_only_selected,
        )
        values["value"] = float(score_fn(z_full, z_value))

    if not has_levels:
        return values

    k_max = _resolve_k_max(metric.config, n_candidates)
    for level in _resolve_sparsity_levels(metric.config):
        k_level = _k_from_level(level, k_max=k_max, n_candidates=n_candidates, ensure_min_one=True)
        selected_level = explanation_eidx_from(
            result,
            candidate_eidx,
            k=k_level,
            context=context,
            model=metric.model,
            config=metric.config,
        )
        z_level = _predict_with_selection(
            metric,
            context,
            candidate_eidx,
            selected_level,
            keep_only_selected=keep_only_selected,
        )
        values[f"@s={float(level):g}"] = float(score_fn(z_full, z_level))

    return values


class SparsityMetric(BaseMetric):
    """Reports explanation sparsity |E_expl| / |E_candidates|."""

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        candidate_eidx = candidate_eidx_from(context, result)
        has_budget = any(self.config.get(key) is not None for key in ("k", "topk", "sparsity"))
        k = _resolve_k(self.config, len(candidate_eidx)) if has_budget else None
        selected_eidx = explanation_eidx_from(
            result,
            candidate_eidx,
            k=k,
            context=context,
            model=self.model,
            config=self.config,
        )
        return MetricResult(
            name=self.name,
            values={"ratio": sparsity(len(selected_eidx), len(candidate_eidx))},
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=result.context_fp,
        )


class FidelityMinusMetric(BaseMetric):
    """Reports Fidelity- = |f(G) - f(G \\ E_expl)|."""

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        values = _compute_fidelity_values(
            self,
            context,
            result,
            keep_only_selected=False,
            score_fn=fidelity_minus,
        )
        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=result.context_fp,
        )


class FidelityPlusMetric(BaseMetric):
    """Reports Fidelity+ = 1 - |f(G) - f(E_expl)|."""

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        values = _compute_fidelity_values(
            self,
            context,
            result,
            keep_only_selected=True,
            score_fn=fidelity_plus,
        )
        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=result.context_fp,
        )


class AufscMetric(BaseMetric):
    """Reports AUFSC over configured sparsity levels."""

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        candidate_eidx = candidate_eidx_from(context, result)
        n_candidates = len(candidate_eidx)
        k_max = _resolve_k_max(self.config, n_candidates)
        levels = _resolve_sparsity_levels(self.config)

        z_full = float(self.model.predict_proba(context.subgraph, context.target))
        mode = str(self.config.get("mode", "minus"))
        keep_only_selected = mode == "plus"
        score_fn = fidelity_plus if keep_only_selected else fidelity_minus

        points: list[tuple[float, float]] = []
        for level in levels:
            k = _k_from_level(level, k_max=k_max, n_candidates=n_candidates, ensure_min_one=False)
            selected_eidx = explanation_eidx_from(
                result,
                candidate_eidx,
                k=k,
                context=context,
                model=self.model,
                config=self.config,
            )
            z_masked = _predict_with_selection(
                self,
                context,
                candidate_eidx,
                selected_eidx,
                keep_only_selected=keep_only_selected,
            )
            points.append((float(level), float(score_fn(z_full, z_masked))))

        max_sparsity = float(self.config.get("max_sparsity", 1.0))
        n_grid = int(self.config.get("n_grid", 101))
        value = aufsc(points, max_sparsity=max_sparsity, n_grid=n_grid)
        return MetricResult(
            name=self.name,
            values={"value": value},
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=result.context_fp,
            extras={
                "points": points,
                "mode": mode,
                "max_sparsity": max_sparsity,
                "n_grid": n_grid,
            },
        )


_LOADED = False


def ensure_builtin_metrics_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True

    register_metric("sparsity")(
        lambda cfg: SparsityMetric(
            name="sparsity",
            direction=MetricDirection.LOWER_IS_BETTER,
            config=cfg,
        )
    )
    register_metric("fidelity_minus")(
        lambda cfg: FidelityMinusMetric(
            name="fidelity_minus",
            direction=MetricDirection.HIGHER_IS_BETTER,
            config=cfg,
        )
    )
    register_metric("fidelity_plus")(
        lambda cfg: FidelityPlusMetric(
            name="fidelity_plus",
            direction=MetricDirection.HIGHER_IS_BETTER,
            config=cfg,
        )
    )
    register_metric("aufsc")(
        lambda cfg: AufscMetric(
            name="aufsc",
            direction=MetricDirection.HIGHER_IS_BETTER,
            config=cfg,
        )
    )


__all__ = [
    "fidelity_minus",
    "fidelity_plus",
    "sparsity",
    "aufsc",
    "SparsityMetric",
    "FidelityMinusMetric",
    "FidelityPlusMetric",
    "AufscMetric",
    "ensure_builtin_metrics_loaded",
]
