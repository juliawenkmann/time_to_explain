from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from time_to_explain.core.metrics import BaseMetric, MetricDirection
from time_to_explain.core.registry import register_metric
from time_to_explain.core.types import ExplanationContext, ExplanationResult, MetricResult


def fidelity_minus(z_full: float, z_removed: float) -> float:
    return abs(z_full - z_removed)


def fidelity_plus(z_full: float, z_expl: float) -> float:
    return 1.0 - abs(z_expl - z_full)


def sparsity(num_edges_expl: int, num_edges_lhop: int) -> float:
    return num_edges_expl / max(1, num_edges_lhop)


def aufsc(points: Iterable[Tuple[float, float]], *, max_sparsity: float = 1.0, n_grid: int = 101) -> float:
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


def candidate_eidx_from(context: ExplanationContext, result: ExplanationResult) -> List[int]:
    if context.subgraph and context.subgraph.payload and "candidate_eidx" in context.subgraph.payload:
        return list(context.subgraph.payload["candidate_eidx"])
    return list(result.extras["candidate_eidx"])


def explanation_eidx_from(
    result: ExplanationResult,
    candidate_eidx: Sequence[int],
    *,
    k: int | None,
) -> List[int]:
    if "selected_eidx" in result.extras:
        return list(result.extras["selected_eidx"])
    importance = result.importance_edges
    order = sorted(range(len(importance)), key=importance.__getitem__, reverse=True)
    if k is None:
        k = len(order)
    return [int(candidate_eidx[i]) for i in order[:k]]


def edge_mask_keep(candidate_eidx: Sequence[int], keep_eidx: Sequence[int]) -> List[float]:
    keep = set(int(e) for e in keep_eidx)
    return [1.0 if int(e) in keep else 0.0 for e in candidate_eidx]


def edge_mask_drop(candidate_eidx: Sequence[int], drop_eidx: Sequence[int]) -> List[float]:
    drop = set(int(e) for e in drop_eidx)
    return [0.0 if int(e) in drop else 1.0 for e in candidate_eidx]


def _resolve_k(config: dict, n_candidates: int) -> int:
    if "k" in config and config["k"] is not None:
        return int(config["k"])
    if "sparsity" in config and config["sparsity"] is not None:
        return int(round(float(config["sparsity"]) * float(n_candidates)))
    return int(n_candidates)


def _resolve_sparsity_levels(config: dict) -> List[float]:
    levels = config.get("sparsity_levels")
    if levels:
        return [float(s) for s in levels]
    return [0.0, 0.25, 0.5, 0.75, 1.0]


def _resolve_k_max(config: dict, n_candidates: int) -> int:
    if "k_max" in config and config["k_max"] is not None:
        return int(config["k_max"])
    return int(n_candidates)


class SparsityMetric(BaseMetric):
    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        candidate_eidx = candidate_eidx_from(context, result)
        k = _resolve_k(self.config, len(candidate_eidx))
        expl_eidx = explanation_eidx_from(result, candidate_eidx, k=k)
        value = sparsity(len(expl_eidx), len(candidate_eidx))
        return MetricResult(
            name=self.name,
            values={"ratio": float(value)},
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=result.context_fp,
        )


class FidelityMinusMetric(BaseMetric):
    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        candidate_eidx = candidate_eidx_from(context, result)
        k = _resolve_k(self.config, len(candidate_eidx))
        expl_eidx = explanation_eidx_from(result, candidate_eidx, k=k)
        z_full = self.model.predict_proba(context.subgraph, context.target)
        z_removed = self.model.predict_proba_with_mask(
            context.subgraph,
            context.target,
            edge_mask=edge_mask_drop(candidate_eidx, expl_eidx),
        )
        value = fidelity_minus(float(z_full), float(z_removed))
        values = {"value": float(value)}

        levels = self.config.get("sparsity_levels") or self.config.get("levels")
        if levels is not None:
            n_candidates = len(candidate_eidx)
            k_max = _resolve_k_max(self.config, n_candidates)
            for lvl in _resolve_sparsity_levels(self.config):
                k_lvl = int(round(float(lvl) * float(k_max)))
                if lvl > 0.0 and k_lvl == 0 and n_candidates > 0:
                    k_lvl = 1
                k_lvl = max(0, min(k_lvl, n_candidates))
                expl_eidx_lvl = explanation_eidx_from(result, candidate_eidx, k=k_lvl)
                z_removed_lvl = self.model.predict_proba_with_mask(
                    context.subgraph,
                    context.target,
                    edge_mask=edge_mask_drop(candidate_eidx, expl_eidx_lvl),
                )
                values[f"@s={float(lvl):g}"] = fidelity_minus(float(z_full), float(z_removed_lvl))
        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=result.context_fp,
        )


class FidelityPlusMetric(BaseMetric):
    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        candidate_eidx = candidate_eidx_from(context, result)
        k = _resolve_k(self.config, len(candidate_eidx))
        expl_eidx = explanation_eidx_from(result, candidate_eidx, k=k)
        z_full = self.model.predict_proba(context.subgraph, context.target)
        z_expl = self.model.predict_proba_with_mask(
            context.subgraph,
            context.target,
            edge_mask=edge_mask_keep(candidate_eidx, expl_eidx),
        )
        value = fidelity_plus(float(z_full), float(z_expl))
        values = {"value": float(value)}

        levels = self.config.get("sparsity_levels") or self.config.get("levels")
        if levels is not None:
            n_candidates = len(candidate_eidx)
            k_max = _resolve_k_max(self.config, n_candidates)
            for lvl in _resolve_sparsity_levels(self.config):
                k_lvl = int(round(float(lvl) * float(k_max)))
                if lvl > 0.0 and k_lvl == 0 and n_candidates > 0:
                    k_lvl = 1
                k_lvl = max(0, min(k_lvl, n_candidates))
                expl_eidx_lvl = explanation_eidx_from(result, candidate_eidx, k=k_lvl)
                z_expl_lvl = self.model.predict_proba_with_mask(
                    context.subgraph,
                    context.target,
                    edge_mask=edge_mask_keep(candidate_eidx, expl_eidx_lvl),
                )
                values[f"@s={float(lvl):g}"] = fidelity_plus(float(z_full), float(z_expl_lvl))
        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=result.context_fp,
        )


class AufscMetric(BaseMetric):
    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        candidate_eidx = candidate_eidx_from(context, result)
        importance = result.importance_edges
        order = sorted(range(len(importance)), key=importance.__getitem__, reverse=True)
        sparsity_levels = _resolve_sparsity_levels(self.config)
        k_max = _resolve_k_max(self.config, len(candidate_eidx))
        z_full = self.model.predict_proba(context.subgraph, context.target)
        points: List[Tuple[float, float]] = []
        mode = str(self.config.get("mode", "minus"))
        max_sparsity = float(self.config.get("max_sparsity", 1.0))
        n_grid = int(self.config.get("n_grid", 101))
        for level in sparsity_levels:
            k = int(round(float(level) * float(k_max)))
            expl_eidx = [int(candidate_eidx[i]) for i in order[:k]]
            if mode == "plus":
                z_expl = self.model.predict_proba_with_mask(
                    context.subgraph,
                    context.target,
                    edge_mask=edge_mask_keep(candidate_eidx, expl_eidx),
                )
                fid = fidelity_plus(float(z_full), float(z_expl))
            else:
                z_removed = self.model.predict_proba_with_mask(
                    context.subgraph,
                    context.target,
                    edge_mask=edge_mask_drop(candidate_eidx, expl_eidx),
                )
                fid = fidelity_minus(float(z_full), float(z_removed))
            points.append((float(level), float(fid)))
        value = aufsc(points, max_sparsity=max_sparsity, n_grid=n_grid)
        return MetricResult(
            name=self.name,
            values={"value": float(value)},
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
