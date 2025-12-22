# time_to_explain/metrics/subgraphx_mcts.py
from __future__ import annotations

from typing import Any, Mapping, Dict, List
import numpy as np

from time_to_explain.core.registry import register_metric
from time_to_explain.core.types import ExplanationContext, ExplanationResult, MetricResult
from time_to_explain.core.metrics import BaseMetric, MetricDirection


@register_metric("subgraphx_mcts")
def build_subgraphx_mcts(config: Mapping[str, Any] | None = None):
    """
    Best-reward-by-sparsity metric mirroring EvaluatorMCTSTG:
    for each sparsity level s, pick the maximum node reward among
    MCTS tree nodes whose sparsity <= s.

    Expects SubgraphXTGAdapter to populate:
      extras["mcts_tree_nodes_sparsity"] -> list[float]
      extras["mcts_tree_nodes_reward"]   -> list[float]
    """
    return SubgraphXMCTSMetric(config=config or {})


class SubgraphXMCTSMetric(BaseMetric):
    def __init__(self, name: str = "subgraphx_mcts", config: Mapping[str, Any] | None = None):
        cfg = dict(config or {})
        super().__init__(
            name=name,
            direction=MetricDirection.HIGHER_IS_BETTER,
            config=cfg,
        )
        levels = cfg.get("sparsity_levels")
        if levels is None:
            levels = np.arange(0.0, 1.05, 0.05).tolist()
        if isinstance(levels, (float, int)):
            levels = [levels]
        self.levels: List[float] = sorted({float(max(0.0, min(1.0, l))) for l in levels})
        self.output_keys = [f"@s={l:g}" for l in self.levels]

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        extras = result.extras or {}
        sparsity = extras.get("mcts_tree_nodes_sparsity")
        reward = extras.get("mcts_tree_nodes_reward")

        if sparsity is None or reward is None or len(sparsity) == 0 or len(sparsity) != len(reward):
            values = {k: float("nan") for k in self.output_keys}
            return MetricResult(
                name=self.name,
                values=values,
                direction=self.direction.value,
                run_id=context.run_id,
                explainer=result.explainer,
                context_fp=context.fingerprint(),
                extras={
                    "reason": "missing_tree_nodes",
                    "has_sparsity": sparsity is not None,
                    "has_reward": reward is not None,
                    "n_nodes": len(sparsity or []),
                },
            )

        s = np.asarray(sparsity, dtype=float)
        r = np.asarray(reward, dtype=float)
        order = np.argsort(s)
        s = s[order]
        r = r[order]

        best_by_level: Dict[str, float] = {}
        for lvl, key in zip(self.levels, self.output_keys):
            mask = s <= lvl + 1e-9
            if not np.any(mask):
                best_by_level[key] = float("nan")
            else:
                best_by_level[key] = float(np.max(r[mask]))

        return MetricResult(
            name=self.name,
            values=best_by_level,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=context.fingerprint(),
            extras={
                "levels": self.levels,
                "n_nodes": len(s),
            },
        )
