# time_to_explain/adapters/random_baseline_adapter.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

from time_to_explain.core.types import BaseExplainer, ExplanationContext, ExplanationResult
from time_to_explain.explainer.random_baseline import RandomEdgeImportanceGenerator


@dataclass
class RandomAdapterConfig:
    alias: str = "random"
    seed: Optional[int] = None


class RandomAdapter(BaseExplainer):
    """
    Minimal adapter that exposes the random baseline through the shared
    `BaseExplainer` interface so it can be scheduled next to SubgraphX, TEMP-ME, etc.
    """

    def __init__(self, cfg: Optional[RandomAdapterConfig] = None) -> None:
        self.cfg = cfg or RandomAdapterConfig()
        super().__init__(name="random_baseline", alias=self.cfg.alias)
        self._generator = RandomEdgeImportanceGenerator(seed=self.cfg.seed)

    # ------------------------------------------------------------------ setup
    def prepare(self, *, model: Any, dataset: Any) -> None:
        """
        No model/dataset dependency beyond tracking the RNG seed for reproducibility,
        but we keep the signature to align with the other explainers.
        """
        super().prepare(model=model, dataset=dataset)
        self._generator.reset(self.cfg.seed)

    # ---------------------------------------------------------------- explain
    def explain(self, context: ExplanationContext) -> ExplanationResult:
        scores, extras = self._generator.generate(context)
        return ExplanationResult(
            run_id=context.run_id,
            explainer=self.alias,
            context_fp=context.fingerprint(),
            importance_edges=scores,
            importance_nodes=None,
            importance_time=None,
            extras=extras,
        )


__all__ = ["RandomBaselineAdapter", "RandomBaselineAdapterConfig"]

