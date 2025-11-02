from __future__ import annotations
from typing import Any, Optional

from ..core.types import BaseExplainer, ExplanationContext, ExplanationResult

class TGNNExplainerAdapter(BaseExplainer):
    """Wrap a legacy TGNN explainer behind the unified interface."""
    def __init__(self, *, legacy_impl: Any, name: str = "tgnn_legacy", alias: Optional[str] = None):
        super().__init__(name=name, alias=alias)
        self.legacy = legacy_impl

    def prepare(self, *, model: Any, dataset: Any) -> None:
        super().prepare(model=model, dataset=dataset)
        if hasattr(self.legacy, "prepare"):
            self.legacy.prepare(model=model, dataset=dataset)

    def explain(self, context: ExplanationContext) -> ExplanationResult:
        if hasattr(self.legacy, "explain"):
            raw = self.legacy.explain(context)
        else:
            t = context.target
            raw = self.legacy.explain_one(t.get("u"), t.get("i"), t.get("ts"), k_hop=context.k_hop)

        if isinstance(raw, dict):
            res = ExplanationResult(
                run_id=context.run_id, explainer=self.alias, context_fp=context.fingerprint(),
                importance_edges=list(raw.get("importance_edges", [])),
                importance_nodes=list(raw.get("importance_nodes", [])),
                importance_time=list(raw.get("importance_time", [])),
                extras={k:v for k,v in raw.items() if k not in ("importance_edges","importance_nodes","importance_time")}
            )
        else:
            res = ExplanationResult(
                run_id=context.run_id, explainer=self.alias, context_fp=context.fingerprint(),
                extras={"raw": raw}
            )
        return res
