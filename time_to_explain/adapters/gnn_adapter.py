# time_to_explain/adapters/gnn_explainer_adapter.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

from time_to_explain.core.types import BaseExplainer, ExplanationContext, ExplanationResult
from time_to_explain.explainer.gnnexplainer.gnnexplainer import GNNExplainer


@dataclass
class GNNExplainerAdapterConfig:
    """
    Configuration for the GNNExplainer adapter.

    Notes
    -----
    - `epochs`/`lr` are forwarded to the underlying PyG GNNExplainer.
    - `feat_mask_type` is used by the legacy API; for most models "scalar"
      (a single learnable scalar per input feature) is a good default.
    - If your model very clearly operates at node level by default, set
      `node_level_by_default=True`. Otherwise the adapter will auto-detect
      from the context (presence of a node index in payload) and fall back
      to graph level.
    """
    alias: str = "gnnexplainer"
    seed: Optional[int] = None

    # Optim config
    epochs: int = 100
    lr: float = 0.01

    # Legacy PyG API option:
    feat_mask_type: str = "scalar"  # {"scalar", "individual"}

    # Try the newer torch_geometric.explain API if available:
    allow_new_pyg_api: bool = True

    # Logging is handled by PyG internally when True (legacy API):
    log: bool = False

    # Scope heuristics:
    node_level_by_default: bool = False
    force_scope: Optional[str] = None  # {"node", "graph"} or None

    # If your model returns probabilities/log-probabilities and you use the
    # new API, you can hint the return type here (e.g., "log_probs", "probs").
    # Leave as None to let the engine best-effort infer.
    return_type: Optional[str] = None


class GNNExplainerAdapter(BaseExplainer):
    """
    Adapter that exposes GNNExplainer through the shared `BaseExplainer` interface
    so it can be scheduled next to TGNNExplainer, TEMP-ME, etc.
    """

    def __init__(self, cfg: Optional[GNNExplainerAdapterConfig] = None) -> None:
        self.cfg = cfg or GNNExplainerAdapterConfig()
        super().__init__(name="gnn_explainer", alias=self.cfg.alias)
        self._engine = GNNExplainer(
            seed=self.cfg.seed,
            epochs=self.cfg.epochs,
            lr=self.cfg.lr,
            feat_mask_type=self.cfg.feat_mask_type,
            allow_new_pyg_api=self.cfg.allow_new_pyg_api,
            log=self.cfg.log,
            default_scope=("node" if self.cfg.node_level_by_default else "graph"),
            return_type=self.cfg.return_type,
            force_scope=self.cfg.force_scope,
        )

    # ------------------------------------------------------------------ setup
    def prepare(self, *, model: Any, dataset: Any) -> None:
        """
        Set the model/dataset and (optionally) reset RNGs for reproducibility.
        """
        super().prepare(model=model, dataset=dataset)
        self._engine.reset(self.cfg.seed)
        self._engine.attach(model=model, dataset=dataset)

    # ---------------------------------------------------------------- explain
    def explain(self, context: ExplanationContext) -> ExplanationResult:
        edge_scores, node_scores, extras = self._engine.generate(context)
        return ExplanationResult(
            run_id=context.run_id,
            explainer=self.alias,
            context_fp=context.fingerprint(),
            importance_edges=edge_scores,
            importance_nodes=node_scores,   # may be None (GNNExplainer returns feature mask)
            importance_time=None,
            extras=extras,
        )


__all__ = ["GNNExplainerAdapter", "GNNExplainerAdapterConfig"]
