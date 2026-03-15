from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from models.base import ExplainSpace
from utils import clone_data

from pathpyG.nn.dbgnn import DBGNN


def _reset_pyg_message_passing_caches(model: torch.nn.Module) -> None:
    """Clear common PyG message-passing caches.

    Why: some conv layers (notably `GCNConv(cached=True)`) store normalized
    adjacency in internal buffers (e.g. `_cached_edge_index`). When we evaluate
    explainers, we *change* the edge_index/edge_weight per node+frac. If caches
    are not cleared, perturbations may have no effect, and all explainers can
    appear to have identical scores.
    """

    for m in model.modules():
        # PyG GCNConv / SGConv
        if hasattr(m, "_cached_edge_index"):
            try:
                setattr(m, "_cached_edge_index", None)
            except Exception:
                pass
        if hasattr(m, "_cached_adj_t"):
            try:
                setattr(m, "_cached_adj_t", None)
            except Exception:
                pass


@dataclass
class DBGNNAdapter:
    """Minimal adapter for the PathpyG DBGNN model.

    This is the model used in `dbgnn.ipynb`.
    """

    model: torch.nn.Module
    edge_index_attr: str = "edge_index_higher_order"
    # PathpyG's DBGNN forwards `data.edge_weights_higher_order` into the higher-order GCN layers.
    # When we perturb/subset `edge_index_higher_order` during explainer evaluation, we MUST also
    # subset the matching edge weights. Otherwise, PyG's normalization utilities will crash with
    # a mask/shape mismatch.
    edge_weight_attr: Optional[str] = "edge_weights_higher_order"

    def explain_space(self) -> ExplainSpace:
        return ExplainSpace(edge_index_attr=self.edge_index_attr, edge_weight_attr=self.edge_weight_attr)

    def reset_caches(self) -> None:
        """Clear common PyG message-passing caches.

        IMPORTANT for explainers:
        Some PyG conv layers (notably `GCNConv(cached=True)`) cache normalized
        adjacency (and sometimes edge weights). If an explainer changes edge
        weights across multiple forward passes without clearing caches, the
        perturbation can have *no effect*.
        """

        _reset_pyg_message_passing_caches(self.model)

    def predict_logits(self, data) -> torch.Tensor:
        self.model.eval()
        # IMPORTANT: explainers evaluate many *perturbed* graphs. Clear any
        # internal adjacency caches so the perturbation actually takes effect.
        self.reset_caches()
        with torch.no_grad():
            return self.model(data)

    def clone_with_perturbed_edges(
        self,
        data,
        new_edge_index: torch.Tensor,
        *,
        new_edge_weight: Optional[torch.Tensor] = None,
    ):
        data2 = clone_data(data)
        setattr(data2, self.edge_index_attr, new_edge_index)
        if self.edge_weight_attr is not None and hasattr(data2, self.edge_weight_attr):
            if new_edge_weight is None:
                raise ValueError(
                    f"edge_weight_attr was set to {self.edge_weight_attr!r} but new_edge_weight was None"
                )
            setattr(data2, self.edge_weight_attr, new_edge_weight)
        return data2


def build_dbgnn_adapter(*, data, assets, device: torch.device, hidden_dims=(16, 32, 8), p_dropout: float = 0.4) -> DBGNNAdapter:
    """Build the DBGNN model exactly as in the tutorial notebook."""

    num_classes = int(data.y.unique().numel())

    # Exact notebook line:
    # model = DBGNN(num_features=[g.n, g2.n], num_classes=len(data.y.unique()), hidden_dims=[16, 32, 8], p_dropout=0.4)
    model = DBGNN(
        num_features=[assets.g.n, assets.g2.n],
        num_classes=num_classes,
        hidden_dims=list(hidden_dims),
        p_dropout=float(p_dropout),
    ).to(device)

    # Explain higher-order edges by default.
    # IMPORTANT: the model uses `data.edge_weights_higher_order`, so the adapter must keep it
    # consistent when we subset edges.
    return DBGNNAdapter(
        model=model,
        edge_index_attr="edge_index_higher_order",
        edge_weight_attr="edge_weights_higher_order",
    )
