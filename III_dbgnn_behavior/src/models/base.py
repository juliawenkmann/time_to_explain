from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import torch


@dataclass(frozen=True)
class ExplainSpace:
    """Defines which edges are being explained/perturbed.

    Many GNNs have a single `edge_index`.
    DBGNN uses `edge_index_higher_order` for higher-order message passing.
    """

    edge_index_attr: str
    edge_weight_attr: Optional[str] = None


class ModelAdapter(Protocol):
    """Adapter interface that makes evaluation/explainers model-agnostic."""

    model: torch.nn.Module

    def explain_space(self) -> ExplainSpace:
        ...

    def predict_logits(self, data) -> torch.Tensor:
        ...

    def clone_with_perturbed_edges(self, data, new_edge_index: torch.Tensor, *, new_edge_weight: Optional[torch.Tensor] = None):
        ...
