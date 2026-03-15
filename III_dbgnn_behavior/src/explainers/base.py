from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class EdgeExplanation:
    """Edge-level explanation for a single node prediction.

    Notes on `candidate_mask`:
        Some explainers are *node-focused* and only assign meaningful scores to a
        subset of edges (e.g., higher-order edges whose triple middle-node equals
        the explained node). In that case, set `candidate_mask` to a boolean mask
        of shape [E] indicating which edges are eligible for top-k selection.

        If `candidate_mask` is None, evaluation will assume all edges are eligible.
    """

    node_idx: int
    target_class: int
    edge_index: torch.Tensor  # [2, E]
    edge_score: torch.Tensor  # [E]
    candidate_mask: Optional[torch.Tensor] = None  # [E] bool


class EdgeExplainer:
    """Base class for node-level explainers that return an edge importance score."""

    def explain_node(self, *, adapter, data, node_idx: int, target_class: int) -> EdgeExplanation:
        raise NotImplementedError
