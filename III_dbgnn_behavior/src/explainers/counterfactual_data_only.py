from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from explainers.base import EdgeExplainer, EdgeExplanation


@dataclass(frozen=True)
class DataOnlyCounterfactualConfig:
    """Data-only counterfactual ranking for higher-order edges.

    This explainer ignores the model and uses only:
      - higher-order triples (u,v,w)
      - higher-order edge weights
      - node labels (cluster IDs) for u

    It ranks edges ending at the target node (w == target) by their weights,
    but only for edges whose source cluster is the dominant cluster for that target.
    """

    triples_attr: str = "ho_triples"
    weight_attr: str = "edge_weights_higher_order"
    label_attr: str = "y"
    irrelevant_score: float = -1e9


class DataOnlyCounterfactualEdgeDeletionExplainer(EdgeExplainer):
    """Data-only counterfactual explainer (no model required)."""

    def __init__(self, *, cfg: DataOnlyCounterfactualConfig = DataOnlyCounterfactualConfig()):
        self.cfg = cfg

    def explain_node(self, *, adapter, data, node_idx: int, target_class: int) -> EdgeExplanation:
        space = adapter.explain_space()
        edge_index = getattr(data, space.edge_index_attr)
        E = int(edge_index.size(1))
        if E == 0:
            raise ValueError("Graph has 0 edges in the explain space")

        triples = getattr(data, self.cfg.triples_attr, None)
        if triples is None or not isinstance(triples, torch.Tensor):
            raise ValueError(f"data.{self.cfg.triples_attr} is required for data-only counterfactuals")
        if triples.ndim != 2 or triples.size(0) != E or triples.size(1) != 3:
            raise ValueError(
                f"data.{self.cfg.triples_attr} must have shape [E,3] aligned with edge_index_higher_order"
            )

        weights = getattr(data, self.cfg.weight_attr, None)
        if weights is None:
            weights = torch.ones(E, device=edge_index.device, dtype=torch.float32)
        weights = weights.detach().float().view(-1)
        if weights.numel() != E:
            raise ValueError(f"data.{self.cfg.weight_attr} must have shape [E]")

        labels = getattr(data, self.cfg.label_attr, None)
        if labels is None or not isinstance(labels, torch.Tensor):
            raise ValueError(f"data.{self.cfg.label_attr} (node labels) is required")
        labels = labels.detach().to(device=edge_index.device).view(-1)

        u = triples[:, 0].to(device=edge_index.device)
        w = triples[:, 2].to(device=edge_index.device)
        mask_target = w == int(node_idx)
        if int(mask_target.sum().item()) == 0:
            scores = torch.full((E,), float(self.cfg.irrelevant_score), device=edge_index.device)
            return EdgeExplanation(
                node_idx=int(node_idx),
                target_class=int(target_class),
                edge_index=edge_index,
                edge_score=scores,
                candidate_mask=mask_target.to(torch.bool),
            )

        clusters_u = labels[u].to(torch.long)
        C = int(labels.max().item()) + 1 if labels.numel() > 0 else 0
        if C <= 0:
            raise ValueError("Could not infer number of clusters from labels")

        # Sum weights for each cluster among edges ending at target.
        S = torch.zeros(C, dtype=weights.dtype, device=edge_index.device)
        S.scatter_add_(0, clusters_u[mask_target], weights[mask_target])

        # Pick dominant cluster and rank its edges by weight.
        y = int(torch.argmax(S).item())
        cand_mask = mask_target & (clusters_u == y)
        idx = cand_mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            scores = torch.full((E,), float(self.cfg.irrelevant_score), device=edge_index.device)
            return EdgeExplanation(
                node_idx=int(node_idx),
                target_class=int(target_class),
                edge_index=edge_index,
                edge_score=scores,
                candidate_mask=cand_mask.to(torch.bool),
            )

        order = torch.argsort(weights[idx], descending=True)
        idx_sorted = idx[order]

        scores = torch.full((E,), float(self.cfg.irrelevant_score), device=edge_index.device)
        scores[idx_sorted] = weights[idx_sorted]

        return EdgeExplanation(
            node_idx=int(node_idx),
            target_class=int(target_class),
            edge_index=edge_index,
            edge_score=scores,
            candidate_mask=cand_mask.to(torch.bool),
        )


__all__ = [
    "DataOnlyCounterfactualConfig",
    "DataOnlyCounterfactualEdgeDeletionExplainer",
]
