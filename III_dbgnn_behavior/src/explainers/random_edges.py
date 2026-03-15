from __future__ import annotations

import torch

from explainers.base import EdgeExplainer, EdgeExplanation


class RandomEdgeExplainer(EdgeExplainer):
    """Sanity-check baseline: assigns random importance to all edges."""

    def __init__(self, *, seed: int = 0):
        self.seed = int(seed)

    def explain_node(self, *, adapter, data, node_idx: int, target_class: int) -> EdgeExplanation:
        space = adapter.explain_space()
        edge_index = getattr(data, space.edge_index_attr)
        E = edge_index.size(1)

        # Random but deterministic for reproducibility (per node)
        g = torch.Generator(device=edge_index.device)
        g.manual_seed(self.seed + int(node_idx))
        scores = torch.rand(E, generator=g, device=edge_index.device)

        # Optional: if higher-order triples are available, restrict the candidate set
        # to triples with middle == node. This makes the random baseline comparable
        # to node-focused explainers (e.g., ShuffleGTOracleExplainer with focus='middle').
        candidate_mask = None
        if hasattr(data, "ho_triples"):
            triples = getattr(data, "ho_triples")
            if triples is not None and triples.ndim == 2 and triples.size(1) == 3 and triples.size(0) == E:
                candidate_mask = triples[:, 1] == int(node_idx)

        return EdgeExplanation(
            node_idx=int(node_idx),
            target_class=int(target_class),
            edge_index=edge_index,
            edge_score=scores,
            candidate_mask=candidate_mask,
        )
