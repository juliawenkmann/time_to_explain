from __future__ import annotations

import torch

from explainers.base import EdgeExplainer, EdgeExplanation


class EdgeWeightExplainer(EdgeExplainer):
    """Baseline: uses (absolute) edge weights as importance.

    For DBGNN, the higher-order GCN layers consume `data.edge_weights_higher_order`.
    When those weights exist, this baseline is a cheap, deterministic alternative to
    random scores.
    """

    def __init__(self, *, abs_value: bool = True):
        self.abs_value = bool(abs_value)

    def explain_node(self, *, adapter, data, node_idx: int, target_class: int) -> EdgeExplanation:
        space = adapter.explain_space()
        edge_index = getattr(data, space.edge_index_attr)
        E = int(edge_index.size(1))

        scores = None
        if space.edge_weight_attr is not None and hasattr(data, space.edge_weight_attr):
            w = getattr(data, space.edge_weight_attr)
            if w is not None:
                w = w.detach().view(-1)
                if w.numel() == E:
                    scores = w.abs() if self.abs_value else w

        if scores is None:
            scores = torch.ones(E, device=edge_index.device, dtype=torch.float32)

        return EdgeExplanation(
            node_idx=int(node_idx),
            target_class=int(target_class),
            edge_index=edge_index,
            edge_score=scores,
        )
