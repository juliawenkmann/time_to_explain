from __future__ import annotations

import torch

from explainers.base import EdgeExplainer, EdgeExplanation
from utils import clone_data


class GradEdgeWeightExplainer(EdgeExplainer):
    """Gradient saliency on edge weights.

    Computes the gradient of the target node logit with respect to the edge
    weights in the chosen explain-space (e.g. `edge_weights_higher_order`).

    This is simple, fast, and works well as a first "real" explainer for DBGNN.
    """

    def __init__(
        self,
        *,
        abs_value: bool = True,
        grad_x_input: bool = False,
    ):
        self.abs_value = bool(abs_value)
        self.grad_x_input = bool(grad_x_input)

    def explain_node(self, *, adapter, data, node_idx: int, target_class: int) -> EdgeExplanation:
        space = adapter.explain_space()
        if space.edge_weight_attr is None:
            raise ValueError(
                "GradEdgeWeightExplainer requires ExplainSpace.edge_weight_attr to be set "
                "(model must accept/use edge weights)."
            )
        if not hasattr(data, space.edge_weight_attr):
            raise AttributeError(f"Data object has no attribute {space.edge_weight_attr!r}")

        edge_index = getattr(data, space.edge_index_attr)
        E = int(edge_index.size(1))

        w = getattr(data, space.edge_weight_attr)
        if w is None:
            w = torch.ones(E, device=edge_index.device, dtype=torch.float32)
        w = w.detach().float().view(-1)
        if w.numel() != E:
            raise ValueError(
                f"Expected {space.edge_weight_attr} to have shape [E] with E={E}, got {tuple(w.shape)}"
            )

        # Make edge weights differentiable.
        w_var = w.clone().detach().requires_grad_(True)

        data_work = clone_data(data)
        setattr(data_work, space.edge_weight_attr, w_var)

        adapter.model.eval()
        adapter.model.zero_grad(set_to_none=True)

        # PyG GCNConv may cache normalized adjacency (and sometimes edge
        # weights). Clear caches so gradients reflect the current `w_var`.
        if hasattr(adapter, "reset_caches"):
            try:
                adapter.reset_caches()  # type: ignore[attr-defined]
            except Exception:
                pass

        logits = adapter.model(data_work)
        target_logit = logits[int(node_idx), int(target_class)]
        target_logit.backward()

        grad = w_var.grad
        if grad is None:
            raise RuntimeError("No gradient computed for edge weights. Does the model use edge weights?")

        scores = grad * w if self.grad_x_input else grad
        scores = scores.abs() if self.abs_value else scores
        scores = scores.detach()

        return EdgeExplanation(
            node_idx=int(node_idx),
            target_class=int(target_class),
            edge_index=edge_index,
            edge_score=scores,
        )
