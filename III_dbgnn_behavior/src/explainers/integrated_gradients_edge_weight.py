from __future__ import annotations

import torch

from explainers.base import EdgeExplainer, EdgeExplanation
from utils import clone_data


class IntegratedGradientsEdgeWeightExplainer(EdgeExplainer):
    """Integrated Gradients (IG) on edge weights.

    This treats the edge weights in the chosen explain-space as continuous
    inputs and computes Integrated Gradients of the target node logit.

    Notes:
        - Works well for DBGNN because it passes `edge_weights_higher_order` into
          GCN layers.
        - This is slower than a single gradient, but often less noisy.
    """

    def __init__(
        self,
        *,
        steps: int = 32,
        abs_value: bool = True,
        baseline: str = "zeros",  # currently only "zeros"
    ):
        self.steps = int(steps)
        if self.steps <= 0:
            raise ValueError("steps must be >= 1")
        self.abs_value = bool(abs_value)
        self.baseline = str(baseline)
        if self.baseline not in {"zeros"}:
            raise ValueError(f"Unknown baseline: {baseline!r}. Supported: 'zeros'")

    def explain_node(self, *, adapter, data, node_idx: int, target_class: int) -> EdgeExplanation:
        space = adapter.explain_space()
        if space.edge_weight_attr is None:
            raise ValueError(
                "IntegratedGradientsEdgeWeightExplainer requires ExplainSpace.edge_weight_attr to be set "
                "(model must accept/use edge weights)."
            )
        if not hasattr(data, space.edge_weight_attr):
            raise AttributeError(f"Data object has no attribute {space.edge_weight_attr!r}")

        edge_index = getattr(data, space.edge_index_attr)
        E = int(edge_index.size(1))

        w_in = getattr(data, space.edge_weight_attr)
        if w_in is None:
            w_in = torch.ones(E, device=edge_index.device, dtype=torch.float32)
        w_in = w_in.detach().float().view(-1)
        if w_in.numel() != E:
            raise ValueError(
                f"Expected {space.edge_weight_attr} to have shape [E] with E={E}, got {tuple(w_in.shape)}"
            )

        if self.baseline == "zeros":
            w_base = torch.zeros_like(w_in)
        else:  # pragma: no cover
            w_base = torch.zeros_like(w_in)

        delta = w_in - w_base
        total_grad = torch.zeros_like(w_in)

        data_work = clone_data(data)

        adapter.model.eval()

        # Riemann sum approximation of the IG integral.
        for s in range(1, self.steps + 1):
            alpha = float(s) / float(self.steps)
            w = (w_base + alpha * delta).detach().clone().requires_grad_(True)
            setattr(data_work, space.edge_weight_attr, w)

            adapter.model.zero_grad(set_to_none=True)

            # PyG conv layers may cache normalized adjacency/weights. Clear
            # caches so each IG step actually uses the current interpolated
            # edge weights.
            if hasattr(adapter, "reset_caches"):
                try:
                    adapter.reset_caches()  # type: ignore[attr-defined]
                except Exception:
                    pass
            logits = adapter.model(data_work)
            target_logit = logits[int(node_idx), int(target_class)]
            target_logit.backward()

            if w.grad is None:
                raise RuntimeError("No gradient computed for edge weights. Does the model use edge weights?")
            total_grad += w.grad.detach()

        ig = delta * (total_grad / float(self.steps))
        scores = ig.abs() if self.abs_value else ig
        scores = scores.detach()

        return EdgeExplanation(
            node_idx=int(node_idx),
            target_class=int(target_class),
            edge_index=edge_index,
            edge_score=scores,
        )
