from __future__ import annotations

import torch

from explainers.base import EdgeExplainer, EdgeExplanation
from utils import clone_data


class MaskOptimEdgeExplainer(EdgeExplainer):
    """A small GNNExplainer-style edge mask optimizer.

    The idea:
      - keep the graph structure fixed
      - learn a soft mask m in (0, 1)^E that scales edge weights
      - optimize m to make the model confident in the target class for the node

    This is intentionally minimal (no PyG explain API), but works well for DBGNN
    because it consumes `edge_weights_higher_order`.
    """

    def __init__(
        self,
        *,
        steps: int = 150,
        lr: float = 0.1,
        lam_size: float = 0.01,
        lam_ent: float = 0.1,
        seed: int = 0,
    ):
        self.steps = int(steps)
        self.lr = float(lr)
        self.lam_size = float(lam_size)
        self.lam_ent = float(lam_ent)
        self.seed = int(seed)
        if self.steps <= 0:
            raise ValueError("steps must be >= 1")

    def explain_node(self, *, adapter, data, node_idx: int, target_class: int) -> EdgeExplanation:
        space = adapter.explain_space()
        if space.edge_weight_attr is None:
            raise ValueError(
                "MaskOptimEdgeExplainer requires ExplainSpace.edge_weight_attr to be set "
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

        # One working copy of the Data object; we only overwrite the edge weights.
        data_work = clone_data(data)

        # Initialize mask logits with small noise (deterministic per node).
        g = torch.Generator(device=edge_index.device)
        g.manual_seed(self.seed + int(node_idx))
        mask_logits = torch.randn(E, generator=g, device=edge_index.device) * 0.1
        mask_logits = torch.nn.Parameter(mask_logits)

        opt = torch.optim.Adam([mask_logits], lr=self.lr)
        eps = 1e-15

        adapter.model.eval()

        for _ in range(self.steps):
            opt.zero_grad(set_to_none=True)

            mask = torch.sigmoid(mask_logits)  # (0, 1)
            w_masked = w_in * mask
            setattr(data_work, space.edge_weight_attr, w_masked)

            # IMPORTANT: clear potential PyG adjacency caches so the model
            # actually consumes the current masked edge weights.
            if hasattr(adapter, "reset_caches"):
                try:
                    adapter.reset_caches()  # type: ignore[attr-defined]
                except Exception:
                    pass

            logits = adapter.model(data_work)
            target_logit = logits[int(node_idx), int(target_class)]

            # Prediction term: maximize target logit => minimize negative.
            loss_pred = -target_logit

            # Sparsity term: encourage small masks.
            loss_size = self.lam_size * mask.mean()

            # Entropy term: push masks towards {0, 1}.
            ent = -mask * torch.log(mask + eps) - (1.0 - mask) * torch.log(1.0 - mask + eps)
            loss_ent = self.lam_ent * ent.mean()

            loss = loss_pred + loss_size + loss_ent
            loss.backward()
            opt.step()

        scores = torch.sigmoid(mask_logits).detach()

        return EdgeExplanation(
            node_idx=int(node_idx),
            target_class=int(target_class),
            edge_index=edge_index,
            edge_score=scores,
        )
