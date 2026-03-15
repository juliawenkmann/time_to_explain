from __future__ import annotations

import contextlib

import torch

from explainers.base import EdgeExplainer, EdgeExplanation
from utils import clone_data


class GNNExplainerDeBruijn(EdgeExplainer):
    """A minimal GNNExplainer-style edge mask optimizer for DBGNN / De Bruijn graphs.

    This implementation is intentionally small and does *not* depend on PyG's
    `torch_geometric.explain` API, because DBGNN consumes a PyG `Data` object and
    expects De Bruijn-specific attributes (e.g. `edge_index_higher_order`,
    `edge_weights_higher_order`).

    Core idea (same as GNNExplainer):
      - learn a soft mask m in (0, 1)^E over edges in the chosen explain-space
      - apply it multiplicatively to the corresponding edge weights
      - optimize m so that the model stays confident in the target class for the
        selected node

    For DBGNN we typically explain the *higher-order* edges. A higher-order node
    corresponds to a first-order edge (u, v), and a higher-order edge corresponds
    to a first-order length-2 walk u->v->w.
    """

    def __init__(
        self,
        *,
        steps: int = 200,
        lr: float = 0.05,
        lam_size: float = 0.01,
        lam_ent: float = 0.1,
        size_mode: str = "mean",
        seed: int = 0,
    ):
        self.steps = int(steps)
        self.lr = float(lr)
        self.lam_size = float(lam_size)
        self.lam_ent = float(lam_ent)
        self.size_mode = str(size_mode)
        self.seed = int(seed)
        if self.steps <= 0:
            raise ValueError("steps must be >= 1")
        if self.size_mode not in {"mean", "sum"}:
            raise ValueError("size_mode must be 'mean' or 'sum'")

    @contextlib.contextmanager
    def _freeze_model(self, model: torch.nn.Module):
        """Temporarily disable grads on model parameters to speed up mask training."""

        reqs = [p.requires_grad for p in model.parameters()]
        try:
            for p in model.parameters():
                p.requires_grad_(False)
            yield
        finally:
            for p, r in zip(model.parameters(), reqs):
                p.requires_grad_(r)

    def explain_node(self, *, adapter, data, node_idx: int, target_class: int) -> EdgeExplanation:
        space = adapter.explain_space()
        if space.edge_weight_attr is None:
            raise ValueError(
                "GNNExplainerDeBruijn requires ExplainSpace.edge_weight_attr to be set "
                "(the model must accept/use edge weights in the explain space)."
            )
        if not hasattr(data, space.edge_weight_attr):
            raise AttributeError(f"Data object has no attribute {space.edge_weight_attr!r}")

        edge_index = getattr(data, space.edge_index_attr)
        E = int(edge_index.size(1))
        if E == 0:
            raise ValueError("Graph has 0 edges in the explain space")

        w_base = getattr(data, space.edge_weight_attr)
        if w_base is None:
            w_base = torch.ones(E, device=edge_index.device, dtype=torch.float32)
        w_base = w_base.detach().float().view(-1)
        if w_base.numel() != E:
            raise ValueError(
                f"Expected {space.edge_weight_attr} to have shape [E] with E={E}, got {tuple(w_base.shape)}"
            )

        # Work on a clone so we don't mutate the caller's Data object.
        data_work = clone_data(data)

        # Initialize mask logits with small noise (deterministic per node).
        g = torch.Generator(device=edge_index.device)
        g.manual_seed(self.seed + int(node_idx))
        mask_logits = torch.randn(E, generator=g, device=edge_index.device) * 0.1
        mask_logits = torch.nn.Parameter(mask_logits)

        opt = torch.optim.Adam([mask_logits], lr=self.lr)
        eps = 1e-15

        adapter.model.eval()  # explain prediction-time behaviour (dropout off)

        with self._freeze_model(adapter.model):
            for _ in range(self.steps):
                opt.zero_grad(set_to_none=True)

                mask = torch.sigmoid(mask_logits)  # (0, 1)
                w_masked = w_base * mask
                setattr(data_work, space.edge_weight_attr, w_masked)

                # IMPORTANT: clear potential PyG adjacency caches so the model
                # actually consumes the current masked edge weights.
                if hasattr(adapter, "reset_caches"):
                    try:
                        adapter.reset_caches()  # type: ignore[attr-defined]
                    except Exception:
                        pass

                logits = adapter.model(data_work)
                logp = torch.log_softmax(logits[int(node_idx)], dim=-1)[int(target_class)]
                loss_pred = -logp

                # Regularize for small explanations (sparsity)
                if self.size_mode == "sum":
                    loss_size = self.lam_size * mask.sum()
                else:
                    loss_size = self.lam_size * mask.mean()

                # Regularize towards hard {0,1} masks (entropy)
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
