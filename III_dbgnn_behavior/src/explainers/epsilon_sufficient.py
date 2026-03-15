from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F

from explainers.base import EdgeExplainer, EdgeExplanation
from eval.metrics import default_candidate_mask_from_triples
from explainers.counterfactual import _candidate_edge_mask_from_khop_ho
from utils import clone_data


@dataclass(frozen=True)
class EpsilonSufficientConfig:
    """Configuration for epsilon-sufficient HO-edge explanations.

    Objective: find a small set of higher-order edges whose *kept* graph
    yields a prediction distribution close to the full graph:

        KL(p_full || p_masked) <= epsilon

    We optimize a continuous keep-mask and then rank edges by keep-probability.
    """

    # Candidate restriction
    k_hops: int = 4
    undirected_khop: bool = True
    triples_attr: str = "ho_triples"

    # Optimization
    steps: int = 300
    lr: float = 0.1
    n_restarts: int = 1
    base_logit: float = 10.0   # non-candidate keep logit (~1)
    init_logit: float = -2.0   # initial candidate keep logit (start sparse)
    clamp_logits: float = 12.0
    lambda_size: float = 0.02
    lambda_entropy: float = 0.001
    size_mode: Literal["sum", "mean"] = "sum"
    early_stop: bool = True

    # Constraint
    epsilon: float = 0.05

    # Scoring
    irrelevant_score: float = -1e9


class EpsilonSufficientEdgeExplainer(EdgeExplainer):
    """Sufficient explainer: keep a small edge set that preserves p(y|v)."""

    def __init__(self, *, cfg: EpsilonSufficientConfig = EpsilonSufficientConfig(), seed: int = 0):
        self.cfg = cfg
        self.seed = int(seed)

        if int(self.cfg.k_hops) <= 0:
            raise ValueError("k_hops must be >= 1")
        if int(self.cfg.steps) <= 0:
            raise ValueError("steps must be >= 1")
        if int(self.cfg.n_restarts) <= 0:
            raise ValueError("n_restarts must be >= 1")
        if self.cfg.size_mode not in {"sum", "mean"}:
            raise ValueError("size_mode must be 'sum' or 'mean'")

    @contextlib.contextmanager
    def _freeze_model(self, model: torch.nn.Module):
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
                "EpsilonSufficientEdgeExplainer requires ExplainSpace.edge_weight_attr to be set "
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

        # Baseline distribution p_full (no gradients needed)
        logits0 = adapter.predict_logits(data)
        row0 = logits0[int(node_idx)]
        p_full = torch.softmax(row0, dim=-1).detach()
        log_p_full = torch.log(p_full + 1e-12)

        # Candidate edges
        cand_mask = _candidate_edge_mask_from_khop_ho(
            data,
            edge_index=edge_index,
            node_idx=int(node_idx),
            k_hops=int(self.cfg.k_hops),
            undirected=bool(self.cfg.undirected_khop),
            triples_attr=str(self.cfg.triples_attr),
        )
        if cand_mask is None:
            cand_mask = default_candidate_mask_from_triples(
                data,
                E=E,
                node_idx=int(node_idx),
                triples_attr=str(self.cfg.triples_attr),
            )
        if cand_mask is None:
            cand_mask = torch.ones(E, dtype=torch.bool, device=edge_index.device)

        cand_mask = cand_mask.to(device=edge_index.device, dtype=torch.bool).view(-1)
        if cand_mask.numel() != E:
            raise ValueError(f"candidate_mask has wrong shape: expected [E]={E}, got {tuple(cand_mask.shape)}")

        cand_idx = cand_mask.nonzero(as_tuple=False).view(-1)
        C = int(cand_idx.numel())
        if C == 0:
            scores = torch.full((E,), float(self.cfg.irrelevant_score), device=edge_index.device)
            return EdgeExplanation(
                node_idx=int(node_idx),
                target_class=int(target_class),
                edge_index=edge_index,
                edge_score=scores,
                candidate_mask=cand_mask,
            )

        data_work = clone_data(data)
        adapter.model.eval()

        best_loss = float("inf")
        best_mask_logits_var: Optional[torch.Tensor] = None
        best_success_key: Optional[tuple[float, float]] = None
        best_success_mask_logits_var: Optional[torch.Tensor] = None

        eps = 1e-8
        clamp = float(self.cfg.clamp_logits)
        epsilon = float(self.cfg.epsilon)

        g = torch.Generator(device=edge_index.device)
        g.manual_seed(self.seed + int(node_idx) * 1009)

        with self._freeze_model(adapter.model):
            for _restart in range(int(self.cfg.n_restarts)):
                init = torch.full((C,), float(self.cfg.init_logit), device=edge_index.device)
                if int(self.cfg.n_restarts) > 1:
                    init = init + 0.1 * torch.randn((C,), generator=g, device=edge_index.device)
                mask_logits_var = torch.nn.Parameter(init)
                opt = torch.optim.Adam([mask_logits_var], lr=float(self.cfg.lr))

                for _step in range(int(self.cfg.steps)):
                    opt.zero_grad(set_to_none=True)

                    base = torch.full((E,), float(self.cfg.base_logit), device=edge_index.device)
                    mask_logits_full = base.index_put((cand_idx,), mask_logits_var)
                    keep_prob_full = torch.sigmoid(mask_logits_full)

                    w_masked = w_base * keep_prob_full
                    setattr(data_work, space.edge_weight_attr, w_masked)

                    if hasattr(adapter, "reset_caches"):
                        try:
                            adapter.reset_caches()  # type: ignore[attr-defined]
                        except Exception:
                            pass

                    logits = adapter.model(data_work)
                    row = logits[int(node_idx)]

                    log_q = F.log_softmax(row, dim=-1)
                    kl = torch.sum(p_full * (log_p_full - log_q))
                    loss_pred = F.relu(kl - epsilon)

                    keep_prob_c = torch.sigmoid(mask_logits_var)
                    if self.cfg.size_mode == "sum":
                        loss_size = keep_prob_c.sum()
                    else:
                        loss_size = keep_prob_c.mean()

                    ent = -(keep_prob_c * torch.log(keep_prob_c + eps) + (1 - keep_prob_c) * torch.log(1 - keep_prob_c + eps))
                    loss_ent = ent.mean()

                    loss = loss_pred + float(self.cfg.lambda_size) * loss_size + float(self.cfg.lambda_entropy) * loss_ent

                    lval = float(loss.item())
                    snap = mask_logits_var.detach().clone()
                    if lval < best_loss:
                        best_loss = lval
                        best_mask_logits_var = snap

                    # Track best solution that satisfies KL <= epsilon
                    if float(kl.item()) <= epsilon:
                        keep_sum = float(keep_prob_c.sum().item())
                        key = (keep_sum, float(loss_pred.item()))
                        if best_success_key is None or key < best_success_key:
                            best_success_key = key
                            best_success_mask_logits_var = snap

                    loss.backward()
                    opt.step()

                    with torch.no_grad():
                        mask_logits_var.clamp_(-clamp, clamp)

                    if self.cfg.early_stop and float(kl.item()) <= epsilon:
                        break

        if best_success_mask_logits_var is not None:
            best_mask_logits_var = best_success_mask_logits_var

        if best_mask_logits_var is None:
            best_mask_logits_var = torch.full((C,), float(self.cfg.init_logit), device=edge_index.device)

        keep_prob_best = torch.sigmoid(best_mask_logits_var).detach()
        scores = torch.full((E,), float(self.cfg.irrelevant_score), device=edge_index.device)
        scores = scores.index_put((cand_idx,), keep_prob_best)

        return EdgeExplanation(
            node_idx=int(node_idx),
            target_class=int(target_class),
            edge_index=edge_index,
            edge_score=scores,  # keep-importance
            candidate_mask=cand_mask,
        )


__all__ = [
    "EpsilonSufficientConfig",
    "EpsilonSufficientEdgeExplainer",
]
