from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from eval.metrics import target_log_prob, target_logit, target_margin, target_prob
from explainers.base import EdgeExplainer, EdgeExplanation


@dataclass(frozen=True)
class LIMEConfig:
    """Configuration for the LIME-style explainer on edges.

    We treat *candidate* edges as binary features (present/absent) and fit a
    weighted linear surrogate model around the original graph.

    This is intentionally minimal and designed for small candidate sets
    (node-focused explanations).
    """

    n_samples: int = 256
    p_keep: float = 0.5
    # Kernel for sample weighting:
    #   - "lime": exp(-d^2 / w^2) where d is normalized Hamming distance to the original.
    #   - "shap": KernelSHAP-style Shapley kernel weight (depends on subset size).
    kernel: str = "lime"  # one of: "lime", "shap"
    kernel_width: float = 0.25
    ridge_lambda: float = 1e-3
    output: str = "logit"  # one of: "logit", "margin", "logprob", "prob"
    score_mode: str = "coef"  # one of: "coef", "abs", "positive"
    irrelevant_score: float = -1e9
    focus: str = "middle"  # only supported: "middle" (triple middle-node)
    triples_attr: str = "ho_triples"


class LIMEEdgeExplainer(EdgeExplainer):
    """LIME-style local surrogate explainer for higher-order edges.

    Explanation atoms: higher-order edges (triples) (u,v)->(v,w), represented
    by the `edge_index` in the model's explain space.

    For a given node `v`, we typically restrict the candidate set to triples
    whose middle node equals `v` (requires `data.ho_triples`).
    """

    def __init__(self, *, seed: int = 0, config: Optional[LIMEConfig] = None, **kwargs: object):
        self.seed = int(seed)
        if config is None:
            config = LIMEConfig(**{k: v for k, v in kwargs.items() if k in LIMEConfig.__annotations__})
        self.cfg = config

    def _candidate_mask(self, *, data, E: int, node_idx: int, device: torch.device) -> torch.Tensor:
        """Return candidate edge mask of shape [E]."""

        # Default: all edges are candidates.
        cand = torch.ones(E, dtype=torch.bool, device=device)

        if self.cfg.focus != "middle":
            raise ValueError(f"Unsupported focus={self.cfg.focus!r}. Only 'middle' is supported.")

        if hasattr(data, self.cfg.triples_attr):
            triples = getattr(data, self.cfg.triples_attr)
            if triples is not None and triples.ndim == 2 and triples.size(1) == 3 and triples.size(0) == E:
                cand = (triples[:, 1] == int(node_idx)).to(device=device)

        return cand

    def _target_value(self, logits: torch.Tensor, node_idx: int, target_class: int) -> float:
        out = self.cfg.output
        if out == "logit":
            return target_logit(logits, node_idx, target_class)
        if out == "margin":
            return target_margin(logits, node_idx, target_class)
        if out == "logprob":
            return target_log_prob(logits, node_idx, target_class)
        if out == "prob":
            return target_prob(logits, node_idx, target_class)
        raise ValueError(f"Unknown output={out!r}. Use one of: logit, margin, logprob, prob")

    def explain_node(self, *, adapter, data, node_idx: int, target_class: int) -> EdgeExplanation:
        space = adapter.explain_space()
        edge_index = getattr(data, space.edge_index_attr)
        E = int(edge_index.size(1))
        device = edge_index.device

        edge_weight_full = None
        if space.edge_weight_attr is not None and hasattr(data, space.edge_weight_attr):
            edge_weight_full = getattr(data, space.edge_weight_attr)

        candidate_mask = self._candidate_mask(data=data, E=E, node_idx=int(node_idx), device=device)
        cand_idx = candidate_mask.nonzero(as_tuple=False).view(-1)
        C = int(cand_idx.numel())

        # Degenerate case: no candidates
        if C == 0 or self.cfg.n_samples <= 1:
            scores = torch.full((E,), float(self.cfg.irrelevant_score), device=device)
            return EdgeExplanation(
                node_idx=int(node_idx),
                target_class=int(target_class),
                edge_index=edge_index,
                edge_score=scores,
                candidate_mask=candidate_mask,
            )

        # Pre-slice edge tensors for faster concatenation.
        non_idx = (~candidate_mask).nonzero(as_tuple=False).view(-1)
        ei_non = edge_index[:, non_idx]
        ei_cand = edge_index[:, cand_idx]
        ew_non = edge_weight_full[non_idx] if edge_weight_full is not None else None
        ew_cand = edge_weight_full[cand_idx] if edge_weight_full is not None else None

        # Sample binary perturbations over candidate edges.
        n_samples = int(self.cfg.n_samples)
        g = torch.Generator(device=device)
        g.manual_seed(self.seed + int(node_idx))

        Z = (torch.rand((n_samples, C), generator=g, device=device) < float(self.cfg.p_keep))
        Z[0, :] = True  # include the original instance

        # Collect surrogate training data.
        y = torch.empty(n_samples, device=device, dtype=torch.float32)

        # Kernel weights for local surrogate fitting.
        kernel = self.cfg.kernel
        if kernel == "lime":
            dist = 1.0 - Z.float().mean(dim=1)  # fraction removed
            kw = float(self.cfg.kernel_width)
            weights = torch.exp(-(dist ** 2) / max(1e-12, kw ** 2)).float()
        elif kernel == "shap":
            # KernelSHAP-style Shapley kernel on subset size.
            # Include the all-zeros sample (baseline) explicitly.
            if n_samples >= 2:
                Z[1, :] = False

            s = Z.sum(dim=1).float()  # subset size
            m = float(C)
            weights = torch.zeros(n_samples, device=device, dtype=torch.float32)
            mid = (s > 0.0) & (s < m)
            if bool(mid.any()):
                # log comb(m, s) = lgamma(m+1) - lgamma(s+1) - lgamma(m-s+1)
                sm = s[mid]
                mt = torch.tensor(m, device=device, dtype=torch.float32)
                log_comb = torch.lgamma(mt + 1.0) - torch.lgamma(sm + 1.0) - torch.lgamma(mt - sm + 1.0)
                log_w = torch.log(torch.tensor(m - 1.0, device=device, dtype=torch.float32)) - (
                    log_comb + torch.log(sm) + torch.log(mt - sm)
                )
                weights[mid] = torch.exp(log_w).float()

            # Enforce baseline/full samples with a large weight.
            weights[~mid] = 1e6
        else:
            raise ValueError(f"Unknown kernel={kernel!r}. Use one of: lime, shap")

        # Numerical safety: keep weights finite and avoid extreme values that
        # can lead to ill-conditioned normal equations.
        weights = torch.clamp(weights, min=1e-12, max=1e6)

        for i in range(n_samples):
            zi = Z[i]
            ei = torch.cat([ei_non, ei_cand[:, zi]], dim=1)
            if edge_weight_full is not None:
                ew = torch.cat([ew_non, ew_cand[zi]], dim=0)  # type: ignore[arg-type]
            else:
                ew = None

            data_i = adapter.clone_with_perturbed_edges(data, ei, new_edge_weight=ew)
            logits_i = adapter.predict_logits(data_i)
            y[i] = float(self._target_value(logits_i, int(node_idx), int(target_class)))

        # Weighted ridge regression: y ≈ b + X w
        X = Z.float()
        ones = torch.ones((n_samples, 1), device=device, dtype=torch.float32)
        Xd = torch.cat([ones, X], dim=1)  # [S, 1+C]
        w = weights.view(-1, 1)

        XtW = (Xd * w).transpose(0, 1)  # [1+C, S]
        A = XtW @ Xd  # [1+C, 1+C]
        b = (XtW @ y.view(-1, 1)).view(-1)  # [1+C]

        # Ridge regularization for numerical stability.
        # In theory, adding lam*I makes A invertible for lam>0, but in practice
        # (especially for KernelSHAP weights) the system can still become
        # singular/ill-conditioned. We therefore:
        #   1) ensure a strictly positive, scale-aware lambda
        #   2) symmetrize A numerically
        #   3) fall back to least-squares if a direct solve fails
        lam = float(self.cfg.ridge_lambda)
        diag_mean = float(torch.diagonal(A).abs().mean().detach().cpu()) if A.numel() else 1.0
        lam_eff = max(lam, 1e-6 * max(1.0, diag_mean))
        A = A + lam_eff * torch.eye(A.size(0), device=device, dtype=A.dtype)
        A = 0.5 * (A + A.transpose(0, 1))

        try:
            beta = torch.linalg.solve(A, b)  # [1+C]
        except Exception:
            beta = torch.linalg.lstsq(A, b.unsqueeze(1)).solution.squeeze(1)
        coef = beta[1:]  # [C]

        mode = self.cfg.score_mode
        if mode == "abs":
            coef = coef.abs()
        elif mode == "positive":
            coef = torch.clamp(coef, min=0.0)
        elif mode != "coef":
            raise ValueError(f"Unknown score_mode={mode!r}. Use one of: coef, abs, positive")

        scores = torch.full((E,), float(self.cfg.irrelevant_score), device=device, dtype=torch.float32)
        scores[cand_idx] = coef.detach().float()

        return EdgeExplanation(
            node_idx=int(node_idx),
            target_class=int(target_class),
            edge_index=edge_index,
            edge_score=scores,
            candidate_mask=candidate_mask,
        )
