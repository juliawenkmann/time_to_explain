from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F

from explainers.base import EdgeExplainer, EdgeExplanation
from eval.metrics import default_candidate_mask_from_triples
from utils import clone_data


FlipTo = Literal[
    "non_targeted",  # flip away from the current prediction (any other class)
    "second_best",   # flip specifically towards the 2nd-best class (baseline)
    "true_label",    # flip towards data.y[node_idx] if available, else non_targeted
]


@dataclass(frozen=True)
class CounterfactualConfig:
    """Configuration for counterfactual edge-deletion explanations.

    This explainer is designed for the DBGNN / De Bruijn setting, where we
    explain *higher-order* edges (triples).

    The explainer learns a continuous **keep-probability** for candidate edges
    (GNNExplainer-style), but with the objective reversed: **make the original
    prediction fail** (or reach a desired target class), while penalizing the
    number of deleted edges.
    """

    # Candidate restriction
    k_hops: int = 4
    undirected_khop: bool = True
    triples_attr: str = "ho_triples"

    # Optimization
    steps: int = 300
    lr: float = 0.1
    n_restarts: int = 1
    base_logit: float = 10.0   # non-candidate keep logit (sigmoid(base_logit) ~ 1)
    init_logit: float = 10.0   # initial candidate keep logit
    clamp_logits: float = 12.0
    lambda_size: float = 0.02
    lambda_entropy: float = 0.001
    size_mode: Literal["sum", "mean"] = "sum"
    early_stop: bool = True

    # Flip target
    flip_to: FlipTo = "non_targeted"
    # Optional explicit desired class. If set, overrides flip_to logic.
    desired_class: Optional[int] = None

    # Scoring
    # When candidate_mask is used, non-candidate edges are set to irrelevant_score
    # to avoid accidental selection under global top-k.
    irrelevant_score: float = -1e9


def _k_hop_node_mask(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    seed_nodes: torch.Tensor,
    k_hops: int,
    undirected: bool = True,
) -> torch.Tensor:
    """k-hop neighbourhood mask for a (higher-order) graph.

    Args:
        edge_index: [2, E]
        num_nodes: number of nodes in this graph
        seed_nodes: [S] node indices
        k_hops: number of BFS hops
        undirected: treat edge_index as undirected

    Returns:
        visited mask [num_nodes] bool.
    """
    device = edge_index.device
    visited = torch.zeros(int(num_nodes), dtype=torch.bool, device=device)
    seed_nodes = seed_nodes.to(device=device, dtype=torch.long).view(-1)
    if seed_nodes.numel() == 0:
        return visited

    visited[seed_nodes] = True
    frontier = seed_nodes

    src, dst = edge_index[0], edge_index[1]

    for _ in range(int(k_hops)):
        if frontier.numel() == 0:
            break

        frontier_mask = torch.zeros(int(num_nodes), dtype=torch.bool, device=device)
        frontier_mask[frontier] = True

        nbrs_fwd = dst[frontier_mask[src]]
        if undirected:
            nbrs_bwd = src[frontier_mask[dst]]
            nbrs = torch.cat([nbrs_fwd, nbrs_bwd], dim=0)
        else:
            nbrs = nbrs_fwd

        if nbrs.numel() == 0:
            break

        nbrs = torch.unique(nbrs)
        nbrs = nbrs[~visited[nbrs]]
        if nbrs.numel() == 0:
            break

        visited[nbrs] = True
        frontier = nbrs

    return visited


def _candidate_edge_mask_from_khop_ho(
    data,
    *,
    edge_index: torch.Tensor,
    node_idx: int,
    k_hops: int,
    undirected: bool,
    triples_attr: str,
) -> Optional[torch.Tensor]:
    """Candidate mask over higher-order edges using a k-hop neighbourhood.

    Assumptions (match the repo notebooks):
      - explain-space edges are higher-order edges corresponding to triples (u,v,w)
      - the underlying order-2 nodes are pairs (u,v) and (v,w)
      - the explained first-order node is the *last element* of those pairs

    We use `data.<triples_attr>` aligned with `edge_index` to find all higher-order
    nodes whose last element equals `node_idx` (i.e., (*, node_idx)), then take a
    k-hop neighbourhood in the higher-order graph and mark all edges incident to
    that neighbourhood as candidates.
    """
    if k_hops <= 0:
        return None
    if not hasattr(data, triples_attr):
        return None
    triples = getattr(data, triples_attr)
    if triples is None or not isinstance(triples, torch.Tensor):
        return None
    if triples.ndim != 2 or triples.size(0) != int(edge_index.size(1)) or triples.size(1) != 3:
        return None

    device = edge_index.device
    triples = triples.to(device=device)

    # Find higher-order node indices that end in node_idx.
    # For each HO edge corresponding to (u,v,w):
    #   src HO node is (u,v)  -> last is v
    #   dst HO node is (v,w)  -> last is w
    src_ho, dst_ho = edge_index[0], edge_index[1]
    v = triples[:, 1]
    w = triples[:, 2]

    seed_src = src_ho[v == int(node_idx)]
    seed_dst = dst_ho[w == int(node_idx)]
    seed = torch.unique(torch.cat([seed_src, seed_dst], dim=0))
    if seed.numel() == 0:
        return None

    # Node count in HO graph
    if hasattr(data, "x_h") and isinstance(getattr(data, "x_h"), torch.Tensor):
        num_ho_nodes = int(getattr(data, "x_h").size(0))
    else:
        num_ho_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else 0
    if num_ho_nodes <= 0:
        return None

    ho_node_mask = _k_hop_node_mask(
        edge_index,
        num_nodes=num_ho_nodes,
        seed_nodes=seed,
        k_hops=int(k_hops),
        undirected=bool(undirected),
    )

    cand_edges = ho_node_mask[src_ho] | ho_node_mask[dst_ho]
    return cand_edges.to(torch.bool)


class CounterfactualEdgeDeletionExplainer(EdgeExplainer):
    """Counterfactual explainer via minimal deletions of higher-order edges.

    Output semantics:
      - `edge_score` is a **deletion priority** score (higher => remove first)
        derived from (1 - keep_prob) after optimization.
      - `candidate_mask` restricts selection to a k-hop neighbourhood in the
        higher-order graph.

    This makes the explainer compatible with the existing benchmark:
      - comprehensiveness: dropping top-k edges should flip more often
      - counterfactual size: the new `k_flip` metric (added in this patch)
        tells you how many top-ranked deletions are needed to flip.
    """

    def __init__(
        self,
        *,
        cfg: CounterfactualConfig = CounterfactualConfig(),
        seed: int = 0,
    ):
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

    def _resolve_desired_class(self, *, data, logits0_row: torch.Tensor, orig_class: int, node_idx: int) -> Optional[int]:
        """Choose the desired flip-to class, if any."""

        if self.cfg.desired_class is not None:
            return int(self.cfg.desired_class)

        mode = self.cfg.flip_to
        if mode == "non_targeted":
            return None

        if mode == "second_best":
            # choose the highest logit among the non-orig classes
            row = logits0_row.detach()
            if row.numel() <= 1:
                return None
            tmp = row.clone()
            tmp[int(orig_class)] = -float("inf")
            return int(tmp.argmax().item())

        if mode == "true_label":
            if hasattr(data, "y") and isinstance(getattr(data, "y"), torch.Tensor):
                y = getattr(data, "y")
                if int(node_idx) < int(y.numel()):
                    yc = int(y[int(node_idx)].item())
                    # If the true label equals the original prediction, fall back to non-targeted.
                    return None if yc == int(orig_class) else yc
            return None

        # Should be unreachable due to typing
        return None

    def explain_node(self, *, adapter, data, node_idx: int, target_class: int) -> EdgeExplanation:
        space = adapter.explain_space()
        if space.edge_weight_attr is None:
            raise ValueError(
                "CounterfactualEdgeDeletionExplainer requires ExplainSpace.edge_weight_attr to be set "
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

        # Baseline logits for flip target selection (no gradients needed)
        logits0 = adapter.predict_logits(data)
        logits0_row = logits0[int(node_idx)]
        orig_class = int(target_class)
        desired = self._resolve_desired_class(data=data, logits0_row=logits0_row, orig_class=orig_class, node_idx=int(node_idx))

        # Candidate edges: always restrict to k-hop (preferred), else fall back to middle-node triples.
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
            # As a last resort, allow all edges.
            cand_mask = torch.ones(E, dtype=torch.bool, device=edge_index.device)

        cand_mask = cand_mask.to(device=edge_index.device, dtype=torch.bool).view(-1)
        if cand_mask.numel() != E:
            raise ValueError(f"candidate_mask has wrong shape: expected [E]={E}, got {tuple(cand_mask.shape)}")

        cand_idx = cand_mask.nonzero(as_tuple=False).view(-1)
        C = int(cand_idx.numel())
        if C == 0:
            # Degenerate: no candidates => no-op scores.
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
        # Track a mask that actually achieves the flip objective (if we ever find one).
        # This makes the explainer more reliable on datasets where a low-loss solution
        # might reduce the margin without changing the argmax.
        best_success_key: Optional[tuple[float, float]] = None
        best_success_mask_logits_var: Optional[torch.Tensor] = None

        eps = 1e-8
        clamp = float(self.cfg.clamp_logits)

        # Deterministic per-node randomness
        g = torch.Generator(device=edge_index.device)
        g.manual_seed(self.seed + int(node_idx) * 1009)

        def _pred_success(pred: int) -> bool:
            if desired is None:
                return pred != orig_class
            return pred == int(desired)

        with self._freeze_model(adapter.model):
            for _restart in range(int(self.cfg.n_restarts)):
                # Learn logits for candidate edge keep-probabilities.
                # Non-candidates are fixed to base_logit (~keep=1).
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

                    # Apply mask multiplicatively to edge weights.
                    w_masked = w_base * keep_prob_full
                    setattr(data_work, space.edge_weight_attr, w_masked)

                    # Clear caches so edge perturbations take effect.
                    if hasattr(adapter, "reset_caches"):
                        try:
                            adapter.reset_caches()  # type: ignore[attr-defined]
                        except Exception:
                            pass

                    logits = adapter.model(data_work)
                    row = logits[int(node_idx)]

                    # Flip objective (margin-based, robust to probability saturation)
                    if desired is None:
                        # non-targeted: make orig_class NOT the argmax
                        if row.numel() <= 1:
                            margin = row[int(orig_class)]
                        else:
                            other = torch.cat([row[: int(orig_class)], row[int(orig_class) + 1 :]], dim=0)
                            margin = row[int(orig_class)] - other.max()
                    else:
                        # targeted: make desired class the argmax
                        d = int(desired)
                        if row.numel() <= 1:
                            margin = -row[d]
                        else:
                            other = torch.cat([row[:d], row[d + 1 :]], dim=0)
                            margin = other.max() - row[d]

                    loss_pred = F.softplus(margin)

                    # Sparsity in terms of deletions (candidate edges only)
                    keep_prob_c = torch.sigmoid(mask_logits_var)
                    del_soft = 1.0 - keep_prob_c
                    if self.cfg.size_mode == "sum":
                        loss_size = del_soft.sum()
                    else:
                        loss_size = del_soft.mean()

                    # Entropy regularizer towards hard {0,1}
                    ent = -(keep_prob_c * torch.log(keep_prob_c + eps) + (1 - keep_prob_c) * torch.log(1 - keep_prob_c + eps))
                    loss_ent = ent.mean()

                    loss = loss_pred + float(self.cfg.lambda_size) * loss_size + float(self.cfg.lambda_entropy) * loss_ent

                    # Snapshot current candidate logits BEFORE the optimizer step so they correspond to this forward pass.
                    pred = int(row.argmax().item())
                    lval = float(loss.item())
                    snap = mask_logits_var.detach().clone()

                    # Track best parameters by scalar loss (may not flip).
                    if lval < best_loss:
                        best_loss = lval
                        best_mask_logits_var = snap

                    # Track best parameters among successful flips (if any).
                    if _pred_success(pred):
                        # Prefer fewer deletions; tie-break by the prediction loss term.
                        del_sum = float(del_soft.sum().item())
                        key = (del_sum, float(loss_pred.item()))
                        if best_success_key is None or key < best_success_key:
                            best_success_key = key
                            best_success_mask_logits_var = snap

                    loss.backward()
                    opt.step()

                    with torch.no_grad():
                        mask_logits_var.clamp_(-clamp, clamp)

                    if self.cfg.early_stop and _pred_success(pred):
                        break

        # If we ever found a successful flip, prefer it over a merely low-loss solution.
        if best_success_mask_logits_var is not None:
            best_mask_logits_var = best_success_mask_logits_var

        if best_mask_logits_var is None:
            # Should never happen, but keep it safe.
            best_mask_logits_var = torch.full((C,), float(self.cfg.init_logit), device=edge_index.device)

        # Convert keep-probabilities into deletion scores.
        keep_prob_best = torch.sigmoid(best_mask_logits_var).detach()
        delete_score = 1.0 - keep_prob_best  # higher => remove first

        scores = torch.full((E,), float(self.cfg.irrelevant_score), device=edge_index.device)
        scores = scores.index_put((cand_idx,), delete_score)

        return EdgeExplanation(
            node_idx=int(node_idx),
            target_class=int(target_class),
            edge_index=edge_index,
            edge_score=scores,
            candidate_mask=cand_mask,
        )
