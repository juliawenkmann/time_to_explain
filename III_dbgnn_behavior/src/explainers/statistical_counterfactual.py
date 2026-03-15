from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from explainers.base import EdgeExplainer, EdgeExplanation
from eval.metrics import default_candidate_mask_from_triples
from explainers.counterfactual import _candidate_edge_mask_from_khop_ho


ScoreTransform = Literal[
    "pos_z",   # max(z, 0)
    "abs_z",   # |z|
    "z",       # raw z (can be negative)
    "binary",  # 1[z > z_thr], else 0
]


@dataclass(frozen=True)
class StatisticalCounterfactualConfig:
    """Configuration for a *statistical* counterfactual edge-deletion explainer.

    This explainer ranks higher-order (De Bruijn) edges by how *unexpected* they are
    under a shuffled-time null model, and uses that ranking as a deletion priority.

    It expects the shuffled-time statistics (z-scores) to be already attached to the
    input PyG `data`, aligned with the explain-space edge list.

    See :func:`eval.shuffle_gt.attach_shuffle_gt_to_data` for a helper
    that computes and attaches these tensors for TemporalClusters / netzschleuder
    temporal datasets.

    Notes
    -----
    - Output `edge_score` is a **deletion priority** score (higher => remove first).
    - `candidate_mask` is (by default) restricted to a k-hop neighbourhood in the
      higher-order graph, matching the counterfactual optimizer's candidate logic.
    """

    # Candidate restriction (same semantics as CounterfactualConfig)
    k_hops: int = 2
    undirected_khop: bool = True
    triples_attr: str = "ho_triples"

    # Where to read null-model statistics from
    z_attr: str = "gt_z_higher_order"

    # How to turn z-scores into deletion scores
    score_transform: ScoreTransform = "pos_z"
    z_thr: float = 2.0  # used only for score_transform="binary"

    # Ranking hygiene
    irrelevant_score: float = -1e9


class StatisticalCounterfactualEdgeDeletionExplainer(EdgeExplainer):
    """Counterfactual explainer that deletes edges based on shuffled-null statistics.

    Conceptually:
      1) compute (or load) z-scores vs shuffled-time null model for each HO edge
      2) rank edges by a transform of z (e.g., positive z or |z|)
      3) delete top-k edges to form a counterfactual

    This explainer only *ranks* edges. Finding a minimal k that flips a prediction
    is handled by :func:`eval.metrics.min_k_to_flip` or the notebook helper
    :func:`explainers.counterfactual_search.find_min_ho_edge_deletions_to_flip`.
    """

    def __init__(self, *, cfg: StatisticalCounterfactualConfig = StatisticalCounterfactualConfig()):
        self.cfg = cfg

        if int(self.cfg.k_hops) < 0:
            raise ValueError("k_hops must be >= 0")

        if self.cfg.score_transform not in {"pos_z", "abs_z", "z", "binary"}:
            raise ValueError("score_transform must be one of {'pos_z','abs_z','z','binary'}")

    def _transform(self, z: torch.Tensor) -> torch.Tensor:
        if self.cfg.score_transform == "abs_z":
            return z.abs()
        if self.cfg.score_transform == "pos_z":
            return torch.clamp(z, min=0.0)
        if self.cfg.score_transform == "binary":
            return (z > float(self.cfg.z_thr)).to(torch.float32)
        # raw
        return z

    def explain_node(self, *, adapter, data, node_idx: int, target_class: int) -> EdgeExplanation:
        space = adapter.explain_space()
        edge_index = getattr(data, space.edge_index_attr)
        E = int(edge_index.size(1))

        if E == 0:
            raise ValueError("Graph has 0 edges in the explain space")

        if not hasattr(data, self.cfg.z_attr):
            raise AttributeError(
                f"Data has no attribute {self.cfg.z_attr!r}. "
                "Attach shuffled-null statistics first (see attach_shuffle_gt_to_data). "
            )

        z = getattr(data, self.cfg.z_attr)
        if z is None or not isinstance(z, torch.Tensor):
            raise ValueError(f"data.{self.cfg.z_attr} was None or not a torch.Tensor")

        z = z.detach().to(device=edge_index.device).float().view(-1)
        if z.numel() != E:
            raise ValueError(
                f"Expected data.{self.cfg.z_attr} to have shape [E] with E={E}, got {tuple(z.shape)}"
            )

        # Candidate edges: prefer k-hop HO neighbourhood; fall back to 'middle node equals v' triples.
        cand_mask = None
        if int(self.cfg.k_hops) > 0:
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

        scores_cand = self._transform(z)

        # Push non-candidates far down the ranking (helpful when people accidentally do global top-k).
        scores = scores_cand.clone()
        scores[~cand_mask] = float(self.cfg.irrelevant_score)

        return EdgeExplanation(
            node_idx=int(node_idx),
            target_class=int(target_class),
            edge_index=edge_index,
            edge_score=scores,
            candidate_mask=cand_mask,
        )
