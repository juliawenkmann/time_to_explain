from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from explainers.base import EdgeExplainer, EdgeExplanation


FocusMode = Literal["middle", "none"]


@dataclass(frozen=True)
class StayGTAttrs:
    """Names of Data attributes needed by StayGTOracleExplainer."""

    score_attr: str = "gt_stay_score_higher_order"   # float tensor [E] in {0,1}
    label_attr: str = "gt_stay_label_higher_order"   # float tensor [E] in {0,1}
    triples_attr: str = "ho_triples"                 # long tensor [E,3] with columns (u,v,w)


class StayGTOracleExplainer(EdgeExplainer):
    """Oracle explainer based on deterministic 'cluster stay' ground truth.

    Explanation atoms:
      - higher-order edges ( (u,v) -> (v,w) ), represented as triples (u,v,w)

    Ground truth definition (temporal_clusters):
      A triple (u,v,w) is GT-positive iff:
          cluster(u) == cluster(v) == cluster(w)
      where cluster(x) = floor(x/10).

    This GT is *class-conditional* through v (because v's cluster is its label),
    but does not depend on the model's predicted class.

    Requirements:
      `data` must have:
        - data.<score_attr> : float tensor [E] aligned with explain-space edges
        - data.<triples_attr> : long tensor [E,3]
    """

    def __init__(
        self,
        *,
        attrs: StayGTAttrs = StayGTAttrs(),
        focus: FocusMode = "middle",
        positive_only: bool = False,
        irrelevant_score: float = -1e9,
    ):
        self.attrs = attrs
        self.focus = focus
        self.positive_only = bool(positive_only)
        self.irrelevant_score = float(irrelevant_score)

        if self.focus not in {"middle", "none"}:
            raise ValueError("focus must be one of {'middle','none'}")

    def explain_node(self, *, adapter, data, node_idx: int, target_class: int) -> EdgeExplanation:
        space = adapter.explain_space()
        edge_index = getattr(data, space.edge_index_attr)

        if not hasattr(data, self.attrs.score_attr):
            raise AttributeError(
                f"data has no attribute {self.attrs.score_attr!r}. "
                "Enable attach_stay_gt in the dataset loader."
            )
        if not hasattr(data, self.attrs.triples_attr):
            raise AttributeError(
                f"data has no attribute {self.attrs.triples_attr!r}. "
                "Enable ho_triples attachment in the dataset loader."
            )

        score = getattr(data, self.attrs.score_attr)
        triples = getattr(data, self.attrs.triples_attr)

        if not isinstance(score, torch.Tensor) or score.dim() != 1:
            raise TypeError(f"data.{self.attrs.score_attr} must be a 1D torch.Tensor")
        if not isinstance(triples, torch.Tensor) or triples.ndim != 2 or triples.size(1) != 3:
            raise TypeError(f"data.{self.attrs.triples_attr} must be a [E,3] torch.Tensor")

        # Node-focused candidate set: triples with middle == node_idx
        cand = None
        if self.focus == "middle":
            cand = (triples[:, 1] == int(node_idx)).to(torch.bool)

        s = score.to(torch.float32)

        if self.positive_only:
            s = torch.where(s > 0, s, torch.zeros_like(s))

        # Keep non-candidates unselectable (for global top-k usage)
        if cand is not None:
            s = torch.where(cand, s, torch.full_like(s, float(self.irrelevant_score)))

        return EdgeExplanation(
            node_idx=int(node_idx),
            target_class=int(target_class),
            edge_index=edge_index,
            edge_score=s,
            candidate_mask=cand,
        )
