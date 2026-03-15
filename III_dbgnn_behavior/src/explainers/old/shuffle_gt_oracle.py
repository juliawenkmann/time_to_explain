from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from explainers.base import EdgeExplainer, EdgeExplanation


FocusMode = Literal["middle", "none"]
ScoreMode = Literal["z", "binary"]


@dataclass(frozen=True)
class ShuffleGTAttrs:
    """Names of Data attributes needed by ShuffleGTOracleExplainer."""

    z_attr: str = "gt_z_higher_order"      # float tensor [E]
    triples_attr: str = "ho_triples"       # long tensor [E, 3] with columns (u, v, w)


class ShuffleGTOracleExplainer(EdgeExplainer):
    """Oracle explainer that returns *ground-truth* importance for higher-order edges.

    This explainer is intentionally "unfair": it uses ground-truth statistics
    (z-scores vs a shuffled-time null model) to score De Bruijn transitions.

    Explanation atoms:
      - higher-order edges ( (u,v) -> (v,w) ), i.e. triples (u, v, w)

    Requirements:
      The input `data` must already have:
        - `data.<z_attr>`: float tensor [E] aligned with `adapter.explain_space()` edges
        - `data.<triples_attr>`: long tensor [E,3] aligned with explain-space edges,
          with columns (u, v, w)

    Typical workflow:
      - call `eval.shuffle_gt.attach_shuffle_gt_to_data(...)` once
      - then run benchmarks with this explainer
    """

    def __init__(
        self,
        *,
        attrs: ShuffleGTAttrs = ShuffleGTAttrs(),
        focus: FocusMode = "middle",
        score_mode: ScoreMode = "z",
        z_thr: float = 2.0,
        positive_only: bool = False,
        irrelevant_score: float = -1e9,
    ):
        self.attrs = attrs
        self.focus = focus
        self.score_mode = score_mode
        self.z_thr = float(z_thr)
        self.positive_only = bool(positive_only)
        self.irrelevant_score = float(irrelevant_score)

        if self.focus not in {"middle", "none"}:
            raise ValueError("focus must be one of {'middle','none'}")
        if self.score_mode not in {"z", "binary"}:
            raise ValueError("score_mode must be one of {'z','binary'}")

    def explain_node(self, *, adapter, data, node_idx: int, target_class: int) -> EdgeExplanation:
        space = adapter.explain_space()
        edge_index = getattr(data, space.edge_index_attr)
        E = int(edge_index.size(1))

        if not hasattr(data, self.attrs.z_attr):
            raise AttributeError(
                f"Data has no attribute {self.attrs.z_attr!r}. "

                "Call attach_shuffle_gt_to_data(...) before using ShuffleGTOracleExplainer."
            )
        z = getattr(data, self.attrs.z_attr)
        if z is None:
            raise ValueError(f"data.{self.attrs.z_attr} was None")
        z = z.detach().float().view(-1)
        if z.numel() != E:
            raise ValueError(
                f"Expected data.{self.attrs.z_attr} to have shape [E] with E={E}, got {tuple(z.shape)}"
            )

        if self.score_mode == "binary":
            scores = (z > self.z_thr).to(torch.float32)
        else:
            scores = z

        if self.positive_only:
            scores = torch.clamp(scores, min=0.0)

        candidate_mask = None

        if self.focus == "middle":
            if not hasattr(data, self.attrs.triples_attr):
                raise AttributeError(
                    f"Data has no attribute {self.attrs.triples_attr!r}. "

                    "Call attach_shuffle_gt_to_data(...) before using ShuffleGTOracleExplainer."
                )
            triples = getattr(data, self.attrs.triples_attr)
            if triples is None:
                raise ValueError(f"data.{self.attrs.triples_attr} was None")
            triples = triples.detach().to(torch.long)
            if triples.ndim != 2 or triples.size(1) != 3 or triples.size(0) != E:
                raise ValueError(
                    f"Expected data.{self.attrs.triples_attr} to have shape [E,3] with E={E}, got {tuple(triples.shape)}"
                )

            v = triples[:, 1]
            keep = v == int(node_idx)
            candidate_mask = keep
            # Focus on edges whose *middle node* is the explained first-order node.
            # Push irrelevant edges far down the ranking.
            scores = scores.clone()
            scores[~keep] = self.irrelevant_score

        return EdgeExplanation(
            node_idx=int(node_idx),
            target_class=int(target_class),
            edge_index=edge_index,
            edge_score=scores,
            candidate_mask=candidate_mask,
        )
