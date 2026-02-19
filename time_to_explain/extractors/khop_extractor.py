from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from time_to_explain.core.registry import register_extractor
from time_to_explain.core.types import Subgraph

from .common import EventIndex, anchor_event_idx


@register_extractor("khop_candidates")
class KHopCandidatesExtractor:
    """CoDy/Greedy-style k-hop candidate set.

    Steps (matching CoDy's fixed-size k-hop temporal subgraph logic):
      1) Restrict to events with row <= anchor_row
      2) Compute k-hop neighborhood of anchor endpoints in the induced graph
      3) Keep events whose *both* endpoints fall in that neighborhood
      4) Keep only the most recent mmax events (tail)
      5) Exclude the anchor event itself

    Notes:
      - candidate_eidx are model edge ids (e.g., TGN's e_idx) because fidelity masking
        passes them into backbone.get_prob(edge_idx_preserve_list=...).
    """

    name = "khop_candidates"

    def __init__(
        self,
        *,
        model: Any,
        events,
        candidates_size: int = 75,
        num_hops: Optional[int] = None,
        directed: bool = False,
        attach_event_meta: bool = True,
        attach_candidate_meta: bool = True,
        max_node_mask_size: int = 5_000_000,
    ) -> None:
        self.model = model
        self.candidates_size = int(candidates_size)
        self.num_hops = int(num_hops) if num_hops is not None else None
        self.directed = bool(directed)
        self.attach_event_meta = bool(attach_event_meta)
        self.attach_candidate_meta = bool(attach_candidate_meta)

        self._idx = EventIndex.from_events(events, max_node_mask_size=max_node_mask_size)

    def extract(
        self,
        dataset: Any,
        anchor: Dict[str, Any],
        *,
        k_hop: int,
        num_neighbors: int,
        window: Optional[Tuple[float, float]] = None,
    ) -> Subgraph:
        # CoDy/Greedy candidate set ignores dataset, num_neighbors, window (no sampling)
        del dataset, num_neighbors, window

        eidx = anchor_event_idx(anchor)

        k = self.num_hops
        if k is None:
            k = int(getattr(self.model, "num_layers", k_hop))

        base_row = self._idx.resolve_row(eidx)

        rows = self._idx.khop_edge_rows(base_row=base_row, k=int(k), directed=self.directed, exclude_base=False)
        rows = EventIndex.tail(rows, self.candidates_size)  # tail first (matches original)
        if rows.size:
            rows = rows[rows != int(base_row)]  # exclude anchor after tail

        cand_eidx = self._idx.rows_to_model_eids(rows)

        payload: Dict[str, Any] = {
            "event_idx": int(eidx),
            "candidate_eidx": cand_eidx.tolist(),
            "candidate_row_idx": rows.astype(np.int64).tolist(),
            "k_hop": int(k),
            "mmax": int(self.candidates_size),
        }

        if self.attach_candidate_meta and rows.size:
            edge_pairs = self._idx.edge_pairs(rows)
            payload.update(
                {
                    "candidate_edge_times": self._idx.edge_times(rows),
                    "candidate_edge_index": edge_pairs,
                    "edge_index": edge_pairs,
                }
            )

        if self.attach_event_meta:
            payload.update(self._idx.event_meta(base_row))

        return Subgraph(node_ids=[], edge_index=payload.get("edge_index", []), payload=payload)
