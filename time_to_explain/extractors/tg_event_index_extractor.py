# time_to_explain/extractors/tg_event_index_extractor.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List

from ..core.types import Subgraph
from ..core.registry import register_extractor

@register_extractor("tg_event_index")
class TGEventIndexExtractor:
    """
    Minimal extractor for TGNN explainers that work from an *event index*.
    It does NOT resample neighbors; it simply freezes the anchor event as the
    common starting point by placing `event_idx` into Subgraph.payload.

    Optionally, you can pass `candidate_eidx` along with the anchor so that
    adapters can produce a fixed-order importance vector:

        anchor = {"event_idx": 1234, "candidate_eidx": [12, 77, 88, ...]}

    Then the adapter will map the chosen edges onto this fixed list.
    """
    name = "tg_event_index"

    def extract(self, dataset: Any, anchor: Dict[str, Any], *, k_hop: int,
                num_neighbors: int, window: Optional[Tuple[float, float]] = None) -> Subgraph:
        eidx = (
            anchor.get("event_idx")
            or anchor.get("index")
            or anchor.get("idx")
        )
        if eidx is None:
            raise ValueError("TGEventIndexExtractor requires anchor['event_idx'|'index'|'idx'].")

        payload: Dict[str, Any] = {"event_idx": int(eidx)}
        # If user supplied a stable candidate edge list, keep it for importance mapping
        if "candidate_eidx" in anchor:
            payload["candidate_eidx"] = list(anchor["candidate_eidx"])

        # We do not enforce node_ids/edge_index here; SubgraphX‑TG will search internally.
        return Subgraph(node_ids=[], edge_index=[], payload=payload)
