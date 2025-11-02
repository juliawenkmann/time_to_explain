# time_to_explain/extractors/tempme_extractor.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch

from ..core.types import Subgraph
from ..core.registry import register_extractor

# uses your project utilities
from utils import get_item, get_item_edge  # type: ignore


@register_extractor("tempme_pack")
class TempMEPackExtractor:
    """
    Minimal extractor that *does not* re-sample neighbors. Instead, it slices
    your precomputed packs (train/test) at a given event index and stashes
    everything TempME needs into Subgraph.payload.

    Usage:
        extractor = TempMEPackExtractor(pack=test_pack, edge_pack=test_edge)
        # anchors must include "index" (or "idx") pointing into the pack.
    """
    name = "tempme_pack"

    def __init__(self, *, pack: Any, edge_pack: Any, device: Optional[torch.device] = None) -> None:
        self.pack = pack
        self.edge_pack = edge_pack
        self.device = device

    def extract(self, dataset: Any, anchor: Dict[str, Any], *, k_hop: int,
                num_neighbors: int, window: Optional[Tuple[float, float]] = None) -> Subgraph:

        # We rely on a stable “event index” to slice precomputed arrays
        idx = anchor.get("index", anchor.get("idx", None))
        if idx is None:
            raise ValueError("TempMEPackExtractor expects anchor['index'] (or 'idx') pointing into the precomputed pack.")

        # get_item / get_item_edge expect a *batch of indices*; we pass a length-1 batch.
        batch_idx = np.asarray([idx], dtype=np.int64)

        (subgraph_src, subgraph_tgt, subgraph_bgd,
         walks_src, walks_tgt, walks_bgd, dst_l_fake) = get_item(self.pack, batch_idx)
        edge_src, edge_tgt, edge_bgd = get_item_edge(self.edge_pack, batch_idx)

        # For batch=1, keep tensors/arrays with a leading dim of 1; the adapter will .squeeze(0).
        payload = {
            "subgraph_src": subgraph_src,
            "subgraph_tgt": subgraph_tgt,
            "subgraph_bgd": subgraph_bgd,
            "walks_src": walks_src,
            "walks_tgt": walks_tgt,
            "walks_bgd": walks_bgd,
            "dst_l_fake": dst_l_fake,
            "edge_src": edge_src,
            "edge_tgt": edge_tgt,
            "edge_bgd": edge_bgd,
        }

        # The framework-agnostic Subgraph container: node_ids / edge_index are optional here
        # because TempME works off the payload structure above.
        return Subgraph(node_ids=[], edge_index=[], payload=payload)
