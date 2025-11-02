from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

from .types import Subgraph

class KHopTemporalExtractor:
    """Replace the body of extract() with your real temporal k-hop neighborhood logic."""
    name = "khop_temporal"

    def __init__(self, *, cache: Optional[Dict[str, Subgraph]] = None):
        self._cache = cache if cache is not None else {}

    def extract(self, dataset: Any, anchor: Dict[str, Any], *, k_hop: int,
                num_neighbors: int, window: Optional[Tuple[float,float]] = None) -> Subgraph:
        key = str((id(dataset), anchor, k_hop, num_neighbors, window))
        if key in self._cache:
            return self._cache[key]

        u = int(anchor.get("u", anchor.get("src", 0)))
        i = int(anchor.get("i", anchor.get("dst", 0)))
        ts = float(anchor.get("ts", 0.0))
        subg = Subgraph(node_ids=[u, i], edge_index=[(0,1)], timestamps=[ts])
        self._cache[key] = subg
        return subg
