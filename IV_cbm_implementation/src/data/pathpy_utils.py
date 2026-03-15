from __future__ import annotations

from typing import Any, List


def idx_to_node_list(graph) -> List[Any]:
    """Return a dense list `idx -> node_id` using a PathpyG mapping.

    Works for `pathpyG.core.graph.Graph` and any object that exposes:
    - `graph.n`
    - `graph.nodes` iterable
    - `graph.mapping.to_idx(node_id)`
    """
    idx_to_node: List[Any] = [None] * int(graph.n)
    for node_id in graph.nodes:
        idx = int(graph.mapping.to_idx(node_id))
        idx_to_node[idx] = node_id
    return idx_to_node
