from __future__ import annotations

from typing import Any, List


def maybe_to_device(obj: Any, device: Any) -> Any:
    """Move an object to device when it exposes `.to(device)`.

    pathpyG APIs differ by version:
    - newer releases expose `.to(...)` on graph/model objects
    - older releases (e.g. 0.2.x) do not

    This keeps loaders version-agnostic while preserving old behavior.
    """
    to_fn = getattr(obj, "to", None)
    if callable(to_fn):
        return to_fn(device)
    return obj


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
