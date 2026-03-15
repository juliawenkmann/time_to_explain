from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


def pathpy_layout(graph, coords_2d: np.ndarray) -> Dict[Any, Tuple[float, float]]:
    """Build a PathpyG layout dict from a [n,2] coordinate array.

    `coords_2d[idx]` is assumed to correspond to the node with index `idx`
    under `graph.mapping.to_idx(node_id)`.
    """
    if coords_2d.ndim != 2 or coords_2d.shape[1] != 2:
        raise ValueError("coords_2d must have shape [n, 2]")
    layout: Dict[Any, Tuple[float, float]] = {}
    for node_id in graph.nodes:
        idx = int(graph.mapping.to_idx(node_id))
        layout[node_id] = (float(coords_2d[idx, 0]), float(coords_2d[idx, 1]))
    return layout


def label_colors(labels: Sequence[int]) -> np.ndarray:
    """Turn integer labels into a matplotlib-friendly numeric array."""
    return np.asarray(list(labels), dtype=float)
