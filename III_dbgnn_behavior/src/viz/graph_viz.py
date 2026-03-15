from __future__ import annotations

from typing import Any, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

from pathpy_utils import idx_to_node_list
from .palette import EDGE_GRAY, EVENT_BLUE, SNAPSHOT_ORANGE


# Use project palette consistently.
_CLASS_COLORS = {0: EVENT_BLUE, 1: SNAPSHOT_ORANGE, 2: EDGE_GRAY, 3: EDGE_GRAY}
_CLASS_OPACITIES = {0: 0.6, 1: 0.6, 2: 0.6, 3: 0.12}


def _first_order_class(node_id: Any) -> int:
    try:
        n = int(node_id)
    except Exception:
        return 3
    if 0 <= n < 10:
        return 0
    if 10 <= n < 20:
        return 1
    if 20 <= n < 30:
        return 2
    return 3


def _higher_order_class(ho_node_id: Any) -> int:
    """Class of a De Bruijn node (u,v)."""
    if not (isinstance(ho_node_id, tuple) and len(ho_node_id) == 2):
        return 3
    try:
        a = int(ho_node_id[0])
        b = int(ho_node_id[1])
    except Exception:
        return 3

    if a < 10 and b < 10:
        return 0
    if 10 <= a < 20 and 10 <= b < 20:
        return 1
    if 20 <= a < 30 and 20 <= b < 30:
        return 2
    return 3


def node_rgba(node_id: Any) -> tuple[float, float, float, float]:
    """RGBA color for a node id (first-order int or higher-order (u,v) tuple)."""
    cls = _higher_order_class(node_id) if isinstance(node_id, tuple) else _first_order_class(node_id)
    color = _CLASS_COLORS.get(cls, EDGE_GRAY)
    alpha = _CLASS_OPACITIES.get(cls, 0.3)
    return mcolors.to_rgba(color, alpha=alpha)


def scale_widths(w: np.ndarray) -> list[float]:
    """Map arbitrary edge scores to visually usable edge widths."""
    w = np.asarray(w, dtype=float)
    if w.size == 0:
        return []
    w = np.abs(w)
    if np.allclose(w.max(), w.min()):
        return [1.5] * int(w.size)
    w = (w - w.min()) / (w.max() - w.min() + 1e-12)
    return list(0.8 + 4.0 * w)


def stable_layouts_from_data(
    *,
    data,
    assets,
    seed: int = 1,
    k: float = 0.5,
    iterations: int = 300,
    ho_edge_index_attr: str = "edge_index_higher_order",
    fo_edge_index_attr: str = "edge_index",
) -> tuple[dict[Any, Any], dict[Any, Any]]:
    """Compute stable (reusable) layouts for FO and HO plots."""

    def _edge_index_tensor(ei) -> torch.Tensor:
        if hasattr(ei, "as_tensor"):
            return ei.as_tensor()
        if isinstance(ei, torch.Tensor):
            return ei
        return torch.as_tensor(ei)

    def _layout(edge_index: torch.Tensor, idx_to_node: list[Any]) -> dict[Any, Any]:
        ei = _edge_index_tensor(edge_index).detach().cpu()
        src = ei[0].tolist()
        dst = ei[1].tolist()
        G = nx.DiGraph()
        for s, d in zip(src, dst):
            G.add_edge(idx_to_node[int(s)], idx_to_node[int(d)])
        return nx.spring_layout(G, seed=int(seed), k=float(k), iterations=int(iterations))

    pos_ho: dict[Any, Any] = {}
    if hasattr(data, ho_edge_index_attr) and hasattr(assets, "g2"):
        idx_to_ho = idx_to_node_list(assets.g2)
        pos_ho = _layout(getattr(data, ho_edge_index_attr), idx_to_ho)

    pos_fo: dict[Any, Any] = {}
    if hasattr(data, fo_edge_index_attr) and hasattr(assets, "g"):
        idx_to_fo = idx_to_node_list(assets.g)
        pos_fo = _layout(getattr(data, fo_edge_index_attr), idx_to_fo)

    return pos_fo, pos_ho


def draw_edges_df(
    edges_df: pd.DataFrame,
    *,
    src_col: str,
    dst_col: str,
    score_col: str = "score",
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    pos: Optional[dict[Any, Any]] = None,
    seed: int = 0,
    max_nodes: int = 80,
    focus_node: Optional[Any] = None,
    edge_colors: Optional[dict[tuple[Any, Any], Any]] = None,
    default_edge_color: Any = EDGE_GRAY,
) -> plt.Axes:
    """Draw a directed graph from an edge list DataFrame with class coloring."""
    if title is None:
        title = ""

    G = nx.DiGraph()
    for _, r in edges_df.iterrows():
        src = r[src_col]
        dst = r[dst_col]
        s = float(r.get(score_col, 1.0))
        color = default_edge_color
        if edge_colors is not None:
            color = edge_colors.get((src, dst), default_edge_color)
        G.add_edge(src, dst, score=s, color=color)

    # Limit to avoid unreadable plots.
    if G.number_of_nodes() > int(max_nodes):
        nodes = list(G.nodes())[: int(max_nodes)]
        G = G.subgraph(nodes).copy()

    if pos is None:
        pos = nx.spring_layout(G, seed=int(seed))

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    widths = scale_widths([d.get("score", 1.0) for _, _, d in G.edges(data=True)])
    ecols = [d.get("color", default_edge_color) for _, _, d in G.edges(data=True)]

    nodes = list(G.nodes())
    ncols = [node_rgba(n) for n in nodes]
    nsizes = [650 if (focus_node is not None and n == focus_node) else 220 for n in nodes]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=ncols, node_size=nsizes, edgecolors=EDGE_GRAY, linewidths=0.7)
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color=ecols, arrows=True, arrowstyle="-|>", arrowsize=12, alpha=0.85)

    if G.number_of_nodes() <= 35:
        nx.draw_networkx_labels(G, pos, ax=ax, labels={n: str(n) for n in nodes}, font_size=8)

    ax.set_title(title)
    ax.axis("off")
    return ax
