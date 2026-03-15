from __future__ import annotations

"""Visualization helpers for counterfactual edge-deletion explanations.

Notebook 02 focuses on finding a *small* set of higher-order (De Bruijn) edges
whose deletion flips a node prediction. This module provides small, reusable
plotting helpers so the notebooks can stay thin.
"""

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch
from .palette import (
    DEFAULT_CLASS_COLORS,
    EDGE_GRAY,
    EVENT_BLUE,
    SNAPSHOT_ORANGE,
    color_for_index,
)


def _to_numpy_probs(logits_row: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits_row, dim=-1).detach().cpu().numpy()


def probs_before_after(
    *,
    adapter,
    data,
    node_idx: int,
    removed_edge_indices: Sequence[int],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return (p_before, p_after) for a node.

    If ``removed_edge_indices`` is empty, p_after is None.
    """

    v = int(node_idx)
    logits0 = adapter.predict_logits(data)
    p0 = _to_numpy_probs(logits0[v])

    if not removed_edge_indices:
        return p0, None

    space = adapter.explain_space()
    edge_index_full = getattr(data, space.edge_index_attr)
    edge_weight_full = (
        getattr(data, space.edge_weight_attr)
        if (space.edge_weight_attr and hasattr(data, space.edge_weight_attr))
        else None
    )

    E = int(edge_index_full.size(1))
    keep_mask = torch.ones(E, dtype=torch.bool, device=edge_index_full.device)
    idx = torch.as_tensor(list(removed_edge_indices), dtype=torch.long, device=edge_index_full.device)
    idx = idx[(idx >= 0) & (idx < E)]
    keep_mask[idx] = False

    ei = edge_index_full[:, keep_mask]
    ew = edge_weight_full[keep_mask] if edge_weight_full is not None else None
    data2 = adapter.clone_with_perturbed_edges(data, ei, new_edge_weight=ew)
    logits1 = adapter.predict_logits(data2)
    p1 = _to_numpy_probs(logits1[v])
    return p0, p1


def prob_curve_by_k(
    *,
    adapter,
    data,
    node_idx: int,
    ranked_edge_indices: Sequence[int],
    k_step: int = 5,
    max_k: int = 200,
    drop_mode: str = "remove",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute p(class | node) as a function of number of deleted edges.

    Args:
        ranked_edge_indices: global edge indices ordered by deletion priority.
        k_step: evaluate every k_step (including k=0).
        max_k: cap the number of deletions.
        drop_mode: "remove" to drop edges from edge_index, "zero" to zero their edge weights.

    Returns:
        (k_values, probs) where probs has shape [len(k_values), C].
        Returns (None, None) if ranked_edge_indices is empty.
    """

    if ranked_edge_indices is None or len(ranked_edge_indices) == 0:
        return None, None

    drop_mode = str(drop_mode).lower()
    if drop_mode not in {"remove", "zero"}:
        raise ValueError("drop_mode must be 'remove' or 'zero'")

    space = adapter.explain_space()
    edge_index_full = getattr(data, space.edge_index_attr)
    edge_weight_full = (
        getattr(data, space.edge_weight_attr)
        if (space.edge_weight_attr and hasattr(data, space.edge_weight_attr))
        else None
    )
    if drop_mode == "zero" and edge_weight_full is None:
        raise ValueError("drop_mode='zero' requires edge weights in the explain space")

    ranked = torch.as_tensor(list(ranked_edge_indices), dtype=torch.long, device=edge_index_full.device)
    ranked = ranked.view(-1)
    cap = int(min(int(max_k), int(ranked.numel())))
    if cap <= 0:
        return None, None

    k_values = np.arange(0, cap + 1, int(max(1, k_step)), dtype=int)
    probs = []
    for k in k_values:
        E = int(edge_index_full.size(1))
        if drop_mode == "remove":
            keep_mask = torch.ones(E, dtype=torch.bool, device=edge_index_full.device)
            if k > 0:
                drop = ranked[: int(k)]
                drop = drop[(drop >= 0) & (drop < E)]
                keep_mask[drop] = False

            ei = edge_index_full[:, keep_mask]
            ew = edge_weight_full[keep_mask] if edge_weight_full is not None else None
            data2 = adapter.clone_with_perturbed_edges(data, ei, new_edge_weight=ew)
        else:
            ew = edge_weight_full.detach().clone()
            if k > 0:
                drop = ranked[: int(k)]
                drop = drop[(drop >= 0) & (drop < E)]
                ew[drop] = 0.0
            data2 = adapter.clone_with_perturbed_edges(data, edge_index_full, new_edge_weight=ew)
        logits = adapter.predict_logits(data2)
        probs.append(_to_numpy_probs(logits[int(node_idx)]))

    return k_values, np.vstack(probs)


def prob_curve_keep_only_by_k(
    *,
    adapter,
    data,
    node_idx: int,
    ranked_edge_indices: Sequence[int],
    k_step: int = 5,
    max_k: int = 200,
    keep_non_ranked: bool = False,
    drop_mode: str = "remove",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute p(class | node) as a function of number of *kept* ranked edges.

    This is the "keep-only" counterpart to :func:`prob_curve_by_k`.

    Args:
        ranked_edge_indices: global edge indices ordered by deletion priority.
        k_step: evaluate every k_step (including k=0).
        max_k: cap the number of kept edges (within ranked_edge_indices).
        keep_non_ranked: if True, edges *not* in ranked_edge_indices are always kept.
        drop_mode: "remove" to drop edges from edge_index, "zero" to zero their edge weights.

    Returns:
        (k_values, probs) where probs has shape [len(k_values), C].
        Returns (None, None) if ranked_edge_indices is empty.
    """

    if ranked_edge_indices is None or len(ranked_edge_indices) == 0:
        return None, None

    drop_mode = str(drop_mode).lower()
    if drop_mode not in {"remove", "zero"}:
        raise ValueError("drop_mode must be 'remove' or 'zero'")

    space = adapter.explain_space()
    edge_index_full = getattr(data, space.edge_index_attr)
    edge_weight_full = (
        getattr(data, space.edge_weight_attr)
        if (space.edge_weight_attr and hasattr(data, space.edge_weight_attr))
        else None
    )
    if drop_mode == "zero" and edge_weight_full is None:
        raise ValueError("drop_mode='zero' requires edge weights in the explain space")

    ranked = torch.as_tensor(list(ranked_edge_indices), dtype=torch.long, device=edge_index_full.device)
    ranked = ranked.view(-1)
    cap = int(min(int(max_k), int(ranked.numel())))
    if cap < 0:
        return None, None

    k_values = np.arange(0, cap + 1, int(max(1, k_step)), dtype=int)
    probs = []
    E = int(edge_index_full.size(1))
    for k in k_values:
        if drop_mode == "remove":
            if keep_non_ranked:
                keep_mask = torch.ones(E, dtype=torch.bool, device=edge_index_full.device)
                keep_mask[ranked] = False
            else:
                keep_mask = torch.zeros(E, dtype=torch.bool, device=edge_index_full.device)

            if k > 0:
                keep_idx = ranked[: int(k)]
                keep_idx = keep_idx[(keep_idx >= 0) & (keep_idx < E)]
                keep_mask[keep_idx] = True

            ei = edge_index_full[:, keep_mask]
            ew = edge_weight_full[keep_mask] if edge_weight_full is not None else None
            data2 = adapter.clone_with_perturbed_edges(data, ei, new_edge_weight=ew)
        else:
            if keep_non_ranked:
                ew = edge_weight_full.detach().clone()
                ew[ranked] = 0.0
            else:
                ew = torch.zeros_like(edge_weight_full)

            if k > 0:
                keep_idx = ranked[: int(k)]
                keep_idx = keep_idx[(keep_idx >= 0) & (keep_idx < E)]
                ew[keep_idx] = edge_weight_full[keep_idx]

            data2 = adapter.clone_with_perturbed_edges(data, edge_index_full, new_edge_weight=ew)
        logits = adapter.predict_logits(data2)
        probs.append(_to_numpy_probs(logits[int(node_idx)]))

    return k_values, np.vstack(probs)

def _get_plotly_go():
    """Import plotly graph_objects lazily.

    Plotly is an optional dependency; we keep imports inside helpers so the core
    package works without it.
    """

    try:  # pragma: no cover
        import plotly.graph_objects as go  # type: ignore

        return go
    except Exception:
        return None


def _try_write_plotly_image(fig, save_path: str | Path) -> Optional[Path]:
    """Best-effort export for plotly figures.

    Notes:
        * Static export requires the optional `kaleido` engine.
        * If static export fails, we fall back to HTML next to the requested path.
    """

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    try:  # pragma: no cover
        fig.write_image(str(p))
        return p
    except Exception:
        # Fallback: write an interactive HTML next to the requested file.
        try:  # pragma: no cover
            html_path = p.with_suffix(".html")
            fig.write_html(str(html_path), include_plotlyjs="cdn")
            return html_path
        except Exception:
            return None


def plot_probs_bar(
    *,
    p_before: np.ndarray,
    p_after: Optional[np.ndarray] = None,
    class_labels: Optional[Sequence[str]] = None,
    title: str = "Class probabilities",
    save_path: str | Path | None = None,
    backend: str = "plotly",
    show: bool = True,
    width: Optional[int] = None,
    height: int = 320,
    template: str = "plotly_white",
):
    """Bar plot for probabilities before/after deletion.

    Args:
        backend: "plotly" (default) or "matplotlib".
        show: Whether to display the figure in the active notebook / backend.

    Returns:
        A plotly Figure (backend="plotly") or a matplotlib Axes (backend="matplotlib").
    """

    p0 = np.asarray(p_before)
    p1 = np.asarray(p_after) if p_after is not None else None

    C = int(p0.shape[0])
    if class_labels is None:
        class_labels = [f"class {c}" for c in range(C)]

    if backend.lower() == "plotly":
        go = _get_plotly_go()
        if go is None:
            backend = "matplotlib"
        else:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(class_labels), y=p0, name="before", marker=dict(color=EVENT_BLUE)))
            if p1 is not None:
                fig.add_trace(go.Bar(x=list(class_labels), y=p1, name="after", marker=dict(color=SNAPSHOT_ORANGE)))

            fig.update_layout(
                barmode="group",
                title=title,
                xaxis_title="class",
                yaxis_title="probability",
                template=template,
                colorway=[EVENT_BLUE, SNAPSHOT_ORANGE, EDGE_GRAY],
                width=(int(width) if width is not None else None),
                height=int(height),
            )

            if save_path is not None:
                _try_write_plotly_image(fig, save_path)
            if show:
                fig.show()
            return fig

    # Matplotlib fallback
    import matplotlib.pyplot as plt

    x = np.arange(C)
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(x - w / 2, p0, width=w, label="before", color=EVENT_BLUE)
    if p1 is not None:
        ax.bar(x + w / 2, p1, width=w, label="after", color=SNAPSHOT_ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(list(class_labels))
    ax.set_ylabel("probability")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return ax

def plot_prob_curve(
    *,
    k_values: np.ndarray,
    probs: np.ndarray,
    vline_k: Optional[int] = None,
    title: str = "Prediction vs. edge deletions",
    x_label: Optional[str] = None,
    class_colors: Optional[Sequence[str]] = None,
    save_path: str | Path | None = None,
    backend: str = "plotly",
    show: bool = True,
    width: Optional[int] = None,
    height: int = 360,
    template: str = "plotly_white",
):
    """Plot the probability curve returned by :func:`prob_curve_by_k`.

    Args:
        backend: "plotly" (default) or "matplotlib".
        show: Whether to display the figure in the active notebook / backend.

    Returns:
        A plotly Figure (backend="plotly") or a matplotlib Axes (backend="matplotlib").
    """

    k_values = np.asarray(k_values).reshape(-1)
    probs = np.asarray(probs)

    if backend.lower() == "plotly":
        go = _get_plotly_go()
        if go is None:
            backend = "matplotlib"
        else:
            if class_colors is None:
                class_colors = list(DEFAULT_CLASS_COLORS)
            fig = go.Figure()
            for c in range(int(probs.shape[1])):
                line_cfg = {"color": color_for_index(c)}
                if class_colors is not None and len(class_colors) > 0:
                    line_cfg["color"] = class_colors[c % len(class_colors)]
                fig.add_trace(
                    go.Scatter(
                        x=k_values,
                        y=probs[:, c],
                        mode="lines+markers",
                        name=f"class {c}",
                        line=line_cfg if line_cfg else None,
                    )
                )
            if vline_k is not None:
                fig.add_vline(x=int(vline_k), line_dash="dash", line_color=EDGE_GRAY, opacity=0.5)

            fig.update_layout(
                title=title,
                xaxis_title=x_label or "Number of higher-order edges removed (k)",
                yaxis_title="P(class | node)",
                template=template,
                colorway=[EVENT_BLUE, SNAPSHOT_ORANGE, EDGE_GRAY],
                width=(int(width) if width is not None else None),
                height=int(height),
            )

            if save_path is not None:
                _try_write_plotly_image(fig, save_path)
            if show:
                fig.show()
            return fig

    # Matplotlib fallback
    import matplotlib.pyplot as plt

    if class_colors is None:
        class_colors = list(DEFAULT_CLASS_COLORS)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    for c in range(probs.shape[1]):
        color = color_for_index(c)
        if class_colors is not None and len(class_colors) > 0:
            color = class_colors[c % len(class_colors)]
        ax.plot(k_values, probs[:, c], marker="o", linewidth=1.5, color=color, label=f"class {c}")
    if vline_k is not None:
        ax.axvline(int(vline_k), linestyle="--", color=EDGE_GRAY, alpha=0.5, label="flip point")
    ax.set_xlabel(x_label or "Number of higher-order edges removed (k)")
    ax.set_ylabel("P(class | node)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return ax


def plot_prob_curves_overlay(
    *,
    curves: Sequence[dict],
    title: str = "Prediction vs. edge deletions (overlay)",
    x_label: str = "Number of higher-order edges removed (k)",
    y_label: str = "P(class | node)",
    class_colors: Optional[Sequence[str]] = None,
    max_k: Optional[int] = None,
    save_path: str | Path | None = None,
    backend: str = "plotly",
    show: bool = True,
    width: Optional[int] = None,
    height: int = 360,
    template: str = "plotly_white",
):
    """Plot multiple probability curves in a single figure.

    Args:
        curves: list of dicts with keys:
            - label: str (prefix for legend)
            - k_values: array-like [K]
            - probs: array-like [K, C]
            - line: optional dict (plotly line style, e.g. {"dash": "dash"})
        class_colors: optional list of colors indexed by class id.
        max_k: optional cap; each curve is truncated to k <= max_k.
        backend: "plotly" (default) or "matplotlib".
        show: Whether to display the figure in the active notebook / backend.
    """

    if not curves:
        return None

    if backend.lower() == "plotly":
        go = _get_plotly_go()
        if go is None:
            backend = "matplotlib"
        else:
            if class_colors is None:
                class_colors = list(DEFAULT_CLASS_COLORS)
            fig = go.Figure()
            for spec in curves:
                label = str(spec.get("label", "curve"))
                k_values = np.asarray(spec.get("k_values", [])).reshape(-1)
                probs = np.asarray(spec.get("probs", []))
                if k_values.size == 0 or probs.size == 0:
                    continue
                if max_k is not None:
                    mask = k_values <= int(max_k)
                    k_values = k_values[mask]
                    probs = probs[mask]
                    if k_values.size == 0:
                        continue
                line = spec.get("line", None)
                for c in range(int(probs.shape[1])):
                    trace_kwargs = dict(
                        x=k_values,
                        y=probs[:, c],
                        mode="lines",
                        name=f"{label} class {c}",
                    )
                    if isinstance(line, dict):
                        trace_kwargs["line"] = dict(line)
                    line_cfg = trace_kwargs.get("line", {})
                    if "color" not in line_cfg:
                        line_cfg["color"] = color_for_index(c)
                        if class_colors is not None and len(class_colors) > 0:
                            line_cfg["color"] = class_colors[c % len(class_colors)]
                    trace_kwargs["line"] = line_cfg
                    fig.add_trace(go.Scatter(**trace_kwargs))

            fig.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                template=template,
                colorway=[EVENT_BLUE, SNAPSHOT_ORANGE, EDGE_GRAY],
                width=(int(width) if width is not None else None),
                height=int(height),
            )

            if save_path is not None:
                _try_write_plotly_image(fig, save_path)
            if show:
                fig.show()
            return fig

    # Matplotlib fallback
    import matplotlib.pyplot as plt
    if class_colors is None:
        class_colors = list(DEFAULT_CLASS_COLORS)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    for spec in curves:
        label = str(spec.get("label", "curve"))
        k_values = np.asarray(spec.get("k_values", [])).reshape(-1)
        probs = np.asarray(spec.get("probs", []))
        if k_values.size == 0 or probs.size == 0:
            continue
        if max_k is not None:
            mask = k_values <= int(max_k)
            k_values = k_values[mask]
            probs = probs[mask]
            if k_values.size == 0:
                continue
        line = spec.get("line", {})
        dash = str(line.get("dash", "solid"))
        linestyle = {"solid": "-", "dash": "--", "dot": ":", "dashdot": "-."}.get(dash, "-")
        for c in range(int(probs.shape[1])):
            color = color_for_index(c)
            if class_colors is not None and len(class_colors) > 0:
                color = class_colors[c % len(class_colors)]
            ax.plot(
                k_values,
                probs[:, c],
                linestyle=linestyle,
                linewidth=1.5,
                color=color,
                label=f"{label} class {c}",
            )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return ax

def plot_removed_edges_graph(
    *,
    removed_edges_as_node_ids: Sequence[Tuple[Any, Any]],
    target_node_idx: Optional[int] = None,
    save_path: str | Path | None = None,
):
    """Plot a tiny graph consisting only of the removed higher-order edges."""

    try:
        import pathpyG as pp
    except Exception as e:  # pragma: no cover
        raise ImportError("plot_removed_edges_graph requires pathpyG") from e

    import matplotlib.pyplot as plt

    if not removed_edges_as_node_ids:
        print("No removed edges to visualize.")
        return None

    # Convert nodes to strings for a stable index map.
    def _to_hashable(v):
        if isinstance(v, list):
            return tuple(_to_hashable(x) for x in v)
        if isinstance(v, tuple):
            return tuple(_to_hashable(x) for x in v)
        return v

    cleaned_edges = [(_to_hashable(u), _to_hashable(v)) for u, v in removed_edges_as_node_ids]
    node_ids = sorted({u for u, v in cleaned_edges} | {v for u, v in cleaned_edges}, key=lambda x: str(x))
    labels = [str(n) for n in node_ids]
    idx_map = pp.IndexMap(labels)

    src_idx = [idx_map.to_idx(str(u)) for u, v in cleaned_edges]
    dst_idx = [idx_map.to_idx(str(v)) for u, v in cleaned_edges]
    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)

    g_removed = pp.Graph.from_edge_index(edge_index, mapping=idx_map, num_nodes=len(labels))

    def _contains_target(n):
        if target_node_idx is None:
            return False
        try:
            tval = int(target_node_idx)
        except Exception:
            return False
        if isinstance(n, (tuple, list)):
            for x in n:
                try:
                    if int(x) == tval:
                        return True
                except Exception:
                    pass
        return False

    node_color = []
    for v in g_removed.nodes:
        orig = node_ids[idx_map.to_idx(v)]
        node_color.append(EVENT_BLUE if _contains_target(orig) else EDGE_GRAY)

    pp.plot(
        g_removed,
        backend="matplotlib",
        show_labels=False,
        node_size=5,
        edge_size=1.5,
        node_color=node_color,
        edge_color=SNAPSHOT_ORANGE,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.gcf().savefig(save_path, bbox_inches="tight")
    return g_removed


def plot_de_bruijn_with_deleted_edges(
    *,
    g2,
    data,
    removed_edge_indices: Sequence[int],
    target_node_idx: Optional[int] = None,
    layout_style: Optional[dict] = None,
    base_node_color: str = EDGE_GRAY,
    target_node_color: str = EVENT_BLUE,
    base_node_size: float = 3.0,
    target_node_size: float = 8.0,
    base_edge_size: float = 0.5,
    base_edge_opacity: float = 0.08,
    deleted_edge_size: float = 2.0,
    deleted_edge_opacity: float = 1.0,
    base_edge_color: str = EDGE_GRAY,
    deleted_edge_color: str = SNAPSHOT_ORANGE,
    show_after_deletion: bool = True,
    undirected: bool = True,
):
    """Plot the full De Bruijn graph and highlight deleted higher-order edges."""

    if g2 is None:
        print("No g2 available.")
        return

    try:
        import pathpyG as pp
    except Exception as e:  # pragma: no cover
        raise ImportError("plot_de_bruijn_with_deleted_edges requires pathpyG") from e

    if target_node_idx is not None:
        target_node_idx = int(target_node_idx)

    removed_idx = [int(x) for x in removed_edge_indices]

    if layout_style is None:
        layout_style = {"layout": "Fruchterman-Reingold", "seed": 1, "k": 0.5, "iterations": 300}

    # Plot without arrows by plotting an undirected view
    g_vis = g2.to_undirected() if undirected else g2
    if undirected and hasattr(g2, "data") and hasattr(g2.data, "node_sequence"):
        # Preserve higher-order node IDs (prevents list nodes after to_undirected).
        g_vis.data.node_sequence = g2.data.node_sequence

    layout = pp.layout(g_vis, **layout_style)

    edge_index_ho = data.edge_index_higher_order.detach().cpu()
    E_ho = int(edge_index_ho.size(1))
    removed_pairs = set()
    for e in removed_idx:
        if 0 <= e < E_ho:
            removed_pairs.add((int(edge_index_ho[0, e]), int(edge_index_ho[1, e])))

    def _to_ho_idx(node):
        try:
            import numpy as _np

            if isinstance(node, (int, _np.integer)):
                return int(node)
        except Exception:
            if isinstance(node, int):
                return int(node)

        m = getattr(g2, "mapping", None)
        if m is not None and hasattr(m, "to_idx"):
            try:
                return int(m.to_idx(node))
            except Exception:
                pass
        return node

    def _to_ho_id_tuple(node):
        if isinstance(node, tuple):
            return node
        if isinstance(node, list):
            return tuple(node)

        m = getattr(g2, "mapping", None)
        if m is not None and hasattr(m, "to_id"):
            try:
                nid = m.to_id(int(node))
                if isinstance(nid, list):
                    return tuple(nid)
                if isinstance(nid, tuple):
                    return nid
            except Exception:
                pass
        return None

    # Node styles: highlight HO nodes that contain the target FO node
    nodes = list(g_vis.nodes)
    node_color = []
    node_size = []
    for n in nodes:
        ho_id = _to_ho_id_tuple(n)
        contains = (
            (ho_id is not None)
            and (target_node_idx is not None)
            and (int(target_node_idx) in ho_id)
        )
        node_color.append(target_node_color if contains else base_node_color)
        node_size.append(target_node_size if contains else base_node_size)

    edges = list(g_vis.edges)
    removed_keys = {frozenset((a, b)) for (a, b) in removed_pairs} if undirected else set()

    def _is_removed(s, d):
        if undirected:
            return frozenset((_to_ho_idx(s), _to_ho_idx(d))) in removed_keys
        return (_to_ho_idx(s), _to_ho_idx(d)) in removed_pairs

    is_removed = [_is_removed(s, d) for (s, d) in edges]

    edge_color = [deleted_edge_color if r else base_edge_color for r in is_removed]
    edge_opacity = [deleted_edge_opacity if r else base_edge_opacity for r in is_removed]
    edge_size = [deleted_edge_size if r else base_edge_size for r in is_removed]

    pp.plot(
        g_vis,
        backend="matplotlib",
        layout=layout,
        node_color=node_color,
        node_size=node_size,
        edge_color=edge_color,
        edge_opacity=edge_opacity,
        edge_size=edge_size,
        show_labels=False,
    )

    if not show_after_deletion:
        return

    # "After deletion": hide removed edges (same layout)
    edge_color2 = [base_edge_color] * len(edges)
    edge_opacity2 = [base_edge_opacity] * len(edges)
    edge_size2 = [base_edge_size] * len(edges)
    for i, r in enumerate(is_removed):
        if r:
            edge_opacity2[i] = 0.0
            edge_size2[i] = 0.0

    pp.plot(
        g_vis,
        backend="matplotlib",
        layout=layout,
        node_color=node_color,
        node_size=node_size,
        edge_color=edge_color2,
        edge_opacity=edge_opacity2,
        edge_size=edge_size2,
        show_labels=False,
    )
