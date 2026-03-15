from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from .common import (
    ROLE_COLORS,
    _avoid_text_overlap,
    _deduplicate_preserve_order,
    _normalize_event_time,
    _role_style,
    _stable_sorted_events,
)
from .loading import (
    assign_event_labels,
    build_clean_node_layout,
    filter_events_at_or_before_target,
    relabel_local_nodes,
)

def _compute_edge_curvatures(events_df: pd.DataFrame) -> Dict[int, float]:
    df = _stable_sorted_events(events_df)
    idx_to_rad: Dict[int, float] = {}

    pairs = sorted({tuple(sorted((int(u), int(i)))) for u, i in zip(df["u"], df["i"])})
    levels = [0.12, 0.24, 0.36, 0.48, 0.60]
    alternating = [0.0, 0.14, -0.14, 0.28, -0.28, 0.42, -0.42]

    for a, b in pairs:
        if a == b:
            loop_ids = df.loc[(df["u"] == a) & (df["i"] == b), "idx"].astype(int).tolist()
            for k, eidx in enumerate(loop_ids):
                idx_to_rad[eidx] = 0.45 + 0.12 * k
            continue

        forward_ids = df.loc[(df["u"] == a) & (df["i"] == b), "idx"].astype(int).tolist()
        backward_ids = df.loc[(df["u"] == b) & (df["i"] == a), "idx"].astype(int).tolist()
        if forward_ids and backward_ids:
            for k, eidx in enumerate(forward_ids):
                idx_to_rad[eidx] = levels[min(k, len(levels) - 1)]
            for k, eidx in enumerate(backward_ids):
                idx_to_rad[eidx] = -levels[min(k, len(levels) - 1)]
        else:
            directed_ids = forward_ids if forward_ids else backward_ids
            sign = 1.0 if forward_ids else -1.0
            for k, eidx in enumerate(directed_ids):
                idx_to_rad[eidx] = sign * alternating[min(k, len(alternating) - 1)]
    return idx_to_rad


def _edge_label_xy(p0: Tuple[float, float], p1: Tuple[float, float], rad: float) -> Tuple[float, float]:
    x0, y0 = p0
    x1, y1 = p1
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    dx, dy = x1 - x0, y1 - y0
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return mx + 0.10, my + 0.10
    perp = np.array([-dy / norm, dx / norm])
    strength = 0.16 + 0.22 * abs(rad)
    sign = 1.0 if rad >= 0 else -1.0
    x, y = np.array([mx, my]) + sign * strength * perp
    return float(x), float(y)


def plot_local_graph_panel(
    ax: plt.Axes,
    events_df: pd.DataFrame,
    node_pos: Dict[int, Tuple[float, float]],
    node_mapping: Dict[int, str],
    target_idx: int,
    dataset_name: str,
) -> None:
    """
    Plot left panel: local directed temporal neighborhood graph.
    """
    df = _stable_sorted_events(events_df)
    target_row = df.loc[df["idx"].astype(int) == int(target_idx)]
    if target_row.empty:
        raise KeyError(f"target_idx={target_idx} is not present in plotted events.")
    t_row = target_row.iloc[0]
    target_nodes = [int(t_row["u"]), int(t_row["i"])]

    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(node_pos))
    for row in df.itertuples(index=False):
        graph.add_edge(int(row.u), int(row.i), idx=int(row.idx), role=str(row.event_role))

    nx.draw_networkx_nodes(
        graph,
        pos=node_pos,
        ax=ax,
        node_size=1200,
        node_color="white",
        edgecolors="black",
        linewidths=1.4,
    )
    nx.draw_networkx_labels(
        graph,
        pos=node_pos,
        ax=ax,
        labels={n: node_mapping[n] for n in graph.nodes},
        font_size=10,
        font_weight="medium",
    )
    nx.draw_networkx_nodes(
        graph,
        pos=node_pos,
        ax=ax,
        nodelist=target_nodes,
        node_size=1450,
        node_color="none",
        edgecolors=ROLE_COLORS["target"],
        linewidths=2.0,
    )

    rad_map = _compute_edge_curvatures(df)
    placed_labels: List[Tuple[float, float]] = []
    for row in df.itertuples(index=False):
        u, v, eidx = int(row.u), int(row.i), int(row.idx)
        role = str(row.event_role)
        style = _role_style(role)
        p0 = node_pos[u]
        p1 = node_pos[v]
        rad = float(rad_map.get(eidx, 0.0))

        if u == v:
            start = (p0[0], p0[1] + 0.04)
            end = (p1[0] + 1e-3, p1[1] + 0.04)
        else:
            start, end = p0, p1

        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            shrinkA=16,
            shrinkB=16,
            connectionstyle=f"arc3,rad={rad}",
            linewidth=style["lw"],
            linestyle=style["linestyle"],
            color=style["color"],
            alpha=style["alpha"],
            zorder=2,
        )
        ax.add_patch(patch)

        label = str(row.event_label)
        if label:
            lx, ly = _edge_label_xy(start, end, rad)
            while any(abs(lx - px) < 0.10 and abs(ly - py) < 0.09 for px, py in placed_labels):
                ly += 0.10
            placed_labels.append((lx, ly))
            ax.text(
                lx,
                ly,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color=style["color"],
                fontweight="bold" if role in {"target", "overlap"} else "medium",
                bbox={
                    "boxstyle": "round,pad=0.24",
                    "fc": "white",
                    "ec": style["color"],
                    "lw": 0.9,
                    "alpha": 0.97,
                },
                zorder=5,
            )

    role_labels = [
        ("context", "normal context"),
        ("ground_truth", "ground truth motif"),
        ("explainer", "explainer-selected"),
        ("overlap", "GT & explainer"),
        ("target", "target event"),
    ]
    handles = [
        Line2D([0], [0], color=_role_style(role)["color"], lw=_role_style(role)["lw"], linestyle=_role_style(role)["linestyle"], label=txt)  # type: ignore[index]
        for role, txt in role_labels
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.95, fontsize=9)

    xs = [p[0] for p in node_pos.values()]
    ys = [p[1] for p in node_pos.values()]
    x_span = max(1e-6, max(xs) - min(xs))
    y_span = max(1e-6, max(ys) - min(ys))
    x_pad = max(0.35, 0.18 * x_span)
    y_pad = max(0.35, 0.18 * y_span)
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Local temporal neighbourhood ({dataset_name}-style message graph)")


def plot_timeline_panel(
    ax: plt.Axes,
    events_df: pd.DataFrame,
    node_mapping: Dict[int, str],
    focus_event_idxs: Optional[Sequence[int]] = None,
    zoom_on_focus: bool = False,
    zoom_padding: float = 0.06,
    min_focus_span: float = 0.06,
    spread_focus_events: bool = False,
    spread_focus_margin: float = 0.14,
    zoom_y_on_focus: bool = False,
) -> None:
    """
    Plot right panel: selected events on a normalized timeline.
    """
    df = _stable_sorted_events(events_df)
    node_order = sorted(node_mapping.keys(), key=lambda nid: int(node_mapping[nid][1:]))
    y_map = {node_id: yi for yi, node_id in enumerate(node_order)}

    x_vals = _normalize_event_time(df["ts"].to_numpy())
    df = df.copy()
    df["x_norm"] = x_vals

    x_lo, x_hi = -0.05, 1.08
    focus_rows = df.iloc[0:0]
    if zoom_on_focus and focus_event_idxs:
        focus_set = set(int(v) for v in focus_event_idxs)
        focus_rows = df.loc[df["idx"].astype(int).isin(list(focus_set))]
        if not focus_rows.empty:
            fx_min = float(focus_rows["x_norm"].min())
            fx_max = float(focus_rows["x_norm"].max())
            pad = max(0.0, float(zoom_padding))
            min_span = max(1e-4, float(min_focus_span))
            if math.isclose(fx_min, fx_max):
                half_span = max(min_span / 2.0, pad * 0.5)
                x_lo, x_hi = fx_min - half_span, fx_max + half_span
            else:
                span = fx_max - fx_min
                x_lo = fx_min - pad * span
                x_hi = fx_max + pad * span
                if (x_hi - x_lo) < min_span:
                    center = (x_hi + x_lo) / 2.0
                    x_lo = center - (min_span / 2.0)
                    x_hi = center + (min_span / 2.0)
            x_lo = max(-0.05, x_lo)
            x_hi = min(1.08, x_hi)
            if x_hi - x_lo < min_span:
                center = (fx_min + fx_max) / 2.0
                center = min(max(center, -0.05 + min_span / 2.0), 1.08 - min_span / 2.0)
                x_lo = max(-0.05, center - min_span / 2.0)
                x_hi = min(1.08, center + min_span / 2.0)

    df["x_plot"] = df["x_norm"]
    if spread_focus_events and not focus_rows.empty:
        focus_sorted = focus_rows.sort_values(["ts", "idx"], kind="mergesort")
        n_focus = len(focus_sorted)
        if n_focus > 0:
            span = max(1e-6, x_hi - x_lo)
            margin = min(0.45, max(0.0, float(spread_focus_margin)))
            left = x_lo + margin * span
            right = x_hi - margin * span
            if right <= left:
                center = (x_lo + x_hi) / 2.0
                half = 0.18 * span
                left, right = center - half, center + half
            spread_positions = (
                np.array([(left + right) / 2.0]) if n_focus == 1 else np.linspace(left, right, n_focus)
            )
            for row, pos in zip(focus_sorted.itertuples(index=False), spread_positions):
                df.loc[df["idx"].astype(int) == int(row.idx), "x_plot"] = float(pos)

    for row in df.itertuples(index=False):
        u, v = int(row.u), int(row.i)
        y0, y1 = y_map[u], y_map[v]
        x = float(row.x_plot)
        style = _role_style(str(row.event_role))

        ax.plot(
            [x, x],
            [y0, y1],
            color=style["color"],
            linewidth=style["lw"],
            linestyle=style["linestyle"],
            alpha=style["alpha"],
            zorder=2,
        )
        ax.scatter(
            [x, x],
            [y0, y1],
            s=34 if str(row.event_role) == "target" else 20,
            color=style["color"],
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )

    labeled = df.loc[df["event_label"].astype(str) != ""].copy().sort_values(["x_plot", "idx"], kind="mergesort")
    base_points: List[Tuple[float, float]] = []
    for row in labeled.itertuples(index=False):
        y_top = max(y_map[int(row.u)], y_map[int(row.i)])
        base_points.append((float(row.x_plot), float(y_top + 0.35)))
    shifted = _avoid_text_overlap(base_points, min_dx=0.035, min_dy=0.25)

    for row, (tx, ty) in zip(labeled.itertuples(index=False), shifted):
        style = _role_style(str(row.event_role))
        x = float(row.x_plot)
        y_anchor = max(y_map[int(row.u)], y_map[int(row.i)])

        ax.plot([x, tx], [y_anchor + 0.03, ty - 0.04], color=style["color"], linewidth=0.9, alpha=0.75)
        ax.text(
            tx,
            ty,
            str(row.event_label),
            ha="center",
            va="center",
            fontsize=9,
            color=style["color"],
            fontweight="bold" if str(row.event_role) in {"target", "overlap"} else "medium",
            bbox={
                "boxstyle": "round,pad=0.22",
                "fc": "white",
                "ec": style["color"],
                "lw": 0.9,
                "alpha": 0.97,
            },
            zorder=5,
        )

    y_lo, y_hi = -0.5, len(node_order) - 0.5
    visible_nodes = list(node_order)
    if zoom_y_on_focus and not focus_rows.empty:
        focus_nodes = sorted(
            set(int(v) for v in focus_rows["u"].tolist()).union(set(int(v) for v in focus_rows["i"].tolist()))
        )
        focus_y = [y_map[nid] for nid in focus_nodes if nid in y_map]
        if focus_y:
            y_pad = 0.75
            y_lo = max(-0.5, min(focus_y) - y_pad)
            y_hi = min(len(node_order) - 0.5, max(focus_y) + y_pad)
            visible_nodes = [nid for nid in node_order if y_lo <= y_map[nid] <= y_hi]
            if not visible_nodes:
                visible_nodes = list(node_order)
                y_lo, y_hi = -0.5, len(node_order) - 0.5

    ax.set_yticks([y_map[nid] for nid in visible_nodes])
    ax.set_yticklabels([node_mapping[nid] for nid in visible_nodes])
    ax.set_ylabel("Local node labels")
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(x_lo, x_hi)
    ax.set_xticks(np.linspace(x_lo, x_hi, 6))
    ax.set_xlabel("Normalized time")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.9, alpha=0.45)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.25)
    ax.set_title("Same events on a time axis")


def compute_overlap_stats(
    motif_event_idxs: Sequence[int],
    explainer_event_idxs: Sequence[int],
) -> Dict[str, float]:
    """
    Compute overlap summary metrics between ground-truth and explainer-selected events.
    """
    gt = set(int(v) for v in motif_event_idxs)
    ex = set(int(v) for v in explainer_event_idxs)
    inter = gt.intersection(ex)

    precision = len(inter) / len(ex) if ex else 0.0
    recall = len(inter) / len(gt) if gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "gt_count": float(len(gt)),
        "explainer_count": float(len(ex)),
        "overlap_count": float(len(inter)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _resolve_timeline_focus_ids(
    resolved_target_idx: int,
    motif_in_view: Sequence[int],
    explainer_in_view: Sequence[int],
    timeline_zoom_focus_mode: str,
) -> List[int]:
    mode = str(timeline_zoom_focus_mode or "key_events").strip().lower().replace("-", "_").replace(" ", "_")
    target_only = [int(resolved_target_idx)]
    motif_only = [int(v) for v in motif_in_view]
    explainer_only = [int(v) for v in explainer_in_view]

    if mode in {"ground_truth", "ground_truth_only", "gt", "motif", "motif_only"}:
        focus_ids = _deduplicate_preserve_order(motif_only)
    elif mode in {"ground_truth_plus_target", "gt_plus_target", "motif_plus_target"}:
        focus_ids = _deduplicate_preserve_order(target_only + motif_only)
    elif mode in {"explainer", "explainer_only"}:
        focus_ids = _deduplicate_preserve_order(explainer_only)
    elif mode in {"explainer_plus_target", "target_plus_explainer"}:
        focus_ids = _deduplicate_preserve_order(target_only + explainer_only)
    elif mode in {"target", "target_only"}:
        focus_ids = _deduplicate_preserve_order(target_only)
    else:
        focus_ids = _deduplicate_preserve_order(target_only + motif_only + explainer_only)

    if not focus_ids:
        focus_ids = target_only
    return focus_ids


def plot_temporal_explanation_figure(
    events_df: pd.DataFrame,
    target_idx: int,
    motif_event_idxs: Optional[Sequence[int]] = None,
    explainer_event_idxs: Optional[Sequence[int]] = None,
    dataset_name: str = "UCI",
    explainer_name: Optional[str] = None,
    timeline_zoom_on_key_events: bool = False,
    timeline_zoom_padding: float = 0.06,
    timeline_min_zoom_span: float = 0.06,
    timeline_spread_focus_events: bool = False,
    timeline_zoom_focus_mode: str = "key_events",
    timeline_zoom_nodes_on_focus: bool = False,
    timeline_emphasize_panel: bool = False,
    panel_layout: str = "horizontal",
    enforce_causal_order: bool = True,
) -> Tuple[plt.Figure, pd.DataFrame, pd.DataFrame, Dict[int, str], int]:
    """
    Plot a two-panel temporal explanation figure with optional explainer overlay.

    Parameters
    ----------
    timeline_zoom_focus_mode:
        Which events define the timeline zoom window when `timeline_zoom_on_key_events=True`.
        Supported values include: `key_events` (default), `ground_truth`,
        `ground_truth_plus_target`, `explainer`, `explainer_plus_target`, `target`.
    timeline_zoom_nodes_on_focus:
        If `True`, zoom the timeline y-axis to rows touched by focused events.
    timeline_emphasize_panel:
        If `True`, allocate more width to the timeline panel.
    panel_layout:
        Figure panel layout: `horizontal` (default, side-by-side) or
        `split`/`vertical` (stacked, larger per-panel view).
    enforce_causal_order:
        If `True`, drop events after target time so target remains rightmost in causal views.
    """
    labeled_df, motif_in_view, explainer_in_view, resolved_target_idx = assign_event_labels(
        events_df=events_df,
        motif_event_idxs=motif_event_idxs,
        target_idx=target_idx,
        explainer_event_idxs=explainer_event_idxs,
    )
    if enforce_causal_order:
        labeled_df = filter_events_at_or_before_target(labeled_df, resolved_target_idx)
        labeled_df, motif_in_view, explainer_in_view, resolved_target_idx = assign_event_labels(
            events_df=labeled_df,
            motif_event_idxs=motif_event_idxs,
            target_idx=resolved_target_idx,
            explainer_event_idxs=explainer_event_idxs,
        )
    relabeled_df, node_mapping, mapping_table = relabel_local_nodes(labeled_df)
    node_pos = build_clean_node_layout(
        events_df=relabeled_df,
        target_idx=resolved_target_idx,
        motif_event_idxs=motif_in_view,
    )

    layout = str(panel_layout or "horizontal").strip().lower().replace("-", "_")
    if layout in {"split", "vertical", "stacked"}:
        fig_w, fig_h = 12.0, 13.6
        if timeline_emphasize_panel:
            fig_w, fig_h = 12.4, 14.2
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(fig_w, fig_h),
            gridspec_kw={"height_ratios": [1.0, 1.0]},
            constrained_layout=False,
        )
        ax_graph = axes[0]
        ax_timeline = axes[1]
    else:
        fig_w, fig_h = 15.8, 7.2
        width_ratios = [1.2, 1.0]
        if timeline_emphasize_panel:
            fig_w, fig_h = 17.2, 7.4
            width_ratios = [1.15, 1.35]
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(fig_w, fig_h),
            gridspec_kw={"width_ratios": width_ratios},
            constrained_layout=False,
        )
        ax_graph = axes[0]
        ax_timeline = axes[1]

    plot_local_graph_panel(
        ax=ax_graph,
        events_df=relabeled_df,
        node_pos=node_pos,
        node_mapping=node_mapping,
        target_idx=resolved_target_idx,
        dataset_name=dataset_name,
    )
    focus_ids = _resolve_timeline_focus_ids(
        resolved_target_idx=resolved_target_idx,
        motif_in_view=motif_in_view,
        explainer_in_view=explainer_in_view,
        timeline_zoom_focus_mode=timeline_zoom_focus_mode,
    )

    plot_timeline_panel(
        ax=ax_timeline,
        events_df=relabeled_df,
        node_mapping=node_mapping,
        focus_event_idxs=focus_ids,
        zoom_on_focus=bool(timeline_zoom_on_key_events),
        zoom_padding=float(timeline_zoom_padding),
        min_focus_span=float(timeline_min_zoom_span),
        spread_focus_events=bool(timeline_spread_focus_events),
        zoom_y_on_focus=bool(timeline_zoom_nodes_on_focus),
    )

    overlay_suffix = f" | explainer={explainer_name}" if explainer_name else ""
    fig.suptitle(
        f"Temporal explanation for target idx={resolved_target_idx}{overlay_suffix}",
        fontsize=13,
        y=0.965,
    )

    stats = compute_overlap_stats(motif_in_view, explainer_in_view)
    stats_txt = (
        f"GT={int(stats['gt_count'])}, Expl={int(stats['explainer_count'])}, "
        f"Overlap={int(stats['overlap_count'])}, P={stats['precision']:.2f}, "
        f"R={stats['recall']:.2f}"
    )
    ax_timeline.text(
        1.0,
        1.015,
        stats_txt,
        transform=ax_timeline.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#333333",
    )
    if layout in {"split", "vertical", "stacked"}:
        fig.subplots_adjust(left=0.06, right=0.985, bottom=0.06, top=0.93, hspace=0.24)
    else:
        fig.subplots_adjust(left=0.04, right=0.985, bottom=0.09, top=0.90, wspace=0.18)

    return fig, relabeled_df, mapping_table, node_mapping, resolved_target_idx


def plot_temporal_explanation_separate_panels(
    events_df: pd.DataFrame,
    target_idx: int,
    motif_event_idxs: Optional[Sequence[int]] = None,
    explainer_event_idxs: Optional[Sequence[int]] = None,
    dataset_name: str = "UCI",
    explainer_name: Optional[str] = None,
    timeline_zoom_on_key_events: bool = False,
    timeline_zoom_padding: float = 0.06,
    timeline_min_zoom_span: float = 0.06,
    timeline_spread_focus_events: bool = False,
    timeline_zoom_focus_mode: str = "key_events",
    timeline_zoom_nodes_on_focus: bool = False,
    graph_figsize: Tuple[float, float] = (12.8, 9.2),
    timeline_figsize: Tuple[float, float] = (22.0, 9.8),
    enforce_causal_order: bool = True,
) -> Tuple[plt.Figure, plt.Figure, pd.DataFrame, pd.DataFrame, Dict[int, str], int]:
    """
    Plot local graph and timeline as two separate larger figures.
    """
    labeled_df, motif_in_view, explainer_in_view, resolved_target_idx = assign_event_labels(
        events_df=events_df,
        motif_event_idxs=motif_event_idxs,
        target_idx=target_idx,
        explainer_event_idxs=explainer_event_idxs,
    )
    if enforce_causal_order:
        labeled_df = filter_events_at_or_before_target(labeled_df, resolved_target_idx)
        labeled_df, motif_in_view, explainer_in_view, resolved_target_idx = assign_event_labels(
            events_df=labeled_df,
            motif_event_idxs=motif_event_idxs,
            target_idx=resolved_target_idx,
            explainer_event_idxs=explainer_event_idxs,
        )
    relabeled_df, node_mapping, mapping_table = relabel_local_nodes(labeled_df)
    node_pos = build_clean_node_layout(
        events_df=relabeled_df,
        target_idx=resolved_target_idx,
        motif_event_idxs=motif_in_view,
    )

    fig_graph, ax_graph = plt.subplots(1, 1, figsize=graph_figsize, constrained_layout=False)
    plot_local_graph_panel(
        ax=ax_graph,
        events_df=relabeled_df,
        node_pos=node_pos,
        node_mapping=node_mapping,
        target_idx=resolved_target_idx,
        dataset_name=dataset_name,
    )
    overlay_suffix = f" | explainer={explainer_name}" if explainer_name else ""
    fig_graph.suptitle(
        f"Local graph for target idx={resolved_target_idx}{overlay_suffix}",
        fontsize=14,
        y=0.98,
    )
    fig_graph.subplots_adjust(left=0.05, right=0.985, bottom=0.06, top=0.92)

    focus_ids = _resolve_timeline_focus_ids(
        resolved_target_idx=resolved_target_idx,
        motif_in_view=motif_in_view,
        explainer_in_view=explainer_in_view,
        timeline_zoom_focus_mode=timeline_zoom_focus_mode,
    )
    fig_timeline, ax_timeline = plt.subplots(1, 1, figsize=timeline_figsize, constrained_layout=False)
    plot_timeline_panel(
        ax=ax_timeline,
        events_df=relabeled_df,
        node_mapping=node_mapping,
        focus_event_idxs=focus_ids,
        zoom_on_focus=bool(timeline_zoom_on_key_events),
        zoom_padding=float(timeline_zoom_padding),
        min_focus_span=float(timeline_min_zoom_span),
        spread_focus_events=bool(timeline_spread_focus_events),
        zoom_y_on_focus=bool(timeline_zoom_nodes_on_focus),
    )
    stats = compute_overlap_stats(motif_in_view, explainer_in_view)
    stats_txt = (
        f"GT={int(stats['gt_count'])}, Expl={int(stats['explainer_count'])}, "
        f"Overlap={int(stats['overlap_count'])}, P={stats['precision']:.2f}, "
        f"R={stats['recall']:.2f}"
    )
    ax_timeline.text(
        1.0,
        1.02,
        stats_txt,
        transform=ax_timeline.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#333333",
    )
    fig_timeline.suptitle(
        f"Timeline for target idx={resolved_target_idx}{overlay_suffix}",
        fontsize=14,
        y=0.99,
    )
    fig_timeline.subplots_adjust(left=0.09, right=0.985, bottom=0.10, top=0.90)
    return fig_graph, fig_timeline, relabeled_df, mapping_table, node_mapping, resolved_target_idx


def save_figure(
    fig: plt.Figure,
    out_prefix: str,
    output_dir: Path | str = "outputs",
) -> Tuple[Path, Path]:
    """
    Save figure as PNG and PDF with 300 dpi and tight bounding box.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{out_prefix}.png"
    pdf_path = out_dir / f"{out_prefix}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    return png_path.resolve(), pdf_path.resolve()

__all__ = [
    "compute_overlap_stats",
    "plot_temporal_explanation_figure",
    "plot_temporal_explanation_separate_panels",
    "save_figure",
]
