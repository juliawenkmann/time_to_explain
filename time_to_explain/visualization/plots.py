from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .utils import (
    COLORS,
    PLOT_STYLE,
    SEQUENTIAL_COLORSCALE,
    _PLOTLY_TEMPLATE,
    _auto_show,
    _ensure_dataframe,
    _histogram_to_bar,
    _map_ratio_to_color,
    _maybe_save,
    _rgba,
    _resolve_event_positions,
    _require_networkx,
    _require_plotly,
    load_dataset_bundle,
    go,
    make_subplots,
    nx,
)


def plot_event_count_over_time(
    df: pd.DataFrame,
    *,
    bins: int = 50,
    show: bool = True,
    save_to: Optional[Union[str, Path]] = None,
) -> "go.Figure":
    _require_plotly()
    if len(df) == 0:
        raise ValueError("No data to plot.")
    centers, counts = _histogram_to_bar(df["ts"].to_numpy(), bins)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=centers, y=counts, marker_color=COLORS["accent2"], name="events"))
    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        title="Event count over time",
        xaxis_title="time",
        yaxis_title="count",
        bargap=0.05,
    )
    _auto_show(fig, show)
    _maybe_save(fig, save_to)
    return fig


def plot_inter_event_time_hist(
    df: pd.DataFrame,
    *,
    bins: int = 50,
    show: bool = True,
    save_to: Optional[Union[str, Path]] = None,
) -> "go.Figure":
    _require_plotly()
    if len(df) < 2:
        raise ValueError("Need at least 2 events to compute inter-event times.")
    ts_sorted = np.sort(df["ts"].to_numpy())
    delta = np.diff(ts_sorted)
    centers, counts = _histogram_to_bar(delta, bins)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=centers, y=counts, marker_color=COLORS["user"], name="Δt histogram"))
    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        title="Inter-event time distribution",
        xaxis_title="Δt",
        yaxis_title="freq",
        bargap=0.05,
    )
    _auto_show(fig, show)
    _maybe_save(fig, save_to)
    return fig


def plot_degree_histograms(
    df: pd.DataFrame,
    *,
    show: bool = True,
    save_to: Optional[Union[str, Path]] = None,
) -> "go.Figure":
    _require_plotly()
    if len(df) == 0:
        raise ValueError("No data to plot.")
    out_deg = df.groupby("u")["i"].nunique()
    in_deg = df.groupby("i")["u"].nunique()

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Out-degree (targets per user)", "In-degree (users per item)"))
    fig.add_trace(go.Histogram(x=out_deg, nbinsx=30, marker_color=COLORS["user"], name="out-degree"), row=1, col=1)
    fig.add_trace(go.Histogram(x=in_deg, nbinsx=30, marker_color=COLORS["item"], name="in-degree"), row=1, col=2)
    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        title="Degree histograms",
        showlegend=False,
        bargap=0.05,
    )
    fig.update_xaxes(title_text="degree", row=1, col=1)
    fig.update_yaxes(title_text="freq", row=1, col=1)
    fig.update_xaxes(title_text="degree", row=1, col=2)
    _auto_show(fig, show)
    _maybe_save(fig, save_to)
    return fig


def plot_dataset_quadrants(
    df: pd.DataFrame,
    *,
    event_bins: int = 50,
    inter_event_bins: int = 50,
    degree_bins: int = 30,
    explain_indices: Optional[Sequence[int]] = None,
    happen_interval: float = 0.5,
    show: bool = True,
    save_to: Optional[Union[str, Path]] = None,
) -> "go.Figure":
    _require_plotly()
    if len(df) == 0:
        raise ValueError("No data to plot.")

    ts = df["ts"].to_numpy(dtype=float)
    event_centers, event_counts = _histogram_to_bar(ts, event_bins)

    if len(ts) >= 2:
        delta = np.diff(np.sort(ts))
        inter_centers, inter_counts = _histogram_to_bar(delta, inter_event_bins)
        has_inter = True
    else:
        inter_centers = np.array([0.0])
        inter_counts = np.array([0.0])
        has_inter = False

    out_deg = df.groupby("u")["i"].nunique().to_numpy()
    in_deg = df.groupby("i")["u"].nunique().to_numpy()

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Event count over time",
            "Inter-event Δt distribution",
            "Explain timeline (anchors)",
            "Out-degree (targets per user)",
            "In-degree (users per item)",
            "Happen rate overview",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )

    fig.add_trace(
        go.Bar(x=event_centers, y=event_counts, marker_color=COLORS["accent2"], name="events"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=inter_centers, y=inter_counts, marker_color=COLORS["user"], name="Δt histogram"),
        row=1,
        col=2,
    )
    if not has_inter:
        fig.add_annotation(
            text="Need ≥2 events",
            xref="x2",
            yref="y2",
            x=0,
            y=float(np.max(inter_counts) if inter_counts.size else 1.0),
            showarrow=False,
            font=dict(color=COLORS["text"], size=PLOT_STYLE["annotation_size"]),
        )

    explain_positions = []
    explain_ts = []
    explain_order = []
    if explain_indices:
        try:
            explain_positions = _resolve_event_positions(df, explain_indices)
            if explain_positions:
                explain_ts = df.iloc[explain_positions]["ts"].to_list()
                explain_order = explain_positions
        except Exception:
            explain_positions = []

    total_events = len(df)
    order = np.arange(total_events)
    if total_events > 20_000:
        sample_idx = np.linspace(0, total_events - 1, 20_000, dtype=int)
        base_ts = df["ts"].to_numpy()[sample_idx]
        base_order = order[sample_idx]
    else:
        base_ts = df["ts"].to_numpy()
        base_order = order

    fig.add_trace(
        go.Scatter(
            x=base_ts,
            y=base_order,
            mode="markers",
            marker=dict(size=4, color=_rgba(COLORS["base"], 0.3)),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    if explain_positions and explain_ts:
        fig.add_trace(
            go.Scatter(
                x=explain_ts,
                y=explain_order,
                mode="markers+text",
                text=[f"idx={idx}" for idx in explain_indices[: len(explain_positions)]],
                textposition="top center",
                marker=dict(color=COLORS["accent"], size=11),
                name="explain anchors",
                hovertemplate="ts=%{x:.2f}<br>row=%{y}<extra></extra>",
            ),
            row=1,
            col=3,
        )
    else:
        fig.add_annotation(
            text="No explain indices",
            xref="x3",
            yref="y3",
            x=0,
            y=0,
            showarrow=False,
            font=dict(color=COLORS["text"], size=PLOT_STYLE["annotation_size"]),
        )

    fig.add_trace(
        go.Histogram(x=out_deg, nbinsx=degree_bins, marker_color=COLORS["item"], name="out-degree"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=in_deg, nbinsx=degree_bins, marker_color=COLORS["accent"], name="in-degree"),
        row=2,
        col=2,
    )

    if "etype" in df.columns and len(df["etype"].unique()) > 0:
        types = sorted(df["etype"].unique())
        ts_by_type = {k: df.loc[df["etype"] == k, "ts"].to_numpy() for k in types}
        H = np.zeros((len(types), len(types)), dtype=float)
        for a, ka in enumerate(types):
            for b, kb in enumerate(types):
                H[a, b] = happen_rate(ts_by_type[ka], ts_by_type[kb], interval=happen_interval, reverse=False)
        fig.add_trace(
            go.Heatmap(
                z=H,
                x=types,
                y=types,
                zmin=0.0,
                zmax=1.0,
                colorscale=SEQUENTIAL_COLORSCALE,
                colorbar=dict(title="rate"),
            ),
            row=2,
            col=3,
        )
    else:
        fig.add_annotation(
            text="No etype column",
            xref="x6",
            yref="y6",
            x=0,
            y=0,
            showarrow=False,
            font=dict(color=COLORS["text"], size=PLOT_STYLE["annotation_size"]),
        )

    fig.update_xaxes(title_text="time", row=1, col=1)
    fig.update_yaxes(title_text="count", row=1, col=1)
    fig.update_xaxes(title_text="Δt", row=1, col=2)
    fig.update_yaxes(title_text="freq", row=1, col=2)
    fig.update_xaxes(title_text="degree", row=2, col=1)
    fig.update_xaxes(title_text="degree", row=2, col=2)
    fig.update_xaxes(title_text="event type", row=2, col=3)
    fig.update_yaxes(title_text="event type", row=2, col=3)
    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        title="Dataset diagnostics",
        bargap=0.05,
        showlegend=False,
        height=820,
        width=1260,
        margin=dict(l=40, r=40, t=90, b=60),
    )
    _auto_show(fig, show)
    _maybe_save(fig, save_to)
    return fig


def plot_adjacency_heatmap(
    df: pd.DataFrame,
    *,
    num_nodes: Optional[int] = None,
    show: bool = True,
    save_to: Optional[Union[str, Path]] = None,
) -> "go.Figure":
    _require_plotly()
    if len(df) == 0:
        raise ValueError("No data to plot.")
    if num_nodes is None:
        num_nodes = int(max(df["u"].max(), df["i"].max()) + 1)
    M = np.zeros((num_nodes, num_nodes), dtype=int)
    for (u, i), cnt in df.value_counts(["u", "i"]).items():
        M[int(u), int(i)] = int(cnt)

    fig = go.Figure(
        data=go.Heatmap(
            z=M,
            colorscale=SEQUENTIAL_COLORSCALE,
            colorbar=dict(title="# interactions"),
        )
    )
    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        title="Adjacency heatmap",
        xaxis_title="i (target)",
        yaxis_title="u (source)",
    )
    _auto_show(fig, show)
    _maybe_save(fig, save_to)
    return fig


def plot_label_balance(
    df: pd.DataFrame,
    *,
    show: bool = True,
    save_to: Optional[Union[str, Path]] = None,
) -> "go.Figure":
    _require_plotly()
    if "label" not in df.columns or len(df) == 0:
        raise ValueError("No 'label' column or no data.")
    vc = df["label"].value_counts().sort_index()
    fig = go.Figure(go.Bar(x=vc.index.astype(str), y=vc.values, marker_color=COLORS["accent"]))
    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        title="Label balance",
        xaxis_title="label",
        yaxis_title="count",
    )
    _auto_show(fig, show)
    _maybe_save(fig, save_to)
    return fig


def happen_rate(source_ts: np.ndarray, target_ts: np.ndarray, interval: float, reverse: bool = False) -> float:
    if reverse:
        source_ts, target_ts = target_ts, source_ts
        source_ts = -1.0 * source_ts
    source_ts = np.sort(np.asarray(source_ts, dtype=float))
    target_ts = np.sort(np.asarray(target_ts, dtype=float))
    if len(source_ts) == 0 or len(target_ts) == 0:
        return 0.0
    j = 0
    count = 0
    for t in source_ts:
        while j < len(target_ts) and target_ts[j] < t:
            j += 1
        ok = False
        if j < len(target_ts) and abs(target_ts[j] - t) <= interval:
            ok = True
        if j > 0 and abs(target_ts[j - 1] - t) <= interval:
            ok = True
        if ok:
            count += 1
    return count / float(len(source_ts))


def plot_happen_rate_matrix(
    df: pd.DataFrame,
    *,
    interval: float = 0.5,
    show: bool = True,
    save_to: Optional[Union[str, Path]] = None,
) -> "go.Figure":
    _require_plotly()
    if "etype" not in df.columns or len(df) == 0:
        raise ValueError("No 'etype' column or no data.")
    types = sorted(df["etype"].unique())
    ts_by_type = {k: df.loc[df["etype"] == k, "ts"].to_numpy() for k in types}
    H = np.zeros((len(types), len(types)), dtype=float)
    for a, ka in enumerate(types):
        for b, kb in enumerate(types):
            H[a, b] = happen_rate(ts_by_type[ka], ts_by_type[kb], interval=interval, reverse=False)

    fig = go.Figure(
        data=go.Heatmap(
            z=H,
            x=types,
            y=types,
            zmin=0.0,
            zmax=1.0,
            colorscale=SEQUENTIAL_COLORSCALE,
            colorbar=dict(title="happen rate"),
        )
    )
    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        title=f"Happen rate matrix (interval={interval})",
        xaxis_title="target type",
        yaxis_title="source type",
    )
    _auto_show(fig, show)
    _maybe_save(fig, save_to)
    return fig


def build_bipartite_graph(
    data: Union[pd.DataFrame, Dict[str, Any], str, Path],
    *,
    max_users: int = 40,
    max_items: int = 40,
    max_edges: int = 500,
    label_column: str = "label",
) -> Tuple["nx.Graph", List[int], List[int], pd.DataFrame]:
    _require_networkx()
    df = _ensure_dataframe(data)
    if len(df) == 0:
        raise ValueError("No interactions available to build a graph.")

    max_users = max(max_users, 1)
    max_items = max(max_items, 1)
    max_edges = max(max_edges, 1)

    user_counts = df["u"].value_counts().sort_values(ascending=False)
    item_counts = df["i"].value_counts().sort_values(ascending=False)
    keep_users = set(user_counts.head(max_users).index.astype(int))
    keep_items = set(item_counts.head(max_items).index.astype(int))

    sub = df[df["u"].isin(keep_users) & df["i"].isin(keep_items)].copy()
    if len(sub) == 0:
        raise ValueError("Filtering removed all edges; increase max_users/max_items.")

    grouped = sub.groupby(["u", "i"], as_index=False)
    agg_ts = grouped["ts"].agg(["size", "mean"]).reset_index()
    agg_ts.rename(columns={"size": "count", "mean": "ts_mean"}, inplace=True)

    if label_column in sub.columns:
        label_stats = grouped[label_column].agg(["mean"]).reset_index()
        label_stats.rename(columns={"mean": "label_mean"}, inplace=True)
        pos_ratio = grouped[label_column].agg(lambda s: float(np.mean(np.asarray(s) > 0))).reset_index()
        pos_ratio.rename(columns={label_column: "positive_ratio"}, inplace=True)
        agg = agg_ts.merge(label_stats, on=["u", "i"], how="left").merge(pos_ratio, on=["u", "i"], how="left")
    else:
        agg = agg_ts
        agg["label_mean"] = np.nan
        agg["positive_ratio"] = np.nan

    agg = agg.sort_values("count", ascending=False)
    if len(agg) > max_edges:
        agg = agg.head(max_edges).copy()

    users = sorted(agg["u"].astype(int).unique())
    items = sorted(agg["i"].astype(int).unique())

    G = nx.Graph()
    for u in users:
        G.add_node(u, bipartite=0, kind="user")
    for i in items:
        G.add_node(i, bipartite=1, kind="item")

    for _, row in agg.iterrows():
        u = int(row["u"])
        v = int(row["i"])
        attrs = {"weight": float(row["count"]), "ts_mean": float(row["ts_mean"])}
        if not np.isnan(row.get("label_mean", np.nan)):
            attrs["label_mean"] = float(row["label_mean"])
            attrs["positive_ratio"] = float(row["positive_ratio"])
        G.add_edge(u, v, **attrs)

    return G, users, items, agg


def plot_bipartite_graph(
    data: Union[pd.DataFrame, Dict[str, Any], str, Path],
    *,
    max_users: int = 40,
    max_items: int = 40,
    max_edges: int = 500,
    edge_cmap: str = "theme_diverging",
    show_labels: bool = True,
    show: bool = True,
    save_to: Optional[Union[str, Path]] = None,
    highlight_users: Optional[Sequence[int]] = None,
    highlight_items: Optional[Sequence[int]] = None,
    highlight_edges: Optional[Sequence[Tuple[int, int]]] = None,
    highlight_size: float = 20.0,
) -> "go.Figure":
    _require_plotly()
    _require_networkx()

    G, users, items, agg = build_bipartite_graph(
        data, max_users=max_users, max_items=max_items, max_edges=max_edges
    )
    if G.number_of_edges() == 0:
        raise ValueError("Graph has no edges to draw.")

    if len(users) == 0 or len(items) == 0:
        raise ValueError("Not enough nodes to draw graph.")

    user_x = np.linspace(-1.0, 1.0, len(users))
    item_x = np.linspace(-1.0, 1.0, len(items))
    pos = {u: (float(x), 1.0) for u, x in zip(users, user_x)}
    pos.update({i: (float(x), -1.0) for i, x in zip(items, item_x)})

    user_sizes = 12.0 * (1.0 + np.log1p([G.degree(u) for u in users]))
    item_sizes = 12.0 * (1.0 + np.log1p([G.degree(i) for i in items]))

    highlight_users = set(highlight_users or [])
    highlight_items = set(highlight_items or [])
    highlight_edges = {(min(u, v), max(u, v)) for u, v in (highlight_edges or [])}

    def _edge_trace(color: str, width: float, filter_edges: Optional[Iterable[Tuple[int, int]]] = None):
        xs: List[float] = []
        ys: List[float] = []
        iterable = filter_edges if filter_edges is not None else [(int(row["u"]), int(row["i"])) for _, row in agg.iterrows()]
        for u, v in iterable:
            if u not in pos or v not in pos:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            xs += [x0, x1, None]
            ys += [y0, y1, None]
        return go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="skip",
            showlegend=False,
        )

    edge_trace = _edge_trace(
        color=COLORS["base"],
        width=1.2,
    )

    highlight_edge_trace = None
    if highlight_edges:
        highlight_edge_trace = _edge_trace(color=COLORS["accent"], width=3.0, filter_edges=highlight_edges)

    user_trace = go.Scatter(
        x=[pos[u][0] for u in users],
        y=[pos[u][1] for u in users],
        mode="markers+text" if show_labels else "markers",
        marker=dict(
            size=user_sizes,
            color=[COLORS["accent"] if u in highlight_users else COLORS["user"] for u in users],
            line=dict(color=COLORS["grid"], width=0.6),
        ),
        text=[str(u) for u in users] if show_labels else None,
        name="users",
        hovertemplate="user=%{text}<extra></extra>" if show_labels else "user=%{x}<extra></extra>",
    )
    item_trace = go.Scatter(
        x=[pos[i][0] for i in items],
        y=[pos[i][1] for i in items],
        mode="markers+text" if show_labels else "markers",
        marker=dict(
            size=item_sizes,
            color=[COLORS["accent2"] if i in highlight_items else COLORS["item"] for i in items],
            symbol="square",
            line=dict(color=COLORS["grid"], width=0.6),
        ),
        text=[str(i) for i in items] if show_labels else None,
        name="items",
        hovertemplate="item=%{text}<extra></extra>" if show_labels else "item=%{x}<extra></extra>",
    )

    fig = go.Figure(data=[edge_trace, user_trace, item_trace])
    if highlight_edge_trace is not None:
        fig.add_trace(highlight_edge_trace)

    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        title="Bipartite interaction snapshot",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    _auto_show(fig, show)
    _maybe_save(fig, save_to)
    return fig


def animate_bipartite_graph(
    data: Union[pd.DataFrame, Dict[str, Any], str, Path],
    *,
    bins: int = 30,
    max_users: int = 40,
    max_items: int = 40,
    cumulative: bool = True,
    edge_cmap: str = "theme_sequential",
    pruned: Optional[float] = None,
    show: bool = True,
    save_to: Optional[Union[str, Path]] = None,
) -> "go.Figure":
    _require_plotly()
    _require_networkx()

    df = _ensure_dataframe(data).sort_values("ts")
    if len(df) == 0:
        raise ValueError("Dataset has no interactions to animate.")

    user_counts_all = df["u"].value_counts()
    item_counts_all = df["i"].value_counts()
    top_users_series = user_counts_all.head(max(1, max_users))
    top_items_series = item_counts_all.head(max(1, max_items))
    if pruned is not None:
        if not (0 < pruned <= 1):
            raise ValueError(f"`pruned` must be in (0, 1], got {pruned!r}")
        n_users_keep = max(1, int(np.ceil(len(top_users_series) * float(pruned))))
        n_items_keep = max(1, int(np.ceil(len(top_items_series) * float(pruned))))
        top_users_series = top_users_series.head(n_users_keep)
        top_items_series = top_items_series.head(n_items_keep)

    df = df[df["u"].isin(top_users_series.index) & df["i"].isin(top_items_series.index)]
    if len(df) == 0:
        raise ValueError("Pruning removed all edges. Increase `pruned`, `max_users`, or `max_items`.")

    bins = max(1, min(int(bins), len(df)))
    t_edges = np.linspace(df["ts"].min(), df["ts"].max(), bins + 1)
    t_labels = [f"{t_edges[i]:.2f} – {t_edges[i+1]:.2f}" for i in range(bins)]

    users = sorted(df["u"].unique().astype(int))
    items = sorted(df["i"].unique().astype(int))

    if len(users) == 0 or len(items) == 0:
        raise ValueError("Not enough nodes to animate graph.")

    user_x = np.linspace(-1.0, 1.0, len(users))
    item_x = np.linspace(-1.0, 1.0, len(items))
    pos = {u: (float(x), 1.0) for u, x in zip(users, user_x)}
    pos.update({i: (float(x), -1.0) for i, x in zip(items, item_x)})

    edges_over_time: List[pd.DataFrame] = []
    for idx in range(bins):
        mask = (df["ts"] >= t_edges[idx]) & (df["ts"] < t_edges[idx + 1])
        edges_over_time.append(df.loc[mask, ["u", "i"]].astype(int))

    user_activity = df["u"].value_counts()
    user_sizes = 12 + 3 * np.log1p([user_activity.get(u, 0) for u in users])
    item_activity = df["i"].value_counts()
    item_sizes = 12 + 3 * np.log1p([item_activity.get(i, 0) for i in items])

    user_trace = go.Scatter(
        x=[pos[u][0] for u in users],
        y=[pos[u][1] for u in users],
        mode="markers",
        marker=dict(
            size=user_sizes,
            color=COLORS["user"],
            line=dict(color=COLORS["grid"], width=0.5),
        ),
        name="Users",
        hovertext=[f"user={u}" for u in users],
        hoverinfo="text",
    )
    item_trace = go.Scatter(
        x=[pos[i][0] for i in items],
        y=[pos[i][1] for i in items],
        mode="markers",
        marker=dict(
            size=item_sizes,
            color=COLORS["item"],
            symbol="square",
            line=dict(color=COLORS["grid"], width=0.5),
        ),
        name="Items",
        hovertext=[f"item={i}" for i in items],
        hoverinfo="text",
    )
    edges_trace = go.Scatter(
        x=[],
        y=[],
        mode="lines",
        line=dict(width=1.5, color=_rgba(COLORS["base"], 0.9)),
        hoverinfo="skip",
        showlegend=False,
    )

    frames = []
    denom = max(1, bins - 1)
    for f in range(bins):
        if cumulative:
            active = pd.concat(edges_over_time[: f + 1], ignore_index=True)
        else:
            active = edges_over_time[f]
        xs: List[float] = []
        ys: List[float] = []
        if not active.empty:
            counts = active.groupby(["u", "i"]).size().reset_index(name="count")
            for _, row in counts.iterrows():
                u, i = int(row["u"]), int(row["i"])
                x0, y0 = pos[u]
                x1, y1 = pos[i]
                xs += [x0, x1, None]
                ys += [y0, y1, None]
        color = _map_ratio_to_color(f / denom, edge_cmap=edge_cmap)
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(width=1.8, color=color),
                        hoverinfo="skip",
                    )
                ]
            )
        )

    fig = go.Figure(data=[edges_trace, user_trace, item_trace], frames=frames)
    title_text = "Temporal bipartite graph (use ▶ to play, or drag the slider)"
    if pruned is not None:
        title_text += f" — pruned={pruned:.2f}"
    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        title=dict(text=title_text, y=0.97, yanchor="top", pad=dict(t=60)),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=110, b=40),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": -0.12,
                "xanchor": "left",
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {"frame": {"duration": 350, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}},
                        ],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}},
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "y": -0.07,
                "x": 0.08,
                "len": 0.85,
                "xanchor": "left",
                "steps": [
                    {
                        "label": f"{i+1}/{bins} · {t_labels[i]}",
                        "method": "animate",
                        "args": [[f"frame{i}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                    }
                    for i in range(bins)
                ],
            }
        ],
    )

    for i, fr in enumerate(fig.frames):
        fr.name = f"frame{i}"

    _auto_show(fig, show)
    _maybe_save(fig, save_to)
    return fig


def animate_stick_figure(
    data: Union[pd.DataFrame, Dict[str, Any], str, Path],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    clip_id: Optional[int] = None,
    max_frames: Optional[int] = None,
    show: bool = True,
    save_to: Optional[Union[str, Path]] = None,
) -> "go.Figure":
    _require_plotly()

    edge_features = None
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        meta = dict(metadata or {})
    elif isinstance(data, (dict, str, Path)):
        bundle = load_dataset_bundle(data, verbose=False)
        df = _ensure_dataframe(bundle.get("interactions"))
        edge_features = bundle.get("edge_features")
        meta = dict(metadata or bundle.get("metadata") or {})
    else:
        raise TypeError(f"Unsupported data type: {type(data)!r}")

    if edge_features is None:
        raise ValueError("Stick-figure animation requires edge_features in the dataset bundle.")

    edge_arr = np.asarray(edge_features)
    if edge_arr.ndim != 2:
        raise ValueError("edge_features must be a 2D array.")

    if "e_idx" not in df.columns:
        df = df.copy()
        df["e_idx"] = np.arange(len(df), dtype=int) + 1

    if edge_arr.shape[0] == len(df) + 1:
        feat = edge_arr[df["e_idx"].astype(int).to_numpy()]
    elif edge_arr.shape[0] == len(df):
        feat = edge_arr
    else:
        raise ValueError("edge_features length does not align with the interactions dataframe.")

    if feat.shape[1] < 4:
        raise ValueError("Stick-figure edge features must include endpoint coordinates (f0..f3).")

    u_raw = df["u"].to_numpy(dtype=int)
    v_raw = df["i"].to_numpy(dtype=int)
    min_node = int(min(u_raw.min(), v_raw.min()))
    node_offset = min_node if min_node > 0 else 0

    joints_per_person = 14
    u_orig = u_raw - node_offset
    v_orig = v_raw - node_offset
    clip_ids = u_orig // joints_per_person
    if clip_id is None:
        clip_id = int(np.min(clip_ids))
    if clip_id not in set(clip_ids.tolist()):
        raise ValueError(f"clip_id {clip_id} not found in the dataset.")

    clip_mask = clip_ids == clip_id
    df_clip = df.loc[clip_mask]
    feat_clip = feat[clip_mask]
    u_clip = u_orig[clip_mask]
    v_clip = v_orig[clip_mask]

    is_query = feat_clip[:, 7] if feat_clip.shape[1] > 7 else np.zeros(len(df_clip), dtype=float)
    bone_mask = is_query < 0.5
    query_mask = ~bone_mask
    if not np.any(bone_mask):
        raise ValueError("No bone edges found to animate for this clip.")

    feat_bone = feat_clip[bone_mask]
    u_bone = u_clip[bone_mask]
    v_bone = v_clip[bone_mask]
    feat_query = feat_clip[query_mask]

    x0 = feat_bone[:, 0]
    y0 = feat_bone[:, 1]
    x1 = feat_bone[:, 2]
    y1 = feat_bone[:, 3]
    has_query = feat_query.size > 0
    if has_query:
        xq0 = feat_query[:, 0]
        yq0 = feat_query[:, 1]
        xq1 = feat_query[:, 2]
        yq1 = feat_query[:, 3]
    else:
        xq0 = yq0 = xq1 = yq1 = np.array([], dtype=float)

    config = meta.get("config") if isinstance(meta.get("config"), dict) else {}
    frames_cfg = int(config.get("frames", 0) or 0)

    if feat_bone.shape[1] > 8:
        frame_norm = feat_bone[:, 8]
        if frames_cfg > 1:
            frame_idx = np.clip(np.round(frame_norm * (frames_cfg - 1)), 0, frames_cfg - 1).astype(int)
            num_frames = frames_cfg
        else:
            unique_norms = sorted({float(v) for v in frame_norm})
            frame_map = {val: idx for idx, val in enumerate(unique_norms)}
            frame_idx = np.array([frame_map[float(v)] for v in frame_norm], dtype=int)
            num_frames = len(unique_norms)
    else:
        if frames_cfg <= 0:
            frames_cfg = max(1, len(feat_bone) // 14)
        index_in_clip = np.arange(len(feat_bone), dtype=int)
        frame_idx = (index_in_clip // 8).astype(int)
        num_frames = max(frames_cfg, int(frame_idx.max()) + 1)

    if has_query:
        if feat_query.shape[1] > 8:
            frame_norm_q = feat_query[:, 8]
            if frames_cfg > 1:
                query_frame_idx = np.clip(
                    np.round(frame_norm_q * (frames_cfg - 1)), 0, frames_cfg - 1
                ).astype(int)
            else:
                unique_norms = sorted({float(v) for v in frame_norm_q})
                frame_map = {val: idx for idx, val in enumerate(unique_norms)}
                query_frame_idx = np.array([frame_map[float(v)] for v in frame_norm_q], dtype=int)
        else:
            query_frame_idx = np.full(len(feat_query), max(frames_cfg - 1, 0), dtype=int)
    else:
        query_frame_idx = np.array([], dtype=int)

    if max_frames is not None:
        max_frames = int(max_frames)
        if max_frames > 0 and max_frames < num_frames:
            keep = frame_idx < max_frames
            frame_idx = frame_idx[keep]
            feat_bone = feat_bone[keep]
            u_bone = u_bone[keep]
            v_bone = v_bone[keep]
            x0 = x0[keep]
            y0 = y0[keep]
            x1 = x1[keep]
            y1 = y1[keep]
            if has_query:
                keep_q = query_frame_idx < max_frames
                query_frame_idx = query_frame_idx[keep_q]
                xq0 = xq0[keep_q]
                yq0 = yq0[keep_q]
                xq1 = xq1[keep_q]
                yq1 = yq1[keep_q]
            num_frames = max_frames

    joint_u = u_bone % joints_per_person
    joint_v = v_bone % joints_per_person

    J_T, J_H, J_LS, J_RS, J_LE, J_RE, J_LW, J_RW, J_LH, J_RH, J_LK, J_RK, J_LA, J_RA = range(14)
    left_edges = {tuple(sorted(pair)) for pair in [(J_T, J_LS), (J_LS, J_LE), (J_LE, J_LW)]}
    right_edges = {tuple(sorted(pair)) for pair in [(J_T, J_RS), (J_RS, J_RE), (J_RE, J_RW)]}
    center_edges = {tuple(sorted(pair)) for pair in [(J_T, J_H), (J_LS, J_RS)]}

    frame_ids = sorted(set(frame_idx.tolist()))
    if not frame_ids:
        raise ValueError("No frames available to animate.")
    all_x = np.concatenate([x0, x1, xq0, xq1])
    all_y = np.concatenate([y0, y1, yq0, yq1])
    x_min = float(np.nanmin(all_x))
    x_max = float(np.nanmax(all_x))
    y_min = float(np.nanmin(all_y))
    y_max = float(np.nanmax(all_y))
    pad_x = max(0.1 * (x_max - x_min), 0.05)
    pad_y = max(0.1 * (y_max - y_min), 0.05)

    def _make_traces(frame_id: int, *, show_legend: bool) -> List["go.Scatter"]:
        query_idxs = [i for i, f in enumerate(query_frame_idx) if f == frame_id]
        query_present = bool(query_idxs)
        query_color = COLORS["item"] if query_present else _rgba(COLORS["item"], 0.2)
        query_width = 3.6 if query_present else 2.0
        query_dash = "solid" if query_present else "dot"
        idxs = [i for i, f in enumerate(frame_idx) if f == frame_id]
        if not idxs:
            return [
                go.Scatter(x=[], y=[], mode="lines", line=dict(color=COLORS["base"], width=2), name="center", showlegend=show_legend),
                go.Scatter(x=[], y=[], mode="lines", line=dict(color=COLORS["accent2"], width=3), name="left arm", showlegend=show_legend),
                go.Scatter(x=[], y=[], mode="lines", line=dict(color=COLORS["accent"], width=3), name="right arm", showlegend=show_legend),
                go.Scatter(x=[], y=[], mode="lines", line=dict(color=query_color, width=query_width, dash=query_dash), name="contact edge", showlegend=show_legend),
                go.Scatter(x=[], y=[], mode="markers", marker=dict(size=8, color=COLORS["text"]), name="joints", showlegend=show_legend),
            ]

        left_x: List[float] = []
        left_y: List[float] = []
        right_x: List[float] = []
        right_y: List[float] = []
        center_x: List[float] = []
        center_y: List[float] = []

        positions: Dict[int, List[Tuple[float, float]]] = {}

        for idx in idxs:
            ju = int(joint_u[idx])
            jv = int(joint_v[idx])
            pair = tuple(sorted((ju, jv)))
            x0i = float(x0[idx])
            y0i = float(y0[idx])
            x1i = float(x1[idx])
            y1i = float(y1[idx])

            if pair in left_edges:
                left_x += [x0i, x1i, None]
                left_y += [y0i, y1i, None]
            elif pair in right_edges:
                right_x += [x0i, x1i, None]
                right_y += [y0i, y1i, None]
            else:
                center_x += [x0i, x1i, None]
                center_y += [y0i, y1i, None]

            positions.setdefault(ju, []).append((x0i, y0i))
            positions.setdefault(jv, []).append((x1i, y1i))

        joint_order = [
            J_T, J_H,
            J_LS, J_RS,
            J_LH, J_RH,
            J_LE, J_RE,
            J_LK, J_RK,
            J_LW, J_RW,
            J_LA, J_RA,
        ]
        joint_x: List[float] = []
        joint_y: List[float] = []
        joint_colors: List[str] = []
        joint_sizes: List[float] = []
        for j in joint_order:
            if j not in positions:
                continue
            coords = positions[j]
            xs = [pt[0] for pt in coords]
            ys = [pt[1] for pt in coords]
            joint_x.append(float(np.mean(xs)))
            joint_y.append(float(np.mean(ys)))
            if j == J_H:
                joint_colors.append(COLORS["item"])
                joint_sizes.append(10.0)
            elif j in (J_LS, J_LE, J_LW):
                joint_colors.append(COLORS["accent2"])
                joint_sizes.append(9.0 if j == J_LW else 8.0)
            elif j in (J_RS, J_RE, J_RW):
                joint_colors.append(COLORS["accent"])
                joint_sizes.append(9.0 if j == J_RW else 8.0)
            elif j in (J_LH, J_LK, J_LA, J_RH, J_RK, J_RA):
                joint_colors.append(COLORS["base"])
                joint_sizes.append(7.5 if j in (J_LA, J_RA) else 7.0)
            else:
                joint_colors.append(COLORS["user"])
                joint_sizes.append(9.0)

        query_x: List[float] = []
        query_y: List[float] = []
        for q_idx in query_idxs:
            query_x += [float(xq0[q_idx]), float(xq1[q_idx]), None]
            query_y += [float(yq0[q_idx]), float(yq1[q_idx]), None]

        return [
            go.Scatter(
                x=center_x,
                y=center_y,
                mode="lines",
                line=dict(color=_rgba(COLORS["base"], 0.8), width=2.2),
                name="torso",
                showlegend=show_legend,
                hoverinfo="skip",
            ),
            go.Scatter(
                x=left_x,
                y=left_y,
                mode="lines",
                line=dict(color=COLORS["accent2"], width=3.2),
                name="left arm",
                showlegend=show_legend,
                hoverinfo="skip",
            ),
            go.Scatter(
                x=right_x,
                y=right_y,
                mode="lines",
                line=dict(color=COLORS["accent"], width=3.2),
                name="right arm",
                showlegend=show_legend,
                hoverinfo="skip",
            ),
            go.Scatter(
                x=query_x,
                y=query_y,
                mode="lines",
                line=dict(color=query_color, width=query_width, dash=query_dash),
                name="contact edge",
                showlegend=show_legend and has_query,
                hoverinfo="skip",
            ),
            go.Scatter(
                x=joint_x,
                y=joint_y,
                mode="markers",
                marker=dict(
                    size=joint_sizes,
                    color=joint_colors,
                    line=dict(color=COLORS["text"], width=0.8),
                ),
                name="joints",
                showlegend=show_legend,
                hoverinfo="skip",
            ),
        ]

    init_traces = _make_traces(frame_ids[0], show_legend=True)
    frames = [
        go.Frame(name=str(fid), data=_make_traces(fid, show_legend=False))
        for fid in frame_ids
    ]

    fig = go.Figure(data=init_traces, frames=frames)

    sliders = [
        {
            "active": 0,
            "y": -0.08,
            "x": 0.1,
            "len": 0.8,
            "pad": {"b": 10, "t": 50},
            "currentvalue": {"prefix": "frame "},
            "steps": [
                {
                    "label": f"{idx + 1}/{len(frame_ids)}",
                    "method": "animate",
                    "args": [
                        [str(fid)],
                        {
                            "frame": {"duration": 160, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                }
                for idx, fid in enumerate(frame_ids)
            ],
        }
    ]

    title = meta.get("dataset_name") or meta.get("recipe") or "Stick figure"
    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        title=f"{title} - waving animation (clip {clip_id})",
        xaxis=dict(visible=False, range=[x_min - pad_x, x_max + pad_x]),
        yaxis=dict(visible=False, range=[y_min - pad_y, y_max + pad_y], scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, t=70, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": -0.14,
                "xanchor": "left",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 160, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=sliders,
    )

    _auto_show(fig, show)
    _maybe_save(fig, save_to)
    return fig


def plot_force_directed_graph(
    data: Union[pd.DataFrame, Dict[str, Any]],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    max_nodes: int = 60,
    max_edges: int = 600,
    highlight_nodes: Optional[Sequence[int]] = None,
    highlight_edges: Optional[Sequence[Tuple[int, int]]] = None,
    show: bool = True,
    save_to: Optional[Union[str, Path]] = None,
) -> "go.Figure":
    """Force-directed view for general graphs (useful for non-bipartite datasets)."""
    _require_plotly()
    _require_networkx()
    df = _ensure_dataframe(data)
    if len(df) == 0:
        raise ValueError("No interactions available to build a graph.")

    meta = dict(metadata or {})
    cols = ["u", "i", "ts"]
    if "label" in df.columns:
        cols.append("label")
    sub = df[cols].copy()

    if max_nodes is not None and max_nodes > 0:
        node_counts = pd.concat([sub["u"], sub["i"]], ignore_index=True).value_counts()
        keep_nodes = set(node_counts.head(max_nodes).index.astype(int))
        filtered = sub[sub["u"].isin(keep_nodes) & sub["i"].isin(keep_nodes)]
        if not filtered.empty:
            sub = filtered

    sub["src"] = sub[["u", "i"]].min(axis=1).astype(int)
    sub["dst"] = sub[["u", "i"]].max(axis=1).astype(int)

    grouped = sub.groupby(["src", "dst"], as_index=False)
    agg = grouped.agg(
        count=("ts", "size"),
        ts_mean=("ts", "mean"),
    )
    if "label" in sub.columns:
        label_stats = grouped.agg(
            label_mean=("label", "mean"),
            positive_ratio=("label", lambda s: float(np.mean(np.asarray(s) > 0))),
        )
        agg = agg.merge(label_stats, on=["src", "dst"], how="left")
    else:
        agg["label_mean"] = np.nan
        agg["positive_ratio"] = np.nan

    agg = agg.sort_values("count", ascending=False)
    if max_edges is not None and max_edges > 0 and len(agg) > max_edges:
        agg = agg.head(max_edges)

    if agg.empty:
        raise ValueError("Graph has no edges to draw.")

    def _edge_tuple(u: Any, v: Any) -> Tuple[int, int]:
        ui, vi = int(u), int(v)
        return (ui, vi) if ui <= vi else (vi, ui)

    highlight_nodes_set = {int(n) for n in (highlight_nodes or [])}
    highlight_edges_set = {_edge_tuple(u, v) for u, v in (highlight_edges or [])}

    ground = meta.get("ground_truth") or {}
    if not highlight_nodes_set:
        highlight_nodes_set = {int(n) for n in ground.get("motif_nodes", []) if n is not None}
    if not highlight_edges_set:
        raw_edges = ground.get("motif_edges_undirected") or ground.get("motif_edges") or []
        highlight_edges_set = {
            _edge_tuple(edge[0], edge[1])
            for edge in raw_edges
            if isinstance(edge, (list, tuple)) and len(edge) == 2
        }

    G = nx.Graph()
    for node in sorted(set(agg["src"].astype(int)).union(agg["dst"].astype(int))):
        G.add_node(int(node))
    for _, row in agg.iterrows():
        u = int(row["src"])
        v = int(row["dst"])
        G.add_edge(
            u,
            v,
            weight=float(row["count"]),
            count=float(row["count"]),
            positive_ratio=float(row.get("positive_ratio", np.nan)),
        )

    if G.number_of_edges() == 0:
        raise ValueError("Graph has no edges to draw.")

    pos = nx.spring_layout(G, seed=42, weight="weight")
    max_count = agg["count"].max() if len(agg) else 1.0

    edge_traces: List["go.Scatter"] = []
    for _, row in agg.iterrows():
        u = int(row["src"])
        v = int(row["dst"])
        euv = _edge_tuple(u, v)
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        base_width = 2.0 + 4.0 * (float(row["count"]) / max_count if max_count > 0 else 0.0)
        ratio = float(row.get("positive_ratio", np.nan))
        if np.isnan(ratio):
            color = _rgba(COLORS["base"], 0.55)
        else:
            color = _map_ratio_to_color(np.clip(ratio, 0.0, 1.0))
        if euv in highlight_edges_set:
            color = COLORS["accent"]
            base_width += 2.5
        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=color, width=base_width),
                hovertemplate=(
                    f"{u}–{v}"
                    f"<br>count={int(row['count'])}"
                    + (
                        f"<br>pos%={float(ratio) * 100:.1f}%"
                        if not np.isnan(ratio)
                        else ""
                    )
                    + "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    node_sizes = {
        node: 10.0 + 6.0 * np.log1p(G.degree(node))
        for node in G.nodes
    }

    def _node_trace(nodes: Sequence[int], *, color: str) -> Optional["go.Scatter"]:
        if not nodes:
            return None
        return go.Scatter(
            x=[pos[n][0] for n in nodes],
            y=[pos[n][1] for n in nodes],
            mode="markers+text",
            marker=dict(
                size=[node_sizes[n] for n in nodes],
                color=color,
                line=dict(color=COLORS["text"], width=1.0),
            ),
            text=[str(n) for n in nodes],
            textposition="top center",
            hovertemplate="node=%{text}<extra></extra>",
            showlegend=False,
        )

    highlight_nodes_final = sorted({n for n in highlight_nodes_set if n in G.nodes})
    regular_nodes = sorted([n for n in G.nodes if n not in highlight_nodes_set])

    fig = go.Figure()
    for trace in edge_traces:
        fig.add_trace(trace)

    regular_trace = _node_trace(regular_nodes, color=COLORS["accent2"])
    if regular_trace is not None:
        fig.add_trace(regular_trace)
    highlight_trace = _node_trace(highlight_nodes_final, color=COLORS["item"])
    if highlight_trace is not None:
        fig.add_trace(highlight_trace)

    title = meta.get("dataset_name") or meta.get("recipe") or "Graph"
    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        title=title + " — force-directed view",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        margin=dict(l=20, r=20, t=70, b=40),
    )

    _auto_show(fig, show)
    _maybe_save(fig, save_to)
    return fig


def plot_triadic_closure_subgraph(
    data: Union[pd.DataFrame, Dict[str, Any]],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    max_targets: int = 1,
    window: int = 3,
    show: bool = True,
    save_to: Optional[Union[str, Path]] = None,
) -> "go.Figure":
    """Visualize consecutive events around triadic-closure targets (non-bipartite)."""
    _require_plotly()
    _require_networkx()

    if isinstance(data, dict):
        df = _ensure_dataframe(data.get("interactions"))
        meta = dict(metadata or data.get("metadata") or {})
    else:
        df = _ensure_dataframe(data)
        meta = dict(metadata or {})

    ground = meta.get("ground_truth") or {}
    raw_targets = ground.get("targets") or []
    raw_rationales = ground.get("rationales") or {}

    targets: List[int] = []
    for t in raw_targets:
        try:
            targets.append(int(t))
        except (TypeError, ValueError):
            continue

    rationales: Dict[int, List[int]] = {}
    for key, value in raw_rationales.items():
        try:
            k_int = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(value, (list, tuple)):
            continue
        cleaned = []
        for item in value:
            try:
                cleaned.append(int(item))
            except (TypeError, ValueError):
                continue
        if cleaned:
            rationales[k_int] = cleaned

    targets = [t for t in targets if 0 <= t < len(df)]
    if not targets or not rationales:
        raise ValueError("Triadic-closure plot requires ground_truth targets and rationales.")

    targets = sorted(set(targets))
    if max_targets and len(targets) > max_targets:
        idx = np.linspace(0, len(targets) - 1, num=max_targets, dtype=int)
        selected_targets = [targets[i] for i in idx]
    else:
        selected_targets = targets

    window = max(0, int(window))
    support_indices: set[int] = set()
    for t_idx in selected_targets:
        support_indices.update(rationales.get(int(t_idx), []))

    event_indices: List[int] = []
    for t_idx in selected_targets:
        start = max(0, int(t_idx) - window)
        end = min(len(df) - 1, int(t_idx) + window)
        event_indices.extend(range(start, end + 1))

    event_indices = sorted({idx for idx in event_indices if 0 <= idx < len(df)})
    if not event_indices:
        raise ValueError("No consecutive events available to plot for the selected targets.")

    edges_info: List[Dict[str, Any]] = []
    for idx in event_indices:
        row = df.iloc[int(idx)]
        try:
            u = int(row["u"])
            v = int(row["i"])
        except (TypeError, ValueError):
            continue
        if u == v:
            continue
        if idx in selected_targets:
            kind = "target"
        elif idx in support_indices:
            kind = "support"
        else:
            kind = "context"
        edges_info.append(
            {
                "u": u,
                "v": v,
                "idx": int(idx),
                "ts": float(row.get("ts", np.nan)),
                "kind": kind,
            }
        )

    if not edges_info:
        raise ValueError("No edges available to plot for the selected consecutive events.")

    G = nx.Graph()
    for info in edges_info:
        G.add_edge(int(info["u"]), int(info["v"]))

    pos = None
    fixed_nodes: Optional[Iterable[int]] = None
    pos_from_features = False
    pos_from_features = False
    if len(selected_targets) == 1:
        target_edges = [
            (int(info["u"]), int(info["v"]))
            for info in edges_info
            if info["kind"] == "target"
        ]
        support_edges = [
            (int(info["u"]), int(info["v"]))
            for info in edges_info
            if info["kind"] == "support"
        ]
        if target_edges and support_edges:
            u_t, v_t = target_edges[0]
            target_nodes = {u_t, v_t}
            support_nodes = {n for edge in support_edges for n in edge}
            support_only = [n for n in support_nodes if n not in target_nodes]
            if len(support_only) == 1 and u_t in G.nodes and v_t in G.nodes and support_only[0] in G.nodes:
                w = support_only[0]
                pos_init = {
                    u_t: (-0.8, -0.4),
                    v_t: (0.8, -0.4),
                    w: (0.0, 0.9),
                }
                try:
                    pos = nx.spring_layout(G, seed=42, pos=pos_init, fixed=pos_init.keys())
                    fixed_nodes = list(pos_init.keys())
                except TypeError:
                    pos = None

    def _min_distance(layout: Dict[int, Tuple[float, float]]) -> float:
        nodes = list(layout.keys())
        if len(nodes) < 2:
            return float("inf")
        min_dist = float("inf")
        for i, u in enumerate(nodes[:-1]):
            x0, y0 = layout[u]
            for v in nodes[i + 1 :]:
                x1, y1 = layout[v]
                dist = float(np.hypot(x1 - x0, y1 - y0))
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def _layout_with_spread(
        graph: "nx.Graph",
        *,
        base_pos: Optional[Dict[int, Tuple[float, float]]] = None,
        fixed: Optional[Iterable[int]] = None,
    ) -> Dict[int, Tuple[float, float]]:
        n_nodes = max(graph.number_of_nodes(), 1)
        base_k = 1.5 / np.sqrt(n_nodes)
        best_pos: Optional[Dict[int, Tuple[float, float]]] = None
        best_min = -1.0
        for attempt in range(4):
            k = base_k * (1.4 + 0.5 * attempt)
            kwargs = dict(seed=42, k=k, iterations=200, scale=1.6)
            if base_pos is not None:
                kwargs["pos"] = base_pos
                if fixed:
                    kwargs["fixed"] = list(fixed)
            layout = nx.spring_layout(graph, **kwargs)
            min_dist = _min_distance(layout)
            if min_dist > best_min:
                best_pos = layout
                best_min = min_dist
            if min_dist >= 0.22:
                return layout
        if best_pos is None:
            best_pos = nx.spring_layout(graph, seed=42, iterations=200, scale=1.6)
        # Deterministic jitter to avoid exact overlaps.
        rng = np.random.default_rng(42)
        for node, (x, y) in best_pos.items():
            dx, dy = rng.normal(scale=0.05, size=2)
            best_pos[node] = (float(x + dx), float(y + dy))
        return best_pos

    if pos is None:
        pos = _layout_with_spread(G)
    else:
        pos = _layout_with_spread(G, base_pos=pos, fixed=fixed_nodes)
    node_sizes = {node: 10.0 + 6.0 * np.log1p(G.degree(node)) for node in G.nodes}

    target_nodes = {info["u"] for info in edges_info if info["kind"] == "target"} | {
        info["v"] for info in edges_info if info["kind"] == "target"
    }
    support_nodes = {info["u"] for info in edges_info if info["kind"] == "support"} | {
        info["v"] for info in edges_info if info["kind"] == "support"
    }

    fig = go.Figure()
    legend_used: set[str] = set()
    for info in edges_info:
        u = int(info["u"])
        v = int(info["v"])
        kind = str(info["kind"])
        if kind == "target":
            color = COLORS["accent"]
            width = 3.2
            label = "target edge"
        elif kind == "support":
            color = COLORS["accent2"]
            width = 2.4
            label = "support edge"
        else:
            color = _rgba(COLORS["base"], 0.45)
            width = 1.6
            label = "context edge"

        show_legend = label not in legend_used
        legend_used.add(label)
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=color, width=width),
                name=label,
                showlegend=show_legend,
                hovertemplate=(
                    f"{u}–{v}"
                    f"<br>event={info['idx']}"
                    + (f"<br>ts={info['ts']:.2f}" if not np.isnan(info["ts"]) else "")
                    + f"<extra>{kind}</extra>"
                ),
            )
        )

    target_nodes_final = sorted({n for n in target_nodes if n in G.nodes})
    support_only_nodes = sorted({n for n in support_nodes if n in G.nodes and n not in target_nodes})
    context_nodes = sorted([n for n in G.nodes if n not in target_nodes and n not in support_nodes])

    def _node_trace(nodes: Sequence[int], *, color: str, name: str) -> Optional["go.Scatter"]:
        if not nodes:
            return None
        return go.Scatter(
            x=[pos[int(n)][0] for n in nodes],
            y=[pos[int(n)][1] for n in nodes],
            mode="markers+text",
            marker=dict(
                size=[node_sizes[int(n)] for n in nodes],
                color=color,
                line=dict(color=COLORS["text"], width=1.0),
            ),
            text=[str(int(n)) for n in nodes],
            textposition="top center",
            hovertemplate="node=%{text}<extra></extra>",
            name=name,
        )

    context_trace = _node_trace(context_nodes, color=COLORS["base"], name="context nodes")
    if context_trace is not None:
        fig.add_trace(context_trace)
    support_trace_nodes = _node_trace(support_only_nodes, color=COLORS["accent2"], name="support nodes")
    if support_trace_nodes is not None:
        fig.add_trace(support_trace_nodes)
    target_trace_nodes = _node_trace(target_nodes_final, color=COLORS["item"], name="target nodes")
    if target_trace_nodes is not None:
        fig.add_trace(target_trace_nodes)

    title = meta.get("dataset_name") or meta.get("recipe") or "Triadic closure"
    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        title=f"{title} - triadic-closure subset (+/-{window} events)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        margin=dict(l=20, r=20, t=70, b=40),
    )

    _auto_show(fig, show)
    _maybe_save(fig, save_to)
    return fig


def plot_ground_truth_subgraph(
    data: Union[pd.DataFrame, Dict[str, Any]],
    event_idx: int,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    event_window: int = 20,
    time_window: Optional[float] = None,
    max_context_edges: int = 40,
    explanation_indices: Optional[Sequence[int]] = None,
    show: bool = True,
    save_to: Optional[Union[str, Path]] = None,
) -> "go.Figure":
    """Highlight ground-truth edges around a target event with context and explanation edges."""
    _require_plotly()
    _require_networkx()

    if isinstance(data, dict):
        df = _ensure_dataframe(data.get("interactions"))
        meta = dict(metadata or data.get("metadata") or {})
    else:
        df = _ensure_dataframe(data)
        meta = dict(metadata or {})

    if len(df) == 0:
        raise ValueError("No interactions to visualise.")

    ground = meta.get("ground_truth") or {}
    raw_targets = ground.get("targets") or []
    raw_rationales = ground.get("rationales") or {}

    targets: List[int] = []
    for t in raw_targets:
        try:
            targets.append(int(t))
        except (TypeError, ValueError):
            continue

    rationales: Dict[int, List[int]] = {}
    for key, value in raw_rationales.items():
        try:
            k_int = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(value, (list, tuple)):
            continue
        cleaned: List[int] = []
        for item in value:
            try:
                cleaned.append(int(item))
            except (TypeError, ValueError):
                continue
        if cleaned:
            rationales[k_int] = cleaned

    targets = sorted({t for t in targets if 0 <= t < len(df)})
    if not targets:
        raise ValueError("Ground-truth targets are missing or out of range.")

    event_idx = int(event_idx)
    if event_idx not in targets:
        event_idx = targets[0]

    support_indices = [i for i in rationales.get(event_idx, []) if 0 <= i < len(df)]
    gt_indices = {event_idx, *support_indices}

    target_row = df.iloc[event_idx]
    target_nodes = {int(target_row["u"]), int(target_row["i"])}
    support_nodes: set[int] = set()
    for idx in support_indices:
        row = df.iloc[int(idx)]
        support_nodes.update([int(row["u"]), int(row["i"])])

    target_ts = float(target_row.get("ts", np.nan))
    if time_window is not None and not np.isnan(target_ts):
        mask = (df["ts"] >= target_ts - time_window) & (df["ts"] <= target_ts + time_window)
        candidate_indices = df.index[mask].tolist()
    else:
        event_window = max(0, int(event_window))
        start = max(0, event_idx - event_window)
        end = min(len(df) - 1, event_idx + event_window)
        candidate_indices = list(range(start, end + 1))

    candidate_indices = [
        idx
        for idx in candidate_indices
        if idx not in support_indices and idx != event_idx
    ]

    gt_nodes = target_nodes.union(support_nodes)
    is_stick = "stick" in str(meta.get("recipe") or meta.get("dataset_name") or "").lower()
    edge_features = data.get("edge_features") if isinstance(data, dict) else None
    edge_arr = None
    feat_map = None
    if edge_features is None and is_stick:
        dataset_name = meta.get("dataset_name") or meta.get("recipe")
        if dataset_name:
            try:
                bundle = load_dataset_bundle(dataset_name, verbose=False)
                edge_features = bundle.get("edge_features")
            except Exception:
                edge_features = None
    if edge_features is not None:
        edge_arr = np.asarray(edge_features)
        if edge_arr.ndim == 2 and edge_arr.shape[1] >= 4:
            idx_col = "e_idx" if "e_idx" in df.columns else ("idx" if "idx" in df.columns else None)
            if idx_col is not None:
                idx_vals = df[idx_col].astype(int).to_numpy()
                if idx_vals.size > 0 and edge_arr.shape[0] > int(np.max(idx_vals)):
                    feat_map = edge_arr[idx_vals]
            if feat_map is None and edge_arr.shape[0] == len(df):
                feat_map = edge_arr

    context_candidates: List[Tuple[int, float]] = []
    for idx in candidate_indices:
        row = df.iloc[int(idx)]
        u = int(row["u"])
        v = int(row["i"])
        if gt_nodes and not (u in gt_nodes or v in gt_nodes):
            continue
        ts = float(row.get("ts", np.nan))
        dt = abs(ts - target_ts) if not np.isnan(ts) and not np.isnan(target_ts) else float("inf")
        context_candidates.append((int(idx), dt))

    if not context_candidates and candidate_indices:
        for idx in candidate_indices:
            row = df.iloc[int(idx)]
            ts = float(row.get("ts", np.nan))
            dt = abs(ts - target_ts) if not np.isnan(ts) and not np.isnan(target_ts) else float("inf")
            context_candidates.append((int(idx), dt))

    context_candidates.sort(key=lambda item: (item[1], item[0]))
    context_indices = [idx for idx, _ in context_candidates[: max(0, int(max_context_edges))]]
    if is_stick and feat_map is not None and feat_map.ndim == 2 and feat_map.shape[1] >= 9:
        try:
            frame_norm = float(feat_map[event_idx][8])
        except (TypeError, ValueError, IndexError):
            frame_norm = np.nan
        if not np.isnan(frame_norm):
            joint_count = 14
            u_vals = df["u"].astype(int).to_numpy()
            v_vals = df["i"].astype(int).to_numpy()
            node_min = int(
                min(
                    int(u_vals.min()) if u_vals.size else 0,
                    int(v_vals.min()) if v_vals.size else 0,
                )
            )
            node_base = 1 if node_min >= 1 else 0
            person_ids = {(int(n) - node_base) // joint_count for n in gt_nodes}
            if person_ids:
                u_person = (u_vals - node_base) // joint_count
                v_person = (v_vals - node_base) // joint_count
                same_person = u_person == v_person
                in_person = np.isin(u_person, list(person_ids))
                same_frame = np.isclose(feat_map[:, 8], frame_norm, atol=1e-6)
                bone_mask = feat_map[:, 7] < 0.5
                extra_mask = same_frame & same_person & in_person & bone_mask
                extra_context_indices = [int(i) for i in np.where(extra_mask)[0]]
                if extra_context_indices:
                    context_indices = sorted({*context_indices, *extra_context_indices})

    edges_by_idx: Dict[int, Dict[str, Any]] = {}

    def _append_edge(idx: int, kind: str) -> None:
        row = df.iloc[int(idx)]
        edges_by_idx[int(idx)] = {
            "u": int(row["u"]),
            "v": int(row["i"]),
            "idx": int(idx),
            "ts": float(row.get("ts", np.nan)),
            "kind": kind,
        }

    def _ensure_edge(idx: int, kind: str) -> None:
        if int(idx) not in edges_by_idx:
            _append_edge(idx, kind)
            return
        existing = edges_by_idx[int(idx)]["kind"]
        if existing == "context" and kind == "explanation":
            edges_by_idx[int(idx)]["kind"] = "explanation"

    def _add_edges(indices: Sequence[int], kind: str) -> None:
        for idx in indices:
            _ensure_edge(int(idx), kind)

    _append_edge(event_idx, "target")
    _add_edges(support_indices, "support")
    _add_edges(context_indices, "context")

    explanation_overlay: List[int] = []
    if explanation_indices:
        cleaned = [
            int(idx)
            for idx in explanation_indices
            if isinstance(idx, (int, np.integer)) and 0 <= int(idx) < len(df)
        ]
        explanation_overlay = cleaned
        extra = [idx for idx in cleaned if idx not in gt_indices]
        _add_edges(extra, "explanation")

    edges_info = list(edges_by_idx.values())

    G = nx.Graph()
    for info in edges_info:
        if info["u"] == info["v"]:
            continue
        G.add_edge(int(info["u"]), int(info["v"]))

    if G.number_of_edges() == 0:
        raise ValueError("No edges available to plot for the selected ground-truth window.")

    pos = None
    fixed_nodes: Optional[Iterable[int]] = None
    pos_from_features = False

    if is_stick and feat_map is not None and feat_map.ndim == 2 and feat_map.shape[1] >= 4:
        pos_samples: Dict[int, List[Tuple[float, float]]] = {}
        for info in edges_info:
            idx = int(info["idx"])
            if idx < 0 or idx >= len(df):
                continue
            row = df.iloc[idx]
            u = int(row["u"])
            v = int(row["i"])
            coords = feat_map[idx]
            pos_samples.setdefault(u, []).append((float(coords[0]), float(coords[1])))
            pos_samples.setdefault(v, []).append((float(coords[2]), float(coords[3])))

        if pos_samples:
            pos = {
                node: (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])))
                for node, pts in pos_samples.items()
            }
            pos_from_features = True
            nodes_in_pos = list(pos.keys())
            if nodes_in_pos:
                node_offset = min(nodes_in_pos) if min(nodes_in_pos) > 0 else 0
                joint_count = 14
                head_nodes = [
                    n for n in nodes_in_pos if (int(n) - node_offset) % joint_count == 1
                ]
                torso_nodes = [
                    n for n in nodes_in_pos if (int(n) - node_offset) % joint_count == 0
                ]

                coords = np.array(list(pos.values()), dtype=float)
                center = coords.mean(axis=0)
                if head_nodes and torso_nodes:
                    head_xy = np.mean([pos[n] for n in head_nodes], axis=0)
                    torso_xy = np.mean([pos[n] for n in torso_nodes], axis=0)
                    vec = np.array(head_xy) - np.array(torso_xy)
                else:
                    top = coords[np.argmax(coords[:, 1])]
                    bottom = coords[np.argmin(coords[:, 1])]
                    vec = top - bottom

                if np.linalg.norm(vec) > 1e-6:
                    angle = float(np.arctan2(vec[1], vec[0]))
                    rot = (np.pi / 2.0) - angle
                    c, s = float(np.cos(rot)), float(np.sin(rot))
                    pos_rot = {}
                    for node, pt in pos.items():
                        rel = np.array(pt) - center
                        x_new = rel[0] * c - rel[1] * s
                        y_new = rel[0] * s + rel[1] * c
                        pos_rot[node] = (float(x_new), float(y_new))
                    if head_nodes and torso_nodes:
                        head_rot = np.mean([pos_rot[n] for n in head_nodes], axis=0)
                        torso_rot = np.mean([pos_rot[n] for n in torso_nodes], axis=0)
                        if head_rot[1] < torso_rot[1]:
                            pos_rot = {n: (x, -y) for n, (x, y) in pos_rot.items()}
                    pos = pos_rot
                    fixed_nodes = list(pos.keys())
                    pos_from_features = True
    target_edges = [
        (int(info["u"]), int(info["v"]))
        for info in edges_info
        if info["kind"] == "target"
    ]
    support_edges = [
        (int(info["u"]), int(info["v"]))
        for info in edges_info
        if info["kind"] == "support"
    ]
    if target_edges and support_edges:
        u_t, v_t = target_edges[0]
        target_nodes = {u_t, v_t}
        support_nodes = {n for edge in support_edges for n in edge}
        support_only = [n for n in support_nodes if n not in target_nodes]
        if len(support_only) == 1 and u_t in G.nodes and v_t in G.nodes and support_only[0] in G.nodes:
            w = support_only[0]
            pos_init = {
                u_t: (-0.8, -0.4),
                v_t: (0.8, -0.4),
                w: (0.0, 0.9),
            }
            try:
                pos = nx.spring_layout(G, seed=42, pos=pos_init, fixed=pos_init.keys())
                fixed_nodes = list(pos_init.keys())
            except TypeError:
                pos = None

    def _min_distance(layout: Dict[int, Tuple[float, float]]) -> float:
        nodes = list(layout.keys())
        if len(nodes) < 2:
            return float("inf")
        min_dist = float("inf")
        for i, u in enumerate(nodes[:-1]):
            x0, y0 = layout[u]
            for v in nodes[i + 1 :]:
                x1, y1 = layout[v]
                dist = float(np.hypot(x1 - x0, y1 - y0))
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def _layout_with_spread(
        graph: "nx.Graph",
        *,
        base_pos: Optional[Dict[int, Tuple[float, float]]] = None,
        fixed: Optional[Iterable[int]] = None,
    ) -> Dict[int, Tuple[float, float]]:
        n_nodes = max(graph.number_of_nodes(), 1)
        base_k = 1.5 / np.sqrt(n_nodes)
        best_pos: Optional[Dict[int, Tuple[float, float]]] = None
        best_min = -1.0
        for attempt in range(4):
            k = base_k * (1.4 + 0.5 * attempt)
            kwargs = dict(seed=42, k=k, iterations=200, scale=1.6)
            if base_pos is not None:
                kwargs["pos"] = base_pos
                if fixed:
                    kwargs["fixed"] = list(fixed)
            layout = nx.spring_layout(graph, **kwargs)
            min_dist = _min_distance(layout)
            if min_dist > best_min:
                best_pos = layout
                best_min = min_dist
            if min_dist >= 0.22:
                return layout
        if best_pos is None:
            best_pos = nx.spring_layout(graph, seed=42, iterations=200, scale=1.6)
        rng = np.random.default_rng(42)
        for node, (x, y) in best_pos.items():
            dx, dy = rng.normal(scale=0.05, size=2)
            best_pos[node] = (float(x + dx), float(y + dy))
        return best_pos

    if pos is None:
        pos = _layout_with_spread(G)
    elif not pos_from_features:
        if fixed_nodes:
            pos = _layout_with_spread(G, base_pos=pos, fixed=fixed_nodes)
    node_sizes = {node: 10.0 + 6.0 * np.log1p(G.degree(node)) for node in G.nodes}

    fig = go.Figure()
    legend_used: set[str] = set()
    for info in edges_info:
        u = int(info["u"])
        v = int(info["v"])
        if u == v:
            continue
        kind = str(info["kind"])
        if kind == "target":
            color = COLORS["accent"]
            width = 3.4
            label = "ground-truth target"
        elif kind == "support":
            color = COLORS["accent2"]
            width = 2.6
            label = "ground-truth support"
        elif kind == "explanation":
            color = COLORS["user"]
            width = 2.4
            label = "explanation (non-GT)"
        else:
            color = _rgba(COLORS["base"], 0.4)
            width = 1.6
            label = "context edge"

        show_legend = label not in legend_used
        legend_used.add(label)
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=color, width=width),
                name=label,
                showlegend=show_legend,
                hovertemplate=(
                    f"{u}–{v}"
                    f"<br>event={info['idx']}"
                    + (f"<br>ts={info['ts']:.2f}" if not np.isnan(info["ts"]) else "")
                    + f"<extra>{kind}</extra>"
                ),
            )
        )

    if explanation_overlay:
        label = "explainer edge"
        show_legend = label not in legend_used
        legend_used.add(label)
        for idx in explanation_overlay:
            row = df.iloc[int(idx)]
            u = int(row["u"])
            v = int(row["i"])
            if u == v or u not in pos or v not in pos:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(color=COLORS["user"], width=2.8, dash="dot"),
                    name=label,
                    showlegend=show_legend,
                    hovertemplate=(
                        f"{u}–{v}"
                        f"<br>event={idx}"
                        + (f"<br>ts={float(row.get('ts', np.nan)):.2f}" if not np.isnan(float(row.get('ts', np.nan))) else "")
                        + "<extra>explainer</extra>"
                    ),
                )
            )
            show_legend = False

    target_nodes_final = sorted({n for n in target_nodes if n in G.nodes})
    support_nodes_final = sorted({n for n in support_nodes if n in G.nodes and n not in target_nodes})
    context_nodes = sorted([n for n in G.nodes if n not in target_nodes and n not in support_nodes])

    def _node_trace(nodes: Sequence[int], *, color: str, name: str) -> Optional["go.Scatter"]:
        if not nodes:
            return None
        return go.Scatter(
            x=[pos[int(n)][0] for n in nodes],
            y=[pos[int(n)][1] for n in nodes],
            mode="markers+text",
            marker=dict(
                size=[node_sizes[int(n)] for n in nodes],
                color=color,
                line=dict(color=COLORS["text"], width=1.0),
            ),
            text=[str(int(n)) for n in nodes],
            textposition="top center",
            hovertemplate="node=%{text}<extra></extra>",
            name=name,
        )

    context_trace = _node_trace(context_nodes, color=COLORS["base"], name="context nodes")
    if context_trace is not None:
        fig.add_trace(context_trace)
    support_trace = _node_trace(support_nodes_final, color=COLORS["accent2"], name="support nodes")
    if support_trace is not None:
        fig.add_trace(support_trace)
    target_trace = _node_trace(target_nodes_final, color=COLORS["item"], name="target nodes")
    if target_trace is not None:
        fig.add_trace(target_trace)

    title = meta.get("dataset_name") or meta.get("recipe") or "Ground truth"
    xaxis_cfg = dict(visible=False)
    yaxis_cfg = dict(visible=False)
    if pos_from_features:
        yaxis_cfg["scaleanchor"] = "x"
        yaxis_cfg["scaleratio"] = 1

    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        title=dict(
            text=f"{title} - ground-truth neighborhood (event {event_idx})",
            x=0.0,
            xanchor="left",
            y=0.98,
            yanchor="top",
        ),
        xaxis=xaxis_cfg,
        yaxis=yaxis_cfg,
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0.0),
        margin=dict(l=20, r=20, t=90, b=70),
    )

    _auto_show(fig, show)
    _maybe_save(fig, save_to)
    return fig


def plot_nicolaus_motif(
    data: Union[pd.DataFrame, Dict[str, Any]],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    show: bool = True,
    save_to: Optional[Union[str, Path]] = None,
) -> "go.Figure":
    """Visualize Haus-des-Nicolaus motif edges with their positive-event intensity."""
    _require_plotly()
    if isinstance(data, dict):
        df = _ensure_dataframe(data.get("interactions"))
        meta = dict(metadata or data.get("metadata") or {})
    else:
        df = _ensure_dataframe(data)
        meta = dict(metadata or {})

    if not meta:
        raise ValueError("plot_nicolaus_motif requires metadata containing motif layout.")

    ground = meta.get("ground_truth") or {}
    positions_raw = ground.get("positions") or {}
    motif_edges_raw = ground.get("motif_edges_undirected") or ground.get("motif_edges")
    motif_nodes_raw = ground.get("motif_nodes") or []
    if not positions_raw or not motif_edges_raw:
        raise ValueError("Metadata missing Nicolaus motif positions or edges.")

    def _normalize_positions(raw: Dict[Any, Any]) -> Dict[int, Tuple[float, float]]:
        norm: Dict[int, Tuple[float, float]] = {}
        for key, value in raw.items():
            try:
                node = int(key)
                x, y = value
                norm[node] = (float(x), float(y))
            except (ValueError, TypeError):
                continue
        return norm

    def _normalize_edges(raw_edges: Iterable[Iterable[Any]]) -> List[Tuple[int, int]]:
        edges: List[Tuple[int, int]] = []
        for edge in raw_edges:
            try:
                u, v = edge
                edges.append(tuple(sorted((int(u), int(v)))))
            except (ValueError, TypeError):
                continue
        return edges

    def _count_edges(frame: pd.DataFrame) -> Counter:
        if frame is None or len(frame) == 0:
            return Counter()
        if not {"u", "i"}.issubset(frame.columns):
            raise ValueError("Nicolaus motif visualisation requires 'u' and 'i' columns.")
        arr = frame[["u", "i"]].to_numpy(dtype=int, copy=True)
        arr.sort(axis=1)
        return Counter(map(tuple, arr))

    positions = _normalize_positions(positions_raw)
    motif_edges = _normalize_edges(motif_edges_raw)
    if not positions or not motif_edges:
        raise ValueError("Could not derive motif structure from metadata.")

    motif_nodes = sorted({int(node) for node in (motif_nodes_raw or positions.keys()) if node in positions})

    total_counts = _count_edges(df)
    if "label" in df.columns:
        pos_counts = _count_edges(df[df["label"] > 0])
    else:
        pos_counts = total_counts
    max_pos = max((pos_counts[edge] for edge in motif_edges), default=0)
    max_total = max((total_counts[edge] for edge in motif_edges), default=0)

    fig = go.Figure()
    for edge in motif_edges:
        u, v = edge
        if u not in positions or v not in positions:
            continue
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        pos_count = pos_counts[edge]
        tot_count = total_counts[edge]
        intensity = (pos_count / max_pos) if max_pos > 0 else 0.0
        base_intensity = (tot_count / max_total) if max_total > 0 else 0.0
        if pos_count > 0:
            color = _rgba(COLORS["accent"], 0.35 + 0.6 * intensity)
            width = 3.0 + 4.0 * intensity
        else:
            color = _rgba(COLORS["user"], 0.25 + 0.5 * base_intensity)
            width = 2.0 + 3.0 * base_intensity
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=color, width=width),
                hovertemplate=(
                    f"{u}–{v}<br>positive events: {pos_count}"
                    f"<br>total events: {tot_count}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    if motif_nodes:
        node_x = [positions[n][0] for n in motif_nodes if n in positions]
        node_y = [positions[n][1] for n in motif_nodes if n in positions]
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(
                size=18,
                color=COLORS["item"],
                line=dict(color=COLORS["text"], width=1.2),
            ),
            text=[str(n) for n in motif_nodes if n in positions],
            textposition="top center",
            hovertemplate="node=%{text}<extra></extra>",
            showlegend=False,
        )
        fig.add_trace(node_trace)

    title = meta.get("dataset_name") or meta.get("recipe") or "Nicolaus motif"
    motif_window = ground.get("motif_window")
    subtitle = None
    if isinstance(motif_window, (list, tuple)) and len(motif_window) == 2:
        try:
            t0 = float(motif_window[0])
            t1 = float(motif_window[1])
            subtitle = f"Motif window: [{t0:.2f}, {t1:.2f}]"
        except (TypeError, ValueError):
            subtitle = None

    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        title=title + " — Haus des Nicolaus motif",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        showlegend=False,
        margin=dict(l=20, r=20, t=70, b=30),
        annotations=(
            [
                dict(
                    text=subtitle,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.05,
                    showarrow=False,
                    font=dict(size=PLOT_STYLE["annotation_size"], color=COLORS["text"]),
                )
            ]
            if subtitle
            else []
        ),
    )

    _auto_show(fig, show)
    _maybe_save(fig, save_to)
    return fig


def plot_explain_timeline(
    data: Union[pd.DataFrame, Dict[str, Any], str, Path],
    event_indices: Sequence[int],
    *,
    window: int = 0,
    max_base_points: int = 50_000,
    show: bool = True,
    save_to: Optional[Union[str, Path]] = None,
) -> "go.Figure":
    _require_plotly()
    df = _ensure_dataframe(data)
    if len(df) == 0:
        raise ValueError("No interactions to visualise.")
    positions = _resolve_event_positions(df, event_indices)
    if not positions:
        raise ValueError("None of the requested explain indices were found in the dataframe.")

    total_events = len(df)
    order = np.arange(total_events)
    if total_events > max_base_points:
        sample_idx = np.linspace(0, total_events - 1, max_base_points, dtype=int)
        base_ts = df["ts"].to_numpy()[sample_idx]
        base_order = order[sample_idx]
    else:
        base_ts = df["ts"].to_numpy()
        base_order = order

    base_trace = go.Scatter(
        x=base_ts,
        y=base_order,
        mode="markers",
        marker=dict(size=5, color=_rgba(COLORS["base"], 0.35)),
        name="events",
        hoverinfo="skip",
    )

    focal = df.iloc[positions]
    focal_trace = go.Scatter(
        x=focal["ts"],
        y=positions,
        mode="markers+text",
        marker=dict(size=12, color=COLORS["accent"]),
        text=[f"idx={p}" for p in positions],
        textposition="top center",
        name="explain idx",
        hovertemplate="ts=%{x}<br>row=%{y}<extra></extra>",
    )

    shapes = []
    if window > 0:
        for pos in positions:
            start = max(0, pos - window)
            end = min(len(df) - 1, pos + window)
            shapes.append(
                dict(
                    type="rect",
                    xref="paper",
                    yref="y",
                    x0=0,
                    x1=1,
                    y0=start,
                    y1=end,
                    fillcolor=_rgba(COLORS["accent"], 0.08),
                    line=dict(width=0),
                    layer="below",
                )
            )

    fig = go.Figure(data=[base_trace, focal_trace])
    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        title="Explain instance timeline",
        xaxis_title="timestamp",
        yaxis_title="event order",
        shapes=shapes,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    _auto_show(fig, show)
    _maybe_save(fig, save_to)
    return fig


def summarize_explain_instances(
    data: Union[pd.DataFrame, Dict[str, Any], str, Path],
    event_indices: Sequence[int],
    *,
    events_before: int = 5,
    events_after: int = 5,
    time_window: Optional[float] = None,
) -> pd.DataFrame:
    df = _ensure_dataframe(data)
    positions = _resolve_event_positions(df, event_indices)
    rows: List[Dict[str, Any]] = []

    if not positions:
        return pd.DataFrame(
            columns=[
                "query_idx",
                "row",
                "ts",
                "label",
                "user",
                "item",
                "events_before",
                "events_after",
                "same_user_context",
                "same_item_context",
                "time_window_count",
            ]
        )

    timestamps = df["ts"].to_numpy()
    for raw_idx, pos in zip(event_indices, positions):
        row = df.iloc[pos]
        start = max(0, pos - events_before)
        end = min(len(df), pos + events_after + 1)
        local = df.iloc[start:end]
        same_user = local[local["u"] == row["u"]]
        same_item = local[local["i"] == row["i"]]

        if time_window is not None:
            t0 = float(row["ts"]) - time_window
            t1 = float(row["ts"]) + time_window
            mask = (timestamps >= t0) & (timestamps <= t1)
            time_count = int(np.count_nonzero(mask)) - 1
        else:
            time_count = np.nan

        rows.append(
            {
                "query_idx": int(raw_idx),
                "row": int(pos),
                "ts": float(row["ts"]),
                "label": row.get("label", np.nan),
                "user": int(row["u"]),
                "item": int(row["i"]),
                "events_before": int(pos - start),
                "events_after": int((end - 1) - pos),
                "same_user_context": int(len(same_user) - 1),
                "same_item_context": int(len(same_item) - 1),
                "time_window_count": time_count,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "plot_event_count_over_time",
    "plot_inter_event_time_hist",
    "plot_degree_histograms",
    "plot_dataset_quadrants",
    "plot_adjacency_heatmap",
    "plot_label_balance",
    "happen_rate",
    "plot_happen_rate_matrix",
    "build_bipartite_graph",
    "plot_bipartite_graph",
    "plot_force_directed_graph",
    "plot_triadic_closure_subgraph",
    "plot_ground_truth_subgraph",
    "plot_nicolaus_motif",
    "animate_bipartite_graph",
    "animate_stick_figure",
    "plot_explain_timeline",
    "summarize_explain_instances",
]
