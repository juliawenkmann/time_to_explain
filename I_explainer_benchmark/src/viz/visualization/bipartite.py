from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .utils import (
    COLORS,
    _PLOTLY_TEMPLATE,
    _auto_show,
    _ensure_dataframe,
    _map_ratio_to_color,
    _maybe_save,
    _rgba,
    _require_networkx,
    _require_plotly,
    load_dataset_bundle,
    go,
    nx,
)

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


__all__ = [
    "animate_bipartite_graph",
    "animate_stick_figure",
    "build_bipartite_graph",
    "plot_bipartite_graph",
]
