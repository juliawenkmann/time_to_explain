from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .utils import (
    COLORS,
    PLOT_STYLE,
    _PLOTLY_TEMPLATE,
    _auto_show,
    _ensure_dataframe,
    _map_ratio_to_color,
    _maybe_save,
    _require_networkx,
    _require_plotly,
    _rgba,
    load_dataset_bundle,
    go,
    nx,
)

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

__all__ = [
    "plot_force_directed_graph",
    "plot_ground_truth_subgraph",
    "plot_nicolaus_motif",
    "plot_triadic_closure_subgraph",
]
