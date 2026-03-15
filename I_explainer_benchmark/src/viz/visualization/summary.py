from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .utils import (
    COLORS,
    _PLOTLY_TEMPLATE,
    _auto_show,
    _ensure_dataframe,
    _histogram_to_bar,
    _maybe_save,
    _resolve_event_positions,
    _require_plotly,
    _rgba,
    go,
    make_subplots,
)

def plot_dataset_quadrants(
    data: Union[pd.DataFrame, Dict[str, Any], str, Path],
    *,
    explain_indices: Sequence[int] = (),
    happen_interval: float = 0.5,
    show: bool = True,
    save_to: Optional[Union[str, Path]] = None,
) -> "go.Figure":
    _require_plotly()
    df = _ensure_dataframe(data)
    if len(df) == 0:
        raise ValueError("No interactions available for plotting.")

    ts = df["ts"].to_numpy(dtype=float)
    ts_min = float(ts.min())
    ts_max = float(ts.max())
    bins = max(1, int(np.ceil((ts_max - ts_min) / happen_interval)))
    counts, edges = np.histogram(ts, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    deltas = np.diff(np.sort(ts))
    delta_bins = min(50, max(5, int(np.sqrt(max(1, len(deltas))))))
    delta_centers, delta_counts = _histogram_to_bar(deltas if len(deltas) else np.array([0.0]), bins=delta_bins)

    u_deg = df["u"].value_counts().to_numpy(dtype=float)
    i_deg = df["i"].value_counts().to_numpy(dtype=float)

    if "label" in df.columns:
        labels = df["label"].to_numpy(dtype=float)
        pos = float(np.sum(labels > 0))
        neg = float(np.sum(labels <= 0))
        label_names = ["pos", "neg"]
        label_vals = [pos, neg]
    else:
        label_names = ["events"]
        label_vals = [float(len(df))]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Events over time", "Inter-event Δt", "Degree histogram", "Label balance"),
    )
    fig.add_trace(
        go.Bar(x=centers, y=counts, marker_color=COLORS["accent2"], name="events"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=delta_centers, y=delta_counts, marker_color=COLORS["user"], name="Δt"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Histogram(x=u_deg, nbinsx=30, marker_color=COLORS["user"], name="out-degree"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=i_deg, nbinsx=30, marker_color=COLORS["item"], name="in-degree"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=label_names, y=label_vals, marker_color=COLORS["accent"]),
        row=2,
        col=2,
    )

    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        title="Dataset diagnostics",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
    )
    _auto_show(fig, show)
    _maybe_save(fig, save_to)
    return fig


def plot_explain_timeline(
    data: Union[pd.DataFrame, Dict[str, Any], str, Path],
    *,
    event_indices: Sequence[int] = (),
    window: int = 15,
    max_base_points: int = 20_000,
    show: bool = True,
    save_to: Optional[Union[str, Path]] = None,
) -> "go.Figure":
    _require_plotly()
    df = _ensure_dataframe(data)
    if len(df) == 0:
        raise ValueError("No interactions available for plotting.")

    positions = _resolve_event_positions(df, event_indices)
    if positions:
        start = max(0, min(positions) - window)
        end = min(len(df), max(positions) + window + 1)
    else:
        start = 0
        end = len(df)

    view = df.iloc[start:end]
    ts = view["ts"].to_numpy(dtype=float)
    idx = np.arange(start, end, dtype=int)

    if len(ts) > max_base_points:
        sample = np.linspace(0, len(ts) - 1, max_base_points, dtype=int)
        ts = ts[sample]
        idx = idx[sample]

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=ts,
            y=idx,
            mode="markers",
            marker=dict(size=5, color=_rgba(COLORS["base"], 0.45)),
            name="events",
        )
    )

    highlight = [pos for pos in positions if start <= pos < end]
    if highlight:
        h_ts = df.iloc[highlight]["ts"].to_numpy(dtype=float)
        fig.add_trace(
            go.Scattergl(
                x=h_ts,
                y=highlight,
                mode="markers",
                marker=dict(size=9, color=COLORS["accent"]),
                name="highlight",
            )
        )

    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        title="Explain timeline",
        xaxis_title="time",
        yaxis_title="event index",
        showlegend=True,
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
    "plot_dataset_quadrants",
    "plot_explain_timeline",
    "summarize_explain_instances",
]
