from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import warnings
import nx 
import numpy as np
import pandas as pd
import numbers

# Plotly for interactive charts
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:  # pragma: no cover
    go = None
    make_subplots = None

# Matplotlib only used for nicer color gradients; fall back gracefully if missing
try:  # pragma: no cover - optional dependency
    import matplotlib.colors as mpl_colors
    from matplotlib import cm as mpl_cm
    _HAS_MPL = True
except ImportError:  # pragma: no cover - optional dependency
    mpl_colors = None
    mpl_cm = None
    _HAS_MPL = False

try:
    import networkx as nx
    try:
        from networkx.algorithms import bipartite as nx_bipartite
    except ImportError:  # pragma: no cover
        nx_bipartite = None
except ImportError:  # pragma: no cover
    nx = None
    nx_bipartite = None

# Your loader
from ..data.io import load_processed_dataset


# --------- Styling & small utilities ---------

_PLOTLY_TEMPLATE = "plotly_dark"
# Palette chosen to be colorblind-friendly & modern
COLORS = {
    "user": "#4C78A8",     # blue
    "item": "#F58518",     # orange
    "base": "#9e9e9e",     # grey
    "accent": "#C44E52",   # red
    "accent2": "#54A24B",  # green
}

def _hex_to_rgb(color: str) -> np.ndarray:
    color = color.lstrip("#")
    if len(color) == 3:
        color = "".join(ch * 2 for ch in color)
    if len(color) != 6:
        raise ValueError(f"Expected #RRGGBB hex colour, got '{color}'")
    return np.array([int(color[i:i+2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def _rgb_to_hex(rgb: np.ndarray) -> str:
    rgb = np.clip(np.asarray(rgb), 0.0, 1.0)
    r, g, b = (int(round(x * 255)) for x in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _blend_hex(color_a: str, color_b: str, weight: float) -> str:
    weight = float(np.clip(weight, 0.0, 1.0))
    ra = _hex_to_rgb(color_a)
    rb = _hex_to_rgb(color_b)
    blended = ra * (1.0 - weight) + rb * weight
    return _rgb_to_hex(blended)


def _map_ratio_to_color(ratio: float, edge_cmap: str = "RdYlGn") -> str:
    ratio = float(np.clip(ratio, 0.0, 1.0))
    if _HAS_MPL:
        cmap = mpl_cm.get_cmap(edge_cmap)
        return mpl_colors.to_hex(cmap(ratio))
    return _blend_hex(COLORS["accent"], COLORS["accent2"], ratio)

def _require_plotly() -> None:
    if go is None:
        raise RuntimeError(
            "This module now uses Plotly for interactive visuals. "
            "Install it via: pip install plotly kaleido"
        )

def _ensure_dataframe(data: Union[pd.DataFrame, Dict[str, Any], str, Path]) -> pd.DataFrame:
    """Return a shallow copy of the interactions dataframe from various inputs."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, dict):
        if "interactions" not in data:
            raise KeyError("Bundle dict must contain an 'interactions' dataframe")
        return data["interactions"].copy()
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.exists():
            if path.is_dir():
                candidates = sorted(path.glob("ml_*.csv"))
                if not candidates:
                    raise FileNotFoundError(f"No ml_*.csv found in {path}")
                return pd.read_csv(candidates[0])
            if path.suffix == ".csv":
                return pd.read_csv(path)
        bundle = load_processed_dataset(str(data))
        return bundle["interactions"].copy()
    raise TypeError(f"Unsupported data type: {type(data)!r}")

def _require_networkx() -> None:
    if nx is None:
        raise RuntimeError("networkx is required for graph visualisations. Install via 'pip install networkx'.")

def _maybe_save(fig: "go.Figure", save_to: Optional[Union[str, Path]]) -> Optional[str]:
    """Save plotly figure to HTML (preferred). Falls back gracefully if kaleido not available for images."""
    if not save_to:
        return None
    save_to = str(save_to)
    if save_to.lower().endswith(".html"):
        try:
            fig.write_html(save_to, include_plotlyjs="cdn", auto_open=False)
        except (ValueError, OverflowError) as exc:
            msg = str(exc).lower()
            if "string length" in msg or "memory" in msg:
                warnings.warn(f"Skipping HTML save for '{save_to}': {exc}", RuntimeWarning)
                return None
            raise
    else:
        # attempt static export if kaleido is installed
        try:
            fig.write_image(save_to, scale=2)
        except Exception as e:  # kaleido not installed or other issue
            warnings.warn(
                f"Static export failed ({e}). Saving HTML instead at '{save_to}.html'. "
                "Install kaleido for static image export: pip install -U kaleido",
                RuntimeWarning,
            )
            fallback = save_to + ".html"
            try:
                fig.write_html(fallback, include_plotlyjs="cdn", auto_open=False)
                save_to = fallback
            except (ValueError, OverflowError) as exc:
                msg = str(exc).lower()
                if "string length" in msg or "memory" in msg:
                    warnings.warn(f"Skipping HTML save for '{fallback}': {exc}", RuntimeWarning)
                    return None
                raise
    return save_to

def _auto_show(fig: "go.Figure", show: bool) -> None:
    if show:
        fig.show()


# --------- Layout & index helpers ---------

# Utility: load multiple metrics tables and plot fidelity vs. sparsity
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def read_tabs_plot(files, name, plot_only_og=False, *,
                   metric='fidelity_best.best',
                   sparsity_col='sparsity.edges.zero_frac',
                   labels=None, markers=None, palette='deep',
                   og_keys=None, save_dir='plots'):
    """Load metric CSVs, aggregate by sparsity, and plot fidelity curves."""
    files = {k: os.fspath(v) for k, v in files.items()}
    sns.set_theme(style='whitegrid')

    tabs = {}
    best_fids = {}
    aufsc = {}

    for key, path in files.items():
        df = pd.read_csv(path)
        if sparsity_col in df:
            sparsity = df[sparsity_col]
        elif 'sparsity' in df:
            sparsity = df['sparsity']
        else:
            raise KeyError(f"{sparsity_col!r} column not found in {path}")

        if metric not in df:
            raise KeyError(f"Metric column {metric!r} not found in {path}")

        tab = (df
               .assign(sparsity=sparsity)
               .groupby('sparsity', as_index=False)[metric]
               .mean()
               .sort_values('sparsity'))
        tabs[key] = tab

        best_fids[key] = tab[metric].max()
        aufsc[key] = float(np.trapz(tab[metric], tab['sparsity']))

    print('Best Fidelity (max across levels):', best_fids)
    print('Area under fidelity-sparsity curve:', aufsc)

    og_defaults = set(og_keys or ['xtg-og', 'attn', 'pbone', 'pg'])
    keys_to_plot = [k for k in tabs if (not plot_only_og or k in og_defaults)]
    if not keys_to_plot:
        keys_to_plot = list(tabs.keys())

    os.makedirs(save_dir, exist_ok=True)
    palette_colors = sns.color_palette(palette, len(keys_to_plot))
    default_markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v']

    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, key in enumerate(keys_to_plot):
        tab = tabs[key]
        label = (labels or {}).get(key, key)
        marker = (markers or {}).get(key, default_markers[idx % len(default_markers)])
        sns.lineplot(data=tab, x='sparsity', y=metric, ax=ax,
                     label=label, color=palette_colors[idx], marker=marker)

    ax.set_xlabel('Sparsity (fraction of zero edges)')
    ax.set_ylabel(metric)
    ax.set_title(f'Fidelity vs sparsity — {name}')
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(save_dir, f'{name}.png')
    fig.savefig(out_path, dpi=200)
    plt.show()

    return tabs, best_fids, aufsc


@dataclass(frozen=True)
class MetricCurveSpec:
    prefix: str
    title: str
    color: str = "tab:blue"
    ylabel: str = "|Δ score|"
    axis_label_percent: Tuple[str, str] = ("", "")
    y_min: Optional[float] = 0.0
    annotation_column: Optional[str] = None
    annotation_label: str = "Mean = {value:.3f}"
    annotation_position: Tuple[float, float] = (0.95, 0.05)
    alpha: float = 0.25
    figsize: Tuple[float, float] = (8, 4)


def _parse_suffix(token: str) -> float | str:
    if isinstance(token, str) and token.startswith("s="):
        try:
            return float(token.split("=", 1)[1])
        except ValueError:
            return token
    try:
        return float(token)
    except (TypeError, ValueError):
        return token


def collect_curve_columns(metrics_df: pd.DataFrame, prefix: str) -> Tuple[List[str], List[Any]]:
    cols = [
        c
        for c in metrics_df.columns
        if c.startswith(prefix) and "@" in c and not c.endswith(".k")
    ]
    cols_sorted = sorted(cols, key=lambda c: _parse_suffix(c.split("@", 1)[1]))
    levels = [_parse_suffix(c.split("@", 1)[1]) for c in cols_sorted]
    return cols_sorted, levels


def levels_to_axis(levels: Sequence[Any]) -> Tuple[List[Any], bool]:
    axis_vals: List[Any] = []
    fractional = False
    for lvl in levels:
        if isinstance(lvl, numbers.Real):
            lvl_float = float(lvl)
            if np.isnan(lvl_float) or np.isinf(lvl_float):
                axis_vals.append(lvl_float)
                continue
            if 0.0 <= lvl_float <= 1.0:
                axis_vals.append(lvl_float * 100.0)
                fractional = True
            else:
                axis_vals.append(lvl_float)
        else:
            axis_vals.append(lvl)
    return axis_vals, fractional


def plot_metric_curves(metrics_df: pd.DataFrame, specs: Sequence[MetricCurveSpec]) -> Dict[str, Dict[str, Any]]:
    """
    Plot metric curves defined by MetricCurveSpec entries and return
    summary metadata for each prefix.
    """
    results: Dict[str, Dict[str, Any]] = {}
    if metrics_df.empty:
        return results

    for spec in specs:
        cols, levels = collect_curve_columns(metrics_df, spec.prefix)
        if not cols:
            continue

        axis, is_fractional = levels_to_axis(levels)
        fig, ax = plt.subplots(figsize=spec.figsize)
        finite_vals: List[float] = []
        level_values: Dict[str, List[float]] = {col: [] for col in cols}
        for _, row in metrics_df.iterrows():
            vals = [row.get(col, np.nan) for col in cols]
            cleaned_row: List[float] = []
            for col, val in zip(cols, vals):
                try:
                    vf = float(val)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(vf):
                    cleaned_row.append(vf)
                    finite_vals.append(vf)
                    level_values[col].append(vf)
            ax.plot(axis, vals, color=spec.color, alpha=spec.alpha)

        mean_curve: List[float] = []
        for col in cols:
            samples = level_values.get(col, [])
            if samples:
                mean_curve.append(float(np.mean(samples)))
            else:
                mean_curve.append(float("nan"))
        mean_curve_array = np.asarray(mean_curve, dtype=float)
        if np.isfinite(mean_curve_array).any():
            ax.plot(
                axis,
                mean_curve,
                color=spec.color,
                linestyle="--",
                linewidth=2.5,
                label="mean",
            )
            ax.legend()

        xlabel = spec.axis_label_percent[0] if is_fractional else spec.axis_label_percent[1]
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_ylabel(spec.ylabel)
        if spec.y_min is not None:
            ax.set_ylim(bottom=spec.y_min)
        ax.grid(True, alpha=0.2)
        ax.set_title(spec.title)

        if spec.annotation_column and spec.annotation_column in metrics_df.columns:
            col_vals = metrics_df[spec.annotation_column].to_numpy(dtype=float)
            col_vals = col_vals[np.isfinite(col_vals)]
            if col_vals.size:
                ax.text(
                    spec.annotation_position[0],
                    spec.annotation_position[1],
                    spec.annotation_label.format(value=float(col_vals.mean())),
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=10,
                    bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none"},
                )
        plt.show()

        results[spec.prefix] = {
            "columns": cols,
            "levels": levels,
            "values": finite_vals,
            "axis_values": axis,
            "is_fractional": is_fractional,
            "mean_curve": mean_curve,
        }

    return results


def _compute_bipartite_layout(
    G: "nx.Graph",
    top_nodes: Iterable[int],
    *,
    scale: float = 1.8,
    vertical: bool = True,
) -> Dict[int, Tuple[float, float]]:
    """Layout helper that works across networkx versions."""
    _require_networkx()
    top_nodes = list(top_nodes)
    bottom_nodes = [n for n in G.nodes if n not in top_nodes]

    layout_fn = None
    if nx_bipartite is not None:
        layout_module = getattr(nx_bipartite, "layout", None)
        if layout_module is not None:
            layout_fn = getattr(layout_module, "bipartite_layout", None)
    if layout_fn is None:
        layout_fn = getattr(nx, "bipartite_layout", None)

    if layout_fn is not None:
        try:
            return layout_fn(
                G,
                top_nodes=top_nodes,
                align="vertical" if vertical else "horizontal",
                scale=scale,
            )
        except TypeError:
            return layout_fn(G, nodes=top_nodes, align="vertical" if vertical else "horizontal")

    # Fallback: simple two rows
    y_top = scale
    y_bottom = -scale if vertical else scale
    x_spacing = scale / max(1, len(top_nodes))
    x_spacing_bottom = scale / max(1, len(bottom_nodes))

    pos: Dict[int, Tuple[float, float]] = {}
    for idx, node in enumerate(top_nodes):
        x = -scale + idx * x_spacing if vertical else 0.0
        pos[node] = (x, y_top) if vertical else (y_top, x)
    for idx, node in enumerate(bottom_nodes):
        x = -scale + idx * x_spacing_bottom if vertical else 0.0
        pos[node] = (x, y_bottom) if vertical else (y_bottom, x)
    return pos

def _resolve_event_positions(df: pd.DataFrame, event_indices: Sequence[int]) -> List[int]:
    if len(event_indices) == 0:
        return []
    if "idx" in df.columns:
        mapping = {int(idx): pos for pos, idx in enumerate(df["idx"].astype(int))}
    else:
        mapping = {}

    positions: List[int] = []
    n = len(df)
    for raw_idx in event_indices:
        idx = int(raw_idx)
        pos: Optional[int] = mapping.get(idx)
        if pos is None:
            if 0 <= idx < n:
                pos = idx
            elif 1 <= idx <= n:
                pos = idx - 1
        if pos is None or not (0 <= pos < n):
            warnings.warn(f"Event index {raw_idx} not found in dataframe; skipping.", RuntimeWarning)
            continue
        positions.append(pos)
    return positions


# --------- Stats & diagnostic plots (interactive) ---------

def _histogram_to_bar(x: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(x, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def plot_event_count_over_time(
    df: pd.DataFrame,
    *,
    bins: int = 50,
    show: bool = True,
    save_to: Optional[Union[str, Path]] = None,
) -> "go.Figure":
    """Interactive histogram of events over time."""
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
    """Interactive histogram of inter-event intervals."""
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
    """Interactive side-by-side degree histograms."""
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

def plot_adjacency_heatmap(
    df: pd.DataFrame,
    *,
    num_nodes: Optional[int] = None,
    show: bool = True,
    save_to: Optional[Union[str, Path]] = None,
) -> "go.Figure":
    """Interactive heatmap of interaction counts (u -> i)."""
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
            colorscale="Blues",
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
    """Interactive bar of label distribution."""
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


# --------- Happen rate ---------

def happen_rate(source_ts: np.ndarray, target_ts: np.ndarray, interval: float, reverse: bool = False) -> float:
    if reverse:
        source_ts, target_ts = target_ts, source_ts
        source_ts = -1.0 * source_ts
    source_ts = np.sort(np.asarray(source_ts, dtype=float))
    target_ts = np.sort(np.asarray(target_ts, dtype=float))
    if len(source_ts) == 0 or len(target_ts) == 0:
        return 0.0
    j = 0; count = 0
    for t in source_ts:
        while j < len(target_ts) and target_ts[j] < t:
            j += 1
        ok = False
        if j < len(target_ts) and abs(target_ts[j] - t) <= interval: ok = True
        if j > 0 and abs(target_ts[j-1] - t) <= interval: ok = True
        if ok: count += 1
    return count / float(len(source_ts))

def plot_happen_rate_matrix(
    df: pd.DataFrame,
    *,
    interval: float = 0.5,
    show: bool = True,
    save_to: Optional[Union[str, Path]] = None,
) -> "go.Figure":
    """Interactive heatmap of happen rate by etype."""
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
            z=H, x=types, y=types, zmin=0.0, zmax=1.0, colorscale="viridis",
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


# --------- Bipartite graph construction ---------

def build_bipartite_graph(
    data: Union[pd.DataFrame, Dict[str, Any], str, Path],
    *,
    max_users: int = 40,
    max_items: int = 40,
    max_edges: int = 500,
    label_column: str = "label",
) -> Tuple["nx.Graph", List[int], List[int], pd.DataFrame]:
    """Create a weighted bipartite graph summarising interactions."""
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
        u = int(row["u"]); v = int(row["i"])
        attrs = {"weight": float(row["count"]), "ts_mean": float(row["ts_mean"])}
        if not np.isnan(row.get("label_mean", np.nan)):
            attrs["label_mean"] = float(row["label_mean"])
            attrs["positive_ratio"] = float(row["positive_ratio"])
        G.add_edge(u, v, **attrs)

    return G, users, items, agg


# --------- Interactive bipartite graph (static snapshot) ---------

def plot_bipartite_graph(
    data: Union[pd.DataFrame, Dict[str, Any], str, Path],
    *,
    max_users: int = 40,
    max_items: int = 40,
    max_edges: int = 500,
    edge_cmap: str = "RdYlGn",
    show_labels: bool = True,
    show: bool = True,
    save_to: Optional[Union[str, Path]] = None,
    highlight_users: Optional[Sequence[int]] = None,
    highlight_items: Optional[Sequence[int]] = None,
    highlight_edges: Optional[Sequence[Tuple[int, int]]] = None,
    highlight_size: float = 20.0,
) -> "go.Figure":
    """Interactive bipartite graph snapshot (hoverable, zoomable)."""
    _require_plotly()
    _require_networkx()

    G, users, items, agg = build_bipartite_graph(
        data, max_users=max_users, max_items=max_items, max_edges=max_edges
    )
    if G.number_of_edges() == 0:
        raise ValueError("Graph has no edges to draw.")

    top_nodes = set(users)
    pos = _compute_bipartite_layout(G, top_nodes, scale=1.8, vertical=True)

    # Node sizes by degree (log scaled)
    user_sizes = 12.0 * (1.0 + np.log1p([G.degree(u) for u in users]))
    item_sizes = 12.0 * (1.0 + np.log1p([G.degree(i) for i in items]))

    # Edge color by positive_ratio if available
    if "positive_ratio" in agg.columns and not agg["positive_ratio"].isna().all():
        ratios = agg["positive_ratio"].fillna(0.5).clip(0.0, 1.0).to_numpy()
        edge_colors = [_map_ratio_to_color(r, edge_cmap=edge_cmap) for r in ratios]
        colorbar_title = "P(label > 0)" if _HAS_MPL else None
    else:
        edge_colors = [COLORS["base"]] * len(agg)
        colorbar_title = None

    # Build edge coordinate arrays
    edge_x, edge_y, edge_color = [], [], []
    for color, (_, row) in zip(edge_colors, agg.iterrows()):
        u, v = int(row["u"]), int(row["i"])
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_color.append(color)

    # One edges trace (uniform width), colored average if heterogeneous
    # To keep code compact: use a single color if heterogeneous; otherwise colorbar is handled via nodes
    edges_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1.5, color=COLORS["base"]),
        hoverinfo="skip",
        showlegend=False,
    )

    # Users
    user_trace = go.Scatter(
        x=[pos[u][0] for u in users], y=[pos[u][1] for u in users],
        mode="markers+text" if show_labels and len(G) <= 120 else "markers",
        marker=dict(size=user_sizes, color=COLORS["user"], line=dict(color="white", width=0.5)),
        text=[str(u) for u in users] if show_labels and len(G) <= 120 else None,
        textposition="top center",
        name="Users",
        hovertemplate="user=%{text}<extra></extra>",
    )
    # Items
    item_trace = go.Scatter(
        x=[pos[i][0] for i in items], y=[pos[i][1] for i in items],
        mode="markers+text" if show_labels and len(G) <= 120 else "markers",
        marker=dict(size=item_sizes, color=COLORS["item"], symbol="square", line=dict(color="white", width=0.5)),
        text=[str(i) for i in items] if show_labels and len(G) <= 120 else None,
        textposition="bottom center",
        name="Items",
        hovertemplate="item=%{text}<extra></extra>",
    )

    traces = [edges_trace, user_trace, item_trace]

    highlight_users_set = {int(u) for u in (highlight_users or []) if int(u) in pos}
    highlight_items_set = {int(i) for i in (highlight_items or []) if int(i) in pos}

    if highlight_users_set:
        traces.append(
            go.Scatter(
                x=[pos[u][0] for u in highlight_users_set],
                y=[pos[u][1] for u in highlight_users_set],
                mode="markers+text",
                marker=dict(
                    size=highlight_size,
                    color=COLORS["accent2"],
                    line=dict(color="white", width=2),
                ),
                text=[str(u) for u in highlight_users_set],
                textposition="middle right",
                name="Explained users",
                hovertemplate="explained user=%{text}<extra></extra>",
            )
        )

    if highlight_items_set:
        traces.append(
            go.Scatter(
                x=[pos[i][0] for i in highlight_items_set],
                y=[pos[i][1] for i in highlight_items_set],
                mode="markers+text",
                marker=dict(
                    size=highlight_size,
                    color=COLORS["accent"],
                    symbol="diamond",
                    line=dict(color="white", width=2),
                ),
                text=[str(i) for i in highlight_items_set],
                textposition="middle left",
                name="Explained items",
                hovertemplate="explained item=%{text}<extra></extra>",
            )
        )

    highlight_edges_seq = [
        (int(u), int(v)) for u, v in (highlight_edges or [])
        if int(u) in pos and int(v) in pos
    ]
    if highlight_edges_seq:
        edge_x_h, edge_y_h = [], []
        for u, v in highlight_edges_seq:
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_x_h += [x0, x1, None]
            edge_y_h += [y0, y1, None]
        traces.append(go.Scatter(
            x=edge_x_h,
            y=edge_y_h,
            mode="lines",
            line=dict(width=3.0, color=COLORS["accent"]),
            name="Edges supporting explanation",
            hoverinfo="skip",
        ))

    fig = go.Figure(data=traces)
    title = "Bipartite interaction graph (node size ∝ degree)"
    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    _auto_show(fig, show)
    _maybe_save(fig, save_to)
    return fig


# --------- Interactive animation over time ---------

def animate_bipartite_graph(
    data: Union[pd.DataFrame, Dict[str, Any], str, Path],
    *,
    bins: int = 30,
    max_users: int = 40,
    max_items: int = 40,
    cumulative: bool = True,
    edge_cmap: str = "viridis",
    pruned: Optional[float] = None,
    show: bool = True,
    save_to: Optional[Union[str, Path]] = None,
) -> "go.Figure":
    """
    Interactive HTML animation of the bipartite graph over time (slider + play).

    Parameters
    ----------
    data : DataFrame | dict | str | Path
        Interactions source (expects columns 'u','i','ts' and optional 'label','etype').
    bins : int, default 30
        Number of time bins (frames). Clipped to [1, len(df)] after filtering.
    max_users : int, default 40
        Keep at most this many most-active users before pruning.
    max_items : int, default 40
        Keep at most this many most-active items before pruning.
    cumulative : bool, default True
        If True, frames accumulate edges; else each frame shows only edges in the bin.
    edge_cmap : str, default "viridis"
        Matplotlib colormap name for edge color across frames.
    pruned : float | None, default None
        If set to a fraction in (0, 1], keep only the top fraction of users and items
        (by global interaction count) *after* the max_users/max_items caps.
        Example: pruned=0.5 keeps the top half of the selected users and items.
    show : bool, default True
        If True, auto-show the figure in the notebook/browser.
    save_to : str | Path | None, default None
        If provided, saves the figure (HTML preferred, PNG if kaleido is installed).

    Returns
    -------
    go.Figure
        Plotly figure with frames and slider.
    """
    _require_plotly()
    _require_networkx()

    df = _ensure_dataframe(data).sort_values("ts")
    if len(df) == 0:
        raise ValueError("Dataset has no interactions to animate.")

    # --- Select top users/items globally by activity, then optionally prune by fraction ---
    user_counts_all = df["u"].value_counts()
    item_counts_all = df["i"].value_counts()

    # Cap by max_users / max_items first (keeps deterministic 'top by count')
    top_users_series = user_counts_all.head(max(1, max_users))
    top_items_series = item_counts_all.head(max(1, max_items))

    # Optional fraction-based pruning
    if pruned is not None:
        if not (0 < pruned <= 1):
            raise ValueError(f"`pruned` must be in (0, 1], got {pruned!r}")
        n_users_keep = max(1, int(np.ceil(len(top_users_series) * float(pruned))))
        n_items_keep = max(1, int(np.ceil(len(top_items_series) * float(pruned))))
        top_users_series = top_users_series.head(n_users_keep)
        top_items_series = top_items_series.head(n_items_keep)

    # Filter the dataframe to the kept users/items
    df = df[df["u"].isin(top_users_series.index) & df["i"].isin(top_items_series.index)]
    if len(df) == 0:
        raise ValueError(
            "Pruning removed all edges. Increase `pruned`, `max_users`, or `max_items`."
        )

    # --- Time binning ---
    bins = max(1, min(int(bins), len(df)))
    t_edges = np.linspace(df["ts"].min(), df["ts"].max(), bins + 1)
    t_labels = [f"{t_edges[i]:.2f} – {t_edges[i+1]:.2f}" for i in range(bins)]

    # Final node sets
    users = sorted(df["u"].unique().astype(int))
    items = sorted(df["i"].unique().astype(int))

    # --- Graph scaffold + layout ---
    G = nx.Graph()
    for u in users:
        G.add_node(int(u), bipartite=0, kind="user")
    for i in items:
        G.add_node(int(i), bipartite=1, kind="item")
    pos = _compute_bipartite_layout(G, users, scale=1.8, vertical=True)

    # --- Pre-split edges per bin ---
    edges_over_time: List[pd.DataFrame] = []
    for idx in range(bins):
        mask = (df["ts"] >= t_edges[idx]) & (df["ts"] < t_edges[idx + 1])
        edges_over_time.append(df.loc[mask, ["u", "i"]].astype(int))

    # --- Static node traces ---
    user_trace = go.Scatter(
        x=[pos[u][0] for u in users], y=[pos[u][1] for u in users],
        mode="markers",
        marker=dict(size=12 + 3*np.log1p([G.degree(u) for u in users]),
                    color=COLORS["user"], line=dict(color="white", width=0.5)),
        name="Users", hovertext=[f"user={u}" for u in users], hoverinfo="text"
    )
    item_trace = go.Scatter(
        x=[pos[i][0] for i in items], y=[pos[i][1] for i in items],
        mode="markers",
        marker=dict(size=12 + 3*np.log1p([G.degree(i) for i in items]),
                    color=COLORS["item"], symbol="square",
                    line=dict(color="white", width=0.5)),
        name="Items", hovertext=[f"item={i}" for i in items], hoverinfo="text"
    )

    # Initial empty edges trace
    edges_trace = go.Scatter(
        x=[], y=[], mode="lines",
        line=dict(width=1.5, color="rgba(120,120,120,0.9)"),
        hoverinfo="skip", showlegend=False
    )

    # --- Frames ---
    frames = []
    denom = max(1, bins - 1)
    for f in range(bins):
        if cumulative:
            active = pd.concat(edges_over_time[:f+1], ignore_index=True)
        else:
            active = edges_over_time[f]
        x, y = [], []
        if not active.empty:
            counts = active.groupby(["u", "i"]).size().reset_index(name="count")
            for _, row in counts.iterrows():
                u, i = int(row["u"]), int(row["i"])
                x0, y0 = pos[u]; x1, y1 = pos[i]
                x += [x0, x1, None]
                y += [y0, y1, None]
        color = _map_ratio_to_color(f / denom, edge_cmap=edge_cmap)
        frames.append(go.Frame(
            data=[go.Scatter(x=x, y=y, mode="lines",
                             line=dict(width=1.8, color=color), hoverinfo="skip")]
        ))

    fig = go.Figure(data=[edges_trace, user_trace, item_trace], frames=frames)
    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        title=(
            "Temporal bipartite graph (use ▶ to play, or drag the slider)"
            + (f" — pruned={pruned:.2f}" if pruned is not None else "")
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=60, b=10),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.05, "y": 1.12, "xanchor": "left",
            "buttons": [
                {"label": "▶ Play", "method": "animate",
                 "args": [None, {"frame": {"duration": 350, "redraw": True},
                                 "fromcurrent": True, "transition": {"duration": 0}}]},
                {"label": "⏸ Pause", "method": "animate",
                 "args": [[None], {"mode": "immediate",
                                   "frame": {"duration": 0, "redraw": False},
                                   "transition": {"duration": 0}}]}
            ],
        }],
        sliders=[{
            "active": 0,
            "y": 1.05, "x": 0.05, "len": 0.9,
            "xanchor": "left",
            "steps": [
                {"label": f"{i+1}/{bins} · {t_labels[i]}",
                 "method": "animate",
                 "args": [[f"frame{i}"],
                          {"mode": "immediate",
                           "frame": {"duration": 0, "redraw": True}}]}
                for i in range(bins)
            ]
        }]
    )

    # Name frames for slider
    for i, fr in enumerate(fig.frames):
        fr.name = f"frame{i}"

    _auto_show(fig, show)
    _maybe_save(fig, save_to)
    return fig


# --------- Explain timeline (interactive) ---------

def plot_explain_timeline(
    data: Union[pd.DataFrame, Dict[str, Any], str, Path],
    event_indices: Sequence[int],
    *,
    window: int = 0,
    max_base_points: int = 50_000,
    show: bool = True,
    save_to: Optional[Union[str, Path]] = None,
) -> "go.Figure":
    """Interactive scatter of event chronology with highlighted explain instances."""
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
        x=base_ts, y=base_order, mode="markers",
        marker=dict(size=5, color="rgba(160,160,160,0.35)"),
        name="events", hoverinfo="skip"
    )

    focal = df.iloc[positions]
    focal_trace = go.Scatter(
        x=focal["ts"], y=positions, mode="markers+text",
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
            shapes.append(dict(
                type="rect",
                xref="paper", yref="y",
                x0=0, x1=1,
                y0=start, y1=end,
                fillcolor="rgba(196,78,82,0.08)",
                line=dict(width=0),
                layer="below",
            ))

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


# --------- Summaries & convenience ---------

def summarize_explain_instances(
    data: Union[pd.DataFrame, Dict[str, Any], str, Path],
    event_indices: Sequence[int],
    *,
    events_before: int = 5,
    events_after: int = 5,
    time_window: Optional[float] = None,
) -> pd.DataFrame:
    """Summarise context around explain indices for notebook display."""
    df = _ensure_dataframe(data)
    positions = _resolve_event_positions(df, event_indices)
    rows: List[Dict[str, Any]] = []

    if not positions:
        return pd.DataFrame(columns=[
            "query_idx","row","ts","label","user","item","events_before","events_after","same_user_context","same_item_context","time_window_count",
        ])

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

        rows.append({
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
        })
    return pd.DataFrame(rows)


def visualize_folder(
    folder: str | Path,
    *,
    num_nodes: int | None = None,
    happen_interval: float = 0.5,
    show: bool = True,
) -> Dict[str, "go.Figure"]:
    """Quick interactive overview for a dataset folder (auto-displays)."""
    _require_plotly()
    bundle = load_processed_dataset(folder)
    df = bundle["interactions"]

    figs = {}
    figs["event_count"] = plot_event_count_over_time(df, show=show)
    if len(df) > 1:
        figs["inter_event"] = plot_inter_event_time_hist(df, show=show)
    figs["degree"] = plot_degree_histograms(df, show=show)
    figs["adjacency"] = plot_adjacency_heatmap(df, num_nodes=num_nodes, show=show)
    if "label" in df.columns:
        figs["label_balance"] = plot_label_balance(df, show=show)
    if "etype" in df.columns:
        figs["happen_rate"] = plot_happen_rate_matrix(df, interval=happen_interval, show=show)
    return figs


def visualize_to_files(
    folder: str | Path,
    out_dir: str | Path,
    *,
    num_nodes: int | None = None,
    happen_interval: float = 0.5,
    include_graph: bool = True,
    graph_max_users: int = 40,
    graph_max_items: int = 40,
    graph_max_edges: int = 500,
    explain_indices: Optional[Sequence[int]] = None,
    explain_events_before: int = 5,
    explain_events_after: int = 5,
    explain_time_window: Optional[float] = None,
    table_max_rows: int = 15,  # kept for API compatibility (unused here)
) -> Dict[str, List[str]]:
    """
    Create interactive HTMLs for standard plots and save them. Returns dict of saved paths.
    Note: static PNG export needs 'kaleido'; otherwise we save HTML.
    """
    _require_plotly()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_processed_dataset(folder)
    df = bundle["interactions"]

    saved: Dict[str, List[str]] = {k: [] for k in [
        "event_count","inter_event","degree","adjacency","label_balance","happen_rate","graph","explain_timeline","explain_summary"
    ]}

    p = out_dir / "01_event_count.html"
    path = _maybe_save(plot_event_count_over_time(df, show=False), p)
    if path:
        saved["event_count"].append(path)

    if len(df) > 1:
        p = out_dir / "02_inter_event.html"
        path = _maybe_save(plot_inter_event_time_hist(df, show=False), p)
        if path:
            saved["inter_event"].append(path)

    p = out_dir / "03_degree.html"
    path = _maybe_save(plot_degree_histograms(df, show=False), p)
    if path:
        saved["degree"].append(path)

    p = out_dir / "04_adjacency.html"
    path = _maybe_save(plot_adjacency_heatmap(df, num_nodes=num_nodes, show=False), p)
    if path:
        saved["adjacency"].append(path)

    if "label" in df.columns:
        p = out_dir / "05_label_balance.html"
        path = _maybe_save(plot_label_balance(df, show=False), p)
        if path:
            saved["label_balance"].append(path)

    if "etype" in df.columns:
        p = out_dir / "06_happen_rate.html"
        path = _maybe_save(plot_happen_rate_matrix(df, interval=happen_interval, show=False), p)
        if path:
            saved["happen_rate"].append(path)

    if include_graph:
        try:
            p = out_dir / "07_interaction_graph.html"
            path = _maybe_save(plot_bipartite_graph(bundle, max_users=graph_max_users, max_items=graph_max_items, max_edges=graph_max_edges, show=False), p)
            if path:
                saved["graph"].append(path)
        except (RuntimeError, ValueError) as exc:
            warnings.warn(f"Graph visualisation skipped: {exc}", RuntimeWarning)

    if explain_indices:
        try:
            p = out_dir / "08_explain_timeline.html"
            path = _maybe_save(
                plot_explain_timeline(
                    bundle,
                    explain_indices,
                    window=max(explain_events_before, explain_events_after),
                    max_base_points=20_000,
                    show=False,
                ),
                p,
            )
            if path:
                saved["explain_timeline"].append(path)

            summary_df = summarize_explain_instances(
                bundle, explain_indices,
                events_before=explain_events_before,
                events_after=explain_events_after,
                time_window=explain_time_window,
            )
            if not summary_df.empty:
                # Save a simple CSV and an HTML table
                csv_path = out_dir / "08_explain_instances.csv"
                summary_df.to_csv(csv_path, index=False)
                table_fig = go.Figure(data=[go.Table(
                    header=dict(values=list(summary_df.columns), fill_color="#f5f5f5"),
                    cells=dict(values=[summary_df[c] for c in summary_df.columns])
                )])
                html_path = _maybe_save(table_fig, out_dir / "08_explain_instances.html")
                saved["explain_summary"].append(str(csv_path))
                if html_path:
                    saved["explain_summary"].append(html_path)
        except (RuntimeError, ValueError) as exc:
            warnings.warn(f"Explain visualisation skipped: {exc}", RuntimeWarning)

    return saved
