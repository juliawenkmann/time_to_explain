from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd

# Plotly for interactive charts
try:  # pragma: no cover
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:  # pragma: no cover
    go = None
    make_subplots = None

# Matplotlib helpers (optional)
try:  # pragma: no cover
    import matplotlib.colors as mpl_colors
    from matplotlib import cm as mpl_cm
except ImportError:  # pragma: no cover
    mpl_colors = None
    mpl_cm = None

try:  # pragma: no cover
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:  # pragma: no cover
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None

# NetworkX is optional but required for several graph views
try:  # pragma: no cover
    import networkx as nx
    from networkx.algorithms import bipartite as nx_bipartite
except ImportError:  # pragma: no cover
    nx = None
    nx_bipartite = None

PLOT_STYLE = {
    "font_family": "Helvetica, Arial, sans-serif",
    "font_size": 12,
    "title_size": 16,
    "axis_title_size": 12,
    "tick_size": 11,
    "legend_size": 11,
    "annotation_size": 11,
}
COLORS = {
    "user": "#4C78A8",
    "item": "#F58518",
    "accent": "#C44E52",
    "accent2": "#54A24B",
    "base": "#9E9E9E",
    "text": "#222222",
    "grid": "#E5E5E5",
    "background": "#FFFFFF",
}
PLOT_COLORWAY = [
    COLORS["user"],
    COLORS["item"],
    COLORS["accent"],
    COLORS["accent2"],
    COLORS["base"],
]


def _build_plotly_template():
    if go is None:
        return "simple_white"
    return go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=COLORS["background"],
            plot_bgcolor=COLORS["background"],
            font=dict(
                family=PLOT_STYLE["font_family"],
                size=PLOT_STYLE["font_size"],
                color=COLORS["text"],
            ),
            title=dict(font=dict(size=PLOT_STYLE["title_size"], color=COLORS["text"])),
            legend=dict(font=dict(size=PLOT_STYLE["legend_size"])),
            colorway=PLOT_COLORWAY,
            xaxis=dict(
                showgrid=True,
                gridcolor=COLORS["grid"],
                zerolinecolor=COLORS["grid"],
                linecolor=COLORS["grid"],
                tickfont=dict(size=PLOT_STYLE["tick_size"], color=COLORS["text"]),
                title_font=dict(size=PLOT_STYLE["axis_title_size"], color=COLORS["text"]),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=COLORS["grid"],
                zerolinecolor=COLORS["grid"],
                linecolor=COLORS["grid"],
                tickfont=dict(size=PLOT_STYLE["tick_size"], color=COLORS["text"]),
                title_font=dict(size=PLOT_STYLE["axis_title_size"], color=COLORS["text"]),
            ),
            annotationdefaults=dict(
                font=dict(size=PLOT_STYLE["annotation_size"], color=COLORS["text"])
            ),
        )
    )


_PLOTLY_TEMPLATE = _build_plotly_template()


def choose_explain_indices(num_events: int, *, count: int = 3) -> List[int]:
    """
    Return up to ``count`` anchor event indices (first, middle, last) for highlighting.

    This mirrors the helper that lived in the data-prep notebook so downstream
    visualisations can consistently highlight representative events.
    """
    if num_events <= 0 or count <= 0:
        return []

    anchors = [0, max(0, num_events // 2), max(0, num_events - 1)]
    selected: List[int] = []
    for idx in anchors:
        if 0 <= idx < num_events and idx not in selected:
            selected.append(int(idx))
        if len(selected) >= count:
            return selected[:count]

    if len(selected) < count and num_events > 0:
        extra = np.linspace(0, num_events - 1, num=min(count, num_events), dtype=int)
        for idx in extra:
            if int(idx) not in selected:
                selected.append(int(idx))
            if len(selected) >= count:
                break

    return selected[: min(count, num_events)]


@dataclass(frozen=True)
class DatasetVizProfile:
    dataset_name: str
    recipe: Optional[str]
    is_bipartite: bool
    is_nicolaus: bool
    is_triadic: bool
    is_stick: bool


def _normalize_dataset_name(value: Union[str, Path]) -> str:
    path = Path(value)
    if path.suffix:
        stem = path.stem
        if stem.startswith("ml_"):
            stem = stem[3:]
        if stem.endswith("_node"):
            stem = stem[: -len("_node")]
        return stem
    return path.name if path.name else str(value)


def infer_dataset_profile(
    metadata: Optional[Dict[str, Any]] = None,
    *,
    dataset_name: Optional[Union[str, Path]] = None,
) -> DatasetVizProfile:
    """Infer dataset flags (bipartite vs. synthetic types) from metadata and name."""
    meta = metadata or {}
    config = meta.get("config") if isinstance(meta.get("config"), dict) else {}

    fallback_name = None
    if dataset_name is not None:
        try:
            fallback_name = _normalize_dataset_name(dataset_name)
        except Exception:
            fallback_name = str(dataset_name)

    raw_dataset = meta.get("dataset_name") if isinstance(meta.get("dataset_name"), str) else None
    raw_recipe = meta.get("recipe") if isinstance(meta.get("recipe"), str) else None
    dataset_label = raw_dataset or fallback_name or raw_recipe or "dataset"

    label_candidates = [
        label.lower()
        for label in (raw_dataset, raw_recipe, fallback_name)
        if isinstance(label, str)
    ]

    nicolaus_tokens = ("nicolaus", "nicholaus", "nikolaus")
    triadic_tokens = ("triadic_closure", "triadic-closure", "triadic", "triad")
    stick_tokens = ("stick", "stick_figure")
    non_bipartite_tokens = triadic_tokens + nicolaus_tokens + ("erdos", "hawkes") + stick_tokens

    def _matches(tokens: Sequence[str]) -> bool:
        return any(token in label for label in label_candidates for token in tokens)

    is_nicolaus = _matches(nicolaus_tokens)
    is_triadic = _matches(triadic_tokens)
    is_stick = _matches(stick_tokens)

    bipartite_flag: Optional[bool] = None
    if "bipartite" in meta:
        bipartite_flag = bool(meta["bipartite"])
    elif "bipartite" in config:
        bipartite_flag = bool(config["bipartite"])

    if bipartite_flag is None:
        bipartite_flag = not _matches(non_bipartite_tokens)

    return DatasetVizProfile(
        dataset_name=dataset_label,
        recipe=raw_recipe,
        is_bipartite=bipartite_flag,
        is_nicolaus=is_nicolaus,
        is_triadic=is_triadic,
        is_stick=is_stick,
    )


def load_dataset_bundle(
    dataset: Union[str, Path, Dict[str, Any]],
    *,
    root_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Resolve a processed dataset bundle from a name/path or pass-through bundle."""
    if isinstance(dataset, dict):
        if "interactions" not in dataset:
            raise KeyError("Dataset bundle must include an 'interactions' dataframe.")
        return dataset
    if isinstance(dataset, (str, Path)):
        from ..data.workflows import load_processed_dataset_safe  # local import to avoid cycles

        return load_processed_dataset_safe(dataset, root_dir=root_dir, verbose=verbose)
    raise TypeError(f"Unsupported dataset type: {type(dataset)!r}")


def _require_plotly() -> None:
    if go is None or make_subplots is None:
        raise RuntimeError("Plotly is required for this visualization. Install via 'pip install plotly kaleido'.")


def _require_networkx() -> None:
    if nx is None:
        raise RuntimeError("networkx is required for graph visualisations. Install via 'pip install networkx'.")


def _require_matplotlib() -> None:
    if plt is None:
        raise RuntimeError("Matplotlib is required for this visualization. Install via 'pip install matplotlib'.")


def _require_seaborn() -> None:
    if sns is None:
        raise RuntimeError("Seaborn is required for this visualization. Install via 'pip install seaborn'.")


def apply_matplotlib_style() -> None:
    if plt is None:
        return
    rc = {
        "axes.edgecolor": COLORS["grid"],
        "grid.color": COLORS["grid"],
        "text.color": COLORS["text"],
        "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "figure.facecolor": COLORS["background"],
        "axes.facecolor": COLORS["background"],
    }
    if sns is not None:
        sns.set_theme(
            style="whitegrid",
            palette=PLOT_COLORWAY,
            font=PLOT_STYLE["font_family"],
            font_scale=1.0,
            rc=rc,
        )
    plt.rcParams.update(
        {
            "font.family": PLOT_STYLE["font_family"],
            "font.size": PLOT_STYLE["font_size"],
            "axes.titlesize": PLOT_STYLE["title_size"],
            "axes.labelsize": PLOT_STYLE["axis_title_size"],
            "xtick.labelsize": PLOT_STYLE["tick_size"],
            "ytick.labelsize": PLOT_STYLE["tick_size"],
            "legend.fontsize": PLOT_STYLE["legend_size"],
            "figure.titlesize": PLOT_STYLE["title_size"],
        }
    )
    try:
        from cycler import cycler  # type: ignore
    except Exception:
        cycler = None
    if cycler is not None:
        plt.rcParams["axes.prop_cycle"] = cycler(color=PLOT_COLORWAY)


def _hex_to_rgb(color: str) -> np.ndarray:
    color = color.lstrip("#")
    if len(color) == 3:
        color = "".join(ch * 2 for ch in color)
    if len(color) != 6:
        raise ValueError(f"Expected #RRGGBB hex colour, got '{color}'")
    return np.array([int(color[i : i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


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


def _rgba(color: str, alpha: float) -> str:
    rgb = (_hex_to_rgb(color) * 255).round().astype(int)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha:.3f})"


def _interpolate_colorscale(
    scale: Sequence[Tuple[float, str]] | Sequence[str],
    value: float,
) -> str:
    if not scale:
        return COLORS["base"]

    if isinstance(scale[0], str):
        scale = list(scale)
        if len(scale) == 1:
            return scale[0]
        scale = [
            (idx / (len(scale) - 1), color) for idx, color in enumerate(scale)
        ]

    stops = sorted((float(pos), color) for pos, color in scale)
    if value <= stops[0][0]:
        return stops[0][1]
    for (p0, c0), (p1, c1) in zip(stops[:-1], stops[1:]):
        if value <= p1:
            if p1 == p0:
                return c1
            weight = (value - p0) / (p1 - p0)
            return _blend_hex(c0, c1, weight)
    return stops[-1][1]


SEQUENTIAL_COLORSCALE = [
    (0.0, _blend_hex(COLORS["background"], COLORS["user"], 0.15)),
    (0.5, _blend_hex(COLORS["background"], COLORS["user"], 0.55)),
    (1.0, COLORS["user"]),
]
DIVERGING_COLORSCALE = [
    (0.0, COLORS["accent"]),
    (0.5, COLORS["background"]),
    (1.0, COLORS["accent2"]),
]


def _map_ratio_to_color(
    ratio: float, edge_cmap: str | Sequence[Tuple[float, str]] | Sequence[str] = "theme_diverging"
) -> str:
    ratio = float(np.clip(ratio, 0.0, 1.0))
    if edge_cmap in {"theme_diverging", "theme"}:
        return _interpolate_colorscale(DIVERGING_COLORSCALE, ratio)
    if edge_cmap in {"theme_sequential", "theme_seq"}:
        return _interpolate_colorscale(SEQUENTIAL_COLORSCALE, ratio)
    if isinstance(edge_cmap, (list, tuple)):
        try:
            return _interpolate_colorscale(edge_cmap, ratio)
        except Exception:
            pass
    if mpl_cm is not None and mpl_colors is not None:
        cmap = mpl_cm.get_cmap(edge_cmap)
        return mpl_colors.to_hex(cmap(ratio))
    return _blend_hex(COLORS["accent"], COLORS["accent2"], ratio)


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
        bundle = load_dataset_bundle(data, verbose=False)
        return bundle["interactions"].copy()
    raise TypeError(f"Unsupported data type: {type(data)!r}")


def _maybe_save(fig: "go.Figure", save_to: Optional[Union[str, Path]]) -> Optional[str]:
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
        try:
            fig.write_image(save_to, scale=2)
        except Exception as exc:  # kaleido missing or other issue
            warnings.warn(
                f"Static export failed ({exc}). Saving HTML instead at '{save_to}.html'. "
                "Install kaleido for static image export: pip install -U kaleido",
                RuntimeWarning,
            )
            fallback = save_to + ".html"
            try:
                fig.write_html(fallback, include_plotlyjs="cdn", auto_open=False)
                save_to = fallback
            except (ValueError, OverflowError) as inner_exc:
                msg = str(inner_exc).lower()
                if "string length" in msg or "memory" in msg:
                    warnings.warn(f"Skipping HTML save for '{fallback}': {inner_exc}", RuntimeWarning)
                    return None
                raise
    return save_to


def _auto_show(fig: "go.Figure", show: bool) -> None:
    if show:
        fig.show()


def _histogram_to_bar(x: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(x, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def _compute_bipartite_layout(
    G: "nx.Graph",
    top_nodes: Sequence[int],
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

    # Fallback to simple two rows
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


def _resolve_event_positions(df: pd.DataFrame, event_indices: Sequence[int]) -> list[int]:
    if len(event_indices) == 0:
        return []
    if "idx" in df.columns:
        mapping = {int(idx): pos for pos, idx in enumerate(df["idx"].astype(int))}
    else:
        mapping = {}

    positions: list[int] = []
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


def select_ground_truth_event(
    df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]],
    explain_indices: Optional[Sequence[int]] = None,
) -> Optional[int]:
    ground = (metadata or {}).get("ground_truth") or {}
    raw_targets = ground.get("targets") or []
    targets: List[int] = []
    for t in raw_targets:
        try:
            targets.append(int(t))
        except (TypeError, ValueError):
            continue
    targets = sorted({t for t in targets if 0 <= t < len(df)})
    if not targets:
        return None

    if explain_indices:
        positions = _resolve_event_positions(df, explain_indices)
        for pos in positions:
            if pos in targets:
                return pos

    return targets[0]


__all__ = [
    "COLORS",
    "PLOT_COLORWAY",
    "PLOT_STYLE",
    "SEQUENTIAL_COLORSCALE",
    "DIVERGING_COLORSCALE",
    "DatasetVizProfile",
    "_PLOTLY_TEMPLATE",
    "_require_plotly",
    "_require_networkx",
    "_require_matplotlib",
    "_require_seaborn",
    "apply_matplotlib_style",
    "infer_dataset_profile",
    "load_dataset_bundle",
    "_ensure_dataframe",
    "_maybe_save",
    "_auto_show",
    "_histogram_to_bar",
    "_compute_bipartite_layout",
    "_resolve_event_positions",
    "_map_ratio_to_color",
    "_rgba",
    "select_ground_truth_event",
    "go",
    "make_subplots",
    "nx",
    "nx_bipartite",
    "mpl_colors",
    "mpl_cm",
    "plt",
    "sns",
]
