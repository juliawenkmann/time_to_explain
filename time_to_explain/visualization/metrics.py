from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import os

import numpy as np
import pandas as pd

from .utils import (
    COLORS,
    PLOT_COLORWAY,
    PLOT_STYLE,
    _require_matplotlib,
    _require_seaborn,
    apply_matplotlib_style,
)

# These imports are guarded; the helper ensures availability before use.
from .utils import plt, sns  # type: ignore  # noqa: F401


def read_tabs_plot(
    files,
    name,
    plot_only_og=False,
    *,
    metric="fidelity_best.best",
    sparsity_col="sparsity.edges.zero_frac",
    labels=None,
    markers=None,
    palette=None,
    og_keys=None,
    save_dir="plots",
):
    """Load metric CSVs, aggregate by sparsity, and plot fidelity curves."""
    _require_seaborn()
    _require_matplotlib()
    apply_matplotlib_style()
    files = {k: os.fspath(v) for k, v in files.items()}

    tabs = {}
    best_fids = {}
    aufsc = {}

    for key, path in files.items():
        df = pd.read_csv(path)
        if sparsity_col in df:
            sparsity = df[sparsity_col]
        elif "sparsity" in df:
            sparsity = df["sparsity"]
        else:
            raise KeyError(f"{sparsity_col!r} column not found in {path}")

        if metric not in df:
            raise KeyError(f"Metric column {metric!r} not found in {path}")

        tab = (
            df.assign(sparsity=sparsity)
            .groupby("sparsity", as_index=False)[metric]
            .mean()
            .sort_values("sparsity")
        )
        tabs[key] = tab

        best_fids[key] = tab[metric].max()
        aufsc[key] = float(np.trapz(tab[metric], tab["sparsity"]))

    print("Best Fidelity (max across levels):", best_fids)
    print("Area under fidelity-sparsity curve:", aufsc)

    og_defaults = set(og_keys or ["xtg-og", "attn", "pbone", "pg"])
    keys_to_plot = [k for k in tabs if (not plot_only_og or k in og_defaults)]
    if not keys_to_plot:
        keys_to_plot = list(tabs.keys())

    os.makedirs(save_dir, exist_ok=True)
    if palette is None or palette == "theme":
        palette_colors = list(PLOT_COLORWAY)
    elif isinstance(palette, (list, tuple)):
        palette_colors = list(palette)
    else:
        palette_colors = sns.color_palette(palette, len(keys_to_plot))
    default_markers = ["o", "s", "^", "D", "P", "X", "*", "v"]

    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, key in enumerate(keys_to_plot):
        tab = tabs[key]
        label = (labels or {}).get(key, key)
        marker = (markers or {}).get(key, default_markers[idx % len(default_markers)])
        sns.lineplot(
            data=tab,
            x="sparsity",
            y=metric,
            ax=ax,
            label=label,
            color=palette_colors[idx % len(palette_colors)],
            marker=marker,
        )

    ax.set_xlabel("Sparsity (fraction of zero edges)")
    ax.set_ylabel(metric)
    ax.set_title(f"Fidelity vs sparsity — {name}")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(save_dir, f"{name}.png")
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


def default_curve_specs() -> List[MetricCurveSpec]:
    """
    Curves commonly used in notebook 04 with a balanced colour palette and labels.
    """
    return [
        MetricCurveSpec(
            prefix="fidelity_drop.@",
            title="Fidelity drop per anchor",
            color=COLORS.get("user", "tab:blue"),
            axis_label_percent=("Drop sparsity level (% of edges removed)", "Drop level (top-k edges removed)"),
        ),
        MetricCurveSpec(
            prefix="fidelity_keep.@",
            title="Fidelity keep per anchor",
            color=COLORS.get("item", "tab:orange"),
            axis_label_percent=("Keep level (% of edges kept)", "Keep level (top-k edges kept)"),
        ),
        MetricCurveSpec(
            prefix="fidelity_tempme.@",
            title="Fidelity TEMP-ME per anchor",
            color=COLORS.get("accent2", "tab:green"),
            ylabel="TEMP-ME fidelity",
            axis_label_percent=("Explanation sparsity (% of edges kept)", "Explanation sparsity (kept edges)"),
            y_min=None,
        ),
        MetricCurveSpec(
            prefix="cohesiveness.@",
            title="Cohesiveness per anchor",
            color=COLORS.get("accent", "tab:red"),
            ylabel="Cohesiveness",
            axis_label_percent=("Explanation sparsity (% of edges kept)", "Explanation sparsity (kept edges)"),
            y_min=0.0,
        ),
        MetricCurveSpec(
            prefix="temgx_fidelity_minus.@",
            title="TemGX fidelity- per anchor",
            color=COLORS.get("user", "tab:blue"),
            ylabel="|Δ score|",
            axis_label_percent=("Explanation sparsity (% of edges kept)", "Explanation sparsity (kept edges)"),
        ),
        MetricCurveSpec(
            prefix="temgx_fidelity_plus.@",
            title="TemGX fidelity+ per anchor",
            color=COLORS.get("accent2", "tab:green"),
            ylabel="1 - |Δ score|",
            axis_label_percent=("Explanation sparsity (% of edges kept)", "Explanation sparsity (kept edges)"),
            y_min=0.0,
        ),
        MetricCurveSpec(
            prefix="singular_value.@",
            title="Singular value per anchor",
            color=COLORS.get("base", "tab:gray"),
            ylabel="Largest singular value",
            axis_label_percent=("Explanation sparsity (% of edges kept)", "Explanation sparsity (kept edges)"),
            y_min=0.0,
        ),
    ]


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
        if isinstance(lvl, (int, float)) and not isinstance(lvl, bool):
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


def prepare_metrics_plotting(
    metrics_df: pd.DataFrame,
    *,
    explainer_order: Optional[Sequence[str]] = None,
    cmap_name: str = "tab10",
) -> Tuple[pd.DataFrame, Optional[List[Any]]]:
    """
    Apply a stable explainer order and build a matching palette for plots.
    """
    df = metrics_df.copy()
    palette = None

    if "explainer" in df.columns:
        if explainer_order:
            df["explainer"] = pd.Categorical(df["explainer"], categories=list(explainer_order), ordered=True)
        unique_explainers = list(pd.unique(df["explainer"].dropna()))
        if plt is not None and unique_explainers:
            try:
                cmap = plt.get_cmap(cmap_name)
                palette = [cmap(i) for i in range(len(explainer_order or unique_explainers))]
            except Exception:
                palette = None

    return df, palette


def plot_metric_curves(
    metrics_df: pd.DataFrame,
    specs: Sequence[MetricCurveSpec],
    *,
    group_col: Optional[str] = None,
    palette: Optional[Sequence[str]] = None,
    show_individual: bool = True,
) -> Dict[str, Dict[str, Any]]:
    _require_matplotlib()
    apply_matplotlib_style()
    results: Dict[str, Dict[str, Any]] = {}
    if metrics_df.empty:
        return results

    group_mode = bool(group_col and group_col in metrics_df.columns)

    grouped = (
        [
            (str(name), group.copy())
            for name, group in metrics_df.groupby(group_col)
            if not group.empty
        ]
        if group_mode
        else []
    )
    if group_mode and not grouped:
        group_mode = False

    for spec in specs:
        cols, levels = collect_curve_columns(metrics_df, spec.prefix)
        if not cols:
            continue

        axis, is_fractional = levels_to_axis(levels)
        fig, ax = plt.subplots(figsize=spec.figsize)
        all_finite: List[float] = []
        group_summaries: Dict[str, Dict[str, Any]] = {}

        if group_mode:
            if palette:
                default_colors = list(palette)
            else:
                default_colors = list(PLOT_COLORWAY)
            color_iter = cycle(default_colors)
            iterable = grouped
        else:
            color_iter = cycle([spec.color])
            iterable = [(None, metrics_df)]

        for group_name, group_df in iterable:
            if group_df.empty:
                continue
            color = next(color_iter)
            finite_vals: List[float] = []
            level_values: Dict[str, List[float]] = {col: [] for col in cols}

            for _, row in group_df.iterrows():
                line_vals: List[float] = []
                for col in cols:
                    val = row.get(col, np.nan)
                    try:
                        vf = float(val)
                    except (TypeError, ValueError):
                        vf = float("nan")
                    if np.isfinite(vf):
                        finite_vals.append(vf)
                        all_finite.append(vf)
                        level_values[col].append(vf)
                    line_vals.append(vf)
                if show_individual:
                    ax.plot(axis, line_vals, color=color, alpha=spec.alpha, linewidth=1.0)

            mean_curve: List[float] = []
            for col in cols:
                samples = level_values.get(col, [])
                mean_curve.append(float(np.mean(samples)) if samples else float("nan"))

            mean_curve_arr = np.asarray(mean_curve, dtype=float)
            if np.isfinite(mean_curve_arr).any():
                label = f"{group_name} mean" if group_mode and group_name is not None else "mean"
                ax.plot(
                    axis,
                    mean_curve,
                    color=color,
                    linestyle="--",
                    linewidth=2.5,
                    label=label,
                )

            group_key = str(group_name) if group_name is not None else "all"
            group_summaries[group_key] = {"values": finite_vals, "mean_curve": mean_curve}

        xlabel = spec.axis_label_percent[0] if is_fractional else spec.axis_label_percent[1]
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_ylabel(spec.ylabel)
        if spec.y_min is not None:
            ax.set_ylim(bottom=spec.y_min)
        ax.grid(True, alpha=0.2)
        ax.set_title(spec.title)
        if group_mode:
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:
                ax.legend()

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
                    fontsize=PLOT_STYLE["annotation_size"],
                    bbox={
                        "facecolor": COLORS["background"],
                        "alpha": 0.75,
                        "edgecolor": "none",
                    },
                )
        plt.show()

        results[spec.prefix] = {
            "columns": cols,
            "levels": levels,
            "values": all_finite,
            "axis_values": axis,
            "is_fractional": is_fractional,
            "mean_curve": None if group_mode else group_summaries.get("all", {}).get("mean_curve"),
            "groups": group_summaries if group_mode else None,
        }

    return results


def summarize_curve_results(curve_results: Dict[str, Dict[str, Any]], labels: Dict[str, str]) -> None:
    """
    Lightweight textual summary for curve outputs keyed by prefix.
    """
    for prefix, label in labels.items():
        spec_res = curve_results.get(prefix, {}) or {}
        group_stats = spec_res.get("groups") or {}
        if not group_stats:
            print(f"No data for {label}.")
            continue
        print(f"\n{label} (prefix={prefix}):")
        for expl_name, stats in group_stats.items():
            vals = np.asarray(stats.get("values", []), dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            print(f"  {expl_name}: mean={vals.mean():.4f}, min={vals.min():.4f}, max={vals.max():.4f}, n={len(vals)}")


def plot_explainer_metric_summary(
    metrics_df: pd.DataFrame,
    metric_columns: Sequence[Union[str, Tuple[str, str]]],
    *,
    agg: Union[str, Callable[[pd.Series], float]] = "mean",
    dropna: bool = True,
    palette: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (10, 5),
    annotate: bool = True,
    legend: bool = True,
) -> pd.DataFrame:
    _require_seaborn()
    _require_matplotlib()
    apply_matplotlib_style()
    if metrics_df.empty:
        raise ValueError("metrics_df is empty; nothing to plot.")
    if "explainer" not in metrics_df.columns:
        raise KeyError("metrics_df must contain an 'explainer' column.")
    if not metric_columns:
        raise ValueError("metric_columns must contain at least one entry.")

    resolved: List[Tuple[str, str]] = []
    for entry in metric_columns:
        if isinstance(entry, str):
            label, col = entry, entry
        elif isinstance(entry, (tuple, list)) and len(entry) == 2:
            label, col = str(entry[0]), str(entry[1])
        else:
            raise TypeError("Each metric spec must be a column name string or (label, column_name) tuple.")
        if col not in metrics_df.columns:
            raise KeyError(f"Column '{col}' not found in metrics_df.")
        resolved.append((label, col))

    value_columns = [col for _, col in resolved]
    work_df = metrics_df.copy()
    if dropna:
        work_df = work_df.dropna(subset=value_columns, how="all")
        if work_df.empty:
            raise ValueError("All rows were dropped due to NaNs in the requested metrics.")

    grouped = work_df.groupby("explainer", dropna=False)
    summary = grouped[value_columns].agg(agg)
    rename_map = {col: label for label, col in resolved}
    summary = summary.rename(columns=rename_map)

    plot_df = summary.reset_index().melt(id_vars="explainer", var_name="metric", value_name="value")

    plt.figure(figsize=figsize)
    palette = palette or PLOT_COLORWAY
    ax = sns.barplot(
        data=plot_df,
        x="metric",
        y="value",
        hue="explainer",
        palette=palette,
    )
    if isinstance(agg, str):
        agg_label = agg.capitalize()
    else:
        agg_label = getattr(agg, "__name__", "metric")
    ax.set_ylabel(f"{agg_label} value")
    ax.set_xlabel("Metric")
    leg_obj = ax.get_legend()
    if not legend and leg_obj is not None:
        leg_obj.remove()
    elif legend:
        ax.legend(title="Explainer")
    ax.set_title("Average metrics per explainer")
    ax.grid(axis="y", alpha=0.2)

    if annotate:
        for patch in ax.patches:
            height = patch.get_height()
            if not np.isfinite(height):
                continue
            ax.annotate(
                f"{height:.3f}",
                (patch.get_x() + patch.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=PLOT_STYLE["annotation_size"],
            )

    plt.tight_layout()
    plt.show()
    return summary


def plot_prediction_match_rate(
    metrics_df: pd.DataFrame,
    *,
    group_col: Optional[str] = "explainer",
    match_prefix: str = "prediction_profile.match_",
    title: str = "Prediction stability vs. sparsity (match rate)",
    ax=None,
    auto_display: bool = True,
) -> Dict[str, Any]:
    """
    Plot how often predictions stay unchanged as sparsity increases.

    Returns a dict with the plotted dataframe, a summary pivot table, and AUC stats.
    """
    _require_matplotlib()
    apply_matplotlib_style()
    if metrics_df.empty:
        print("metrics_df is empty; nothing to plot.")
        return {"match_df": None, "summary_table": None, "auc_table": None}

    match_cols = [c for c in metrics_df.columns if c.startswith(match_prefix)]
    if not match_cols:
        print("No prediction_profile match columns found in metrics.csv — rerun the evaluation with the new metric enabled.")
        return {"match_df": None, "summary_table": None, "auc_table": None}

    plot_records: List[Dict[str, Any]] = []
    for col in sorted(match_cols, key=lambda c: c.split("@", 1)[-1]):
        if "@" not in col:
            continue
        suffix = col.split("@", 1)[1]
        mode_tag = col.split(".")[1] if "." in col else "match"
        axis_kind = "fraction"
        try:
            if suffix.startswith("s="):
                level = float(suffix.split("=", 1)[1])
                axis_value = level * 100.0
            else:
                level = float(suffix)
                axis_value = level
                axis_kind = "count"
        except ValueError:
            continue

        grouped = (
            metrics_df.groupby(group_col)[col].mean()
            if group_col and group_col in metrics_df.columns
            else {None: metrics_df[col].mean()}
        )
        for expl_name, frac in grouped.items():
            plot_records.append(
                {
                    "explainer": expl_name or "all",
                    "level_value": axis_value,
                    "level_raw": level,
                    "fraction": frac,
                    "column": col,
                    "mode": mode_tag,
                    "axis_kind": axis_kind,
                }
            )

    if not plot_records:
        print("No valid match data to plot.")
        return {"match_df": None, "summary_table": None, "auc_table": None}

    match_df = pd.DataFrame(plot_records)
    axis_kind = "fraction" if (match_df["axis_kind"] == "fraction").all() else "count"

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.figure

    for expl_name, group in match_df.groupby("explainer"):
        ordered = group.sort_values("level_value")
        ax.plot(
            ordered["level_value"],
            ordered["fraction"] * 100.0,
            marker="o",
            label=expl_name,
        )

    if axis_kind == "fraction":
        ax.set_xlabel("Sparsity level (% of edges kept)")
    else:
        ax.set_xlabel("Top-k edges kept")
    ax.set_ylabel("Anchors with unchanged prediction (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    if match_df["explainer"].nunique() > 1:
        ax.legend(title="Explainer")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    plt.show()

    summary = (
        match_df.pivot_table(index="explainer", columns="level_value", values="fraction") * 100.0
    ).round(1)
    summary.columns = [
        (f"{col:.0f}%" if axis_kind == "fraction" else f"k={int(col)}") for col in summary.columns
    ]

    auc_rows = []
    for expl_name, group in match_df.groupby("explainer"):
        ordered = group.sort_values("level_raw")
        if ordered["level_raw"].nunique() < 2:
            area = float("nan")
            norm_area = float("nan")
        else:
            x = ordered["level_raw"].values
            y = ordered["fraction"].values
            area = float(np.trapz(y, x))
            span = float(x.max() - x.min())
            norm_area = (area / span) if span > 0 else float("nan")
        auc_rows.append(
            {
                "explainer": expl_name,
                "auc_raw": area,
                "auc_normalized_percent": norm_area * 100.0 if np.isfinite(norm_area) else float("nan"),
            }
        )
    auc_df = pd.DataFrame(auc_rows).set_index("explainer")

    if auto_display:
        print("Match rate table (% of anchors with unchanged prediction):")
        try:  # IPython friendly
            from IPython.display import display  # type: ignore
        except Exception:
            print(summary)
            print("\nNormalized area under stability curve (100% = perfect stability across the range)")
            print(auc_df)
        else:
            display(summary)
            print("\nNormalized area under stability curve (100% = perfect stability across the range)")
            display(auc_df)

    return {"match_df": match_df, "summary_table": summary, "auc_table": auc_df}


def plot_explainer_runtime(
    metrics_df: pd.DataFrame,
    *,
    time_col: str = "elapsed_sec",
    group_col: str = "explainer",
    palette: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (7, 4),
    annotate: bool = True,
) -> pd.DataFrame:
    """
    Plot average explanation time per explainer and return summary stats.
    """
    _require_seaborn()
    _require_matplotlib()
    apply_matplotlib_style()

    if time_col not in metrics_df.columns:
        print(f"Column '{time_col}' not found in metrics_df; skipping runtime plot.")
        return pd.DataFrame()
    if group_col not in metrics_df.columns:
        print(f"Column '{group_col}' not found in metrics_df; skipping runtime plot.")
        return pd.DataFrame()

    grouped = (
        metrics_df.groupby(group_col)[time_col]
        .agg(["mean", "median", "std", "count"])
        .sort_values("mean")
    )

    plt.figure(figsize=figsize)
    palette = palette or PLOT_COLORWAY
    ax = sns.barplot(
        data=grouped.reset_index(),
        x=group_col,
        y="mean",
        palette=palette,
    )
    ax.set_ylabel("Average explanation time (s)")
    ax.set_xlabel("Explainer")
    ax.set_title("Average runtime per explainer")
    ax.grid(axis="y", alpha=0.2)
    if annotate:
        for patch in ax.patches:
            height = patch.get_height()
            if not np.isfinite(height):
                continue
            ax.annotate(
                f"{height:.2f}s",
                (patch.get_x() + patch.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=PLOT_STYLE["annotation_size"],
            )
    plt.tight_layout()
    plt.show()
    return grouped


__all__ = [
    "MetricCurveSpec",
    "default_curve_specs",
    "read_tabs_plot",
    "collect_curve_columns",
    "levels_to_axis",
    "prepare_metrics_plotting",
    "plot_metric_curves",
    "plot_explainer_metric_summary",
    "plot_explainer_runtime",
    "plot_prediction_match_rate",
    "summarize_curve_results",
]
