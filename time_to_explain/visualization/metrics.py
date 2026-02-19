from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import PLOT_COLORWAY, apply_matplotlib_style

DEFAULT_METRIC_LABELS: Dict[str, str] = {
    "fidelity_minus.value": "Fidelity- (drop explanation edges)",
    "fidelity_plus.value": "Fidelity+ (explanation only)",
    "sparsity.ratio": "Sparsity (|E_expl|/|E_candidates|)",
    "aufsc.value": "AUFSC",
    "elapsed_sec": "Runtime (s)",
}


def _resolve_metric_spec(
    entry: Union[str, Tuple[str, str], List[str]],
    *,
    metric_labels: Dict[str, str],
) -> Tuple[str, str]:
    if isinstance(entry, (tuple, list)) and len(entry) == 2:
        label, metric_id = entry
        return str(label), str(metric_id)
    metric_id = str(entry)
    label = metric_labels.get(metric_id, metric_id.replace("_", " ").title())
    return label, metric_id


def filter_explainers(
    metrics_df: pd.DataFrame,
    *,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
) -> pd.DataFrame:
    if "explainer" not in metrics_df.columns:
        raise KeyError("metrics_df must contain an 'explainer' column.")
    out = metrics_df.copy()
    if include:
        include_set = {str(x) for x in include}
        out = out[out["explainer"].astype(str).isin(include_set)]
    if exclude:
        exclude_set = {str(x) for x in exclude}
        out = out[~out["explainer"].astype(str).isin(exclude_set)]
    return out


def prepare_metrics_plotting(
    metrics_df: pd.DataFrame,
    *,
    explainer_order: Sequence[str] | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    if "explainer" not in metrics_df.columns:
        raise KeyError("metrics_df must contain an 'explainer' column.")
    order = list(explainer_order or sorted(metrics_df["explainer"].astype(str).unique()))
    out = metrics_df.copy()
    out["explainer"] = pd.Categorical(out["explainer"].astype(str), categories=order, ordered=True)
    out = out.sort_values("explainer")
    palette = list(PLOT_COLORWAY)
    palette = (palette * (len(order) // len(palette) + 1))[: len(order)]
    return out, palette


def _aggregate_metrics(
    metrics_df: pd.DataFrame,
    *,
    group_col: str,
    metric_columns: Sequence[str],
    agg: str,
) -> pd.DataFrame:
    return metrics_df.groupby(group_col, dropna=False)[list(metric_columns)].agg(agg).reset_index()


def _plot_bar_chart(
    ax: Any,
    *,
    labels: Sequence[str],
    values: Sequence[float],
    colors: Sequence[str],
    title: str,
    ylabel: str,
) -> None:
    ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("explainer")
    ax.tick_params(axis="x", rotation=45)


def plot_explainer_metric_summary(
    metrics_df: pd.DataFrame,
    *,
    metric_columns: Sequence[Tuple[str, str]],
    agg: str = "mean",
    palette: Sequence[str] | None = None,
    group_col: str = "explainer",
    show: bool = True,
) -> pd.DataFrame:
    apply_matplotlib_style()
    metric_ids = [col for _, col in metric_columns]
    summary = _aggregate_metrics(metrics_df, group_col=group_col, metric_columns=metric_ids, agg=agg)

    colors = list(palette or PLOT_COLORWAY)
    if len(colors) < len(summary):
        colors = (colors * (len(summary) // len(colors) + 1))[: len(summary)]

    fig, axes = plt.subplots(1, max(1, len(metric_columns)), figsize=(4 * max(1, len(metric_columns)), 4))
    axes_list = axes if isinstance(axes, (list, np.ndarray)) else [axes]

    for ax, (label, col) in zip(axes_list, metric_columns):
        values = summary[col].to_numpy(dtype=float)
        labels = summary[group_col].astype(str).tolist()
        _plot_bar_chart(
            ax,
            labels=labels,
            values=values,
            colors=colors,
            title=label,
            ylabel=col,
        )

    fig.tight_layout()
    if show:
        plt.show()

    return summary


def plot_explainer_runtime(
    metrics_df: pd.DataFrame,
    *,
    palette: Sequence[str] | None = None,
    group_col: str = "explainer",
    agg: str = "mean",
) -> pd.DataFrame:
    return plot_explainer_metric_summary(
        metrics_df,
        metric_columns=[("Runtime (s)", "elapsed_sec")],
        agg=agg,
        palette=palette,
        group_col=group_col,
    )


def plot_prediction_match_rate(
    metrics_df: pd.DataFrame,
    *,
    palette: Sequence[str] | None = None,
    group_col: str = "explainer",
    agg: str = "mean",
    prefix: str = "prediction_profile.match_",
) -> pd.DataFrame:
    match_cols = [c for c in metrics_df.columns if c.startswith(prefix)]
    if not match_cols:
        raise KeyError("No prediction_profile.match_ columns found in metrics_df.")
    temp = metrics_df.copy()
    temp["match_rate"] = temp[match_cols].mean(axis=1)
    return plot_explainer_metric_summary(
        temp,
        metric_columns=[("Prediction match rate", "match_rate")],
        agg=agg,
        palette=palette,
        group_col=group_col,
    )


def plot_selected_metrics(
    metrics_df: pd.DataFrame,
    metrics: Sequence[Union[str, Tuple[str, str]]],
    *,
    palette: Sequence[str] | None = None,
    group_col: str = "explainer",
    metric_labels: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    labels = dict(DEFAULT_METRIC_LABELS)
    if metric_labels:
        labels.update(metric_labels)

    metric_columns: List[Tuple[str, str]] = []
    missing: List[str] = []
    for entry in metrics:
        label, metric_id = _resolve_metric_spec(entry, metric_labels=labels)
        if metric_id in metrics_df.columns:
            metric_columns.append((label, metric_id))
        else:
            missing.append(metric_id)

    results: Dict[str, Any] = {}
    if metric_columns:
        summary = plot_explainer_metric_summary(
            metrics_df,
            metric_columns=metric_columns,
            palette=palette,
            group_col=group_col,
        )
        results["summary"] = summary

    if missing:
        print("Skipped missing metrics:", ", ".join(missing))

    return results


__all__ = [
    "DEFAULT_METRIC_LABELS",
    "filter_explainers",
    "prepare_metrics_plotting",
    "plot_explainer_metric_summary",
    "plot_explainer_runtime",
    "plot_prediction_match_rate",
    "plot_selected_metrics",
]
