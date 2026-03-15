from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from eval.summary import add_fidelity_columns, summarize_benchmark
from .palette import EDGE_GRAY, color_for_index


def _cycle_colors(n: int) -> list[str]:
    return [color_for_index(i) for i in range(int(max(0, n)))]


def plot_metric_curves(
    df: pd.DataFrame,
    metric: str,
    *,
    x: str = "frac",
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
):
    """Line plot: metric vs x (typically frac) with one line per explainer."""

    piv = df.pivot_table(index=x, columns="explainer", values=metric, aggfunc="mean").sort_index()
    ax = piv.plot(marker="o", color=_cycle_colors(piv.shape[1]))
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or metric)
    ax.axhline(0, linewidth=1, color=EDGE_GRAY)
    plt.show()
    return ax


def plot_explain_time(df: pd.DataFrame, *, title: str = "Mean explanation time per node"):
    """Bar plot of mean explain_time_s per explainer."""

    t = (
        df.groupby(["explainer", "node"], sort=False)["explain_time_s"]
        .first()
        .groupby("explainer")
        .mean()
        .sort_values()
    )
    ax = t.plot(kind="bar", color=_cycle_colors(len(t)))
    ax.set_ylabel("seconds")
    ax.set_title(title)
    plt.show()
    return ax


def plot_fidelity_sparsity(df: pd.DataFrame, *, title: str = "Fidelity–sparsity curve (keep)"):
    """Plot keep-fidelity vs sparsity (=1-frac).

    We use a bounded agreement-style fidelity:
        fidelity_keep_agreement = 1 - |p_keep - p0|

    This avoids a common pitfall where p_keep > p0 is clipped to 1.0, making
    random explainers look artificially perfect.
    """

    df2 = add_fidelity_columns(df)
    piv = (
        df2.pivot_table(
            index="sparsity",
            columns="explainer",
            values="fidelity_keep_agreement",
            aggfunc="mean",
        )
        .sort_index()
    )
    ax = piv.plot(marker="o", color=_cycle_colors(piv.shape[1]))
    ax.set_xlabel("sparsity (1 - frac)")
    ax.set_ylabel("keep-fidelity (1 - |p_keep - p0|)")
    ax.set_title(title)
    plt.show()
    return ax


def plot_fidelity_sparsity_margin(
    df: pd.DataFrame,
    *,
    title: str = "Fidelity–sparsity curve (keep, margin agreement)",
):
    """Plot margin-based keep fidelity vs sparsity.

    Uses:
        fidelity_keep_margin_agreement = 1 - |m_keep - m0| / |m0|

    This is often more discriminative than probability-based fidelities when
    p0 is saturated close to 1.0.
    """

    df2 = add_fidelity_columns(df)
    if "fidelity_keep_margin_agreement" not in df2.columns:
        raise ValueError("Margin columns not available in df (need margin0/margin_keep/margin_drop).")

    piv = (
        df2.pivot_table(
            index="sparsity",
            columns="explainer",
            values="fidelity_keep_margin_agreement",
            aggfunc="mean",
        )
        .sort_index()
    )
    ax = piv.plot(marker="o", color=_cycle_colors(piv.shape[1]))
    ax.set_xlabel("sparsity (1 - frac)")
    ax.set_ylabel("keep-fidelity (margin agreement)")
    ax.set_title(title)
    plt.show()
    return ax


def plot_deletion_curve(df: pd.DataFrame, *, title: str = "Deletion curve (relative drop)"):
    """Plot fidelity_drop_ratio vs frac (fraction of top edges removed)."""

    df2 = add_fidelity_columns(df)
    piv = (
        df2.pivot_table(
            index="frac",
            columns="explainer",
            values="fidelity_drop_ratio",
            aggfunc="mean",
        )
        .sort_index()
    )
    ax = piv.plot(marker="o", color=_cycle_colors(piv.shape[1]))
    ax.set_xlabel("frac (top edges removed)")
    ax.set_ylabel("(p0 - p_drop) / p0")
    ax.set_title(title)
    ax.axhline(0, linewidth=1, color=EDGE_GRAY)
    plt.show()
    return ax


def plot_deletion_curve_margin(df: pd.DataFrame, *, title: str = "Deletion curve (margin drop ratio)"):
    """Plot (m0 - m_drop)/|m0| vs frac.

    This can be a better indicator of deletion faithfulness when probabilities
    saturate.
    """

    df2 = add_fidelity_columns(df)
    if "fidelity_drop_margin_ratio" not in df2.columns:
        raise ValueError("Margin columns not available in df (need margin0/margin_keep/margin_drop).")

    piv = (
        df2.pivot_table(
            index="frac",
            columns="explainer",
            values="fidelity_drop_margin_ratio",
            aggfunc="mean",
        )
        .sort_index()
    )
    ax = piv.plot(marker="o", color=_cycle_colors(piv.shape[1]))
    ax.set_xlabel("frac (top edges removed)")
    ax.set_ylabel("(m0 - m_drop) / |m0|")
    ax.set_title(title)
    ax.axhline(0, linewidth=1, color=EDGE_GRAY)
    plt.show()
    return ax


def plot_auc_bars(df: pd.DataFrame, *, title: str = "AUFSC (AUC of fidelity–sparsity curve)"):
    """Bar plot of AUFSC per explainer (averaged across nodes)."""

    summary = summarize_benchmark(df)
    col = "AUFSC_mean" if "AUFSC_mean" in summary.columns else "auc_fidelity_sparsity_mean"
    s = summary.set_index("explainer")[col].sort_values()
    ax = s.plot(kind="bar", color=_cycle_colors(len(s)))
    ax.set_ylabel("AUFSC (AUC of 1 - |p_keep - p0|)")
    ax.set_title(title)
    plt.show()
    return ax


def plot_auc_bars_margin(df: pd.DataFrame, *, title: str = "AUFSC (margin agreement)"):
    """Bar plot of margin-based AUFSC per explainer."""

    summary = summarize_benchmark(df)
    if "AUFSC_margin_mean" not in summary.columns:
        raise ValueError("Margin AUFSC not available (need margin columns during benchmark).")

    s = summary.set_index("explainer")["AUFSC_margin_mean"].sort_values()
    ax = s.plot(kind="bar", color=_cycle_colors(len(s)))
    ax.set_ylabel("AUFSC_margin")
    ax.set_title(title)
    plt.show()
    return ax


def plot_faithfulness_bars(df: pd.DataFrame, *, title: str = "Faithfulness (deletion AUC)"):
    """Bar plot of deletion-based faithfulness per explainer."""

    summary = summarize_benchmark(df)
    col = "Faithfulness_mean" if "Faithfulness_mean" in summary.columns else "faithfulness_mean"
    s = summary.set_index("explainer")[col].sort_values()
    ax = s.plot(kind="bar", color=_cycle_colors(len(s)))
    ax.set_ylabel("faithfulness (normalized trapz)")
    ax.set_title(title)
    plt.show()
    return ax


def plot_gt_ap_bars(df: pd.DataFrame, *, title: str = "GT agreement (Average Precision)"):
    """Bar plot of GT Average Precision (AP) per explainer.

    Requires that the benchmark dataframe contains a `gt_ap` column.
    """

    summary = summarize_benchmark(df)
    if "GT_AP_mean" not in summary.columns:
        raise ValueError("GT_AP_mean not available (need gt_ap column in benchmark df).")

    s = summary.set_index("explainer")["GT_AP_mean"].sort_values()
    ax = s.plot(kind="bar", color=_cycle_colors(len(s)))
    ax.set_ylabel("Average Precision (AP)")
    ax.set_title(title)
    plt.show()
    return ax


def plot_faithfulness_bars_margin(df: pd.DataFrame, *, title: str = "Faithfulness (margin deletion AUC)"):
    """Bar plot of margin-based faithfulness per explainer."""

    summary = summarize_benchmark(df)
    if "Faithfulness_margin_mean" not in summary.columns:
        raise ValueError("Margin faithfulness not available (need margin columns during benchmark).")

    s = summary.set_index("explainer")["Faithfulness_margin_mean"].sort_values()
    ax = s.plot(kind="bar", color=_cycle_colors(len(s)))
    ax.set_ylabel("faithfulness_margin")
    ax.set_title(title)
    plt.show()
    return ax


def plot_all(df: pd.DataFrame):
    """Convenience: common plots for quick inspection."""

    # Probability-space
    plot_metric_curves(df, "sufficiency_gap", ylabel="p0 - p_keep", title="Sufficiency gap (probability)")
    plot_metric_curves(df, "comprehensiveness_gap", ylabel="p0 - p_drop", title="Comprehensiveness gap (probability)")
    plot_explain_time(df)
    plot_fidelity_sparsity(df)
    plot_auc_bars(df)
    plot_deletion_curve(df)
    plot_faithfulness_bars(df)

    if "gt_ap" in df.columns:
        plot_gt_ap_bars(df)

    # Margin-space (more informative when p0 saturates)
    if "margin0" in df.columns and "margin_keep" in df.columns:
        plot_metric_curves(df, "sufficiency_gap_margin", ylabel="m0 - m_keep", title="Sufficiency gap (margin)")
        plot_metric_curves(df, "comprehensiveness_gap_margin", ylabel="m0 - m_drop", title="Comprehensiveness gap (margin)")
        plot_fidelity_sparsity_margin(df)
        plot_auc_bars_margin(df)
        plot_deletion_curve_margin(df)
        plot_faithfulness_bars_margin(df)
