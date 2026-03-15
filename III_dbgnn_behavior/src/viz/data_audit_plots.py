
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .palette import EVENT_BLUE, SNAPSHOT_ORANGE, continuous_cmap


def plot_cluster_matrix(
    mat: np.ndarray,
    *,
    title: str = "Static cluster mixing (fractions)",
    cluster_labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    annotate: bool = True,
):
    """Heatmap for a small cluster mixing matrix."""
    mat = np.asarray(mat)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("mat must be a square 2D array")

    n = mat.shape[0]
    if cluster_labels is None:
        cluster_labels = [f"C{i}" for i in range(n)]

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    im = ax.imshow(mat, interpolation="nearest", cmap=continuous_cmap())
    ax.set_title(title)
    ax.set_xlabel("target cluster (v)")
    ax.set_ylabel("source cluster (u)")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cluster_labels)
    ax.set_yticklabels(cluster_labels)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if annotate:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center")

    plt.show()
    return ax


def plot_causal_stay_probs(
    stats_df: pd.DataFrame,
    *,
    title: str = "Temporal signal: stay-in-cluster probability",
    ax: Optional[plt.Axes] = None,
):
    """Compare key causal stats for real vs shuffled.

    Expects stats_df with index ["real","shuffled"] and columns:
      - p_stay_given_prev_within
      - p_stay_given_prev_cross
    """
    required = {"p_stay_given_prev_within", "p_stay_given_prev_cross"}
    missing = required - set(stats_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = stats_df.loc[:, ["p_stay_given_prev_within", "p_stay_given_prev_cross"]].copy()
    df.columns = ["stay | prev within", "stay | prev cross"]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    df.plot(kind="bar", ax=ax, color=[EVENT_BLUE, SNAPSHOT_ORANGE])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("probability")
    ax.set_title(title)
    ax.legend(loc="best")
    plt.xticks(rotation=0)
    plt.show()
    return ax
