from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np

from .palette import EDGE_GRAY, continuous_cmap, discrete_cmap


@dataclass(frozen=True)
class BlockBoundaries:
    """Helper for drawing boundary lines between grouped nodes."""

    boundaries: np.ndarray


def group_boundaries(labels: np.ndarray) -> BlockBoundaries:
    """Return indices where a sorted label vector changes.

    Args:
        labels: sorted integer labels.

    Returns:
        BlockBoundaries with `boundaries` containing the cut positions (excluding 0 and n).
    """
    if len(labels) == 0:
        return BlockBoundaries(boundaries=np.zeros((0,), dtype=int))
    cuts = np.nonzero(labels[1:] != labels[:-1])[0] + 1
    return BlockBoundaries(boundaries=cuts.astype(int))


def plot_adjacency_heatmap(
    A: np.ndarray,
    *,
    title: str,
    ax: Optional[plt.Axes] = None,
    log1p: bool = True,
    boundaries: Optional[BlockBoundaries] = None,
):
    """Plot a dense adjacency matrix as an image."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    M = np.log1p(A) if log1p else A
    im = ax.imshow(M, interpolation="nearest", cmap=continuous_cmap())
    ax.set_title(title)
    ax.set_xlabel("destination (sorted)")
    ax.set_ylabel("source (sorted)")

    if boundaries is not None and len(boundaries.boundaries):
        for b in boundaries.boundaries:
            ax.axhline(b - 0.5, linewidth=0.8, color=EDGE_GRAY)
            ax.axvline(b - 0.5, linewidth=0.8, color=EDGE_GRAY)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_block_matrix(
    B: np.ndarray,
    *,
    title: str,
    ax: Optional[plt.Axes] = None,
    normalize: bool = True,
):
    """Plot a small block matrix (e.g. 9x9) as an image."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    M = B.astype(float)
    if normalize and M.sum() > 0:
        M = M / M.sum()

    im = ax.imshow(M, interpolation="nearest", cmap=continuous_cmap())
    ax.set_title(title)
    ax.set_xlabel("dst group")
    ax.set_ylabel("src group")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_embedding(
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    ax: Optional[plt.Axes] = None,
):
    """Scatter plot for 2D embeddings."""
    if coords.shape[1] != 2:
        raise ValueError("coords must have shape [N, 2]")
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    labels = np.asarray(labels).reshape(-1)
    if labels.shape[0] != coords.shape[0]:
        raise ValueError("labels must have the same length as coords")

    uniq = np.unique(labels)
    lut = {v: i for i, v in enumerate(uniq.tolist())}
    encoded = np.asarray([lut[v] for v in labels], dtype=int)
    cmap = discrete_cmap(len(uniq))

    sc = ax.scatter(coords[:, 0], coords[:, 1], c=encoded, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("eigenvector 1")
    ax.set_ylabel("eigenvector 2")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks(np.arange(len(uniq)))
    cbar.set_ticklabels([str(v) for v in uniq.tolist()])
    return ax
