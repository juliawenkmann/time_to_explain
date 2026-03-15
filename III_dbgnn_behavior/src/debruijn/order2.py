from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


Pair = Tuple[int, int]


@dataclass(frozen=True)
class Order2DeBruijn:
    """A tiny, inspectable order-2 De Bruijn graph built from (u,v,w) triples.

    Nodes are ordered pairs (u, v).
    Each observed causal triple (u, v, w) induces a directed transition:

        (u, v)  ->  (v, w)

    We store both a dense adjacency matrix (counts) and the usual edge list
    representation (edge_index + edge_weight).
    """

    pairs: List[Pair]
    """Index -> (u, v) pair."""

    edge_index: np.ndarray
    """Shape [2, E] (src_idx, dst_idx)."""

    edge_weight: np.ndarray
    """Shape [E] with integer counts."""

    adjacency: np.ndarray
    """Shape [N, N] dense count matrix."""

    base: int
    """Encoding base used for pairs: code = u*base + v."""


def build_order2_debruijn(triples: np.ndarray) -> Order2DeBruijn:
    """Build an order-2 De Bruijn graph from a list/array of (u, v, w) triples."""
    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError("triples must have shape [K, 3]")

    if len(triples) == 0:
        return Order2DeBruijn(
            pairs=[],
            edge_index=np.zeros((2, 0), dtype=int),
            edge_weight=np.zeros((0,), dtype=int),
            adjacency=np.zeros((0, 0), dtype=int),
            base=0,
        )

    u = triples[:, 0].astype(int)
    v = triples[:, 1].astype(int)
    w = triples[:, 2].astype(int)

    base = int(triples.max()) + 1
    src_code = u * base + v
    dst_code = v * base + w

    # Build a compact index over all observed pair states.
    all_codes = np.concatenate([src_code, dst_code])
    uniq_codes, inv = np.unique(all_codes, return_inverse=True)
    n = int(len(uniq_codes))

    src_idx = inv[: len(src_code)].astype(np.int64)
    dst_idx = inv[len(src_code) :].astype(np.int64)

    # Count transitions between pair states.
    edge_code = src_idx * n + dst_idx
    uniq_edge_code, counts = np.unique(edge_code, return_counts=True)
    src_e = (uniq_edge_code // n).astype(int)
    dst_e = (uniq_edge_code % n).astype(int)

    edge_index = np.stack([src_e, dst_e], axis=0)
    edge_weight = counts.astype(int)

    adjacency = np.zeros((n, n), dtype=int)
    adjacency[src_e, dst_e] = edge_weight

    pairs: List[Pair] = [(int(code // base), int(code % base)) for code in uniq_codes]

    return Order2DeBruijn(
        pairs=pairs,
        edge_index=edge_index,
        edge_weight=edge_weight,
        adjacency=adjacency,
        base=base,
    )


def pair_cluster(pair: Pair, *, cluster_size: int = 10) -> Tuple[int, int]:
    """Return (cluster(u), cluster(v)) for a pair (u, v)."""
    u, v = pair
    return int(u) // int(cluster_size), int(v) // int(cluster_size)


def pair_group_id(
    pair: Pair,
    *,
    n_clusters: int = 3,
    cluster_size: int = 10,
) -> int:
    """Return a single group id in {0..n_clusters^2-1} for (cluster(u), cluster(v))."""
    cu, cv = pair_cluster(pair, cluster_size=cluster_size)
    if not (0 <= cu < n_clusters and 0 <= cv < n_clusters):
        return -1
    return int(cu) * int(n_clusters) + int(cv)


def pair_group_labels(
    pairs: Iterable[Pair],
    *,
    n_clusters: int = 3,
    cluster_size: int = 10,
) -> np.ndarray:
    """Vectorized pair_group_id for a list of pairs."""
    return np.asarray(
        [pair_group_id(p, n_clusters=n_clusters, cluster_size=cluster_size) for p in pairs],
        dtype=int,
    )


def sort_pairs_by_group(
    pairs: List[Pair],
    *,
    n_clusters: int = 3,
    cluster_size: int = 10,
) -> np.ndarray:
    """Return an index order that groups pairs by (cluster(u), cluster(v))."""
    if not pairs:
        return np.zeros((0,), dtype=int)
    u = np.asarray([p[0] for p in pairs], dtype=int)
    v = np.asarray([p[1] for p in pairs], dtype=int)
    g = pair_group_labels(pairs, n_clusters=n_clusters, cluster_size=cluster_size)
    # Primary key: group id; then u; then v (stable-ish, deterministic).
    return np.lexsort((v, u, g))


def reorder_adjacency(A: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Return A with rows/cols permuted by `order`."""
    if A.size == 0:
        return A
    return A[np.ix_(order, order)]


def block_matrix_from_edges(
    *,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    node_groups: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """Aggregate edge weights into a (n_groups x n_groups) block matrix."""
    if edge_index.size == 0:
        return np.zeros((n_groups, n_groups), dtype=float)
    src = edge_index[0].astype(int)
    dst = edge_index[1].astype(int)
    gs = node_groups[src]
    gd = node_groups[dst]

    B = np.zeros((int(n_groups), int(n_groups)), dtype=float)
    np.add.at(B, (gs, gd), edge_weight.astype(float))
    return B


def spectral_embedding_undirected(
    A: np.ndarray,
    *,
    k: int = 2,
    eps: float = 1e-12,
) -> np.ndarray:
    """Simple spectral embedding of the symmetrized adjacency A + A^T.

    Returns the first k non-trivial eigenvectors of the normalized Laplacian.
    This is meant for small graphs (N <= ~1k) where a dense eigendecomposition is OK.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    n = int(A.shape[0])
    if n == 0:
        return np.zeros((0, k), dtype=float)

    W = A.astype(float) + A.astype(float).T
    deg = W.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, eps))
    D_inv_sqrt = np.diag(inv_sqrt)
    L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

    # eigh returns eigenvalues in ascending order.
    vals, vecs = np.linalg.eigh(L)

    # Skip the first (approximately) constant eigenvector.
    start = 1 if n > 1 else 0
    end = min(start + int(k), n)
    return vecs[:, start:end]
