from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from data.audit import extract_causal_triples, shuffle_timestamps
from debruijn.order2 import build_order2_debruijn, spectral_embedding_undirected


def infer_n_nodes(df: pd.DataFrame) -> int:
    """Infer the number of nodes from a tedge dataframe."""
    if not {"u", "v"} <= set(df.columns):
        raise ValueError("Expected columns {'u','v'}")
    return int(max(df["u"].max(), df["v"].max())) + 1


def make_temporal_clusters_labels(*, n_nodes: int, cluster_size: int = 10) -> np.ndarray:
    """Synthetic labels used by the tutorial dataset: y(i) = i // cluster_size."""
    nodes = np.arange(int(n_nodes), dtype=int)
    return (nodes // int(cluster_size)).astype(int)


def make_split(
    y: np.ndarray,
    *,
    test_frac: float = 0.3,
    seed: int = 0,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a reproducible (train_mask, test_mask) split.

    For the toy dataset we default to a **stratified** split so each class
    contributes ~test_frac of its nodes to the test set.
    """
    y = np.asarray(y, dtype=int)
    n = int(len(y))
    if not (0.0 < float(test_frac) < 1.0):
        raise ValueError("test_frac must be in (0,1)")

    rng = np.random.default_rng(int(seed))
    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    if stratify:
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            n_test = int(round(float(test_frac) * len(idx)))
            n_test = max(1, min(len(idx) - 1, n_test)) if len(idx) > 1 else 0
            test_idx = idx[:n_test]
            train_idx = idx[n_test:]
            test_mask[test_idx] = True
            train_mask[train_idx] = True
    else:
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = int(round(float(test_frac) * n))
        n_test = max(1, min(n - 1, n_test)) if n > 1 else 0
        test_mask[idx[:n_test]] = True
        train_mask[idx[n_test:]] = True

    # Sanity: disjoint, cover all nodes.
    if np.any(train_mask & test_mask):
        raise RuntimeError("train/test masks overlap")
    if not np.all(train_mask | test_mask):
        raise RuntimeError("train/test masks do not cover all nodes")

    return train_mask, test_mask


def _static_adjacency_counts(df: pd.DataFrame, *, n_nodes: int) -> np.ndarray:
    """Dense adjacency matrix with integer edge-event counts (ignoring time)."""
    u = df["u"].to_numpy(dtype=int)
    v = df["v"].to_numpy(dtype=int)
    A = np.zeros((int(n_nodes), int(n_nodes)), dtype=float)
    np.add.at(A, (u, v), 1.0)
    return A


def compute_static_degree_features(df: pd.DataFrame, *, n_nodes: int) -> np.ndarray:
    """Static baseline features: log(1+in_degree), log(1+out_degree)."""
    A = _static_adjacency_counts(df, n_nodes=n_nodes)
    out_deg = A.sum(axis=1)
    in_deg = A.sum(axis=0)
    X = np.stack([np.log1p(in_deg), np.log1p(out_deg)], axis=1)
    return X.astype(float)


def _pad_cols(X: np.ndarray, k: int) -> np.ndarray:
    """Pad feature matrix with zeros so it has exactly k columns."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] == int(k):
        return X
    out = np.zeros((X.shape[0], int(k)), dtype=float)
    out[:, : min(X.shape[1], int(k))] = X[:, : min(X.shape[1], int(k))]
    return out


def compute_static_spectral_features(df: pd.DataFrame, *, n_nodes: int, k: int = 5) -> np.ndarray:
    """Static baseline: spectral embedding of the symmetrized static graph."""
    A = _static_adjacency_counts(df, n_nodes=n_nodes)
    emb = spectral_embedding_undirected(A, k=int(k))
    return _pad_cols(emb, int(k))


def compute_debruijn_spectral_node_features(
    df: pd.DataFrame,
    *,
    n_nodes: int,
    delta: int = 1,
    k: int = 5,
    aggregate: str = "end",
) -> np.ndarray:
    """Second-order baseline: spectral embedding of the order-2 De Bruijn graph.

    Steps:
      1) Extract causal triples (u, v, w) from the temporal edge list.
      2) Build the order-2 De Bruijn graph over pair states (u, v).
      3) Spectrally embed the (symmetrized) pair graph.
      4) Aggregate pair-state embeddings back to first-order nodes.

    Args:
        df: Temporal edge list with columns (u,v,t).
        n_nodes: Number of first-order nodes.
        delta: Maximum time difference for a causal length-2 path.
        k: Number of spectral dimensions to keep.
        aggregate: "end" aggregates by the second element of (u,v), i.e. the
            current node v. "start" aggregates by u. "both" averages the two.
    """
    triples, _ = extract_causal_triples(df, delta=int(delta))
    g2 = build_order2_debruijn(triples)

    if len(g2.pairs) == 0:
        return np.zeros((int(n_nodes), int(k)), dtype=float)

    emb_pairs = spectral_embedding_undirected(g2.adjacency.astype(float), k=int(k))
    emb_pairs = _pad_cols(emb_pairs, int(k))

    X = np.zeros((int(n_nodes), int(k)), dtype=float)
    counts = np.zeros((int(n_nodes),), dtype=float)

    def _add(idx: int, vec: np.ndarray):
        X[idx] += vec
        counts[idx] += 1.0

    mode = str(aggregate).lower().strip()
    for i, (u, v) in enumerate(g2.pairs):
        if mode == "end":
            _add(int(v), emb_pairs[i])
        elif mode == "start":
            _add(int(u), emb_pairs[i])
        elif mode == "both":
            _add(int(u), emb_pairs[i])
            _add(int(v), emb_pairs[i])
        else:
            raise ValueError("aggregate must be one of {'end','start','both'}")

    X = np.divide(X, counts[:, None], out=np.zeros_like(X), where=counts[:, None] > 0)
    return X


def _standardize(X: np.ndarray, train_mask: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Z-score standardization using only training nodes."""
    X = np.asarray(X, dtype=float)
    mu = X[train_mask].mean(axis=0, keepdims=True)
    sd = X[train_mask].std(axis=0, keepdims=True)
    sd = np.where(sd < float(eps), 1.0, sd)
    return (X - mu) / sd


def _fit_lr_and_score(
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    seed: int,
    C: float = 1.0,
    max_iter: int = 5000,
) -> Tuple[float, float]:
    """Fit multinomial logistic regression and return (acc, balanced_acc)."""
    Xs = _standardize(X, train_mask)

    clf = LogisticRegression(
        max_iter=int(max_iter),
        multi_class="multinomial",
        C=float(C),
        solver="lbfgs",
        random_state=int(seed),
    )
    clf.fit(Xs[train_mask], y[train_mask])
    pred = clf.predict(Xs[test_mask])

    acc = float(accuracy_score(y[test_mask], pred))
    bacc = float(balanced_accuracy_score(y[test_mask], pred))
    return acc, bacc


def run_phase_c_baselines(
    df: pd.DataFrame,
    *,
    n_nodes: Optional[int] = None,
    cluster_size: int = 10,
    delta: int = 1,
    test_frac: float = 0.3,
    split_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    stratify: bool = True,
    k_static: int = 5,
    k_debruijn: int = 5,
    include_static_spectral: bool = True,
    include_shuffled_debruijn: bool = False,
    shuffle_seed: int = 0,
) -> pd.DataFrame:
    """Run the Phase C baseline suite and return a compact results table."""
    if n_nodes is None:
        n_nodes = infer_n_nodes(df)

    y = make_temporal_clusters_labels(n_nodes=int(n_nodes), cluster_size=int(cluster_size))

    # Precompute features (same across splits).
    X_deg = compute_static_degree_features(df, n_nodes=int(n_nodes))
    X_static_spec = (
        compute_static_spectral_features(df, n_nodes=int(n_nodes), k=int(k_static))
        if include_static_spectral
        else None
    )
    X_db = compute_debruijn_spectral_node_features(
        df,
        n_nodes=int(n_nodes),
        delta=int(delta),
        k=int(k_debruijn),
        aggregate="end",
    )

    X_db_shuf = None
    if include_shuffled_debruijn:
        df_shuf = shuffle_timestamps(df, seed=int(shuffle_seed))
        X_db_shuf = compute_debruijn_spectral_node_features(
            df_shuf,
            n_nodes=int(n_nodes),
            delta=int(delta),
            k=int(k_debruijn),
            aggregate="end",
        )

    rows = []
    for seed in split_seeds:
        train_mask, test_mask = make_split(y, test_frac=float(test_frac), seed=int(seed), stratify=bool(stratify))

        acc, bacc = _fit_lr_and_score(X_deg, y, train_mask, test_mask, seed=int(seed))
        rows.append(
            {
                "baseline": "static_degree_lr",
                "split_seed": int(seed),
                "acc": acc,
                "balanced_acc": bacc,
            }
        )

        if X_static_spec is not None:
            acc, bacc = _fit_lr_and_score(X_static_spec, y, train_mask, test_mask, seed=int(seed))
            rows.append(
                {
                    "baseline": "static_spectral_lr",
                    "split_seed": int(seed),
                    "acc": acc,
                    "balanced_acc": bacc,
                }
            )

        acc, bacc = _fit_lr_and_score(X_db, y, train_mask, test_mask, seed=int(seed))
        rows.append(
            {
                "baseline": "debruijn_spectral_lr",
                "split_seed": int(seed),
                "acc": acc,
                "balanced_acc": bacc,
            }
        )

        if X_db_shuf is not None:
            acc, bacc = _fit_lr_and_score(X_db_shuf, y, train_mask, test_mask, seed=int(seed))
            rows.append(
                {
                    "baseline": "debruijn_spectral_lr__shuffled_times",
                    "split_seed": int(seed),
                    "acc": acc,
                    "balanced_acc": bacc,
                }
            )

    return pd.DataFrame(rows)
