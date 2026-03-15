"""Core CBM utilities extracted from notebooks/cbm.ipynb."""


# --- Imports + helper utilities ---
from __future__ import annotations

import inspect
import re
from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import networkx as nx
import warnings

from scipy import sparse
from difflib import SequenceMatcher
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

warnings.filterwarnings(
    "ignore",
    message=".*'penalty' was deprecated.*",
    category=FutureWarning,
)

# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------
def to_numpy(x):
    """Best-effort conversion: torch.Tensor / list / np -> np.ndarray."""
    if x is None:
        return None
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def get_asset(assets, key, default=None):
    """assets may be dict-like or object-like; support both."""
    if assets is None:
        return default
    if isinstance(assets, dict):
        return assets.get(key, default)
    return getattr(assets, key, default)

def get_any_asset(assets, keys: List[str], default=None):
    for k in keys:
        v = get_asset(assets, k, None)
        if v is not None:
            return v
    return default

def stable_coalesce_edges(edge_index: np.ndarray, edge_weight: Optional[np.ndarray] = None):
    """
    Coalesce duplicate edges by summing weights, keeping *first occurrence* order stable.
    edge_index: [2, E]
    edge_weight: [E] or None (treated as ones)
    Returns:
      edge_index_u: [2, E_unique]
      edge_weight_u: [E_unique]
    """
    edge_index = np.asarray(edge_index, dtype=int)
    if edge_index.size == 0:
        return edge_index.reshape(2, 0), np.zeros((0,), dtype=np.float32)

    E = edge_index.shape[1]
    if edge_weight is None:
        edge_weight = np.ones(E, dtype=np.float32)
    else:
        edge_weight = np.asarray(edge_weight, dtype=np.float32).reshape(-1)
        assert edge_weight.shape[0] == E

    mapping: Dict[Tuple[int, int], int] = {}
    uniq_edges: List[Tuple[int, int]] = []
    weights: List[float] = []

    for (u, v), w in zip(edge_index.T.tolist(), edge_weight.tolist()):
        key = (int(u), int(v))
        if key in mapping:
            weights[mapping[key]] += float(w)
        else:
            mapping[key] = len(uniq_edges)
            uniq_edges.append(key)
            weights.append(float(w))

    ei = np.asarray(uniq_edges, dtype=int).T
    ew = np.asarray(weights, dtype=np.float32)
    return ei, ew


# ---------------------------------------------------------------------
# Core container: first- and second-order graphs
# ---------------------------------------------------------------------
@dataclass
class FirstSecondOrderGraphs:
    """Holds the first-order graph g and second-order transition graph g2."""
    n_nodes: int                          # number of nodes in INTERNAL node-id space
    g_edge_index: np.ndarray              # [2, E1] edges in INTERNAL node-id space
    g_edge_weight: np.ndarray             # [E1]
    g2_node_ids: np.ndarray               # [T, 2] (u, v) per edge-token, in INTERNAL node-id space
    g2_edge_index: np.ndarray             # [2, E2] transitions between edge-tokens (token indices)
    g2_edge_weight: np.ndarray            # [E2]


# ---------------------------------------------------------------------
# Optional: build g and g2 from raw temporal edges (src, dst, t)
# ---------------------------------------------------------------------
def build_first_second_order_from_temporal_edges(
    src: np.ndarray,
    dst: np.ndarray,
    t: np.ndarray,
    n_nodes: int,
    delta: int = 1,
) -> FirstSecondOrderGraphs:
    """
    Build:
      - g: first-order edge counts
      - g2: second-order De Bruijn transition counts for time-respecting 2-step paths

    We create a transition (u->v at time t1) -> (v->w at time t2) if:
      - t2 > t1
      - t2 - t1 <= delta
      - center node overlaps: v == v
    """
    src = np.asarray(src, dtype=int)
    dst = np.asarray(dst, dtype=int)
    t = np.asarray(t, dtype=int)

    order = np.argsort(t)
    src = src[order]
    dst = dst[order]
    t = t[order]

    # First-order edges: frequencies of observed edges
    edge_counter = Counter(zip(src, dst))
    edge_types = list(edge_counter.keys())  # keep insertion order from Counter? -> not guaranteed
    # Make ordering stable across runs:
    edge_types = sorted(edge_types)

    type_to_tok = {e: i for i, e in enumerate(edge_types)}
    g_edge_index = np.array(edge_types, dtype=int).T  # [2, E1]
    g_edge_weight = np.array([edge_counter[e] for e in edge_types], dtype=np.float32)

    # Second-order edges: transitions between edge-types along time-respecting 2-step paths
    queues = [deque() for _ in range(int(n_nodes))]  # queues[v] stores events that ENDED at v
    trans_counter = Counter()

    for u, v, ti in zip(src, dst, t):
        tok = type_to_tok[(u, v)]

        # candidate predecessors are events that ended at current start node u
        q = queues[u]

        # evict too-old events: keep only times >= ti - delta
        while q and q[0][0] < ti - delta:
            q.popleft()

        # add transitions from each remaining predecessor event to current event (strictly increasing time)
        for tj, tok_prev in q:
            if tj < ti and (ti - tj) <= delta:
                trans_counter[(tok_prev, tok)] += 1

        # add current event as predecessor for events starting at its destination v
        queues[v].append((ti, tok))

    trans_types = sorted(trans_counter.keys())
    if len(trans_types) == 0:
        g2_edge_index = np.zeros((2, 0), dtype=int)
        g2_edge_weight = np.zeros((0,), dtype=np.float32)
    else:
        g2_edge_index = np.array(trans_types, dtype=int).T
        g2_edge_weight = np.array([trans_counter[e] for e in trans_types], dtype=np.float32)

    g2_node_ids = np.array(edge_types, dtype=int)  # [T,2]

    return FirstSecondOrderGraphs(
        n_nodes=int(n_nodes),
        g_edge_index=g_edge_index,
        g_edge_weight=g_edge_weight,
        g2_node_ids=g2_node_ids,
        g2_edge_index=g2_edge_index,
        g2_edge_weight=g2_edge_weight,
    )


# ---------------------------------------------------------------------
# Concept discovery: learn node groups from g
# ---------------------------------------------------------------------
def build_sparse_adj_from_g(graphs: FirstSecondOrderGraphs) -> sparse.csr_matrix:
    n = int(graphs.n_nodes)
    src = graphs.g_edge_index[0].astype(int)
    dst = graphs.g_edge_index[1].astype(int)
    w = graphs.g_edge_weight.astype(np.float32)
    return sparse.csr_matrix((w, (src, dst)), shape=(n, n))


def learn_node_groups_spring_physics(
    graphs: FirstSecondOrderGraphs,
    n_groups: int = 8,
    emb_dim: int = 8,
    seed: int = 0,
    spring_iterations: int = 200,
    spring_k: Optional[float] = None,
    use_g2_layout: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discover node groups from a force-directed (spring) simulation.

    If use_g2_layout=False:
      - simulate on first-order graph g (node-level directly).

    If use_g2_layout=True:
      - simulate on second-order token graph g2 (token-level),
      - then aggregate token coordinates back to original nodes using token incidence.

    Returns:
      group_of: [n_nodes] in {0..G-1}
      node_emb: node-level spring coordinates used for clustering
    """
    n = int(graphs.n_nodes)
    if n < 2:
        return np.zeros(n, dtype=int), np.zeros((n, 2), dtype=np.float32)

    n_groups = int(max(1, min(n_groups, n)))
    dim = int(max(2, min(int(emb_dim), 8)))
    rng = np.random.default_rng(int(seed))

    if not use_g2_layout:
        # Spring simulation on node-level first-order graph g.
        A = build_sparse_adj_from_g(graphs)
        A_sym = (A + A.T).multiply(0.5).tocsr()

        G = nx.Graph()
        G.add_nodes_from(range(n))
        tri = sparse.triu(A_sym, k=1).tocoo()
        for u, v, w in zip(tri.row.tolist(), tri.col.tolist(), tri.data.tolist()):
            ww = float(w)
            if ww > 0.0:
                G.add_edge(int(u), int(v), weight=ww)

        if G.number_of_edges() == 0:
            node_emb = rng.normal(0.0, 1.0, size=(n, dim)).astype(np.float32)
        else:
            pos = nx.spring_layout(
                G,
                dim=dim,
                seed=int(seed),
                iterations=int(max(10, spring_iterations)),
                weight="weight",
                k=spring_k,
            )
            node_emb = np.zeros((n, dim), dtype=np.float32)
            for i in range(n):
                node_emb[i] = np.asarray(pos[i], dtype=np.float32)
    else:
        # Spring simulation on token-level second-order graph g2.
        tok_uv = np.asarray(graphs.g2_node_ids, dtype=int)
        T = int(tok_uv.shape[0])

        if T == 0:
            node_emb = rng.normal(0.0, 1.0, size=(n, dim)).astype(np.float32)
        else:
            g2s = np.asarray(graphs.g2_edge_index[0], dtype=int).reshape(-1)
            g2t = np.asarray(graphs.g2_edge_index[1], dtype=int).reshape(-1)
            g2w = np.asarray(graphs.g2_edge_weight, dtype=np.float32).reshape(-1)

            Gt = nx.Graph()
            Gt.add_nodes_from(range(T))
            for a, b, w in zip(g2s.tolist(), g2t.tolist(), g2w.tolist()):
                aa, bb = int(a), int(b)
                ww = float(w)
                if aa == bb or ww <= 0.0:
                    continue
                if Gt.has_edge(aa, bb):
                    Gt[aa][bb]["weight"] += ww
                else:
                    Gt.add_edge(aa, bb, weight=ww)

            if Gt.number_of_edges() == 0:
                tok_emb = rng.normal(0.0, 1.0, size=(T, dim)).astype(np.float32)
            else:
                pos_t = nx.spring_layout(
                    Gt,
                    dim=dim,
                    seed=int(seed),
                    iterations=int(max(10, spring_iterations)),
                    weight="weight",
                    k=spring_k,
                )
                tok_emb = np.zeros((T, dim), dtype=np.float32)
                for t in range(T):
                    tok_emb[t] = np.asarray(pos_t[t], dtype=np.float32)

            # Aggregate token coordinates back to original nodes.
            tok_w = np.ones((T,), dtype=np.float32)
            ge = np.asarray(graphs.g_edge_weight, dtype=np.float32).reshape(-1)
            if ge.shape[0] == T:
                tok_w = np.maximum(ge, 1e-8)

            node_sum = np.zeros((n, dim), dtype=np.float32)
            node_mass = np.zeros((n,), dtype=np.float32)
            for t in range(T):
                u, v = int(tok_uv[t, 0]), int(tok_uv[t, 1])
                ww = float(tok_w[t])
                node_sum[u] += ww * tok_emb[t]
                node_sum[v] += ww * tok_emb[t]
                node_mass[u] += ww
                node_mass[v] += ww

            node_emb = np.zeros((n, dim), dtype=np.float32)
            nz = node_mass > 0
            node_emb[nz] = node_sum[nz] / node_mass[nz, None]
            if np.any(~nz):
                node_emb[~nz] = rng.normal(0.0, 1.0, size=(int((~nz).sum()), dim)).astype(np.float32)

    if n_groups == 1:
        labels = np.zeros(n, dtype=int)
    else:
        try:
            clusterer = AgglomerativeClustering(
                n_clusters=n_groups,
                metric="euclidean",
                linkage="ward",
            )
        except TypeError:
            clusterer = AgglomerativeClustering(
                n_clusters=n_groups,
                affinity="euclidean",
                linkage="ward",
            )
        labels = clusterer.fit_predict(node_emb).astype(int)

    return labels.astype(int), node_emb.astype(np.float32)




def _fit_node2vec_embedding(
    G: nx.Graph,
    n_items: int,
    emb_dim: int,
    seed: int,
    walk_length: int,
    num_walks: int,
    window: int,
    epochs: int,
    p: float,
    q: float,
) -> np.ndarray:
    """Fit node2vec on graph G and return embeddings aligned with node ids [0..n_items-1]."""
    n_items = int(max(0, n_items))
    emb_dim = int(max(2, emb_dim))
    rng = np.random.default_rng(int(seed))

    if n_items == 0:
        return np.zeros((0, emb_dim), dtype=np.float32)

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return rng.normal(0.0, 1.0, size=(n_items, emb_dim)).astype(np.float32)

    try:
        from node2vec import Node2Vec
    except Exception as e:
        raise RuntimeError(
            "Missing dependency `node2vec`. Install it in the notebook environment to use node2vec-based concept discovery."
        ) from e

    n2v = Node2Vec(
        G,
        dimensions=int(emb_dim),
        walk_length=int(max(4, walk_length)),
        num_walks=int(max(4, num_walks)),
        p=float(max(p, 1e-6)),
        q=float(max(q, 1e-6)),
        weight_key="weight",
        workers=1,
        quiet=True,
        seed=int(seed),
    )
    model = n2v.fit(
        window=int(max(2, window)),
        min_count=1,
        batch_words=64,
        epochs=int(max(1, epochs)),
    )

    emb = np.zeros((n_items, emb_dim), dtype=np.float32)
    missing = []
    for i in range(n_items):
        key_str = str(i)
        if key_str in model.wv:
            emb[i] = np.asarray(model.wv[key_str], dtype=np.float32)
        elif i in model.wv:
            emb[i] = np.asarray(model.wv[i], dtype=np.float32)
        else:
            missing.append(i)

    if len(missing) > 0:
        emb[np.asarray(missing, dtype=int)] = rng.normal(0.0, 1.0, size=(len(missing), emb_dim)).astype(np.float32)

    return emb.astype(np.float32)


def _zscore_cols(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return ((x - mu) / sd).astype(np.float32)


def learn_node_groups_node2vec(
    graphs: FirstSecondOrderGraphs,
    n_groups: int = 8,
    emb_dim: int = 16,
    seed: int = 0,
    walk_length: int = 18,
    num_walks: int = 120,
    window: int = 8,
    epochs: int = 2,
    p: float = 1.0,
    q: float = 0.75,
    use_g2_layout: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discover node groups from node2vec random-walk embeddings.

    If use_g2_layout=False:
      - fit node2vec on first-order graph g (node-level directly).

    If use_g2_layout=True:
      - fit node2vec on second-order token graph g2 (token-level),
      - aggregate token embeddings back to original nodes by token incidence.

    Returns:
      group_of: [n_nodes] in {0..G-1}
      node_emb: node-level node2vec embeddings used for clustering/visualization
    """
    n = int(graphs.n_nodes)
    if n < 2:
        return np.zeros(n, dtype=int), np.zeros((n, 2), dtype=np.float32)

    n_groups = int(max(1, min(n_groups, n)))
    dim = int(max(2, min(int(emb_dim), 64)))
    rng = np.random.default_rng(int(seed))

    if not use_g2_layout:
        A = build_sparse_adj_from_g(graphs)
        A_sym = (A + A.T).multiply(0.5).tocsr()

        G = nx.Graph()
        G.add_nodes_from(range(n))
        tri = sparse.triu(A_sym, k=1).tocoo()
        for u, v, w in zip(tri.row.tolist(), tri.col.tolist(), tri.data.tolist()):
            ww = float(w)
            if ww > 0.0:
                G.add_edge(int(u), int(v), weight=ww)

        node_emb = _fit_node2vec_embedding(
            G,
            n_items=n,
            emb_dim=dim,
            seed=int(seed),
            walk_length=int(walk_length),
            num_walks=int(num_walks),
            window=int(window),
            epochs=int(epochs),
            p=float(p),
            q=float(q),
        )
    else:
        tok_uv = np.asarray(graphs.g2_node_ids, dtype=int)
        T = int(tok_uv.shape[0])

        if T == 0:
            node_emb = rng.normal(0.0, 1.0, size=(n, dim)).astype(np.float32)
        else:
            g2s = np.asarray(graphs.g2_edge_index[0], dtype=int).reshape(-1)
            g2t = np.asarray(graphs.g2_edge_index[1], dtype=int).reshape(-1)
            g2w = np.asarray(graphs.g2_edge_weight, dtype=np.float32).reshape(-1)

            Gt = nx.Graph()
            Gt.add_nodes_from(range(T))
            for a, b, w in zip(g2s.tolist(), g2t.tolist(), g2w.tolist()):
                aa, bb = int(a), int(b)
                ww = float(w)
                if aa == bb or ww <= 0.0:
                    continue
                if Gt.has_edge(aa, bb):
                    Gt[aa][bb]["weight"] += ww
                else:
                    Gt.add_edge(aa, bb, weight=ww)

            tok_emb = _fit_node2vec_embedding(
                Gt,
                n_items=T,
                emb_dim=dim,
                seed=int(seed),
                walk_length=int(walk_length),
                num_walks=int(num_walks),
                window=int(window),
                epochs=int(epochs),
                p=float(p),
                q=float(q),
            )

            tok_mass = np.ones((T,), dtype=np.float32)
            if g2s.size > 0:
                np.add.at(tok_mass, g2s.astype(int), np.maximum(g2w, 1e-6))
                np.add.at(tok_mass, g2t.astype(int), np.maximum(g2w, 1e-6))

            node_sum = np.zeros((n, dim), dtype=np.float32)
            node_mass = np.zeros((n,), dtype=np.float32)
            for t in range(T):
                u, v = int(tok_uv[t, 0]), int(tok_uv[t, 1])
                ww = float(max(tok_mass[t], 1e-6))
                node_sum[u] += ww * tok_emb[t]
                node_sum[v] += ww * tok_emb[t]
                node_mass[u] += ww
                node_mass[v] += ww

            node_emb = np.zeros((n, dim), dtype=np.float32)
            nz = node_mass > 0
            node_emb[nz] = node_sum[nz] / node_mass[nz, None]
            if np.any(~nz):
                node_emb[~nz] = rng.normal(0.0, 1.0, size=(int((~nz).sum()), dim)).astype(np.float32)

    node_emb = _zscore_cols(node_emb)

    if n_groups == 1:
        labels = np.zeros(n, dtype=int)
    else:
        try:
            clusterer = AgglomerativeClustering(
                n_clusters=n_groups,
                metric="euclidean",
                linkage="ward",
            )
        except TypeError:
            clusterer = AgglomerativeClustering(
                n_clusters=n_groups,
                affinity="euclidean",
                linkage="ward",
            )
        labels = clusterer.fit_predict(node_emb).astype(int)

    return labels.astype(int), node_emb.astype(np.float32)


def node_embedding_to_plot2d(node_emb: np.ndarray, seed: int = 0) -> np.ndarray:
    """Stable 2D projection for plotting: direct if 2D, else PCA to 2D."""
    x = np.asarray(node_emb, dtype=np.float32)
    if x.ndim != 2:
        x = np.asarray(x).reshape(len(x), -1).astype(np.float32)

    n = int(x.shape[0]) if x.ndim == 2 else 0
    d = int(x.shape[1]) if x.ndim == 2 and x.shape[0] > 0 else 0

    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if d == 0:
        return np.zeros((n, 2), dtype=np.float32)
    if d == 1:
        return np.concatenate([x, np.zeros((n, 1), dtype=np.float32)], axis=1)
    if d == 2:
        return x.astype(np.float32)

    try:
        pca2 = PCA(n_components=2, random_state=int(seed))
        return pca2.fit_transform(x).astype(np.float32)
    except Exception:
        return x[:, :2].astype(np.float32)

def _row_signature(row: sparse.csr_matrix, topk: int = 8) -> str:
    """Compact string for one sparse row: 'col:weight|col:weight|...'"""
    if row.nnz == 0:
        return "none"
    pairs = sorted(
        zip(row.indices.tolist(), row.data.tolist()),
        key=lambda t: (-float(t[1]), int(t[0])),
    )[: int(topk)]
    return "|".join([f"{int(j)}:{float(w):.3f}" for j, w in pairs])

def _node_string_signatures(
    A_out: sparse.csr_matrix,
    A_in: sparse.csr_matrix,
    topk: int = 8,
) -> List[str]:
    """Build one structural signature string per node from outgoing/incoming patterns."""
    n = int(A_out.shape[0])
    sigs: List[str] = []
    for i in range(n):
        out_sig = _row_signature(A_out.getrow(i), topk=topk)
        in_sig = _row_signature(A_in.getrow(i), topk=topk)
        sigs.append(f"out[{out_sig}]::in[{in_sig}]")
    return sigs

def _pairwise_string_distance(signatures: List[str]) -> np.ndarray:
    """Distance matrix from SequenceMatcher similarity: d = 1 - ratio."""
    n = len(signatures)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        si = signatures[i]
        for j in range(i + 1, n):
            sim = SequenceMatcher(None, si, signatures[j]).ratio()
            d = float(1.0 - sim)
            D[i, j] = d
            D[j, i] = d
    return D

def _classical_mds_embedding(D: np.ndarray, n_components: int = 8) -> np.ndarray:
    """Classical MDS embedding from a precomputed distance matrix."""
    D = np.asarray(D, dtype=np.float64)
    n = int(D.shape[0])
    if n < 2:
        return np.zeros((n, 1), dtype=np.float32)

    n_components = int(max(1, min(n_components, n - 1)))

    # Double-centering: B = -1/2 * J * D^2 * J
    J = np.eye(n, dtype=np.float64) - (1.0 / n)
    B = -0.5 * (J @ (D ** 2) @ J)

    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    keep = evals > 1e-12
    if not np.any(keep):
        return np.zeros((n, 1), dtype=np.float32)

    evals = evals[keep][:n_components]
    evecs = evecs[:, keep][:, :n_components]
    X = evecs * np.sqrt(np.clip(evals, 0.0, None))[None, :]
    return X.astype(np.float32)

def learn_node_groups_svd_string_similarity(
    graphs: FirstSecondOrderGraphs,
    n_groups: int = 8,
    emb_dim: int = 32,
    seed: int = 0,
    signature_topk: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discover node groups (concepts) from the first-order graph g.

    Steps:
      1) Build per-node structural strings from outgoing/incoming neighborhoods.
      2) Compute a string-similarity distance matrix.
      3) Cluster nodes via average-linkage agglomerative clustering.
      4) Build a visualization embedding (classical MDS) from the same distances.

    Returns:
      group_of: shape [n_nodes] values in {0..G-1}
      node_emb: embedding derived from the same distance matrix used for clustering
    """
    A = build_sparse_adj_from_g(graphs)
    n = int(A.shape[0])
    if n < 2:
        return np.zeros(n, dtype=int), np.zeros((n, 1), dtype=np.float32)

    # cap to safe values
    n_groups = int(max(1, min(n_groups, n)))
    n_components = int(min(emb_dim, max(2, n - 1)))

    # row-normalize outgoing/incoming patterns
    A_out = normalize(A, norm="l1", axis=1).tocsr()
    A_in = normalize(A.T, norm="l1", axis=1).tocsr()

    signatures = _node_string_signatures(A_out, A_in, topk=max(1, int(signature_topk)))
    D = _pairwise_string_distance(signatures)

    if n_groups == 1:
        labels = np.zeros(n, dtype=int)
    else:
        try:
            clusterer = AgglomerativeClustering(
                n_clusters=n_groups,
                metric="precomputed",
                linkage="average",
            )
        except TypeError:
            # Older sklearn versions use 'affinity' instead of 'metric'.
            clusterer = AgglomerativeClustering(
                n_clusters=n_groups,
                affinity="precomputed",
                linkage="average",
            )
        labels = clusterer.fit_predict(D).astype(int)

    node_emb = _classical_mds_embedding(D, n_components=n_components)
    return labels.astype(int), node_emb.astype(np.float32)


# ---------------------------------------------------------------------
# Concept post-processing + causal concept construction from (g, g2)
# ---------------------------------------------------------------------
def _concept_family_key(name: str) -> str:
    s = str(name)
    if "__" in s:
        s = s.split("__", 1)[1]
    s = re.sub(r"_grp\d+$", "", s)
    for suf in ["_strength", "_entropy", "_top1_mass", "_top_margin"]:
        if s.endswith(suf):
            return s[: -len(suf)] + suf
    return s


def prune_redundant_concepts(
    C: np.ndarray,
    names: List[str],
    min_var: float = 1e-7,
    corr_thresh: float = 0.995,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """Drop near-constant and near-duplicate concept dimensions (leakage mitigation)."""
    C = np.asarray(C, dtype=np.float32)
    names = [str(n) for n in names]

    if C.shape[1] == 0:
        return C, names, {"n_input": 0, "n_kept": 0, "drop_var": 0, "drop_corr": 0}

    var = np.var(C, axis=0)
    keep_var = var > float(min_var)
    idx_var = np.where(keep_var)[0]

    C1 = C[:, idx_var]
    names1 = [names[int(j)] for j in idx_var.tolist()]

    if C1.shape[1] <= 1:
        info = {
            "n_input": int(C.shape[1]),
            "n_kept": int(C1.shape[1]),
            "drop_var": int(C.shape[1] - C1.shape[1]),
            "drop_corr": 0,
        }
        return C1.astype(np.float32), names1, info

    mu = C1.mean(axis=0, keepdims=True)
    sd = C1.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (C1 - mu) / sd

    order = np.argsort(-np.var(C1, axis=0))
    keep_local = []
    for j in order.tolist():
        if not keep_local:
            keep_local.append(int(j))
            continue
        corr = np.abs((Z[:, j][:, None] * Z[:, keep_local]).mean(axis=0))
        if np.all(corr < float(corr_thresh)):
            keep_local.append(int(j))

    keep_local = sorted(keep_local)
    C2 = C1[:, keep_local].astype(np.float32)
    names2 = [names1[j] for j in keep_local]

    info = {
        "n_input": int(C.shape[1]),
        "n_kept": int(C2.shape[1]),
        "drop_var": int(C.shape[1] - C1.shape[1]),
        "drop_corr": int(C1.shape[1] - C2.shape[1]),
    }
    return C2, names2, info


def balance_concept_families(C: np.ndarray, names: List[str], eps: float = 1e-8):
    """Scale each concept family by sqrt(#features in family) to prevent family dominance."""
    C = np.asarray(C, dtype=np.float32).copy()
    fam_to_idx: Dict[str, List[int]] = {}
    for j, nm in enumerate(names):
        fam = _concept_family_key(str(nm))
        fam_to_idx.setdefault(fam, []).append(int(j))

    for fam, idxs in fam_to_idx.items():
        scale = float(np.sqrt(max(len(idxs), 1)))
        C[:, idxs] = C[:, idxs] / max(scale, eps)

    info = {"n_families": int(len(fam_to_idx)), "max_family_size": int(max(len(v) for v in fam_to_idx.values())) if fam_to_idx else 0}
    return C.astype(np.float32), info


def _distribution_descriptors(dist: np.ndarray, eps: float = 1e-9):
    G = int(dist.shape[1])
    if G <= 1:
        zeros = np.zeros((dist.shape[0], 1), dtype=np.float32)
        return zeros, zeros, zeros

    d = np.clip(np.asarray(dist, dtype=np.float32), 0.0, 1.0)
    ent = -np.sum(d * np.log(np.clip(d, eps, 1.0)), axis=1, keepdims=True) / np.log(float(G))

    top1 = np.max(d, axis=1, keepdims=True)
    top2 = np.partition(d, kth=G - 2, axis=1)[:, -2][:, None]
    margin = top1 - top2
    return ent.astype(np.float32), top1.astype(np.float32), margin.astype(np.float32)


def compute_causal_concepts(
    graphs: FirstSecondOrderGraphs,
    group_of: np.ndarray,
    include_strengths: bool = True,
    include_descriptors: bool = True,
    add_temporal_motifs: bool = True,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build graph concepts with richer descriptors:
      - neighbor-group distributions (1-step, 2-step)
      - distribution descriptors (entropy, top-1 mass, top-margin)
      - temporal motif ratios (return/bridge) + reciprocity ratios
    """
    n = int(graphs.n_nodes)
    group_of = np.asarray(group_of, dtype=int)
    assert group_of.shape == (n,)
    G = int(group_of.max()) + 1

    direct_in = np.zeros((n, G), dtype=np.float32)
    direct_out = np.zeros((n, G), dtype=np.float32)

    src = graphs.g_edge_index[0].astype(int)
    dst = graphs.g_edge_index[1].astype(int)
    w1 = graphs.g_edge_weight.astype(np.float32)

    for u, v, w in zip(src, dst, w1):
        direct_out[u, group_of[v]] += w
        direct_in[v, group_of[u]] += w

    in2_start = np.zeros((n, G), dtype=np.float32)
    in2_mid = np.zeros((n, G), dtype=np.float32)
    out2_end = np.zeros((n, G), dtype=np.float32)
    out2_mid = np.zeros((n, G), dtype=np.float32)

    out2_total = np.zeros((n, 1), dtype=np.float32)
    in2_total = np.zeros((n, 1), dtype=np.float32)
    out2_return = np.zeros((n, 1), dtype=np.float32)
    in2_return = np.zeros((n, 1), dtype=np.float32)
    out2_bridge = np.zeros((n, 1), dtype=np.float32)
    in2_bridge = np.zeros((n, 1), dtype=np.float32)

    tok = graphs.g2_node_ids.astype(int)
    g2s = graphs.g2_edge_index[0].astype(int)
    g2t = graphs.g2_edge_index[1].astype(int)
    w2 = graphs.g2_edge_weight.astype(np.float32)

    for a, b, w in zip(g2s, g2t, w2):
        x, _m1 = tok[a]
        m2, v = tok[b]

        in2_start[v, group_of[x]] += w
        in2_mid[v, group_of[m2]] += w
        out2_end[x, group_of[v]] += w
        out2_mid[x, group_of[m2]] += w

        out2_total[x, 0] += w
        in2_total[v, 0] += w
        if int(x) == int(v):
            out2_return[x, 0] += w
            in2_return[v, 0] += w
        if (int(m2) != int(x)) and (int(m2) != int(v)):
            out2_bridge[x, 0] += w
            in2_bridge[v, 0] += w

    families = [
        ("direct_in", direct_in),
        ("direct_out", direct_out),
        ("in2_start", in2_start),
        ("in2_mid", in2_mid),
        ("out2_end", out2_end),
        ("out2_mid", out2_mid),
    ]

    parts: List[np.ndarray] = []
    names: List[str] = []

    for fname, F in families:
        s = F.sum(axis=1, keepdims=True)
        dist = F / (s + eps)

        parts.append(dist)
        names.extend([f"{fname}_grp{g}" for g in range(G)])

        if include_strengths:
            parts.append(np.log1p(s))
            names.append(f"{fname}_strength")

        if include_descriptors:
            ent, top1, margin = _distribution_descriptors(dist, eps=eps)
            parts.extend([ent, top1, margin])
            names.extend([
                f"{fname}_entropy",
                f"{fname}_top1_mass",
                f"{fname}_top_margin",
            ])

    if add_temporal_motifs:
        edge_set = set((int(u), int(v)) for u, v in zip(src.tolist(), dst.tolist()))
        recip_out = np.zeros((n, 1), dtype=np.float32)
        recip_in = np.zeros((n, 1), dtype=np.float32)
        for u, v, w in zip(src.tolist(), dst.tolist(), w1.tolist()):
            if (int(v), int(u)) in edge_set:
                recip_out[int(u), 0] += float(w)
                recip_in[int(v), 0] += float(w)

        out_strength = direct_out.sum(axis=1, keepdims=True)
        in_strength = direct_in.sum(axis=1, keepdims=True)

        motif_feats = [
            recip_out / (out_strength + eps),
            recip_in / (in_strength + eps),
            out2_return / (out2_total + eps),
            in2_return / (in2_total + eps),
            out2_bridge / (out2_total + eps),
            in2_bridge / (in2_total + eps),
            np.log1p(out_strength) - np.log1p(in_strength),
        ]
        motif_names = [
            "fo_reciprocity_out_ratio",
            "fo_reciprocity_in_ratio",
            "ho_return_out_ratio",
            "ho_return_in_ratio",
            "ho_bridge_out_ratio",
            "ho_bridge_in_ratio",
            "fo_out_minus_in_log_strength",
        ]

        parts.extend([m.astype(np.float32) for m in motif_feats])
        names.extend(motif_names)

    C = np.concatenate(parts, axis=1).astype(np.float32)
    return C, names


# ---------------------------------------------------------------------
# Sparse CBM head (L1 Logistic Regression) with CV
# ---------------------------------------------------------------------
def tune_and_train_sparse_logreg_l1(
    X: np.ndarray,
    y: np.ndarray,
    Cs: Optional[np.ndarray] = None,
    n_splits: int = 5,
    seed: int = 0,
    max_iter: int = 8000,
    l1_ratios: Tuple[float, ...] = (1.0,),
    target_active_concepts: int = 12,
    diversity_weight: float = 0.05,
    class_weight_balanced: bool = False,
) -> Tuple[LogisticRegression, Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    CV selection for sparse/elastic-net logistic regression with a concept-diversity objective.

    objective = balanced_accuracy + diversity_weight * (0.7 * nnz_alignment + 0.3 * coef_entropy)
    where nnz_alignment encourages using a moderate number of active concepts.
    """
    if Cs is None:
        Cs = np.logspace(-2, 2, 9)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)

    _, counts = np.unique(y, return_counts=True)
    n_splits = int(min(n_splits, counts.min()))
    n_splits = int(max(n_splits, 2))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    lr_sig = inspect.signature(LogisticRegression)

    def make_clf(Cval: float, l1_ratio_val: float):
        common = dict(C=float(Cval), max_iter=int(max_iter))
        if "class_weight" in lr_sig.parameters and bool(class_weight_balanced):
            common["class_weight"] = "balanced"

        # Preferred: saga + elastic-net (or l1 when l1_ratio=1) for modern sklearn.
        preferred = dict(common)
        if "solver" in lr_sig.parameters:
            preferred["solver"] = "saga"
        if "penalty" in lr_sig.parameters:
            preferred["penalty"] = "l1" if float(l1_ratio_val) >= 0.999 else "elasticnet"
        if "l1_ratio" in lr_sig.parameters:
            preferred["l1_ratio"] = float(l1_ratio_val)
        if "multi_class" in lr_sig.parameters:
            preferred["multi_class"] = "multinomial"
        try:
            return LogisticRegression(**preferred)
        except (TypeError, ValueError):
            pass

        # Compatibility fallback: l1_ratio without explicit penalty (new API variants).
        fallback_modern = dict(common)
        if "solver" in lr_sig.parameters:
            fallback_modern["solver"] = "saga"
        if "l1_ratio" in lr_sig.parameters:
            fallback_modern["l1_ratio"] = float(l1_ratio_val)
        if "multi_class" in lr_sig.parameters:
            fallback_modern["multi_class"] = "multinomial"
        try:
            return LogisticRegression(**fallback_modern)
        except (TypeError, ValueError):
            pass

        # Last fallback: strict L1 with liblinear.
        legacy = dict(common)
        if "solver" in lr_sig.parameters:
            legacy["solver"] = "liblinear"
        if "penalty" in lr_sig.parameters:
            legacy["penalty"] = "l1"
        if "multi_class" in lr_sig.parameters:
            legacy["multi_class"] = "ovr"
        return LogisticRegression(**legacy)

    scores: Dict[str, Dict[str, float]] = {}
    best_cfg: Optional[Dict[str, float]] = None
    best_obj = -1e18

    target = max(float(target_active_concepts), 1.0)

    for Cval in Cs:
        for l1r in l1_ratios:
            fold_ba = []
            fold_nnz = []
            fold_entropy = []

            for tr, te in cv.split(X, y):
                clf = make_clf(Cval, l1r)
                clf.fit(X[tr], y[tr])
                pred = clf.predict(X[te])
                fold_ba.append(float(balanced_accuracy_score(y[te], pred)))

                coef_abs = np.abs(np.asarray(clf.coef_, dtype=np.float64)).reshape(-1)
                nnz = float((coef_abs > 1e-8).sum())
                fold_nnz.append(nnz)

                if coef_abs.size == 0 or float(coef_abs.sum()) <= 1e-12:
                    fold_entropy.append(0.0)
                else:
                    p = coef_abs / float(np.sum(coef_abs))
                    ent = -float(np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))
                    ent /= max(float(np.log(float(coef_abs.size))), 1e-12)
                    fold_entropy.append(float(ent))

            mean_ba = float(np.mean(fold_ba))
            mean_nnz = float(np.mean(fold_nnz))
            mean_ent = float(np.mean(fold_entropy))

            nnz_alignment = 1.0 - min(1.0, abs(mean_nnz - target) / target)
            objective = mean_ba + float(diversity_weight) * (0.7 * nnz_alignment + 0.3 * mean_ent)

            key = f"C={float(Cval):.6g}|l1_ratio={float(l1r):.3f}"
            scores[key] = {
                "balanced_acc": mean_ba,
                "mean_nnz": mean_nnz,
                "coef_entropy": mean_ent,
                "nnz_alignment": float(nnz_alignment),
                "objective": float(objective),
                "C": float(Cval),
                "l1_ratio": float(l1r),
            }

            if (
                (objective > best_obj + 1e-8)
                or (abs(objective - best_obj) <= 1e-8 and best_cfg is not None and mean_ba > float(best_cfg.get("balanced_acc", -1e9)) + 1e-8)
                or (abs(objective - best_obj) <= 1e-8 and best_cfg is not None and abs(mean_ba - float(best_cfg.get("balanced_acc", -1e9))) <= 1e-8 and float(Cval) < float(best_cfg.get("C", 1e18)))
            ):
                best_obj = float(objective)
                best_cfg = {
                    "C": float(Cval),
                    "l1_ratio": float(l1r),
                    "balanced_acc": mean_ba,
                    "mean_nnz": mean_nnz,
                    "coef_entropy": mean_ent,
                    "objective": float(objective),
                }

    assert best_cfg is not None
    clf_final = make_clf(float(best_cfg["C"]), float(best_cfg["l1_ratio"]))
    clf_final.fit(X, y)
    return clf_final, best_cfg, scores



# ---------------------------------------------------------------------
# Concept discovery: mask-learned node groups (GIB-style)
# ---------------------------------------------------------------------
def _torch_sparse_adj_from_edges(n_nodes: int, src: np.ndarray, dst: np.ndarray, w: np.ndarray):
    import torch

    n = int(n_nodes)
    src = np.asarray(src, dtype=int).reshape(-1)
    dst = np.asarray(dst, dtype=int).reshape(-1)
    w = np.asarray(w, dtype=np.float32).reshape(-1)

    if src.size == 0 or dst.size == 0:
        idx = torch.zeros((2, 0), dtype=torch.long)
        val = torch.zeros((0,), dtype=torch.float32)
        A = torch.sparse_coo_tensor(idx, val, size=(n, n)).coalesce()
        At = torch.sparse_coo_tensor(idx, val, size=(n, n)).coalesce()
        return A, At, src, dst, w

    if w.shape[0] != src.shape[0]:
        w = np.ones((src.shape[0],), dtype=np.float32)

    idx = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    val = torch.tensor(w, dtype=torch.float32)
    A = torch.sparse_coo_tensor(idx, val, size=(n, n)).coalesce()
    At = torch.sparse_coo_tensor(idx[[1, 0]], val, size=(n, n)).coalesce()
    return A, At, src, dst, w


def _torch_sparse_adj_from_graphs(graphs: FirstSecondOrderGraphs):
    src = np.asarray(graphs.g_edge_index[0], dtype=int).reshape(-1)
    dst = np.asarray(graphs.g_edge_index[1], dtype=int).reshape(-1)
    w = np.asarray(graphs.g_edge_weight, dtype=np.float32).reshape(-1)
    return _torch_sparse_adj_from_edges(int(graphs.n_nodes), src, dst, w)


def _induced_node_edges_from_g2(graphs: FirstSecondOrderGraphs):
    """
    Build a node-level directed graph induced by second-order transitions.
    For each transition token (x,m)->(m,v), add x->m, m->v, and x->v edges.
    """
    tok = np.asarray(graphs.g2_node_ids, dtype=int)
    if tok.ndim != 2 or tok.shape[1] != 2 or tok.shape[0] == 0:
        return np.zeros((0,), dtype=int), np.zeros((0,), dtype=int), np.zeros((0,), dtype=np.float32)

    g2s = np.asarray(graphs.g2_edge_index[0], dtype=int).reshape(-1)
    g2t = np.asarray(graphs.g2_edge_index[1], dtype=int).reshape(-1)
    if g2s.size == 0 or g2t.size == 0:
        return np.zeros((0,), dtype=int), np.zeros((0,), dtype=int), np.zeros((0,), dtype=np.float32)

    g2w = np.asarray(graphs.g2_edge_weight, dtype=np.float32).reshape(-1)
    if g2w.shape[0] != g2s.shape[0]:
        g2w = np.ones((g2s.shape[0],), dtype=np.float32)

    x = tok[g2s, 0].astype(int)
    m = tok[g2s, 1].astype(int)
    v = tok[g2t, 1].astype(int)

    src = np.concatenate([x, m, x], axis=0).astype(int)
    dst = np.concatenate([m, v, v], axis=0).astype(int)
    w = np.concatenate([g2w, g2w, g2w], axis=0).astype(np.float32)
    return src, dst, w


def _mask_group_features_torch(A, At, assign_soft):
    import torch

    eps = 1e-6
    # direct_out_grp / direct_in_grp in soft-group form
    out_grp = torch.sparse.mm(A, assign_soft)
    in_grp = torch.sparse.mm(At, assign_soft)

    out_strength = torch.sum(out_grp, dim=1, keepdim=True)
    in_strength = torch.sum(in_grp, dim=1, keepdim=True)

    def _top_stats(x):
        g = int(x.shape[1])
        if g <= 1:
            top1 = x
            top2 = torch.zeros_like(top1)
        else:
            vals, _ = torch.topk(x, k=2, dim=1)
            top1 = vals[:, :1]
            top2 = vals[:, 1:2]
        total = torch.clamp(torch.sum(x, dim=1, keepdim=True), min=eps)
        top1_mass = top1 / total
        top_margin = (top1 - top2) / total
        return top1_mass, top_margin

    in_top1, in_margin = _top_stats(in_grp)
    out_top1, out_margin = _top_stats(out_grp)
    out_minus_in = torch.log1p(torch.clamp(out_strength, min=0.0)) - torch.log1p(torch.clamp(in_strength, min=0.0))

    feats = torch.cat(
        [
            torch.log1p(torch.clamp(in_grp, min=0.0)),
            torch.log1p(torch.clamp(out_grp, min=0.0)),
            torch.log1p(torch.clamp(in_strength, min=0.0)),
            torch.log1p(torch.clamp(out_strength, min=0.0)),
            in_top1,
            in_margin,
            out_top1,
            out_margin,
            out_minus_in,
        ],
        dim=1,
    )
    return feats


def learn_node_groups_mask_gib(
    graphs: FirstSecondOrderGraphs,
    y: np.ndarray,
    train_nodes: np.ndarray,
    val_nodes: Optional[np.ndarray] = None,
    *,
    n_groups: int = 10,
    seed: int = 0,
    steps: int = 900,
    lr: float = 0.08,
    temp_start: float = 2.0,
    temp_end: float = 0.45,
    entropy_reg: float = 0.004,
    smooth_reg: float = 0.03,
    balance_reg: float = 0.02,
    class_weight_balanced: bool = False,
    use_g2_layout: bool = False,
    eval_every: int = 25,
) -> Dict[str, Any]:
    """Learn node-group assignments via stochastic mask optimization (GIB-style)."""
    import torch
    import torch.nn.functional as F

    y = np.asarray(y, dtype=int).reshape(-1)
    train_nodes = np.asarray(train_nodes, dtype=int).reshape(-1)
    val_nodes = np.asarray([] if val_nodes is None else val_nodes, dtype=int).reshape(-1)

    n = int(graphs.n_nodes)
    if n < 2:
        return {
            "group_of": np.zeros((n,), dtype=int),
            "assign_prob": np.ones((n, 1), dtype=np.float32),
            "history": [],
            "best_train_ba": np.nan,
            "best_val_ba": np.nan,
            "best_step": -1,
            "n_groups": 1,
            "graph_source": "empty",
            "use_g2_layout": bool(use_g2_layout),
        }

    n_groups = int(max(2, min(int(n_groups), n)))
    steps = int(max(50, int(steps)))
    eval_every = int(max(1, int(eval_every)))

    # Class mapping on train labels only.
    y_train_raw = y[train_nodes]
    y_train_raw = y_train_raw[y_train_raw >= 0]
    classes = np.unique(y_train_raw)
    if classes.size < 2:
        # fallback: pseudo groups from spring layout clustering
        gg, emb = learn_node_groups_spring_physics(
            graphs,
            n_groups=n_groups,
            emb_dim=min(8, n_groups),
            seed=seed,
            spring_iterations=220,
            use_g2_layout=bool(use_g2_layout),
        )
        probs = np.eye(n_groups, dtype=np.float32)[np.asarray(gg, dtype=int)]
        return {
            "group_of": np.asarray(gg, dtype=int),
            "assign_prob": probs,
            "history": [],
            "best_train_ba": np.nan,
            "best_val_ba": np.nan,
            "best_step": -1,
            "n_groups": n_groups,
            "fallback": True,
            "graph_source": "g2-induced-fallback-spring" if bool(use_g2_layout) else "g-fallback-spring",
            "use_g2_layout": bool(use_g2_layout),
        }

    class_to_idx = {int(c): i for i, c in enumerate(classes.tolist())}
    y_idx = np.full(y.shape, -1, dtype=int)
    for c, i in class_to_idx.items():
        y_idx[y == int(c)] = int(i)

    train_nodes = train_nodes[y_idx[train_nodes] >= 0]
    val_nodes = val_nodes[y_idx[val_nodes] >= 0]

    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))

    if bool(use_g2_layout):
        src_ho, dst_ho, w_ho = _induced_node_edges_from_g2(graphs)
        if src_ho.size > 0:
            A, At, src_np, dst_np, w_np = _torch_sparse_adj_from_edges(int(graphs.n_nodes), src_ho, dst_ho, w_ho)
            graph_source = "g2-induced-node-flow"
        else:
            A, At, src_np, dst_np, w_np = _torch_sparse_adj_from_graphs(graphs)
            graph_source = "g-fallback-no-g2-edges"
    else:
        A, At, src_np, dst_np, w_np = _torch_sparse_adj_from_graphs(graphs)
        graph_source = "g-first-order"

    # Undirected smoothness edges.
    u_und = np.concatenate([src_np, dst_np]).astype(int)
    v_und = np.concatenate([dst_np, src_np]).astype(int)
    w_und = np.concatenate([w_np, w_np]).astype(np.float32)

    u_t = torch.tensor(u_und, dtype=torch.long)
    v_t = torch.tensor(v_und, dtype=torch.long)
    w_t = torch.tensor(w_und, dtype=torch.float32)

    train_idx_t = torch.tensor(train_nodes, dtype=torch.long)
    val_idx_t = torch.tensor(val_nodes, dtype=torch.long)
    y_idx_t = torch.tensor(y_idx, dtype=torch.long)

    n_classes = int(len(classes))
    feat_dim = int(2 * n_groups + 7)

    init_logits = rng.normal(0.0, 0.05, size=(n, n_groups)).astype(np.float32)
    logits_param = torch.nn.Parameter(torch.tensor(init_logits, dtype=torch.float32))
    head = torch.nn.Linear(feat_dim, n_classes)

    params = [logits_param] + list(head.parameters())
    opt = torch.optim.Adam(params, lr=float(lr))

    class_weight_t = None
    if bool(class_weight_balanced):
        ytr = y_idx[train_nodes]
        cnt = np.bincount(ytr, minlength=n_classes).astype(np.float32)
        cnt = np.clip(cnt, 1.0, None)
        cw = cnt.sum() / (float(n_classes) * cnt)
        class_weight_t = torch.tensor(cw, dtype=torch.float32)

    best_score = -1e18
    best_probs = None
    best_step = -1
    best_train_ba = np.nan
    best_val_ba = np.nan
    history = []

    def _safe_ba(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        if y_true.size == 0:
            return np.nan
        try:
            return float(balanced_accuracy_score(y_true, y_pred))
        except Exception:
            return np.nan

    for step in range(steps):
        t = float(step) / float(max(steps - 1, 1))
        tau = float(temp_start) * (float(temp_end) / float(temp_start + 1e-12)) ** t
        tau = float(max(tau, 1e-3))

        # Stochastic node-group masks via Gumbel-Softmax.
        assign = F.gumbel_softmax(logits_param, tau=tau, hard=False, dim=1)
        feats = _mask_group_features_torch(A, At, assign)
        logits_cls = head(feats)

        loss_cls = F.cross_entropy(logits_cls[train_idx_t], y_idx_t[train_idx_t], weight=class_weight_t)

        p_soft = torch.softmax(logits_param, dim=1)
        ent = -torch.sum(p_soft * torch.log(torch.clamp(p_soft, min=1e-8)), dim=1).mean()
        ent = ent / float(np.log(max(n_groups, 2)))

        usage = torch.mean(p_soft, dim=0)
        balance = torch.mean((usage - (1.0 / float(n_groups))) ** 2)

        if u_t.numel() > 0:
            diff = p_soft[u_t] - p_soft[v_t]
            smooth = torch.mean(torch.sum(diff * diff, dim=1) * w_t)
        else:
            smooth = torch.tensor(0.0)

        loss = (
            loss_cls
            + float(entropy_reg) * ent
            + float(smooth_reg) * smooth
            + float(balance_reg) * balance
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        opt.step()

        if (step % eval_every == 0) or (step == steps - 1):
            with torch.no_grad():
                p_eval = torch.softmax(logits_param, dim=1)
                feats_eval = _mask_group_features_torch(A, At, p_eval)
                pred_eval = torch.argmax(head(feats_eval), dim=1).cpu().numpy().astype(int)
                p_eval_np = p_eval.cpu().numpy().astype(np.float32)

            tr_ba = _safe_ba(y_idx[train_nodes], pred_eval[train_nodes])
            va_ba = _safe_ba(y_idx[val_nodes], pred_eval[val_nodes]) if val_nodes.size > 0 else np.nan
            score = va_ba if np.isfinite(va_ba) else tr_ba

            history.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach().cpu().item()),
                    "train_ba": float(tr_ba) if np.isfinite(tr_ba) else np.nan,
                    "val_ba": float(va_ba) if np.isfinite(va_ba) else np.nan,
                    "tau": float(tau),
                }
            )

            if np.isfinite(score) and (score > best_score + 1e-8):
                best_score = float(score)
                best_probs = p_eval_np.copy()
                best_step = int(step)
                best_train_ba = float(tr_ba) if np.isfinite(tr_ba) else np.nan
                best_val_ba = float(va_ba) if np.isfinite(va_ba) else np.nan

    if best_probs is None:
        with torch.no_grad():
            best_probs = torch.softmax(logits_param, dim=1).cpu().numpy().astype(np.float32)

    group_of = np.asarray(np.argmax(best_probs, axis=1), dtype=int)

    return {
        "group_of": group_of,
        "assign_prob": np.asarray(best_probs, dtype=np.float32),
        "history": history,
        "best_train_ba": float(best_train_ba) if np.isfinite(best_train_ba) else np.nan,
        "best_val_ba": float(best_val_ba) if np.isfinite(best_val_ba) else np.nan,
        "best_step": int(best_step),
        "n_groups": int(n_groups),
        "graph_source": str(graph_source),
        "use_g2_layout": bool(use_g2_layout),
    }


def build_mask_concepts_with_config(
    graphs: FirstSecondOrderGraphs,
    y: np.ndarray,
    train_nodes: np.ndarray,
    val_nodes: np.ndarray,
    *,
    n_groups: int,
    seed: int,
    mask_steps: int,
    mask_lr: float,
    temp_start: float,
    temp_end: float,
    entropy_reg: float,
    smooth_reg: float,
    balance_reg: float,
    class_weight_balanced: bool,
    include_strengths: bool,
    use_both_layout_concepts: bool,
    layout_on_g2: bool,
    use_concept_pruning: bool,
    concept_min_var: float,
    concept_max_corr: float,
    balance_concept_families_flag: bool,
) -> Dict[str, Any]:
    """Build concepts from mask-learned groups with optional pruning/family balancing."""
    if bool(use_both_layout_concepts):
        mask_info_g = learn_node_groups_mask_gib(
            graphs,
            y=y,
            train_nodes=train_nodes,
            val_nodes=val_nodes,
            n_groups=int(n_groups),
            seed=int(seed),
            steps=int(mask_steps),
            lr=float(mask_lr),
            temp_start=float(temp_start),
            temp_end=float(temp_end),
            entropy_reg=float(entropy_reg),
            smooth_reg=float(smooth_reg),
            balance_reg=float(balance_reg),
            class_weight_balanced=bool(class_weight_balanced),
            use_g2_layout=False,
        )
        mask_info_g2 = learn_node_groups_mask_gib(
            graphs,
            y=y,
            train_nodes=train_nodes,
            val_nodes=val_nodes,
            n_groups=int(n_groups),
            seed=int(seed),
            steps=int(mask_steps),
            lr=float(mask_lr),
            temp_start=float(temp_start),
            temp_end=float(temp_end),
            entropy_reg=float(entropy_reg),
            smooth_reg=float(smooth_reg),
            balance_reg=float(balance_reg),
            class_weight_balanced=bool(class_weight_balanced),
            use_g2_layout=True,
        )

        group_of_g = np.asarray(mask_info_g["group_of"], dtype=int)
        assign_prob_g = np.asarray(mask_info_g["assign_prob"], dtype=np.float32)
        group_of_g2 = np.asarray(mask_info_g2["group_of"], dtype=int)
        assign_prob_g2 = np.asarray(mask_info_g2["assign_prob"], dtype=np.float32)

        C_full_g, concept_names_g = compute_causal_concepts(
            graphs,
            group_of_g,
            include_strengths=bool(include_strengths),
        )
        C_full_g2, concept_names_g2 = compute_causal_concepts(
            graphs,
            group_of_g2,
            include_strengths=bool(include_strengths),
        )
        C_full = np.concatenate(
            [
                np.asarray(C_full_g, dtype=np.float32),
                np.asarray(C_full_g2, dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        concept_names = [f"g__{n}" for n in concept_names_g] + [f"g2__{n}" for n in concept_names_g2]

        if bool(layout_on_g2):
            group_of = np.asarray(group_of_g2, dtype=int)
            assign_prob = np.asarray(assign_prob_g2, dtype=np.float32)
            mask_info_active = mask_info_g2
            layout_source = "mask-gib(g+g2, active=g2)"
        else:
            group_of = np.asarray(group_of_g, dtype=int)
            assign_prob = np.asarray(assign_prob_g, dtype=np.float32)
            mask_info_active = mask_info_g
            layout_source = "mask-gib(g+g2, active=g)"
    else:
        mask_info_active = learn_node_groups_mask_gib(
            graphs,
            y=y,
            train_nodes=train_nodes,
            val_nodes=val_nodes,
            n_groups=int(n_groups),
            seed=int(seed),
            steps=int(mask_steps),
            lr=float(mask_lr),
            temp_start=float(temp_start),
            temp_end=float(temp_end),
            entropy_reg=float(entropy_reg),
            smooth_reg=float(smooth_reg),
            balance_reg=float(balance_reg),
            class_weight_balanced=bool(class_weight_balanced),
            use_g2_layout=bool(layout_on_g2),
        )

        group_of = np.asarray(mask_info_active["group_of"], dtype=int)
        assign_prob = np.asarray(mask_info_active["assign_prob"], dtype=np.float32)
        C_full, concept_names = compute_causal_concepts(
            graphs,
            group_of,
            include_strengths=bool(include_strengths),
        )
        C_full = np.asarray(C_full, dtype=np.float32)
        concept_names = [str(n) for n in concept_names]
        layout_source = "mask-gib(g2-induced)" if bool(layout_on_g2) else "mask-gib(g)"

    prune_info = None
    if bool(use_concept_pruning):
        C_full, concept_names, prune_info = prune_redundant_concepts(
            C_full,
            concept_names,
            min_var=float(concept_min_var),
            corr_thresh=float(concept_max_corr),
        )

    family_balance_info = None
    if bool(balance_concept_families_flag):
        C_full, family_balance_info = balance_concept_families(C_full, concept_names)

    uniq, cnt = np.unique(group_of, return_counts=True)
    order = np.argsort(-cnt)

    out = {
        "C_full": np.asarray(C_full, dtype=np.float32),
        "concept_names": [str(n) for n in concept_names],
        "group_of": group_of,
        "node_emb": assign_prob.astype(np.float32),
        "layout_source": str(layout_source),
        "uniq": np.asarray(uniq, dtype=int),
        "cnt": np.asarray(cnt, dtype=int),
        "order": np.asarray(order, dtype=int),
        "prune_info": prune_info,
        "family_balance_info": family_balance_info,
        "mask_train_info": mask_info_active,
        "n_groups": int(n_groups),
        "mask_steps": int(mask_steps),
        "mask_lr": float(mask_lr),
        "temp_start": float(temp_start),
        "temp_end": float(temp_end),
        "entropy_reg": float(entropy_reg),
        "smooth_reg": float(smooth_reg),
        "balance_reg": float(balance_reg),
        "class_weight_balanced": bool(class_weight_balanced),
        "use_both_layout_concepts": bool(use_both_layout_concepts),
        "layout_on_g2": bool(layout_on_g2),
    }
    if bool(use_both_layout_concepts):
        out.update(
            {
                "group_of_g": np.asarray(group_of_g, dtype=int),
                "node_emb_g": np.asarray(assign_prob_g, dtype=np.float32),
                "group_of_g2": np.asarray(group_of_g2, dtype=int),
                "node_emb_g2": np.asarray(assign_prob_g2, dtype=np.float32),
                "mask_train_info_g": mask_info_g,
                "mask_train_info_g2": mask_info_g2,
            }
        )
    return out


# ---------------------------------------------------------------------
# Split + val-selection helpers
# ---------------------------------------------------------------------
def make_train_val_test_split(
    y: np.ndarray,
    labeled_mask: np.ndarray,
    seed: int,
    test_size: float = 0.30,
    val_size: float = 0.20,
    train_mask: Optional[np.ndarray] = None,
    test_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Build train/val/test node splits in INTERNAL node-id space.

    Strategy:
      1) Use loader train/test masks when valid and class coverage is acceptable.
      2) Otherwise use stratified split on labeled nodes for train+val vs test.
      3) Split train+val into train/val (stratified when possible).
    """
    y = np.asarray(y, dtype=int).reshape(-1)
    labeled_mask = np.asarray(labeled_mask, dtype=bool).reshape(-1)

    def _can_stratify(labels: np.ndarray) -> bool:
        labels = np.asarray(labels, dtype=int).reshape(-1)
        if labels.size < 2:
            return False
        cls, cnt = np.unique(labels, return_counts=True)
        return (cls.size >= 2) and (cnt.min() >= 2)

    def _split_idx(indices: np.ndarray, labels: np.ndarray, frac: float):
        indices = np.asarray(indices, dtype=int)
        labels = np.asarray(labels, dtype=int)
        frac = float(np.clip(frac, 1e-6, 1.0 - 1e-6))

        strat = labels if _can_stratify(labels) else None
        try:
            tr, te = train_test_split(
                np.arange(indices.shape[0], dtype=int),
                test_size=frac,
                random_state=int(seed),
                stratify=strat,
            )
        except ValueError:
            tr, te = train_test_split(
                np.arange(indices.shape[0], dtype=int),
                test_size=frac,
                random_state=int(seed),
                stratify=None,
            )
        return indices[tr], indices[te]

    split_info: Dict[str, Any] = {
        "source": "stratified_fallback",
        "used_loader_split": False,
    }

    use_loader_split = False
    train_nodes = np.zeros((0,), dtype=int)
    test_nodes = np.zeros((0,), dtype=int)

    if train_mask is not None and test_mask is not None:
        trm = np.asarray(train_mask).astype(bool).reshape(-1)
        tem = np.asarray(test_mask).astype(bool).reshape(-1)
        if trm.shape[0] == y.shape[0] and tem.shape[0] == y.shape[0]:
            cand_train = np.where(trm & labeled_mask)[0]
            cand_test = np.where(tem & labeled_mask)[0]
            if cand_train.size > 0 and cand_test.size > 0:
                cls_train = set(np.unique(y[cand_train]).tolist())
                cls_test = set(np.unique(y[cand_test]).tolist())
                missing = sorted(list(cls_train - cls_test))
                if len(missing) == 0:
                    use_loader_split = True
                    train_nodes = cand_train
                    test_nodes = cand_test
                    split_info["source"] = "loader_train_test_masks"
                    split_info["used_loader_split"] = True
                else:
                    split_info["loader_missing_test_classes"] = missing

    if not use_loader_split:
        idx_labeled = np.where(labeled_mask)[0]
        y_labeled = y[idx_labeled]
        train_nodes, test_nodes = _split_idx(idx_labeled, y_labeled, float(test_size))

    # Train/val split from train nodes
    val_nodes = np.zeros((0,), dtype=int)
    val_size = float(np.clip(val_size, 0.0, 0.9))
    if val_size > 0.0 and train_nodes.size >= 4:
        y_train_nodes = y[train_nodes]
        train_nodes, val_nodes = _split_idx(train_nodes, y_train_nodes, float(val_size))

    split_info.update(
        {
            "n_train": int(train_nodes.size),
            "n_val": int(val_nodes.size),
            "n_test": int(test_nodes.size),
        }
    )
    return train_nodes.astype(int), val_nodes.astype(int), test_nodes.astype(int), split_info


def make_sparse_logreg_clf(
    C: float,
    l1_ratio: float,
    max_iter: int = 8000,
    seed: int = 0,
    class_weight_balanced: bool = False,
) -> LogisticRegression:
    """Build a sparse/elastic-net LogisticRegression with compatibility fallbacks."""
    lr_sig = inspect.signature(LogisticRegression)

    common = dict(C=float(C), max_iter=int(max_iter))
    if "random_state" in lr_sig.parameters:
        common["random_state"] = int(seed)
    if "class_weight" in lr_sig.parameters and bool(class_weight_balanced):
        common["class_weight"] = "balanced"

    preferred = dict(common)
    if "solver" in lr_sig.parameters:
        preferred["solver"] = "saga"
    if "penalty" in lr_sig.parameters:
        preferred["penalty"] = "l1" if float(l1_ratio) >= 0.999 else "elasticnet"
    if "l1_ratio" in lr_sig.parameters:
        preferred["l1_ratio"] = float(l1_ratio)
    if "multi_class" in lr_sig.parameters:
        preferred["multi_class"] = "multinomial"

    try:
        return LogisticRegression(**preferred)
    except (TypeError, ValueError):
        pass

    fallback_modern = dict(common)
    if "solver" in lr_sig.parameters:
        fallback_modern["solver"] = "saga"
    if "l1_ratio" in lr_sig.parameters:
        fallback_modern["l1_ratio"] = float(l1_ratio)
    if "multi_class" in lr_sig.parameters:
        fallback_modern["multi_class"] = "multinomial"
    try:
        return LogisticRegression(**fallback_modern)
    except (TypeError, ValueError):
        pass

    legacy = dict(common)
    if "solver" in lr_sig.parameters:
        legacy["solver"] = "liblinear"
    if "penalty" in lr_sig.parameters:
        legacy["penalty"] = "l1"
    if "multi_class" in lr_sig.parameters:
        legacy["multi_class"] = "ovr"
    return LogisticRegression(**legacy)


def tune_sparse_logreg_on_val(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    Cs: Optional[np.ndarray] = None,
    seed: int = 0,
    max_iter: int = 8000,
    l1_ratios: Tuple[float, ...] = (1.0,),
    target_active_concepts: int = 12,
    diversity_weight: float = 0.05,
    class_weight_balanced: bool = False,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Select C/l1_ratio using validation balanced-accuracy + sparsity/diversity objective."""
    if Cs is None:
        Cs = np.logspace(-2, 2, 9)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=int)
    y_val = np.asarray(y_val, dtype=int)

    target = max(float(target_active_concepts), 1.0)

    scores: Dict[str, Dict[str, float]] = {}
    best_cfg: Optional[Dict[str, float]] = None
    best_obj = -1e18

    for Cval in Cs:
        for l1r in l1_ratios:
            clf = make_sparse_logreg_clf(
                C=float(Cval),
                l1_ratio=float(l1r),
                max_iter=int(max_iter),
                seed=int(seed),
                class_weight_balanced=bool(class_weight_balanced),
            )
            clf.fit(X_train, y_train)

            pred_val = clf.predict(X_val)
            val_ba = float(balanced_accuracy_score(y_val, pred_val))

            coef_abs = np.abs(np.asarray(clf.coef_, dtype=np.float64)).reshape(-1)
            mean_nnz = float((coef_abs > 1e-8).sum())

            if coef_abs.size == 0 or float(coef_abs.sum()) <= 1e-12:
                coef_entropy = 0.0
            else:
                p = coef_abs / float(np.sum(coef_abs))
                coef_entropy = -float(np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))
                coef_entropy /= max(float(np.log(float(coef_abs.size))), 1e-12)

            nnz_alignment = 1.0 - min(1.0, abs(mean_nnz - target) / target)
            objective = val_ba + float(diversity_weight) * (0.7 * nnz_alignment + 0.3 * coef_entropy)

            key = f"C={float(Cval):.6g}|l1_ratio={float(l1r):.3f}"
            scores[key] = {
                "balanced_acc": val_ba,
                "mean_nnz": mean_nnz,
                "coef_entropy": float(coef_entropy),
                "nnz_alignment": float(nnz_alignment),
                "objective": float(objective),
                "C": float(Cval),
                "l1_ratio": float(l1r),
            }

            if (
                (objective > best_obj + 1e-8)
                or (abs(objective - best_obj) <= 1e-8 and best_cfg is not None and val_ba > float(best_cfg.get("balanced_acc", -1e9)) + 1e-8)
                or (
                    abs(objective - best_obj) <= 1e-8
                    and best_cfg is not None
                    and abs(val_ba - float(best_cfg.get("balanced_acc", -1e9))) <= 1e-8
                    and float(Cval) < float(best_cfg.get("C", 1e18))
                )
            ):
                best_obj = float(objective)
                best_cfg = {
                    "C": float(Cval),
                    "l1_ratio": float(l1r),
                    "balanced_acc": val_ba,
                    "mean_nnz": mean_nnz,
                    "coef_entropy": float(coef_entropy),
                    "objective": float(objective),
                }

    if best_cfg is None:
        raise RuntimeError("Validation search could not produce any candidate.")
    return best_cfg, scores


def build_node2vec_concepts_with_config(
    graphs: FirstSecondOrderGraphs,
    *,
    n_groups: int,
    emb_dim: int,
    seed: int,
    walk_length: int,
    num_walks: int,
    window: int,
    epochs: int,
    p: float,
    q: float,
    use_both_layout_concepts: bool,
    layout_on_g2: bool,
    include_strengths: bool,
    use_concept_pruning: bool,
    concept_min_var: float,
    concept_max_corr: float,
    balance_concept_families_flag: bool,
) -> Dict[str, Any]:
    """Build concepts from node2vec grouping with optional pruning/family balancing."""
    if use_both_layout_concepts:
        group_of_g, node_emb_g = learn_node_groups_node2vec(
            graphs,
            n_groups=int(n_groups),
            emb_dim=int(emb_dim),
            seed=int(seed),
            walk_length=int(walk_length),
            num_walks=int(num_walks),
            window=int(window),
            epochs=int(epochs),
            p=float(p),
            q=float(q),
            use_g2_layout=False,
        )
        group_of_g2, node_emb_g2 = learn_node_groups_node2vec(
            graphs,
            n_groups=int(n_groups),
            emb_dim=int(emb_dim),
            seed=int(seed),
            walk_length=int(walk_length),
            num_walks=int(num_walks),
            window=int(window),
            epochs=int(epochs),
            p=float(p),
            q=float(q),
            use_g2_layout=True,
        )

        if bool(layout_on_g2):
            group_of = np.asarray(group_of_g2, dtype=int)
            node_emb = np.asarray(node_emb_g2, dtype=np.float32)
            layout_source = "g2-node2vec"
        else:
            group_of = np.asarray(group_of_g, dtype=int)
            node_emb = np.asarray(node_emb_g, dtype=np.float32)
            layout_source = "g-node2vec"

        C_full_g, concept_names_g = compute_causal_concepts(graphs, group_of_g, include_strengths=include_strengths)
        C_full_g2, concept_names_g2 = compute_causal_concepts(graphs, group_of_g2, include_strengths=include_strengths)

        C_full = np.concatenate([C_full_g, C_full_g2], axis=1).astype(np.float32)
        concept_names = [f"g__{n}" for n in concept_names_g] + [f"g2__{n}" for n in concept_names_g2]
    else:
        group_of_raw, node_emb_raw = learn_node_groups_node2vec(
            graphs,
            n_groups=int(n_groups),
            emb_dim=int(emb_dim),
            seed=int(seed),
            walk_length=int(walk_length),
            num_walks=int(num_walks),
            window=int(window),
            epochs=int(epochs),
            p=float(p),
            q=float(q),
            use_g2_layout=bool(layout_on_g2),
        )
        group_of = np.asarray(group_of_raw, dtype=int)
        node_emb = np.asarray(node_emb_raw, dtype=np.float32)
        layout_source = "g2-node2vec" if bool(layout_on_g2) else "g-node2vec"

        C_full, concept_names = compute_causal_concepts(graphs, group_of, include_strengths=include_strengths)
        C_full = np.asarray(C_full, dtype=np.float32)
        concept_names = [str(n) for n in concept_names]
        group_of_g = None
        node_emb_g = None
        group_of_g2 = None
        node_emb_g2 = None

    prune_info = None
    if bool(use_concept_pruning):
        C_full, concept_names, prune_info = prune_redundant_concepts(
            C_full,
            concept_names,
            min_var=float(concept_min_var),
            corr_thresh=float(concept_max_corr),
        )

    family_balance_info = None
    if bool(balance_concept_families_flag):
        C_full, family_balance_info = balance_concept_families(C_full, concept_names)

    uniq, cnt = np.unique(group_of, return_counts=True)
    order = np.argsort(-cnt)

    out = {
        "C_full": np.asarray(C_full, dtype=np.float32),
        "concept_names": [str(n) for n in concept_names],
        "group_of": np.asarray(group_of, dtype=int),
        "node_emb": np.asarray(node_emb, dtype=np.float32),
        "layout_source": str(layout_source),
        "uniq": np.asarray(uniq, dtype=int),
        "cnt": np.asarray(cnt, dtype=int),
        "order": np.asarray(order, dtype=int),
        "prune_info": prune_info,
        "family_balance_info": family_balance_info,
        "layout_on_g2": bool(layout_on_g2),
        "n_groups": int(n_groups),
        "walk_length": int(walk_length),
        "num_walks": int(num_walks),
        "window": int(window),
        "epochs": int(epochs),
        "p": float(p),
        "q": float(q),
    }

    if use_both_layout_concepts:
        out.update(
            {
                "group_of_g": np.asarray(group_of_g, dtype=int),
                "node_emb_g": np.asarray(node_emb_g, dtype=np.float32),
                "group_of_g2": np.asarray(group_of_g2, dtype=int),
                "node_emb_g2": np.asarray(node_emb_g2, dtype=np.float32),
            }
        )

    return out


# ---------------------------------------------------------------------
# Explanation helpers
# ---------------------------------------------------------------------
def _weights_for_class(clf: LogisticRegression, class_id: int) -> Tuple[np.ndarray, float]:
    coef = np.asarray(clf.coef_, dtype=np.float32)
    intercept = np.asarray(clf.intercept_, dtype=np.float32)
    classes = np.asarray(clf.classes_, dtype=int)

    if len(classes) == 2 and coef.shape[0] == 1:
        # binary: coef is for classes_[1]
        w = coef[0]
        b = float(intercept[0]) if intercept.size else 0.0
        if int(class_id) == int(classes[0]):
            w = -w
            b = -b
        return w, b

    # multi-class: one row per class
    row = int(np.where(classes == int(class_id))[0][0]) if int(class_id) in set(classes.tolist()) else 0
    w = coef[row]
    b = float(intercept[row]) if intercept.size else 0.0
    return w, b

def explain_node_prediction(
    clf: LogisticRegression,
    X: np.ndarray,
    feature_names: List[str],
    node_idx: int,
    topk: int = 10,
) -> Dict[str, Any]:
    x = np.asarray(X[node_idx], dtype=np.float32)
    pred_class = int(clf.predict(x[None, :])[0])
    proba = clf.predict_proba(x[None, :])[0] if hasattr(clf, "predict_proba") else None

    w, b = _weights_for_class(clf, pred_class)
    contrib = w * x

    order = np.argsort(-np.abs(contrib))[: int(topk)]
    top = []
    for j in order:
        top.append(
            dict(
                feature_idx=int(j),
                name=str(feature_names[int(j)]),
                value=float(x[int(j)]),
                weight=float(w[int(j)]),
                contribution=float(contrib[int(j)]),
            )
        )

    return dict(
        node_idx=int(node_idx),
        pred_class=int(pred_class),
        proba=None if proba is None else np.asarray(proba, dtype=float),
        intercept=float(b),
        top=top,
    )



def _colors_by_sign(vals, pos="#0000A6", neg="#D98C00", zero="#737373"):
    return [pos if v > 0 else neg if v < 0 else zero for v in vals]

def plot_explanation(expl: Dict[str, Any], title: Optional[str] = None):
    names = [t["name"] for t in expl["top"]][::-1]
    contrib = [t["contribution"] for t in expl["top"]][::-1]
    colors = _colors_by_sign(contrib)

    plt.figure(figsize=(10, 3.5))
    plt.barh(range(len(contrib)), contrib, color=colors)
    plt.yticks(range(len(contrib)), names)
    plt.axvline(0.0, linewidth=1)
    plt.xlabel("Contribution to predicted-class score (w · x)")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()

def annotate_scatter(xs, ys, labels, fontsize=8):
    for x, y, lab in zip(xs, ys, labels):
        plt.text(x, y, str(lab), fontsize=fontsize, ha="center", va="center")


# ---------------------------------------------------------------------
# Faithfulness utilities (concept-level ablation on node predictions)
# ---------------------------------------------------------------------
def select_active_indices(
    concept_names: List[str],
    use_both_layout_concepts: bool,
    spring_layout_on_g2: bool,
    view: str = "layout",
) -> np.ndarray:
    """
    view:
      - "layout": if both-layout concepts are enabled, keep only the active layout prefix (g__ or g2__)
      - "all": keep all concept dimensions
    """
    out = []
    pref = "g2__" if spring_layout_on_g2 else "g__"
    for j, nm in enumerate(concept_names):
        s = str(nm)
        if view == "all":
            out.append(j)
            continue
        if use_both_layout_concepts and not s.startswith(pref):
            continue
        out.append(j)
    return np.asarray(out, dtype=int)


def _logits_from_clf(clf, x_vec: np.ndarray, classes_arr: np.ndarray, pred_pos: int) -> np.ndarray:
    if hasattr(clf, "decision_function"):
        d = np.asarray(clf.decision_function(x_vec[None, :])).reshape(-1)
        if d.size == 1 and classes_arr.size == 2:
            s = float(d[0])
            return np.asarray([-s, s], dtype=float)
        if d.size == classes_arr.size:
            return d.astype(float)

    if hasattr(clf, "predict_proba"):
        p = np.asarray(clf.predict_proba(x_vec[None, :])[0], dtype=float)
        return np.log(np.clip(p, 1e-12, 1.0))

    z = np.zeros((classes_arr.size,), dtype=float)
    w, b = _weights_for_class(clf, int(classes_arr[pred_pos]))
    z[pred_pos] = float(np.dot(x_vec, w) + b)
    return z


def faithfulness_profile_for_node(
    clf,
    C_full: np.ndarray,
    node_idx: int,
    active_idx: np.ndarray,
    max_k: int = 18,
    random_trials: int = 32,
    random_seed: int = 0,
) -> Dict[str, Any]:
    classes_arr = np.asarray(clf.classes_, dtype=int)
    node = int(node_idx)

    x0 = np.asarray(C_full[node], dtype=np.float32)
    pred = int(clf.predict(x0[None, :])[0])
    pred_pos = int(np.where(classes_arr == pred)[0][0])

    w_full, _ = _weights_for_class(clf, pred)
    w_active = np.asarray(w_full[active_idx], dtype=np.float32)
    x_active = np.asarray(x0[active_idx], dtype=np.float32)

    contrib = x_active * w_active
    pos = np.where(contrib > 0)[0]
    if pos.size > 0:
        rank_local = pos[np.argsort(-contrib[pos])]
    else:
        rank_local = np.argsort(-np.abs(contrib))

    k_max = int(min(int(max_k), int(rank_local.size)))
    if k_max < 2:
        return dict(node=node, pred=pred, k_max=k_max, auc_gap=np.nan, flip_score=0.0, p_drop_gap=0.0, margin_drop_gap=0.0)

    k_vals = np.arange(k_max + 1, dtype=int)

    def _curve(order_local: np.ndarray):
        probs = []
        margins = []
        preds = []
        for k in k_vals.tolist():
            x_mod = x0.copy()
            if k > 0:
                drop_global = active_idx[order_local[:k]]
                x_mod[drop_global] = 0.0
            p = np.asarray(clf.predict_proba(x_mod[None, :])[0], dtype=float)
            z = _logits_from_clf(clf, x_mod, classes_arr, pred_pos)
            z_pred = float(z[pred_pos])
            z_other = np.delete(z, pred_pos)
            margin = z_pred - (float(np.max(z_other)) if z_other.size else 0.0)
            probs.append(float(p[pred_pos]))
            margins.append(float(margin))
            preds.append(int(clf.predict(x_mod[None, :])[0]))
        return np.asarray(probs), np.asarray(margins), np.asarray(preds)

    p_top, m_top, pred_top = _curve(rank_local)
    p_drop_top = p_top[0] - p_top
    m_drop_top = m_top[0] - m_top

    rng = np.random.default_rng(int(random_seed))
    rand_pool = np.arange(contrib.shape[0], dtype=int)
    rand_p = []
    rand_m = []
    for _ in range(int(max(4, random_trials))):
        order_rand = rng.permutation(rand_pool)[:k_max]
        p_r, m_r, _ = _curve(order_rand)
        rand_p.append(p_r[0] - p_r)
        rand_m.append(m_r[0] - m_r)

    rand_p = np.asarray(rand_p, dtype=float)
    rand_m = np.asarray(rand_m, dtype=float)
    p_drop_rand = rand_p.mean(axis=0)
    m_drop_rand = rand_m.mean(axis=0)

    auc_top = float(np.trapz(p_drop_top, k_vals)) / max(float(k_max), 1.0)
    auc_rand = float(np.trapz(p_drop_rand, k_vals)) / max(float(k_max), 1.0)
    auc_gap = float(auc_top - auc_rand)

    flip_k = None
    for k, pk in zip(k_vals.tolist(), pred_top.tolist()):
        if int(pk) != int(pred):
            flip_k = int(k)
            break
    flip_score = 0.0 if flip_k is None else 1.0 - (float(flip_k) / float(k_max))

    return dict(
        node=node,
        pred=pred,
        k_max=k_max,
        auc_gap=auc_gap,
        flip_score=float(flip_score),
        p_drop_gap=float(p_drop_top[-1] - p_drop_rand[-1]),
        margin_drop_gap=float(m_drop_top[-1] - m_drop_rand[-1]),
    )


def aggregate_faithfulness_report(
    clf,
    C_full: np.ndarray,
    eval_nodes: np.ndarray,
    concept_names: List[str],
    use_both_layout_concepts: bool,
    spring_layout_on_g2: bool,
    view: str = "all",
    max_k: int = 18,
    random_trials: int = 32,
    random_seed: int = 0,
):
    active_idx = select_active_indices(
        concept_names,
        use_both_layout_concepts=bool(use_both_layout_concepts),
        spring_layout_on_g2=bool(spring_layout_on_g2),
        view=str(view),
    )
    rows = []
    for n in np.asarray(eval_nodes, dtype=int).tolist():
        r = faithfulness_profile_for_node(
            clf,
            C_full,
            node_idx=int(n),
            active_idx=active_idx,
            max_k=max_k,
            random_trials=random_trials,
            random_seed=int(random_seed) + int(n),
        )
        if np.isfinite(r.get("auc_gap", np.nan)):
            rows.append(r)

    if len(rows) == 0:
        return active_idx, [], dict(mean_auc_gap=np.nan, mean_flip=np.nan, mean_p_drop=np.nan, mean_margin_drop=np.nan)

    aucs = np.asarray([r["auc_gap"] for r in rows], dtype=float)
    flips = np.asarray([r["flip_score"] for r in rows], dtype=float)
    pdrop = np.asarray([r["p_drop_gap"] for r in rows], dtype=float)
    mdrop = np.asarray([r["margin_drop_gap"] for r in rows], dtype=float)

    summary = dict(
        mean_auc_gap=float(np.mean(aucs)),
        std_auc_gap=float(np.std(aucs)),
        mean_flip=float(np.mean(flips)),
        mean_p_drop=float(np.mean(pdrop)),
        mean_margin_drop=float(np.mean(mdrop)),
        n_nodes=int(len(rows)),
        view=str(view),
        n_active_concepts=int(active_idx.size),
    )
    return active_idx, rows, summary
