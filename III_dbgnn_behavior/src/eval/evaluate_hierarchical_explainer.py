#!/usr/bin/env python3
"""
evaluate_hierarchical_explainer.py

Evaluation utilities for the hierarchical DBGNN explainer (Level 1: HO nodes (u,v),
Level 2: HO edges (u,v)->(v,w)).

It provides TWO kinds of "ground truth":

(A) Model-ground-truth (inference-time):
    Level 1: exact/MC Shapley values for incoming HO nodes of v (feasible: 15-21 players).
    Level 2: leave-one-out (LOO) ground truth for edges in the k-hop neighborhood of an HO node
             (or MC Shapley if you want, but LOO is much cheaper).

(B) Data-generator ground truth (optional, if you pass node labels):
    Label each HO node/edge as "within-cluster" vs "cross-cluster" and compute precision@k, AUROC.

In addition, it computes deletion curves (AUC) for both levels, which do not require any external GT.

Usage:
  python evaluate_hierarchical_explainer.py --node 19 --top_m 5 --k 2 --lvl1_gt shapley --lvl2_gt loo

It expects hierarchical_dbgnn_explainer.py to be in the same folder (or PYTHONPATH).
"""

import argparse
import json
import math
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from explainers.hierarchical_dbgnn_explainer import HierarchicalDBGNNExplainer


# ----------------------------
# Simple ranking / metrics
# ----------------------------
def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks for ties, like scipy.stats.rankdata(method='average'), 1..n."""
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    n = len(a)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        # average rank for ties
        r = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = r
        i = j + 1
    return ranks


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation without scipy."""
    if x.size != y.size or x.size < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = (np.sqrt((rx ** 2).sum()) * np.sqrt((ry ** 2).sum()))
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def ndcg_at_k(rel: np.ndarray, score: np.ndarray, k: int) -> float:
    """NDCG@k with non-negative relevance rel."""
    k = min(k, len(rel))
    if k <= 0:
        return float("nan")
    # predicted ordering
    idx = np.argsort(score)[::-1][:k]
    dcg = 0.0
    for rank, i in enumerate(idx, start=1):
        dcg += (2 ** rel[i] - 1.0) / math.log2(rank + 1)
    # ideal ordering
    idx2 = np.argsort(rel)[::-1][:k]
    idcg = 0.0
    for rank, i in enumerate(idx2, start=1):
        idcg += (2 ** rel[i] - 1.0) / math.log2(rank + 1)
    return float(dcg / idcg) if idcg > 0 else float("nan")


def auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal AUC."""
    # x must be increasing
    return float(np.trapz(y, x))


def _margin_from_logits_row(row: torch.Tensor, y_ref: int) -> torch.Tensor:
    mask = torch.ones(row.numel(), dtype=torch.bool, device=row.device)
    mask[y_ref] = False
    return row[y_ref] - row[mask].max()


# ----------------------------
# Level 1: Shapley ground truth for incoming HO nodes
# ----------------------------
def shapley_level1_permutation(
    expl: HierarchicalDBGNNExplainer,
    v: int,
    num_perm: int = 5000,
    seed: int = 0,
) -> Tuple[np.ndarray, List[int], int, float]:
    """
    Approximate Shapley values for incoming HO nodes of v using random permutations.
    Value function: f(S) = margin_y( ELU( base + sum_{q in S} Z_h[q] ) ), where
    base = deg_bip[v]*Z_fo[v], y is the full-graph predicted class.

    Returns:
      phi: shapley values aligned to incoming list order
      incoming: list of HO node ids (q)
      y_ref: reference class
      margin_full: margin on full set
    """
    if "logits" not in expl._cache:
        expl._compute_and_cache_full_forward()

    logits_full_all = expl._cache["logits"]
    Z_h = expl._cache["Z_h"]
    Z_fo = expl._cache["Z_fo"]

    v = int(v)
    y_ref = int(logits_full_all[v].argmax().item())

    incoming_t = (expl.v_last == v).nonzero(as_tuple=False).view(-1)
    incoming = [int(q) for q in incoming_t.tolist()]
    m = len(incoming)

    if m == 0:
        return np.array([]), incoming, y_ref, float("nan")

    base = (expl.deg_bip[v] * Z_fo[v]).detach()              # [8]
    contrib = Z_h[incoming_t].detach()                       # [m,8]

    W = expl.state["lin.weight"]
    b = expl.state["lin.bias"]

    def f_from_z(z: torch.Tensor) -> float:
        row = F.elu(z) @ W.t() + b
        return float(_margin_from_logits_row(row, y_ref).item())

    # full margin (all incoming)
    margin_full = f_from_z(base + contrib.sum(dim=0))

    rng = np.random.default_rng(seed)
    phi = np.zeros(m, dtype=float)

    # permutation sampling
    idx = np.arange(m)
    for _ in range(num_perm):
        rng.shuffle(idx)
        z = base.clone()
        f_prev = f_from_z(z)
        for j in idx:
            z = z + contrib[j]
            f_new = f_from_z(z)
            phi[j] += (f_new - f_prev)
            f_prev = f_new

    phi /= float(num_perm)
    return phi, incoming, y_ref, margin_full


# ----------------------------
# Level 2: Ground truth for HO edges around a given HO node q
# ----------------------------
def loo_level2_edges(
    expl: HierarchicalDBGNNExplainer,
    q: int,
    k: int,
    direction: str = "both",
    restrict_to_neighborhood: bool = True,
) -> Tuple[List[int], np.ndarray, float]:
    """
    Leave-one-out (LOO) importance for HO edges in the k-hop neighborhood of HO node q,
    relative to embedding change scalar used by the explainer:

        delta = h_full - h_base
        f(w) = (h(w) - h_base) · delta

    LOO score for edge e: f(full) - f(full\{e})

    Returns:
      edges_k: list of HO edge ids in neighborhood
      scores: LOO scores aligned with edges_k
      f_full: scalar at full
    """
    q = int(q)
    nodes_k, edges_k = expl.k_hop_neighborhood_ho(q, k=k, direction=direction)
    if len(edges_k) == 0:
        return edges_k, np.array([]), float("nan")

    edges_k_t = torch.tensor(edges_k, dtype=torch.long, device=expl.device)

    if restrict_to_neighborhood:
        w_full = torch.zeros_like(expl.g2_edge_weight)
        w_full[edges_k_t] = expl.g2_edge_weight[edges_k_t]
        w_base = torch.zeros_like(expl.g2_edge_weight)
    else:
        w_full = expl.g2_edge_weight.clone()
        w_base = expl.g2_edge_weight.clone()
        w_base[edges_k_t] = 0.0

    with torch.no_grad():
        H_base = expl._forward_ho_embeddings(w_base)
        H_full = expl._forward_ho_embeddings(w_full)
        h_base = H_base[q].detach()
        h_full = H_full[q].detach()
        delta = (h_full - h_base).detach()

        # scalar at full
        f_full = float(((h_full - h_base) * delta).sum().item())

    scores = np.zeros(len(edges_k), dtype=float)

    # LOO: drop one edge at a time
    for i, e in enumerate(edges_k):
        w_minus = w_full.clone()
        w_minus[int(e)] = 0.0
        with torch.no_grad():
            H_minus = expl._forward_ho_embeddings(w_minus)
            h_minus = H_minus[q].detach()
            f_minus = float(((h_minus - h_base) * delta).sum().item())
        scores[i] = f_full - f_minus

    return edges_k, scores, f_full


# ----------------------------
# Deletion curves
# ----------------------------
def deletion_curve_level1(
    expl: HierarchicalDBGNNExplainer,
    v: int,
    ranking_q: List[int],
    y_ref: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove incoming HO node contributions one by one in the given ranking order,
    measure margin after each removal.

    Semantics matches explainer's Level 1 margin_drop: subtract Z_h[q] only.
    """
    if "Z_h" not in expl._cache:
        expl._compute_and_cache_full_forward()

    Z_h = expl._cache["Z_h"]
    z_pre = expl._cache["z_pre"]

    v = int(v)
    z = z_pre[v].detach().clone()
    W = expl.state["lin.weight"]
    b = expl.state["lin.bias"]

    def margin_from_z(zvec: torch.Tensor) -> float:
        row = F.elu(zvec) @ W.t() + b
        return float(_margin_from_logits_row(row, y_ref).item())

    m0 = margin_from_z(z)
    margins = [m0]
    fracs = [0.0]

    total = len(ranking_q)
    for t, q in enumerate(ranking_q, start=1):
        z = z - Z_h[int(q)].detach()
        margins.append(margin_from_z(z))
        fracs.append(t / total)

    return np.array(fracs), np.array(margins)


def deletion_curve_level2_embedding(
    expl: HierarchicalDBGNNExplainer,
    q: int,
    edges_k: List[int],
    ranking_edges: List[int],
    restrict_to_neighborhood: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove HO edges in ranking order and track cosine similarity between H_q and full-local H_q.
    """
    q = int(q)
    edges_k_t = torch.tensor(edges_k, dtype=torch.long, device=expl.device)

    if restrict_to_neighborhood:
        w_full = torch.zeros_like(expl.g2_edge_weight)
        w_full[edges_k_t] = expl.g2_edge_weight[edges_k_t]
    else:
        w_full = expl.g2_edge_weight.clone()

    with torch.no_grad():
        h_full = expl._forward_ho_embeddings(w_full)[q].detach()

    def cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a @ b) / ((a.norm() + 1e-9) * (b.norm() + 1e-9)))

    w = w_full.clone()
    sims = [cos(expl._forward_ho_embeddings(w)[q].detach(), h_full)]
    fracs = [0.0]

    total = len(ranking_edges)
    for t, e in enumerate(ranking_edges, start=1):
        w[int(e)] = 0.0
        with torch.no_grad():
            h = expl._forward_ho_embeddings(w)[q].detach()
        sims.append(cos(h, h_full))
        fracs.append(t / total)

    return np.array(fracs), np.array(sims)


# ----------------------------
# Optional generator-ground-truth metrics (cluster consistency)
# ----------------------------
def within_cluster_ho_node(q_pair: Tuple[int, int], labels: np.ndarray) -> bool:
    u, v = q_pair
    return int(labels[u]) == int(labels[v])

def within_cluster_ho_edge(src_pair: Tuple[int, int], dst_pair: Tuple[int, int], labels: np.ndarray) -> bool:
    u, v = src_pair
    v2, w = dst_pair
    if v != v2:
        return False
    c = int(labels[v])
    return int(labels[u]) == c and int(labels[w]) == c


def precision_at_k(binary_gt: np.ndarray, score: np.ndarray, k: int) -> float:
    k = min(k, len(binary_gt))
    if k <= 0:
        return float("nan")
    idx = np.argsort(score)[::-1][:k]
    return float(binary_gt[idx].mean())


def auc_roc(binary_gt: np.ndarray, score: np.ndarray) -> float:
    """AUROC without sklearn (handles ties crudely)."""
    y = binary_gt.astype(int)
    s = score.astype(float)
    # rank by score
    order = np.argsort(s)
    y = y[order]
    # compute ROC via ranks: AUROC = (sum ranks of positives - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    n_pos = y.sum()
    n = len(y)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(s[order])  # ranks in sorted order
    sum_pos_ranks = float(ranks[y == 1].sum())
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _load_labels_from_data_pt(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        data = torch.load(path, map_location="cpu")
    except Exception:
        return None
    y = getattr(data, "y", None)
    if torch.is_tensor(y) and y.numel() > 0:
        return y.detach().cpu().numpy()
    return None


def _infer_labels_from_meta(meta_path: Path) -> Optional[np.ndarray]:
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    dataset_name = str(meta.get("dataset_name", "")).strip()
    if dataset_name not in {"temporal_clusters", "temporal_clusters_connected", "dbgnn_explainable_dataset_connected"}:
        return None
    try:
        num_nodes = int(meta.get("num_nodes"))
        num_classes = int(meta.get("num_classes"))
    except Exception:
        return None
    if num_classes <= 0 or num_nodes <= 0 or num_nodes % num_classes != 0:
        return None
    block = num_nodes // num_classes
    labels = np.arange(num_nodes, dtype=int) // block
    return labels


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", type=int, default=19)
    ap.add_argument("--top_m", type=int, default=5)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--direction", type=str, default="both", choices=["in", "out", "both"])
    ap.add_argument("--lvl1_gt", type=str, default="shapley", choices=["shapley"])
    ap.add_argument("--lvl2_gt", type=str, default="loo", choices=["loo"])
    ap.add_argument("--lvl1_perm", type=int, default=5000, help="Permutations for Level 1 Shapley")
    ap.add_argument("--top_e_all", type=int, default=100000, help="Request this many edges from explainer (to get all)")
    ap.add_argument("--labels", type=str, default=None, help="Optional .npy file with node labels (len=30) for generator-GT metrics")
    ap.add_argument("--data_pt", type=str, default=None, help="Optional data.pt path to read node labels from data.y")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")

    # paths
    ap.add_argument("--model_state", type=str, default="model_state.pt")
    ap.add_argument("--g_edge_index", type=str, default="g_edge_index.pt")
    ap.add_argument("--g_edge_weight", type=str, default="g_edge_weight.pt")
    ap.add_argument("--g2_edge_index", type=str, default="g2_edge_index.pt")
    ap.add_argument("--g2_edge_weight", type=str, default="g2_edge_weight.pt")
    ap.add_argument("--g2_node_ids", type=str, default="g2_node_ids.npy")

    args = ap.parse_args()

    default_data_dir = None
    if not Path(args.model_state).exists():
        default_run_dir = ROOT / "notebooks" / "runs" / "temporal_clusters_dbgnn"
        default_data_dir = default_run_dir / "dataset"
        candidate_model = default_run_dir / "model_state.pt"
        if candidate_model.exists():
            args.model_state = str(candidate_model)
            args.g_edge_index = str(default_data_dir / "g_edge_index.pt")
            args.g_edge_weight = str(default_data_dir / "g_edge_weight.pt")
            args.g2_edge_index = str(default_data_dir / "g2_edge_index.pt")
            args.g2_edge_weight = str(default_data_dir / "g2_edge_weight.pt")
            args.g2_node_ids = str(default_data_dir / "g2_node_ids.npy")
            print("[info] Using temporal_clusters_dbgnn artifacts from notebooks/runs.")

    device = torch.device(args.device)

    labels = None
    if args.labels is not None:
        labels = np.load(args.labels)
        print(f"[info] Loaded labels from {args.labels}")
    else:
        if args.data_pt is not None:
            labels = _load_labels_from_data_pt(Path(args.data_pt))
            if labels is not None:
                print(f"[info] Loaded labels from {args.data_pt}")
        if labels is None and default_data_dir is not None:
            data_pt = default_data_dir / "data.pt"
            labels = _load_labels_from_data_pt(data_pt)
            if labels is not None:
                print("[info] Loaded labels from temporal_clusters_dbgnn dataset/data.pt")
        if labels is None and default_data_dir is not None:
            meta_path = default_data_dir / "meta.json"
            labels = _infer_labels_from_meta(meta_path)
            if labels is not None:
                print("[info] Inferred labels from temporal_clusters meta.json")
        if labels is None:
            print("[warn] No labels found; pass --labels or --data_pt to enable signal/noise metrics.")

    expl = HierarchicalDBGNNExplainer.from_files(
        args.model_state,
        args.g_edge_index, args.g_edge_weight,
        args.g2_edge_index, args.g2_edge_weight,
        args.g2_node_ids,
        device=device,
        cache_full_forward=True,
    )
    print(f"[info] #HO edges (E_ho) = {int(expl.g2_edge_index.size(1))}")

    v = int(args.node)
    lvl1_items, y_ref, margin0, M = expl.explain_level1(v=v, top_m=10_000, method="margin_drop")
    incoming = [it.ho_node_id for it in lvl1_items]
    scores_expl = np.array([it.score_margin_drop for it in lvl1_items], dtype=float)

    print(f"\n=== Level 1 (node v={v}) ===")
    print(f"predicted class y_ref = {y_ref}, full margin = {margin0:.6f}, #incoming HO nodes = {M}")

    # Ground truth: Shapley
    phi, incoming_gt, y_ref_gt, margin_full_gt = shapley_level1_permutation(
        expl, v=v, num_perm=args.lvl1_perm, seed=args.seed
    )
    # Align Shapley values to explainer's ordering (lvl1_items are sorted by score).
    phi_map = {int(q): float(phi[i]) for i, q in enumerate(incoming_gt)}
    phi_aligned = np.array([phi_map[int(q)] for q in incoming], dtype=float)

    # Compare rankings
    rho = spearmanr(scores_expl, phi_aligned)
    ndcg10 = ndcg_at_k(np.maximum(phi_aligned, 0.0), scores_expl, k=min(10, len(phi_aligned)))

    print(f"Level1: Spearman(explainer, Shapley) = {rho:.4f}")
    print(f"Level1: NDCG@10 (pos Shapley rel) = {ndcg10:.4f}")

    # Deletion curve AUC
    frac, margins = deletion_curve_level1(expl, v=v, ranking_q=incoming, y_ref=y_ref)
    auc = auc_trapz(frac, margins)
    print(f"Level1 deletion AUC (margin vs removed fraction) = {auc:.6f}")

    # Optional generator-ground-truth: within-cluster for HO nodes
    if labels is not None:
        gt_bin = np.array([within_cluster_ho_node((it.u, it.v), labels) for it in lvl1_items], dtype=int)
        p10 = precision_at_k(gt_bin, scores_expl, k=min(10, len(gt_bin)))
        auroc = auc_roc(gt_bin, scores_expl)
        print(f"Level1 generator-GT: precision@10(intra-cluster)={p10:.3f}, AUROC={auroc:.3f}")

    # Level 2 evaluation for top_m HO nodes
    print(f"\n=== Level 2 (top {args.top_m} HO nodes from Level 1) ===")
    top_lvl1 = lvl1_items[: args.top_m]

    for rank, it in enumerate(top_lvl1, start=1):
        q = int(it.ho_node_id)
        pair = (int(it.u), int(it.v))
        print(f"\n-- HO node #{rank}: q={q}, pair={pair}, lvl1_score={it.score_margin_drop:.6f}")

        # Explainer Level 2 (IG by default)
        lvl2 = expl.explain_level2(
            ho_node_id=q,
            k=args.k,
            direction=args.direction,
            method="ig",
            steps=50,
            top_e=args.top_e_all,
            restrict_to_neighborhood=True,
        )

        nodes_k, edges_k = expl.k_hop_neighborhood_ho(q, k=args.k, direction=args.direction)
        if len(edges_k) == 0:
            print("   (no HO edges in neighborhood)")
            continue

        # Extract explainer scores for ALL neighborhood edges:
        # top_edges contains all edges if top_e_all is large enough; otherwise only top subset.
        top_edges = lvl2.get("top_edges", [])
        expl_edge_score: Dict[int, float] = {}
        for d in top_edges:
            expl_edge_score[int(d["edge_id"])] = float(d["score"])

        # Ensure we have scores for all edges_k; if not, fill missing with 0
        scores_expl2 = np.array([expl_edge_score.get(int(e), 0.0) for e in edges_k], dtype=float)

        # Ground truth for Level 2: LOO
        edges_k2, loo_scores, f_full = loo_level2_edges(
            expl, q=q, k=args.k, direction=args.direction, restrict_to_neighborhood=True
        )
        assert edges_k2 == edges_k, "Neighborhood edge list mismatch (unexpected)."

        rho2 = spearmanr(np.abs(scores_expl2), np.abs(loo_scores))
        ndcg2 = ndcg_at_k(np.maximum(loo_scores, 0.0), np.abs(scores_expl2), k=min(10, len(edges_k)))
        print(f"   #edges in k-hop neighborhood: {len(edges_k)}")
        print(f"   Level2: Spearman(|explainer|, |LOO|) = {rho2:.4f}")
        print(f"   Level2: NDCG@10 (pos LOO rel) = {ndcg2:.4f}")

        # Deletion curve on embedding similarity
        ranking_edges = [e for e in edges_k]  # order by explainer score descending
        ranking_edges.sort(key=lambda e: abs(expl_edge_score.get(int(e), 0.0)), reverse=True)
        frac2, sims = deletion_curve_level2_embedding(expl, q=q, edges_k=edges_k, ranking_edges=ranking_edges)
        auc_sim = auc_trapz(frac2, sims)
        print(f"   Level2 deletion AUC (cos sim vs removed fraction) = {auc_sim:.6f}")

        # Optional generator-ground-truth for HO edges (within-cluster u,v,w)
        if labels is not None:
            gt_bin_e = []
            for e in edges_k:
                s = int(expl.g2_edge_index[0, int(e)].item())
                t = int(expl.g2_edge_index[1, int(e)].item())
                src_pair = tuple(expl.g2_node_ids[s].detach().cpu().numpy().tolist())
                dst_pair = tuple(expl.g2_node_ids[t].detach().cpu().numpy().tolist())
                gt_bin_e.append(within_cluster_ho_edge(src_pair, dst_pair, labels))
            gt_bin_e = np.array(gt_bin_e, dtype=int)
            p10e = precision_at_k(gt_bin_e, np.abs(scores_expl2), k=min(10, len(gt_bin_e)))
            auroce = auc_roc(gt_bin_e, np.abs(scores_expl2))
            print(f"   Level2 generator-GT: precision@10(intra)={p10e:.3f}, AUROC={auroce:.3f}")


if __name__ == "__main__":
    main()
