from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data

import pathpyG as pp

from pathpy_utils import idx_to_node_list


NodeLike = Union[int, str]
Pair = Tuple[int, int]
Triple = Tuple[int, int, int]


def _edge_index_tensor(ei) -> torch.Tensor:
    """Return edge_index as a plain torch.Tensor shape [2, E]."""
    if hasattr(ei, "as_tensor"):
        return ei.as_tensor()
    if isinstance(ei, torch.Tensor):
        return ei
    return torch.as_tensor(ei)


def _clone_temporal_graph_minimal(t: pp.TemporalGraph) -> pp.TemporalGraph:
    """
    Minimal safe clone: only edge_index, time, num_nodes and mapping.
    (No node/edge attrs needed for MultiOrderModel counts.)
    """
    ei = _edge_index_tensor(t.data.edge_index).clone()
    tt = t.data.time.clone()
    num_nodes = int(t.data.num_nodes) if "num_nodes" in t.data else int(ei.max().item() + 1)
    data = Data(edge_index=ei, time=tt, num_nodes=num_nodes)
    # TemporalGraph.__init__ sorts by time, so cloning also guarantees sorted order.
    return pp.TemporalGraph(data=data, mapping=t.mapping)


def shuffle_temporal_graph_sorted(t: pp.TemporalGraph, seed: Optional[int] = None) -> pp.TemporalGraph:
    """
    Timestamp-shuffle null model:
    - permute the timestamps among events
    - then re-sort by time by reconstructing TemporalGraph (required for TRP extraction)

    This returns a NEW TemporalGraph and does not mutate the input.
    """
    g = _clone_temporal_graph_minimal(t)

    if seed is not None:
        # Make shuffle reproducible (uses torch.randperm below)
        gen = torch.Generator(device=g.data.time.device)
        gen.manual_seed(int(seed))
        perm = torch.randperm(g.data.time.numel(), generator=gen, device=g.data.time.device)
    else:
        perm = torch.randperm(g.data.time.numel(), device=g.data.time.device)

    # Shuffle timestamps only (keeps which (u,v) event exists; randomizes time assignments)
    g.data.time = g.data.time[perm]

    # Rebuild to enforce sorting by the *new* times (TemporalGraph.__init__ sorts)
    ei = _edge_index_tensor(g.data.edge_index)
    data = Data(edge_index=ei, time=g.data.time, num_nodes=g.data.num_nodes)
    return pp.TemporalGraph(data=data, mapping=g.mapping)


def ho2_triples_and_counts(t: pp.TemporalGraph, delta: int, device: Optional[torch.device] = None) -> Tuple[List[Triple], torch.Tensor]:
    """
    Compute order-2 De Bruijn edges as triples (u,v,w) and their counts (edge_weight).
    Uses pathpyG MultiOrderModel.from_temporal_graph.
    """
    g = _clone_temporal_graph_minimal(t)
    if device is not None:
        g = g.to(device)  # NOTE: .to mutates and returns self in pathpyG

    m = pp.MultiOrderModel.from_temporal_graph(g, delta=delta, max_order=2)
    g2 = m.layers[2]  # order-2 De Bruijn graph

    ei = _edge_index_tensor(g2.data.edge_index)
    src, dst = ei[0], ei[1]

    node_seq = g2.data.node_sequence  # [N2, 2] where each row is (u,v)
    src_pair = node_seq[src]         # [E, 2] -> (u,v)
    dst_pair = node_seq[dst]         # [E, 2] -> (v,w)

    u = src_pair[:, 0].to(torch.long)
    v = src_pair[:, 1].to(torch.long)
    w = dst_pair[:, 1].to(torch.long)

    triples: List[Triple] = list(zip(u.tolist(), v.tolist(), w.tolist()))
    counts = g2.data.edge_weight.to(torch.float32)  # [E]
    return triples, counts


@dataclass
class ShuffleGT:
    """
    Ground truth defined by deviation from shuffled-time null model on order-2 transitions.
    The 'universe' of edges is the order-2 edges observed in the REAL temporal graph.
    """
    triples: List[Triple]                 # length E_real
    real_counts: torch.Tensor             # [E_real]
    null_mean: torch.Tensor               # [E_real]
    null_std: torch.Tensor                # [E_real]
    z: torch.Tensor                       # [E_real]
    outgoing: Dict[Pair, np.ndarray]      # (u,v) -> indices into triples

    def indices_for_pair(self, u: int, v: int) -> np.ndarray:
        return self.outgoing.get((u, v), np.array([], dtype=np.int64))


def compute_shuffle_ground_truth(
    t: pp.TemporalGraph,
    delta: int,
    n_shuffles: int = 30,
    seed: int = 0,
    device: Optional[torch.device] = None,
    eps: float = 1e-9,
) -> ShuffleGT:
    """
    Build z-score ground truth for order-2 transitions vs shuffled-time null model.

    z(e) = (C_real(e) - mean(C_null(e))) / (std(C_null(e)) + eps)
    where e is a triple (u,v,w) for (u,v)->(v,w).

    Returns:
      ShuffleGT with z per REAL order-2 edge + index lists per De Bruijn node (u,v).
    """
    if n_shuffles < 2:
        raise ValueError("n_shuffles must be >= 2 to estimate a std dev reliably.")

    triples_real, counts_real = ho2_triples_and_counts(t, delta=delta, device=device)
    E = len(triples_real)

    # Map triple -> index in the REAL edge list
    triple_to_i: Dict[Triple, int] = {tr: i for i, tr in enumerate(triples_real)}

    # Precompute outgoing edge indices per (u,v)
    outgoing: Dict[Pair, List[int]] = {}
    for i, (u, v, w) in enumerate(triples_real):
        outgoing.setdefault((u, v), []).append(i)

    # Null counts matrix [S, E]
    null = torch.zeros((n_shuffles, E), dtype=torch.float32, device="cpu")

    # Compute null samples
    base = _clone_temporal_graph_minimal(t)  # keep CPU copy for shuffling
    for s in range(n_shuffles):
        t_shuf = shuffle_temporal_graph_sorted(base, seed=seed + s)
        triples_s, counts_s = ho2_triples_and_counts(t_shuf, delta=delta, device=device)

        # Fill only edges that exist in REAL (ignore new edges appearing only in null)
        for tr, c in zip(triples_s, counts_s.detach().cpu()):
            i = triple_to_i.get(tr)
            if i is not None:
                null[s, i] = float(c.item())

    null_mean = null.mean(dim=0)
    null_std = null.std(dim=0, unbiased=True)

    real_cpu = counts_real.detach().cpu()
    z = (real_cpu - null_mean) / (null_std + eps)

    outgoing_np = {k: np.asarray(v, dtype=np.int64) for k, v in outgoing.items()}

    return ShuffleGT(
        triples=triples_real,
        real_counts=real_cpu,
        null_mean=null_mean,
        null_std=null_std,
        z=z,
        outgoing=outgoing_np,
    )


# ---------- Evaluation over the FULL sparsity range (k = 0..deg_out) ----------

@dataclass
class PRCurve:
    k: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    ap: float


def precision_recall_over_k(scores: np.ndarray, labels: np.ndarray) -> PRCurve:
    """
    Full-range Precision/Recall@k for k=0..N, plus Average Precision (AP).
    labels must be 0/1.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    assert scores.ndim == 1 and labels.ndim == 1 and scores.shape[0] == labels.shape[0]

    n = scores.shape[0]
    pos = labels.sum()

    # Edge case: no positives in GT -> AP=0, recall always 0
    if pos == 0:
        k = np.arange(0, n + 1, dtype=int)
        precision = np.ones_like(k, dtype=float)
        recall = np.zeros_like(k, dtype=float)
        return PRCurve(k=k, precision=precision, recall=recall, ap=0.0)

    order = np.argsort(-scores)  # descending
    y = labels[order]

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)

    prec_at_k = tp / (tp + fp + 1e-12)
    rec_at_k = tp / pos

    # include k=0 point
    k = np.arange(0, n + 1, dtype=int)
    precision = np.concatenate([[1.0], prec_at_k])
    recall = np.concatenate([[0.0], rec_at_k])

    # Average Precision (stepwise integral of precision over recall)
    # AP = sum_{i: y_i=1} precision@i / (#positives)
    ap = float((prec_at_k[y == 1].sum() / pos))

    return PRCurve(k=k, precision=precision, recall=recall, ap=ap)


def gt_labels_for_pair(gt: ShuffleGT, u: int, v: int, z_thr: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ground-truth labels for outgoing transitions of De Bruijn node (u,v),
    using z-score threshold on (u,v,w).
    Returns (edge_indices, labels)
    """
    idx = gt.indices_for_pair(u, v)
    if idx.size == 0:
        return idx, np.zeros((0,), dtype=int)

    labels = (gt.z[idx].numpy() > z_thr).astype(int)
    return idx, labels

# ---------- Convenience: attach ground-truth tensors to a DBGNN Data object ----------

def attach_shuffle_gt_to_data(
    data,
    assets,
    *,
    delta: int = 1,
    n_shuffles: int = 30,
    seed: int = 0,
    z_thr: float = 2.0,
    device: Optional[torch.device] = None,
    z_attr: str = "gt_z_higher_order",
    label_attr: str = "gt_label_higher_order",
    triples_attr: str = "ho_triples",
) -> ShuffleGT:
    """Compute shuffle-time GT and attach it to `data` in-place.

    Adds three attributes aligned with `data.edge_index_higher_order`:

      - `data.<triples_attr>`: LongTensor [E, 3] with columns (u, v, w)
      - `data.<z_attr>`: FloatTensor [E] with z-scores
      - `data.<label_attr>`: FloatTensor [E] with 1.0 if z > z_thr else 0.0

    Why this exists:
      Explain- and eval-code only gets the PyG `Data` object. By attaching the
      ground-truth tensors once, oracle explainers can work inside the normal
      benchmark loop (no need to thread `assets` everywhere).

    Args:
      data: DBGNN PyG Data (must have edge_index_higher_order).
      assets: TemporalClustersAssets (must have assets.t and assets.g2).
      delta: causal delta for MultiOrderModel extraction (must match training).
      n_shuffles: number of timestamp shuffles for the null model.
      seed: base seed for shuffles.
      z_thr: threshold used to build binary labels.
      device: optional device for MultiOrderModel computation; defaults to data's device.
      z_attr / label_attr / triples_attr: attribute names to attach.

    Returns:
      ShuffleGT object (also useful for plotting/inspection).
    """
    if device is None:
        device = data.y.device if hasattr(data, "y") else None

    gt = compute_shuffle_ground_truth(
        assets.t,
        delta=delta,
        n_shuffles=n_shuffles,
        seed=seed,
        device=device,
    )

    # Map triple -> z-score for fast lookup
    triple_to_z: Dict[Triple, float] = {tr: float(gt.z[i].item()) for i, tr in enumerate(gt.triples)}

    if not hasattr(data, "edge_index_higher_order"):
        raise AttributeError("Data object has no attribute 'edge_index_higher_order'")

    ei = _edge_index_tensor(getattr(data, "edge_index_higher_order"))
    E = int(ei.size(1))

    # Build (u,v,w) for each higher-order edge in the SAME ORDER as edge_index_higher_order.
    idx_to_ho = idx_to_node_list(assets.g2)
    u_list: List[int] = []
    v_list: List[int] = []
    w_list: List[int] = []
    z_list: List[float] = []

    for e in range(E):
        src = idx_to_ho[int(ei[0, e])]
        dst = idx_to_ho[int(ei[1, e])]

        # Higher-order node IDs are expected to be (u,v) tuples for order-2 De Bruijn.
        if not (isinstance(src, tuple) and isinstance(dst, tuple) and len(src) == 2 and len(dst) == 2):
            raise TypeError(
                "Expected order-2 higher-order node IDs to be tuples (u,v). "
                f"Got src={type(src)} {src!r}, dst={type(dst)} {dst!r}"
            )

        u = int(src[0])
        v = int(src[1])
        w = int(dst[1])  # dst is (v,w)

        u_list.append(u)
        v_list.append(v)
        w_list.append(w)

        z_list.append(triple_to_z.get((u, v, w), 0.0))

    triples = torch.tensor(list(zip(u_list, v_list, w_list)), dtype=torch.long, device=ei.device)
    z = torch.tensor(z_list, dtype=torch.float32, device=ei.device)

    setattr(data, triples_attr, triples)
    setattr(data, z_attr, z)
    setattr(data, label_attr, (z > float(z_thr)).to(torch.float32))

    return gt