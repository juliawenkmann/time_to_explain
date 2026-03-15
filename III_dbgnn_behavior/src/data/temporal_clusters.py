from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from urllib import request

import torch
from torch_geometric.transforms import RandomNodeSplit

import pathpyG as pp

from pathpy_utils import maybe_to_device


def _edge_index_tensor(ei) -> torch.Tensor:
    """Return edge_index as a plain torch.Tensor shape [2, E]."""
    if hasattr(ei, "as_tensor"):
        return ei.as_tensor()
    if isinstance(ei, torch.Tensor):
        return ei
    return torch.as_tensor(ei)


@dataclass(frozen=True)
class TemporalClustersAssets:
    """Extra objects derived from the dataset that the DBGNN builder needs."""

    t: pp.TemporalGraph
    m: pp.MultiOrderModel
    g: pp.Graph
    g2: pp.Graph
    # Optional: shuffle-time ground truth object (attached when requested).
    gt: Optional[object] = None


def generate_connected_hub_chain_dataset(
    *,
    device: torch.device,
    n_clusters: int = 3,
    cluster_size: int = 10,
    total_events: int = 60_000,
    intra_block_len: int = 80,
    inter_block_len: int = 120,
    seed: int = 42,
):
    """Generate a *connected* explainable temporal cluster dataset.

    This generator is based on the user's `dbgnn_explainable_dataset_connected` notebook.

    Key idea:
      - Each cluster ``c`` has a hub node ``h_c``.
      - In "intra" blocks we generate many causal (delta=1) 2-step paths that
        repeatedly pass through the hub and stay inside one cluster:
            u -> h_c @t,  h_c -> v @t+1,  v -> h_c @t+2
      - In "inter" blocks we add random edges *but* break causality by ensuring
        the next edge does not start at the previous target.

    This creates a temporal graph where the *static* aggregated topology is
    noisy/mixed, but the order-2 De Bruijn graph contains clear cluster
    structure.

    Returns:
        (t_graph, labels_by_node, hubs_by_class)
    """

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    # We use NumPy's Generator for convenience, but seed it from the torch gen.
    import numpy as np

    np_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
    nrng = np.random.default_rng(np_seed)

    N = int(n_clusters) * int(cluster_size)
    labels_by_node = {u: int(u // int(cluster_size)) for u in range(N)}
    hubs_by_class = {c: int(c * int(cluster_size)) for c in range(int(n_clusters))}

    def nodes_in_cluster(c: int):
        return list(range(c * int(cluster_size), (c + 1) * int(cluster_size)))

    tedges = []
    t = 0
    prev_target = None

    while t < int(total_events):
        # (A) Intra-cluster causal block
        c = int(nrng.integers(0, int(n_clusters)))
        hub = int(hubs_by_class[c])
        cluster_nodes = nodes_in_cluster(c)

        # pick a start node that does not create a causal link from previous block
        u = int(nrng.choice(cluster_nodes))
        if prev_target is not None and u == prev_target:
            u = int(nrng.choice([x for x in cluster_nodes if x != prev_target]))

        for _ in range(int(intra_block_len)):
            v = int(nrng.choice(cluster_nodes))

            tedges.append((u, hub, t))
            t += 1
            if t >= int(total_events):
                break

            tedges.append((hub, v, t))
            t += 1
            if t >= int(total_events):
                break

            # This helps keep the higher-order component connected.
            tedges.append((v, hub, t))
            t += 1
            if t >= int(total_events):
                break

            u = v

        prev_target = tedges[-1][1] if tedges else prev_target
        if t >= int(total_events):
            break

        # (B) Inter-cluster (static) noise block (break causality)
        for _ in range(int(inter_block_len)):
            a = int(nrng.integers(0, N))
            b = int(nrng.integers(0, N))
            # Ensure we do NOT get a causal 2-step path.
            if prev_target is not None and a == prev_target:
                a = (a + 1) % N

            tedges.append((a, b, t))
            t += 1
            prev_target = b
            if t >= int(total_events):
                break

    t_graph = pp.TemporalGraph.from_edge_list(tedges)
    t_graph = maybe_to_device(t_graph, device)
    return t_graph, labels_by_node, hubs_by_class


def load_temporal_clusters(
    *,
    device: torch.device,
    local_path: str | os.PathLike = "data/temporal_clusters.tedges",
    remote_url: str = "https://raw.githubusercontent.com/pathpy/pathpyG/refs/heads/main/docs/data/temporal_clusters.tedges",
    max_order: int = 2,
    num_test: float = 0.3,
    seed: Optional[int] = None,
    # Attach deterministic cluster-stay GT (cheap; recommended).
    attach_stay_gt: bool = True,
    stay_label_attr: str = "gt_stay_label_higher_order",
    stay_score_attr: str = "gt_stay_score_higher_order",
    triples_attr: str = "ho_triples",
    # Optional: compute and attach shuffle-time ground truth tensors for oracle explainers.
    attach_shuffle_gt: bool = False,
    delta_gt: int = 1,
    n_shuffles_gt: int = 10,
    z_thr_gt: float = 2.0,
    seed_gt: Optional[int] = None,
) -> Tuple[object, TemporalClustersAssets]:
    """Load the temporal_clusters toy dataset and produce the exact PyG Data used in dbgnn.ipynb.

    This mirrors the tutorial notebook cells:
    - load temporal graph from local_path else download from GitHub
    - move `t` to device when supported by the installed pathpyG version
    - m = MultiOrderModel.from_temporal_graph(t, max_order=2)
    - data = m.to_dbgnn_data(max_order=2, mapping="last")
    - data.y = torch.tensor([int(i)//10 for i in t.nodes], device=device)
    - RandomNodeSplit(num_val=0, num_test=0.3)(data)

    Extra convenience:
    - Attaches `data.<triples_attr>` aligned with `edge_index_higher_order` (triples (u,v,w))
    - Attaches deterministic cluster-stay GT (`gt_stay_*`) by default

    Args:
        device: torch device.
        local_path: Local dataset path (downloaded if missing).
        remote_url: Fallback URL if local_path doesn't exist.
        max_order: MultiOrderModel max order (2 for DBGNN tutorial).
        num_test: test node fraction.
        seed: RNG seed for the train/test split.
        attach_stay_gt: whether to attach deterministic stay-in-cluster GT.
        attach_shuffle_gt: whether to compute shuffle-time GT z-scores (slower).

    Returns:
        (data, assets)
    """
    path = Path(local_path)
    if path.exists():
        t = pp.io.read_csv_temporal_graph(path, header=False)
    else:
        # download to a temp location
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "temporal_clusters.tedges")
            request.urlretrieve(remote_url, file_path)  # nosec - controlled URL
            t = pp.io.read_csv_temporal_graph(file_path, header=False)

    t = maybe_to_device(t, device)

    # Exact notebook code:
    m = pp.MultiOrderModel.from_temporal_graph(t, max_order=max_order)
    g = m.layers[1]
    g2 = m.layers[2]

    data = m.to_dbgnn_data(max_order=max_order, mapping="last")
    data.y = torch.tensor([int(i) // 10 for i in t.nodes], device=device)

    # Reproducible random split (if seed is provided)
    #
    # Torch Geometric's `RandomNodeSplit` API changed across versions:
    # - some versions support passing a `generator=`
    # - others don't
    #
    # We try the reproducible variant first.
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        try:
            data = RandomNodeSplit(num_val=0, num_test=num_test)(data, generator=gen)
        except TypeError:
            data = RandomNodeSplit(num_val=0, num_test=num_test)(data)
    else:
        data = RandomNodeSplit(num_val=0, num_test=num_test)(data)

    # Ensure tensors live on the correct device.
    if hasattr(data, "to"):
        data = data.to(device)

    assets_base = TemporalClustersAssets(t=t, m=m, g=g, g2=g2, gt=None)

    # --- Attach higher-order triples aligned with edge_index_higher_order ---
    # Useful for:
    # - node-focused candidate masks (triples with middle==v)
    # - deterministic ground truth over triples
    # - oracle explainers
    from pathpy_utils import idx_to_node_list

    if hasattr(data, "edge_index_higher_order"):
        idx_to_ho = idx_to_node_list(g2)
        ei2 = _edge_index_tensor(getattr(data, "edge_index_higher_order"))
        E2 = int(ei2.size(1))

        u_list, v_list, w_list = [], [], []
        for e in range(E2):
            src = idx_to_ho[int(ei2[0, e])]
            dst = idx_to_ho[int(ei2[1, e])]
            # order-2 De Bruijn node IDs are tuples (u,v)
            u = int(src[0])
            v_ = int(src[1])
            w = int(dst[1])
            u_list.append(u)
            v_list.append(v_)
            w_list.append(w)

        triples = torch.tensor(list(zip(u_list, v_list, w_list)), dtype=torch.long, device=ei2.device)
        setattr(data, triples_attr, triples)

        if attach_stay_gt:
            from eval.stay_gt import attach_stay_gt_to_data

            attach_stay_gt_to_data(
                data,
                triples_attr=triples_attr,
                label_attr=stay_label_attr,
                score_attr=stay_score_attr,
            )

    if attach_shuffle_gt:
        # NOTE: this can be slow when n_shuffles_gt is large.
        from eval.shuffle_gt import attach_shuffle_gt_to_data

        base_seed = int(seed_gt) if seed_gt is not None else int(seed or 0)
        print(f"Attaching shuffle-time GT (delta={int(delta_gt)}, n_shuffles={int(n_shuffles_gt)}) ...")
        gt_obj = attach_shuffle_gt_to_data(
            data,
            assets_base,
            delta=int(delta_gt),
            n_shuffles=int(n_shuffles_gt),
            seed=base_seed,
            z_thr=float(z_thr_gt),
        )

        assets = TemporalClustersAssets(t=t, m=m, g=g, g2=g2, gt=gt_obj)
    else:
        assets = assets_base

    return data, assets


def load_synthetic_tedges(
    *,
    device: torch.device,
    local_path: str | os.PathLike = "data/synthetic.tedges",
    max_order: int = 2,
    num_test: float = 0.3,
    seed: Optional[int] = None,
    # Attach deterministic cluster-stay GT (cheap; recommended).
    attach_stay_gt: bool = True,
    stay_label_attr: str = "gt_stay_label_higher_order",
    stay_score_attr: str = "gt_stay_score_higher_order",
    triples_attr: str = "ho_triples",
    # Optional: compute and attach shuffle-time ground truth tensors for oracle explainers.
    attach_shuffle_gt: bool = False,
    delta_gt: int = 1,
    n_shuffles_gt: int = 10,
    z_thr_gt: float = 2.0,
    seed_gt: Optional[int] = None,
) -> Tuple[object, TemporalClustersAssets]:
    """Load a local synthetic temporal graph from data/synthetic.tedges.

    This mirrors :func:`load_temporal_clusters` but does not download anything.
    The file must exist locally.
    """

    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(f"Synthetic dataset not found: {path}")

    t = pp.io.read_csv_temporal_graph(path, header=False)
    t = maybe_to_device(t, device)

    m = pp.MultiOrderModel.from_temporal_graph(t, max_order=max_order)
    g = m.layers[1]
    g2 = m.layers[2]

    data = m.to_dbgnn_data(max_order=max_order, mapping="last")
    data.y = torch.tensor([int(i) // 10 for i in t.nodes], device=device)

    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        try:
            data = RandomNodeSplit(num_val=0, num_test=num_test)(data, generator=gen)
        except TypeError:
            data = RandomNodeSplit(num_val=0, num_test=num_test)(data)
    else:
        data = RandomNodeSplit(num_val=0, num_test=num_test)(data)

    if hasattr(data, "to"):
        data = data.to(device)

    assets_base = TemporalClustersAssets(t=t, m=m, g=g, g2=g2, gt=None)

    from pathpy_utils import idx_to_node_list

    if hasattr(data, "edge_index_higher_order"):
        idx_to_ho = idx_to_node_list(g2)
        ei2 = _edge_index_tensor(getattr(data, "edge_index_higher_order"))
        E2 = int(ei2.size(1))

        u_list, v_list, w_list = [], [], []
        for e in range(E2):
            src = idx_to_ho[int(ei2[0, e])]
            dst = idx_to_ho[int(ei2[1, e])]
            u = int(src[0])
            v_ = int(src[1])
            w = int(dst[1])
            u_list.append(u)
            v_list.append(v_)
            w_list.append(w)

        triples = torch.tensor(list(zip(u_list, v_list, w_list)), dtype=torch.long, device=ei2.device)
        setattr(data, triples_attr, triples)

        if attach_stay_gt:
            from eval.stay_gt import attach_stay_gt_to_data

            attach_stay_gt_to_data(
                data,
                triples_attr=triples_attr,
                label_attr=stay_label_attr,
                score_attr=stay_score_attr,
            )

    if attach_shuffle_gt:
        from eval.shuffle_gt import attach_shuffle_gt_to_data

        base_seed = int(seed_gt) if seed_gt is not None else int(seed or 0)
        print(f"Attaching shuffle-time GT (delta={int(delta_gt)}, n_shuffles={int(n_shuffles_gt)}) ...")
        gt_obj = attach_shuffle_gt_to_data(
            data,
            assets_base,
            delta=int(delta_gt),
            n_shuffles=int(n_shuffles_gt),
            seed=base_seed,
            z_thr=float(z_thr_gt),
        )

        assets = TemporalClustersAssets(t=t, m=m, g=g, g2=g2, gt=gt_obj)
    else:
        assets = assets_base

    return data, assets


def load_temporal_clusters_connected(
    *,
    device: torch.device,
    # Generator parameters (mirrors dbgnn_explainable_dataset_connected notebook)
    n_clusters: int = 3,
    cluster_size: int = 10,
    total_events: int = 60_000,
    intra_block_len: int = 80,
    inter_block_len: int = 120,
    seed_graph: int = 42,
    # DBGNN / pipeline parameters
    max_order: int = 2,
    num_test: float = 0.3,
    seed: Optional[int] = None,
    # Attach deterministic "stay-in-label" GT on higher-order edges
    attach_stay_gt: bool = True,
    stay_label_attr: str = "gt_stay_label_higher_order",
    stay_score_attr: str = "gt_stay_score_higher_order",
    triples_attr: str = "ho_triples",
) -> Tuple[object, TemporalClustersAssets]:
    """Generate a *connected* synthetic explainable dataset.

    This is a drop-in alternative to :func:`load_temporal_clusters`.

    In contrast to the original `temporal_clusters` toy dataset (which can
    produce several disconnected higher-order components), this generator
    explicitly keeps the order-2 structure connected by routing intra-cluster
    causal chains through a per-cluster hub.

    The dataset is meant for:
      - training DBGNN end-to-end
      - sanity checking explainers (counterfactual edge deletions should matter)

    Returns:
        (data, assets)
    """

    t, labels_by_node, _hubs = generate_connected_hub_chain_dataset(
        device=device,
        n_clusters=int(n_clusters),
        cluster_size=int(cluster_size),
        total_events=int(total_events),
        intra_block_len=int(intra_block_len),
        inter_block_len=int(inter_block_len),
        seed=int(seed_graph),
    )

    # Exact notebook line:
    # m = MultiOrderModel.from_temporal_graph(t, max_order=2)
    m = pp.MultiOrderModel.from_temporal_graph(t, max_order=max_order)
    g = m.layers[1]
    g2 = m.layers[2]

    data = m.to_dbgnn_data(max_order=max_order, mapping="last")

    # Labels are stored in labels_by_node. Ensure ordering follows t.nodes.
    node_ids = [int(i) for i in t.nodes]
    data.y = torch.tensor([int(labels_by_node[int(i)]) for i in node_ids], device=device)

    # Reproducible random split (see load_temporal_clusters for details)
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        try:
            data = RandomNodeSplit(num_val=0, num_test=num_test)(data, generator=gen)
        except TypeError:
            data = RandomNodeSplit(num_val=0, num_test=num_test)(data)
    else:
        data = RandomNodeSplit(num_val=0, num_test=num_test)(data)

    if hasattr(data, "to"):
        data = data.to(device)

    assets = TemporalClustersAssets(t=t, m=m, g=g, g2=g2, gt=None)

    # Attach higher-order triples aligned with edge_index_higher_order
    from pathpy_utils import idx_to_node_list

    if hasattr(data, "edge_index_higher_order"):
        idx_to_ho = idx_to_node_list(g2)
        ei2 = _edge_index_tensor(getattr(data, "edge_index_higher_order"))
        E2 = int(ei2.size(1))

        u_list, v_list, w_list = [], [], []
        for e in range(E2):
            src = idx_to_ho[int(ei2[0, e])]
            dst = idx_to_ho[int(ei2[1, e])]
            u = int(src[0])
            v_ = int(src[1])
            w = int(dst[1])
            u_list.append(u)
            v_list.append(v_)
            w_list.append(w)

        triples = torch.tensor(list(zip(u_list, v_list, w_list)), dtype=torch.long, device=ei2.device)
        setattr(data, triples_attr, triples)

        if attach_stay_gt:
            from eval.stay_gt import attach_label_stay_gt_to_data

            attach_label_stay_gt_to_data(
                data,
                triples_attr=triples_attr,
                y_attr="y",
                label_attr=stay_label_attr,
                score_attr=stay_score_attr,
            )

    return data, assets
