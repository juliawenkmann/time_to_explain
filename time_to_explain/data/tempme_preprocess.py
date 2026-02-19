from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple
import random

import h5py
import numpy as np
import pandas as pd
try:
    from submodules.explainer.tempme.utils import NeighborFinder, RandEdgeSampler
except ModuleNotFoundError:  # fallback when submodules isn't importable as a package
    import importlib
    import sys
    tempme_root = Path(__file__).resolve().parents[2] / "submodules" / "explainer" / "tempme"
    if not tempme_root.exists():
        raise
    if str(tempme_root) not in sys.path:
        sys.path.insert(0, str(tempme_root))
    # Avoid picking up a different "utils" already imported (e.g., TGN submodule).
    if "utils" in sys.modules:
        del sys.modules["utils"]
    utils_mod = importlib.import_module("utils")
    NeighborFinder = getattr(utils_mod, "NeighborFinder")  # type: ignore
    RandEdgeSampler = getattr(utils_mod, "RandEdgeSampler")  # type: ignore



_DEGREE_DEFAULTS = {
    "wikipedia": 20,
    "reddit": 20,
    "uci": 30,
    "mooc": 60,
    "enron": 30,
    "canparl": 30,
    "uslegis": 30,
}


@dataclass
class TempMEPreprocessConfig:
    dataset_name: str
    processed_dir: Path = Path("resources/datasets/processed")
    output_dir: Optional[Path] = None
    n_degree: Optional[int] = None
    batch_size: int = 1024
    seed: Optional[int] = 2023
    mask_ratio: float = 0.1
    min_train: int = 100
    min_val: int = 10
    min_test: int = 10
    validate_existing: bool = False
    overwrite: bool = False
    verbose: bool = True


def prepare_tempme_dataset(cfg: TempMEPreprocessConfig) -> Dict[str, Path]:
    processed_dir = Path(cfg.processed_dir)
    output_dir = Path(cfg.output_dir) if cfg.output_dir is not None else processed_dir
    dataset = cfg.dataset_name

    csv_path = processed_dir / f"ml_{dataset}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"TempME preprocessing requires {csv_path}")

    n_degree = cfg.n_degree
    if n_degree is None:
        n_degree = _DEGREE_DEFAULTS.get(dataset, 20)

    train_h5 = output_dir / f"{dataset}_train_cat.h5"
    test_h5 = output_dir / f"{dataset}_test_cat.h5"
    train_edge = output_dir / f"{dataset}_train_edge.npy"
    test_edge = output_dir / f"{dataset}_test_edge.npy"

    outputs = {
        "train_h5": train_h5,
        "test_h5": test_h5,
        "train_edge": train_edge,
        "test_edge": test_edge,
    }

    if not cfg.overwrite and all(p.exists() for p in outputs.values()):
        if not cfg.validate_existing:
            return outputs
        if _outputs_valid(outputs, min_train=cfg.min_train, min_test=cfg.min_test):
            return outputs
        if cfg.verbose:
            print("[TempME] Existing preprocessed files are too small; regenerating.")

    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    df = pd.read_csv(csv_path)
    splits = _split_events(
        df,
        seed=cfg.seed,
        mask_ratio=cfg.mask_ratio,
        min_train=cfg.min_train,
        min_val=cfg.min_val,
        min_test=cfg.min_test,
        verbose=cfg.verbose,
    )

    train_src, train_dst, train_ts, train_eidx = splits["train"]
    test_src, test_dst, test_ts, test_eidx = splits["test"]
    val_src, val_dst = splits["val_nodes"]

    train_finder = _build_neighbor_finder(train_src, train_dst, train_eidx, train_ts)
    full_finder = _build_neighbor_finder(
        df["u"].to_numpy(dtype=np.int64),
        df["i"].to_numpy(dtype=np.int64),
        _edge_index_array(df),
        df["ts"].to_numpy(dtype=np.float32),
    )

    train_sampler = RandEdgeSampler((train_src,), (train_dst,))
    test_sampler = RandEdgeSampler((train_src, val_src, test_src), (train_dst, val_dst, test_dst))

    _write_pack(
        split_name="train",
        out_h5=train_h5,
        out_edge=train_edge,
        src=train_src,
        dst=train_dst,
        ts=train_ts,
        eidx=train_eidx,
        ngh_finder=train_finder,
        rand_sampler=train_sampler,
        n_degree=n_degree,
        batch_size=cfg.batch_size,
        verbose=cfg.verbose,
    )
    _write_pack(
        split_name="test",
        out_h5=test_h5,
        out_edge=test_edge,
        src=test_src,
        dst=test_dst,
        ts=test_ts,
        eidx=test_eidx,
        ngh_finder=full_finder,
        rand_sampler=test_sampler,
        n_degree=n_degree,
        batch_size=cfg.batch_size,
        verbose=cfg.verbose,
    )

    return outputs


def _edge_index_array(df: pd.DataFrame) -> np.ndarray:
    if "idx" in df.columns:
        return df["idx"].to_numpy(dtype=np.int64)
    if "e_idx" in df.columns:
        return df["e_idx"].to_numpy(dtype=np.int64)
    return np.arange(1, len(df) + 1, dtype=np.int64)


def _split_events(
    df: pd.DataFrame,
    seed: Optional[int],
    *,
    mask_ratio: float = 0.1,
    min_train: int = 100,
    min_val: int = 100,
    min_test: int = 100,
    verbose: bool = False,
) -> Dict[str, Tuple[np.ndarray, ...]]:
    ts = df["ts"].to_numpy(dtype=np.float32)
    val_time, test_time = np.quantile(ts, [0.70, 0.85]).tolist()
    src = df["u"].to_numpy(dtype=np.int64)
    dst = df["i"].to_numpy(dtype=np.int64)
    eidx = _edge_index_array(df)

    if seed is not None:
        random.seed(seed)

    n_events = len(ts)
    min_train, min_val, min_test = _sanitize_minima(n_events, min_train, min_val, min_test)
    mask_ratio = float(max(0.0, min(1.0, mask_ratio)))

    total_nodes = np.unique(np.concatenate([src, dst]))
    num_total_nodes = len(total_nodes)
    future_nodes = set(src[ts > val_time]).union(set(dst[ts > val_time]))
    sample_size = int(mask_ratio * num_total_nodes)
    if sample_size > len(future_nodes):
        sample_size = len(future_nodes)
    future_nodes_seq = sorted(future_nodes)
    mask_nodes = set(random.sample(future_nodes_seq, sample_size)) if sample_size > 0 else set()
    if mask_nodes:
        mask_src = np.isin(src, list(mask_nodes))
        mask_dst = np.isin(dst, list(mask_nodes))
    else:
        mask_src = np.zeros_like(src, dtype=bool)
        mask_dst = np.zeros_like(dst, dtype=bool)

    none_node_flag = (~mask_src) & (~mask_dst)
    train_flag = (ts <= val_time) & none_node_flag
    val_flag = (ts > val_time) & (ts <= test_time)
    test_flag = ts > test_time

    if (
        train_flag.sum() < min_train
        or test_flag.sum() < min_test
        or val_flag.sum() < min_val
    ):
        if verbose:
            print(
                "[TempME] Inductive split too small; falling back to chronological split "
                f"(train={train_flag.sum()}, val={val_flag.sum()}, test={test_flag.sum()})."
            )
        train_flag, val_flag, test_flag = _fallback_time_split(
            ts,
            min_train=min_train,
            min_val=min_val,
            min_test=min_test,
        )

    train_src = src[train_flag]
    train_dst = dst[train_flag]
    train_ts = ts[train_flag]
    train_eidx = eidx[train_flag]

    val_src = src[val_flag]
    val_dst = dst[val_flag]

    test_src = src[test_flag]
    test_dst = dst[test_flag]
    test_ts = ts[test_flag]
    test_eidx = eidx[test_flag]

    return {
        "train": (train_src, train_dst, train_ts, train_eidx),
        "val_nodes": (val_src, val_dst),
        "test": (test_src, test_dst, test_ts, test_eidx),
    }


def _sanitize_minima(
    n_events: int, min_train: int, min_val: int, min_test: int
) -> Tuple[int, int, int]:
    if n_events <= 0:
        return 0, 0, 0
    min_train = max(1, int(min_train))
    min_val = max(0, int(min_val))
    min_test = max(0, int(min_test))

    if n_events == 1:
        return 1, 0, 0

    min_val = min(min_val, n_events - 1)
    min_test = min(min_test, n_events - 1)
    max_train = n_events - min_val - min_test
    if max_train < 1:
        # Reduce val/test to guarantee at least one training instance.
        deficit = 1 - max_train
        reduce_val = min(deficit, min_val)
        min_val -= reduce_val
        deficit -= reduce_val
        reduce_test = min(deficit, min_test)
        min_test -= reduce_test
    max_train = n_events - min_val - min_test
    min_train = min(min_train, max_train)
    if min_train < 1:
        min_train = 1
        if n_events - min_train - min_val - min_test < 0:
            min_val = max(0, n_events - min_train - min_test)
    return min_train, min_val, min_test


def _fallback_time_split(
    ts: np.ndarray,
    *,
    min_train: int,
    min_val: int,
    min_test: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_events = len(ts)
    if n_events == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, empty, empty

    min_train, min_val, min_test = _sanitize_minima(n_events, min_train, min_val, min_test)
    n_test = max(min_test, int(round(0.15 * n_events)))
    n_val = max(min_val, int(round(0.15 * n_events)))
    n_train = n_events - n_val - n_test
    if n_train < min_train:
        deficit = min_train - n_train
        reduce_val = min(deficit, n_val)
        n_val -= reduce_val
        deficit -= reduce_val
        reduce_test = min(deficit, n_test)
        n_test -= reduce_test
        deficit -= reduce_test
        n_train = n_events - n_val - n_test
    if n_train <= 0:
        n_train = max(1, n_events - n_test)
        n_val = max(0, n_events - n_train - n_test)
        n_test = n_events - n_train - n_val
    if n_events >= 2 and n_test == 0:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        elif n_val > 0:
            n_val -= 1

    order = np.argsort(ts, kind="stable")
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]

    train_flag = np.zeros(n_events, dtype=bool)
    val_flag = np.zeros(n_events, dtype=bool)
    test_flag = np.zeros(n_events, dtype=bool)
    train_flag[train_idx] = True
    if len(val_idx) > 0:
        val_flag[val_idx] = True
    if len(test_idx) > 0:
        test_flag[test_idx] = True
    return train_flag, val_flag, test_flag


def _outputs_valid(
    outputs: Dict[str, Path],
    *,
    min_train: int,
    min_test: int,
) -> bool:
    try:
        with h5py.File(str(outputs["train_h5"]), "r") as h5:
            train_n = int(h5["subgraph_src_0"].shape[0])
        with h5py.File(str(outputs["test_h5"]), "r") as h5:
            test_n = int(h5["subgraph_src_0"].shape[0])
        if train_n < max(1, min_train) or test_n < max(1, min_test):
            return False
        train_edge_n = int(np.load(outputs["train_edge"], mmap_mode="r").shape[1])
        test_edge_n = int(np.load(outputs["test_edge"], mmap_mode="r").shape[1])
        if train_edge_n < train_n or test_edge_n < test_n:
            return False
        return True
    except Exception:
        return False


def _build_neighbor_finder(
    src: np.ndarray,
    dst: np.ndarray,
    eidx: np.ndarray,
    ts: np.ndarray,
) -> NeighborFinder:
    max_src = int(src.max()) if src.size else 0
    max_dst = int(dst.max()) if dst.size else 0
    max_idx = max(max_src, max_dst)
    adj_list = [[] for _ in range(max_idx + 1)]
    for s, d, e, t in zip(src, dst, eidx, ts):
        s = int(s)
        d = int(d)
        adj_list[s].append((d, int(e), float(t)))
        adj_list[d].append((s, int(e), float(t)))
    return NeighborFinder(adj_list)


def _write_pack(
    *,
    split_name: str,
    out_h5: Path,
    out_edge: Path,
    src: np.ndarray,
    dst: np.ndarray,
    ts: np.ndarray,
    eidx: np.ndarray,
    ngh_finder: NeighborFinder,
    rand_sampler: RandEdgeSampler,
    n_degree: int,
    batch_size: int,
    verbose: bool,
) -> None:
    out_h5 = Path(out_h5)
    out_edge = Path(out_edge)
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    out_edge.parent.mkdir(parents=True, exist_ok=True)

    n_events = len(src)
    n_walk = int(n_degree)
    if n_events == 0:
        raise ValueError(f"TempME preprocessing split '{split_name}' is empty.")

    dst_fake = rand_sampler.sample(n_events)[1].astype(np.int64)

    chunk = min(batch_size, n_events)
    with h5py.File(str(out_h5), "w") as h5:
        sub0 = h5.create_dataset(
            "subgraph_src_0",
            shape=(n_events, n_degree * 3),
            dtype="float32",
            chunks=(chunk, n_degree * 3),
            compression="gzip",
            compression_opts=4,
        )
        sub1 = h5.create_dataset(
            "subgraph_src_1",
            shape=(n_events, (n_degree ** 2) * 3),
            dtype="float32",
            chunks=(chunk, (n_degree ** 2) * 3),
            compression="gzip",
            compression_opts=4,
        )
        tgt0 = h5.create_dataset(
            "subgraph_tgt_0",
            shape=(n_events, n_degree * 3),
            dtype="float32",
            chunks=(chunk, n_degree * 3),
            compression="gzip",
            compression_opts=4,
        )
        tgt1 = h5.create_dataset(
            "subgraph_tgt_1",
            shape=(n_events, (n_degree ** 2) * 3),
            dtype="float32",
            chunks=(chunk, (n_degree ** 2) * 3),
            compression="gzip",
            compression_opts=4,
        )
        bgd0 = h5.create_dataset(
            "subgraph_bgd_0",
            shape=(n_events, n_degree * 3),
            dtype="float32",
            chunks=(chunk, n_degree * 3),
            compression="gzip",
            compression_opts=4,
        )
        bgd1 = h5.create_dataset(
            "subgraph_bgd_1",
            shape=(n_events, (n_degree ** 2) * 3),
            dtype="float32",
            chunks=(chunk, (n_degree ** 2) * 3),
            compression="gzip",
            compression_opts=4,
        )
        walks_src_ds = h5.create_dataset(
            "walks_src_new",
            shape=(n_events, n_walk, 14),
            dtype="float32",
            chunks=(chunk, n_walk, 14),
            compression="gzip",
            compression_opts=4,
        )
        walks_tgt_ds = h5.create_dataset(
            "walks_tgt_new",
            shape=(n_events, n_walk, 14),
            dtype="float32",
            chunks=(chunk, n_walk, 14),
            compression="gzip",
            compression_opts=4,
        )
        walks_bgd_ds = h5.create_dataset(
            "walks_bgd_new",
            shape=(n_events, n_walk, 14),
            dtype="float32",
            chunks=(chunk, n_walk, 14),
            compression="gzip",
            compression_opts=4,
        )
        dst_fake_ds = h5.create_dataset(
            "dst_fake",
            shape=(n_events,),
            dtype="int64",
            chunks=(chunk,),
            compression="gzip",
            compression_opts=4,
        )

        edge_mm = np.lib.format.open_memmap(
            out_edge,
            mode="w+",
            dtype="float32",
            shape=(3, n_events, n_walk, 3, 3),
        )

        if verbose:
            from tqdm import tqdm
            batch_iter = tqdm(range(0, n_events, batch_size), desc=f"TempME {split_name}")
        else:
            batch_iter = range(0, n_events, batch_size)

        for start in batch_iter:
            end = min(n_events, start + batch_size)
            sl = slice(start, end)
            src_b = src[sl]
            dst_b = dst[sl]
            ts_b = ts[sl]
            eidx_b = eidx[sl]
            dst_fake_b = dst_fake[sl]

            sub_src = ngh_finder.find_k_hop(2, src_b, ts_b, num_neighbors=n_degree, e_idx_l=eidx_b)
            sub_tgt = ngh_finder.find_k_hop(2, dst_b, ts_b, num_neighbors=n_degree, e_idx_l=eidx_b)
            sub_bgd = ngh_finder.find_k_hop(2, dst_fake_b, ts_b, num_neighbors=n_degree, e_idx_l=None)

            walks_src = ngh_finder.find_k_walks(n_degree, src_b, num_neighbors=1, subgraph_src=sub_src)
            walks_tgt = ngh_finder.find_k_walks(n_degree, dst_b, num_neighbors=1, subgraph_src=sub_tgt)
            walks_bgd = ngh_finder.find_k_walks(n_degree, dst_fake_b, num_neighbors=1, subgraph_src=sub_bgd)

            sub0[sl] = _pack_subgraph(sub_src, hop=0)
            sub1[sl] = _pack_subgraph(sub_src, hop=1)
            tgt0[sl] = _pack_subgraph(sub_tgt, hop=0)
            tgt1[sl] = _pack_subgraph(sub_tgt, hop=1)
            bgd0[sl] = _pack_subgraph(sub_bgd, hop=0)
            bgd1[sl] = _pack_subgraph(sub_bgd, hop=1)

            walks_src_ds[sl], edge_src = _pack_walks(walks_src)
            walks_tgt_ds[sl], edge_tgt = _pack_walks(walks_tgt)
            walks_bgd_ds[sl], edge_bgd = _pack_walks(walks_bgd)

            dst_fake_ds[sl] = dst_fake_b
            edge_mm[0, sl, :, :, :] = edge_src
            edge_mm[1, sl, :, :, :] = edge_tgt
            edge_mm[2, sl, :, :, :] = edge_bgd
        edge_mm.flush()


def _pack_subgraph(subgraph, hop: int) -> np.ndarray:
    node_records, eidx_records, t_records = subgraph
    n = node_records[hop]
    e = eidx_records[hop]
    t = t_records[hop]
    packed = np.concatenate([n, e, t], axis=1)
    return packed.astype(np.float32, copy=False)


def _pack_walks(walks) -> Tuple[np.ndarray, np.ndarray]:
    node_records, eidx_records, t_records, out_anony = walks
    cat_feat = _map_anony_to_cat(out_anony)
    marginal = np.zeros_like(cat_feat, dtype=np.float32)
    walks_new = np.concatenate(
        [
            node_records.astype(np.float32, copy=False),
            eidx_records.astype(np.float32, copy=False),
            t_records.astype(np.float32, copy=False),
            cat_feat.astype(np.float32, copy=False),
            marginal,
        ],
        axis=2,
    )
    edge_id = _edge_identity(node_records)
    return walks_new, edge_id


def _map_anony_to_cat(out_anony: np.ndarray) -> np.ndarray:
    # Matches the 12 motif categories order in TempME's null_model.pre_processing.
    out_anony = out_anony.astype(np.int64, copy=False)
    second = out_anony[..., 1]
    third = out_anony[..., 2]
    cat = np.zeros(second.shape, dtype=np.int64)
    map_23 = np.array([0, 1, 3, 2], dtype=np.int64)
    mask2 = second == 2
    mask3 = second == 3
    mask1 = second == 1
    if mask2.any():
        cat[mask2] = map_23[third[mask2]]
    if mask3.any():
        cat[mask3] = map_23[third[mask3]] + 4
    if mask1.any():
        cat[mask1] = third[mask1] + 8
    return cat[..., None]


def _edge_identity(node_records: np.ndarray) -> np.ndarray:
    # Edge-identity feature: for each walk edge, mark equality to the other walk edges.
    edges = np.stack(
        [
            node_records[:, :, 0:2],
            node_records[:, :, 2:4],
            node_records[:, :, 4:6],
        ],
        axis=2,
    )
    edges = np.sort(edges, axis=-1)
    eq = (edges[:, :, :, None, :] == edges[:, :, None, :, :]).all(-1).astype(np.float32)
    mask = (edges[..., 0] == 0) & (edges[..., 1] == 0)
    if mask.any():
        eq[mask] = 0.0
    return eq


__all__ = ["TempMEPreprocessConfig", "prepare_tempme_dataset"]
