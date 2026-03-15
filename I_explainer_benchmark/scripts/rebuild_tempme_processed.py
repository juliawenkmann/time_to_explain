from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


DEGREE_BY_DATASET: dict[str, int] = {
    "wikipedia": 20,
    "reddit": 20,
    "simulate_v1": 20,
    "simulate_v2": 20,
    "uci": 30,
    "ucim": 30,
    "mooc": 60,
    "enron": 30,
    "canparl": 30,
    "uslegis": 30,
}

ANONYMOUS_PATTERNS = [
    "1,2,1",
    "1,2,2",
    "1,2,3",
    "1,2,0",
    "1,3,1",
    "1,3,3",
    "1,3,2",
    "1,3,0",
    "1,1,3",
    "1,1,2",
    "1,1,1",
    "1,1,0",
]


def _vendor_root(project_root: Path, tempme_root: str | Path | None) -> Path:
    if tempme_root is not None:
        return Path(tempme_root).expanduser().resolve()
    return (project_root / "I_explainer_benchmark" / "submodules" / "explainer" / "TempME").resolve()


def _bootstrap_vendor(tempme_root: Path) -> tuple[object, object]:
    root_str = str(tempme_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from utils import NeighborFinder, RandEdgeSampler

    return NeighborFinder, RandEdgeSampler


def _load_split(
    *,
    tempme_root: Path,
    dataset: str,
    mode: str,
    NeighborFinder: object,
    RandEdgeSampler: object,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    g_df = pd.read_csv(tempme_root / "processed" / f"ml_{dataset}.csv")
    val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

    src_l = g_df.u.values
    dst_l = g_df.i.values
    e_idx_l = g_df.idx.values
    label_l = g_df.label.values
    ts_l = g_df.ts.values

    max_idx = int(max(src_l.max(), dst_l.max()))

    random.seed(2023)
    total_node_set = set(np.unique(np.hstack([src_l, dst_l])))
    num_total_unique_nodes = len(total_node_set)
    candidate_nodes = sorted(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])))
    sample_size = min(int(0.1 * num_total_unique_nodes), len(candidate_nodes))
    mask_node_set = set(random.sample(candidate_nodes, sample_size)) if sample_size > 0 else set()

    mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
    mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
    none_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)

    valid_train_flag = (ts_l <= val_time) * (none_node_flag > 0)
    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]
    train_label_l = label_l[valid_train_flag]

    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
    valid_test_flag = ts_l > test_time
    val_src_l = src_l[valid_val_flag]
    val_dst_l = dst_l[valid_val_flag]
    test_src_l = src_l[valid_test_flag]
    test_dst_l = dst_l[valid_test_flag]
    test_ts_l = ts_l[valid_test_flag]
    test_e_idx_l = e_idx_l[valid_test_flag]
    test_label_l = label_l[valid_test_flag]

    adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
        adj_list[int(src)].append((int(dst), int(eidx), float(ts)))
        adj_list[int(dst)].append((int(src), int(eidx), float(ts)))
    train_ngh_finder = NeighborFinder(adj_list)

    full_adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
        full_adj_list[int(src)].append((int(dst), int(eidx), float(ts)))
        full_adj_list[int(dst)].append((int(src), int(eidx), float(ts)))
    full_ngh_finder = NeighborFinder(full_adj_list)

    train_rand_sampler = RandEdgeSampler((train_src_l,), (train_dst_l,))
    test_rand_sampler = RandEdgeSampler((train_src_l, val_src_l, test_src_l), (train_dst_l, val_dst_l, test_dst_l))

    if str(mode).strip().lower() == "test":
        return test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l, full_ngh_finder
    return train_rand_sampler, train_src_l, train_dst_l, train_ts_l, train_label_l, train_e_idx_l, train_ngh_finder


def _marginalize_walks(
    walks_src: np.ndarray,
    walks_tgt: np.ndarray,
    walks_bgd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_records_src = walks_src[:, :, :6]
    eidx_records_src = walks_src[:, :, 6:9]
    t_records_src = walks_src[:, :, 9:12]
    out_anony_src = walks_src[:, :, 12:15].astype(int)

    node_records_tgt = walks_tgt[:, :, :6]
    eidx_records_tgt = walks_tgt[:, :, 6:9]
    t_records_tgt = walks_tgt[:, :, 9:12]
    out_anony_tgt = walks_tgt[:, :, 12:15].astype(int)

    node_records_bgd = walks_bgd[:, :, :6]
    eidx_records_bgd = walks_bgd[:, :, 6:9]
    t_records_bgd = walks_bgd[:, :, 9:12]
    out_anony_bgd = walks_bgd[:, :, 12:15].astype(int)

    num_data, num_walks = out_anony_src.shape[:2]
    marginal_repr_src = np.empty((num_data, num_walks, 1), dtype=np.float32)
    marginal_repr_tgt = np.empty((num_data, num_walks, 1), dtype=np.float32)
    marginal_repr_bgd = np.empty((num_data, num_walks, 1), dtype=np.float32)
    cate_feat_src = np.empty((num_data, num_walks, 1), dtype=np.float32)
    cate_feat_tgt = np.empty((num_data, num_walks, 1), dtype=np.float32)
    cate_feat_bgd = np.empty((num_data, num_walks, 1), dtype=np.float32)

    sat = {item: 0 for item in ANONYMOUS_PATTERNS}
    strint_rep: dict[str, str] = {}
    strint_id: dict[str, int] = {}
    for idx, item in enumerate(ANONYMOUS_PATTERNS):
        arr = np.array(list(eval(item)))
        key = np.array2string(arr)
        strint_rep[key] = item
        strint_id[key] = idx

    for i in range(num_data):
        samples_src = out_anony_src[i]
        samples_tgt = out_anony_tgt[i]
        samples_bgd = out_anony_bgd[i]
        for j in range(samples_src.shape[0]):
            sat[strint_rep[np.array2string(samples_src[j])]] += 1
            sat[strint_rep[np.array2string(samples_tgt[j])]] += 1
            sat[strint_rep[np.array2string(samples_bgd[j])]] += 1

    total = float(num_data * num_walks * 3)
    for key in list(sat):
        sat[key] = float(sat[key]) / total if total > 0 else 0.0

    for i in tqdm(range(num_data), desc="Annotating walk motifs"):
        samples_src = out_anony_src[i]
        samples_tgt = out_anony_tgt[i]
        samples_bgd = out_anony_bgd[i]
        for j in range(samples_src.shape[0]):
            src_key = np.array2string(samples_src[j])
            tgt_key = np.array2string(samples_tgt[j])
            bgd_key = np.array2string(samples_bgd[j])
            marginal_repr_src[i, j, 0] = sat[strint_rep[src_key]]
            marginal_repr_tgt[i, j, 0] = sat[strint_rep[tgt_key]]
            marginal_repr_bgd[i, j, 0] = sat[strint_rep[bgd_key]]
            cate_feat_src[i, j, 0] = strint_id[src_key]
            cate_feat_tgt[i, j, 0] = strint_id[tgt_key]
            cate_feat_bgd[i, j, 0] = strint_id[bgd_key]

    walks_src_new = np.concatenate(
        [node_records_src, eidx_records_src, t_records_src, cate_feat_src, marginal_repr_src],
        axis=-1,
    ).astype(np.float32, copy=False)
    walks_tgt_new = np.concatenate(
        [node_records_tgt, eidx_records_tgt, t_records_tgt, cate_feat_tgt, marginal_repr_tgt],
        axis=-1,
    ).astype(np.float32, copy=False)
    walks_bgd_new = np.concatenate(
        [node_records_bgd, eidx_records_bgd, t_records_bgd, cate_feat_bgd, marginal_repr_bgd],
        axis=-1,
    ).astype(np.float32, copy=False)
    return walks_src_new, walks_tgt_new, walks_bgd_new


def _edge_info(edge_ids: np.ndarray) -> np.ndarray:
    bsz, n_walks, walk_len = edge_ids.shape
    emb = np.zeros((bsz, n_walks, walk_len, walk_len), dtype=np.float32)
    for k in tqdm(range(bsz), desc="Building edge features"):
        walk_edge_ids = edge_ids[k]
        unique_ids, inverse = np.unique(walk_edge_ids, return_inverse=True)
        counts = np.zeros((unique_ids.shape[0], walk_edge_ids.shape[-1]), dtype=np.float32)
        for idx, item in enumerate(unique_ids):
            counts[idx] = np.count_nonzero(walk_edge_ids == item, axis=0)
        emb[k] = counts[inverse.reshape(n_walks * walk_len)].reshape(n_walks, walk_len, walk_len)
    return emb


def _calculate_edge_features(
    walks_src_new: np.ndarray,
    walks_tgt_new: np.ndarray,
    walks_bgd_new: np.ndarray,
) -> np.ndarray:
    edge_src = _edge_info(walks_src_new[:, :, 6:9].astype(int))
    edge_tgt = _edge_info(walks_tgt_new[:, :, 6:9].astype(int))
    edge_bgd = _edge_info(walks_bgd_new[:, :, 6:9].astype(int))
    return np.stack([edge_src, edge_tgt, edge_bgd], axis=0).astype(np.float32, copy=False)


def _extract_processed_pack(
    *,
    sampler: object,
    src: np.ndarray,
    dst: np.ndarray,
    ts: np.ndarray,
    e_idx: np.ndarray,
    ngh_finder: object,
    degree: int,
    limit: int | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    num_rows = len(src) - 1
    if limit is not None:
        num_rows = min(num_rows, int(limit))

    load_dict: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "subgraph_src_0",
            "subgraph_src_1",
            "subgraph_tgt_0",
            "subgraph_tgt_1",
            "subgraph_bgd_0",
            "subgraph_bgd_1",
            "walks_src",
            "walks_tgt",
            "walks_bgd",
            "dst_fake",
        )
    }

    for k in tqdm(range(num_rows), desc="Extracting subgraphs"):
        src_l_cut = src[k : k + 1]
        dst_l_cut = dst[k : k + 1]
        ts_l_cut = ts[k : k + 1]
        e_l_cut = e_idx[k : k + 1]
        _, dst_l_fake = sampler.sample(len(src_l_cut))
        load_dict["dst_fake"].append(dst_l_fake.astype(np.int64, copy=False))

        subgraph_src = ngh_finder.find_k_hop(2, src_l_cut, ts_l_cut, num_neighbors=degree, e_idx_l=e_l_cut)
        node_records, eidx_records, t_records = subgraph_src
        load_dict["subgraph_src_0"].append(
            np.concatenate([node_records[0], eidx_records[0], t_records[0]], axis=-1).astype(np.float32, copy=False)
        )
        load_dict["subgraph_src_1"].append(
            np.concatenate([node_records[1], eidx_records[1], t_records[1]], axis=-1).astype(np.float32, copy=False)
        )

        subgraph_tgt = ngh_finder.find_k_hop(2, dst_l_cut, ts_l_cut, num_neighbors=degree, e_idx_l=e_l_cut)
        node_records, eidx_records, t_records = subgraph_tgt
        load_dict["subgraph_tgt_0"].append(
            np.concatenate([node_records[0], eidx_records[0], t_records[0]], axis=-1).astype(np.float32, copy=False)
        )
        load_dict["subgraph_tgt_1"].append(
            np.concatenate([node_records[1], eidx_records[1], t_records[1]], axis=-1).astype(np.float32, copy=False)
        )

        subgraph_bgd = ngh_finder.find_k_hop(2, dst_l_fake, ts_l_cut, num_neighbors=degree, e_idx_l=None)
        node_records, eidx_records, t_records = subgraph_bgd
        load_dict["subgraph_bgd_0"].append(
            np.concatenate([node_records[0], eidx_records[0], t_records[0]], axis=-1).astype(np.float32, copy=False)
        )
        load_dict["subgraph_bgd_1"].append(
            np.concatenate([node_records[1], eidx_records[1], t_records[1]], axis=-1).astype(np.float32, copy=False)
        )

        walks_src = ngh_finder.find_k_walks(degree, src_l_cut, num_neighbors=3, subgraph_src=subgraph_src)
        walks_tgt = ngh_finder.find_k_walks(degree, dst_l_cut, num_neighbors=3, subgraph_src=subgraph_tgt)
        walks_bgd = ngh_finder.find_k_walks(degree, dst_l_fake, num_neighbors=3, subgraph_src=subgraph_bgd)

        node_records, eidx_records, t_records, out_anony = walks_src
        load_dict["walks_src"].append(
            np.concatenate([node_records, eidx_records, t_records, out_anony], axis=-1).astype(np.float32, copy=False)
        )
        node_records, eidx_records, t_records, out_anony = walks_tgt
        load_dict["walks_tgt"].append(
            np.concatenate([node_records, eidx_records, t_records, out_anony], axis=-1).astype(np.float32, copy=False)
        )
        node_records, eidx_records, t_records, out_anony = walks_bgd
        load_dict["walks_bgd"].append(
            np.concatenate([node_records, eidx_records, t_records, out_anony], axis=-1).astype(np.float32, copy=False)
        )

    saved = {
        "subgraph_src_0": np.concatenate(load_dict["subgraph_src_0"], axis=0),
        "subgraph_src_1": np.concatenate(load_dict["subgraph_src_1"], axis=0),
        "subgraph_tgt_0": np.concatenate(load_dict["subgraph_tgt_0"], axis=0),
        "subgraph_tgt_1": np.concatenate(load_dict["subgraph_tgt_1"], axis=0),
        "subgraph_bgd_0": np.concatenate(load_dict["subgraph_bgd_0"], axis=0),
        "subgraph_bgd_1": np.concatenate(load_dict["subgraph_bgd_1"], axis=0),
    }
    walks_src_new, walks_tgt_new, walks_bgd_new = _marginalize_walks(
        np.concatenate(load_dict["walks_src"], axis=0),
        np.concatenate(load_dict["walks_tgt"], axis=0),
        np.concatenate(load_dict["walks_bgd"], axis=0),
    )
    saved["walks_src_new"] = walks_src_new
    saved["walks_tgt_new"] = walks_tgt_new
    saved["walks_bgd_new"] = walks_bgd_new
    saved["dst_fake"] = np.concatenate(load_dict["dst_fake"], axis=0).astype(np.int64, copy=False)
    edge_features = _calculate_edge_features(walks_src_new, walks_tgt_new, walks_bgd_new)
    return saved, edge_features


def _write_outputs(
    *,
    tempme_root: Path,
    dataset: str,
    mode: str,
    saved: dict[str, np.ndarray],
    edge_features: np.ndarray,
) -> None:
    processed_dir = tempme_root / "processed"
    h5_path = processed_dir / f"{dataset}_{mode}_cat.h5"
    edge_path = processed_dir / f"{dataset}_{mode}_edge.npy"

    with h5py.File(h5_path, "w") as hf:
        for key, value in saved.items():
            hf.create_dataset(key, data=value)
    np.save(edge_path, edge_features.astype(np.float32, copy=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild TempME processed explainer caches.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. ucim")
    parser.add_argument("--mode", choices=("train", "test", "both"), default="both")
    parser.add_argument("--degree", type=int, default=None, help="Override neighborhood degree")
    parser.add_argument("--tempme-root", type=str, default=None, help="Path to the TempME vendor root")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for a quick smoke rebuild")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[2]
    tempme_root = _vendor_root(project_root, args.tempme_root)
    dataset = str(args.dataset).strip().lower()
    degree = int(args.degree if args.degree is not None else DEGREE_BY_DATASET[dataset])
    NeighborFinder, RandEdgeSampler = _bootstrap_vendor(tempme_root)

    modes = ("train", "test") if args.mode == "both" else (str(args.mode),)
    print(f"TempME root: {tempme_root}")
    print(f"Dataset    : {dataset}")
    print(f"Degree     : {degree}")
    print(f"Modes      : {', '.join(modes)}")
    if args.limit is not None:
        print(f"Limit      : {int(args.limit)}")

    for mode in modes:
        print(f"\nRebuilding {dataset}_{mode}_cat.h5 and {dataset}_{mode}_edge.npy")
        sampler, src, dst, ts, _label, e_idx, ngh_finder = _load_split(
            tempme_root=tempme_root,
            dataset=dataset,
            mode=mode,
            NeighborFinder=NeighborFinder,
            RandEdgeSampler=RandEdgeSampler,
        )
        saved, edge_features = _extract_processed_pack(
            sampler=sampler,
            src=src,
            dst=dst,
            ts=ts,
            e_idx=e_idx,
            ngh_finder=ngh_finder,
            degree=degree,
            limit=args.limit,
        )
        _write_outputs(
            tempme_root=tempme_root,
            dataset=dataset,
            mode=mode,
            saved=saved,
            edge_features=edge_features,
        )
        print(
            f"Wrote {dataset}_{mode}_cat.h5 with subgraph_src_0 shape {saved['subgraph_src_0'].shape} "
            f"and {dataset}_{mode}_edge.npy with shape {edge_features.shape}"
        )


if __name__ == "__main__":
    main()
