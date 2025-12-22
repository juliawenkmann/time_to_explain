import numpy as np
import random
from tqdm import tqdm
import os.path as osp
from pathlib import Path
import pandas as pd
import numpy as np
from .graph import NeighborFinder
from .batch_loader import RandEdgeSampler

degree_dict = {"wikipedia": 20, "reddit": 20, "uci": 30, "mooc": 60, "enron": 30, "canparl": 30, "uslegis": 30}
_MIN_TRAIN = 100
_MIN_VAL = 10
_MIN_TEST = 10


def _sanitize_minima(n_events: int, min_train: int, min_val: int, min_test: int):
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


def _fallback_time_split(ts, min_train, min_val, min_test):
    n_events = len(ts)
    if n_events == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)
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
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]
    train_flag = np.zeros(n_events, dtype=bool)
    val_flag = np.zeros(n_events, dtype=bool)
    test_flag = np.zeros(n_events, dtype=bool)
    train_flag[train_idx] = True
    if len(val_idx) > 0:
        val_flag[val_idx] = True
    if len(test_idx) > 0:
        test_flag[test_idx] = True
    return train_flag, val_flag, test_flag


def _resolve_csv_path(data: str) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        project_root / "resources" / "datasets" / "processed" / f"ml_{data}.csv",
        project_root / "submodules" / "explainer" / "TempME" / "processed" / f"ml_{data}.csv",
        Path(osp.join(osp.dirname(osp.realpath(__file__)), "..", f"processed/ml_{data}.csv")),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "TempME null-model csv not found. Tried:\n"
        + "\n".join(f" - {p}" for p in candidates)
    )


def load_data_shuffle(mode, data):
    csv_path = _resolve_csv_path(data)
    g_df = pd.read_csv(csv_path)
    val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

    src_l = g_df.u.values
    dst_l = g_df.i.values
    e_idx_l = g_df.idx.values
    label_l = g_df.label.values
    ts_l = g_df.ts.values
    length = len(ts_l)
    permutation = np.random.permutation(length)

    src_l = np.array(src_l)[permutation]
    dst_l = np.array(dst_l)[permutation]
    label_l = np.array(label_l)[permutation]

    max_src_index = src_l.max()
    max_idx = max(src_l.max(), dst_l.max())
    random.seed(2023)
    total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
    num_total_unique_nodes = len(total_node_set)
    future_nodes = set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time]))
    future_nodes_seq = sorted(future_nodes)
    mask_node_set = set(random.sample(future_nodes_seq, int(0.1 * num_total_unique_nodes))) if future_nodes_seq else set()
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

    min_train, min_val, min_test = _sanitize_minima(
        len(ts_l), _MIN_TRAIN, _MIN_VAL, _MIN_TEST
    )
    if (
        valid_train_flag.sum() < min_train
        or valid_val_flag.sum() < min_val
        or valid_test_flag.sum() < min_test
    ):
        valid_train_flag, valid_val_flag, valid_test_flag = _fallback_time_split(
            ts_l, min_train=min_train, min_val=min_val, min_test=min_test
        )
    val_src_l = src_l[valid_val_flag]
    val_dst_l = dst_l[valid_val_flag]
    test_src_l = src_l[valid_test_flag]
    test_dst_l = dst_l[valid_test_flag]
    test_ts_l = ts_l[valid_test_flag]
    test_e_idx_l = e_idx_l[valid_test_flag]
    test_label_l = label_l[valid_test_flag]
    adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
        adj_list[src].append((dst, eidx, ts))
        adj_list[dst].append((src, eidx, ts))
    train_ngh_finder = NeighborFinder(adj_list)
    # full graph with all the data for the test and validation purpose
    full_adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
        full_adj_list[src].append((dst, eidx, ts))
        full_adj_list[dst].append((src, eidx, ts))
    full_ngh_finder = NeighborFinder(full_adj_list)
    train_rand_sampler = RandEdgeSampler((train_src_l,), (train_dst_l,))
    # val_rand_sampler = RandEdgeSampler((train_src_l, val_src_l), (train_dst_l, val_dst_l))
    test_rand_sampler = RandEdgeSampler((train_src_l, val_src_l, test_src_l), (train_dst_l, val_dst_l, test_dst_l))
    if mode == "test":
        return test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l, full_ngh_finder
    else:
        return train_rand_sampler, train_src_l, train_dst_l, train_ts_l, train_label_l, train_e_idx_l, train_ngh_finder


def statistic(out_anony, sat, strint_rep):
    batch = out_anony.shape[0]
    for i in range(batch):
        samples = out_anony[i]  #[N, 3]
        for t in range(samples.shape[0]):
            anony_string = np.array2string(samples[t])
            sat[strint_rep[anony_string]] += 1
    return sat



def pre_processing(ngh_finder, sampler, src, dst, ts, val_e_idx_l, num_neighbors):
    strint_rep = {}
    t = 1
    sat = {}
    for item in ["1,2,0", "1,2,1","1,2,3","1,2,2","1,3,0","1,3,1","1,3,3","1,3,2","1,1,0","1,1,1","1,1,2","1,1,3"]:
        array_rep = np.array(list(eval(item)))
        string = np.array2string(array_rep)
        strint_rep[string] = t
        sat[t] = 0
        t = t + 1
    degree = num_neighbors
    batch_size = 10
    total_sample = 50 * batch_size
    for k in range(50):
        s_id = k*batch_size
        src_l_cut = src[s_id:s_id+batch_size]
        dst_l_cut = dst[s_id:s_id+batch_size]
        ts_l_cut = ts[s_id:s_id+batch_size]
        e_l_cut = val_e_idx_l[s_id:s_id+batch_size] if (val_e_idx_l is not None) else None
        size = len(src_l_cut)
        src_l_fake, dst_l_fake = sampler.sample(size)
        subgraph_src = ngh_finder.find_k_hop(2, src_l_cut, ts_l_cut, num_neighbors=num_neighbors, e_idx_l=e_l_cut)  #first: (batch, num_neighbors), second: [batch, num_neighbors * num_neighbors]
        subgraph_tgt = ngh_finder.find_k_hop(2, dst_l_cut, ts_l_cut, num_neighbors=num_neighbors, e_idx_l=e_l_cut) 
        subgraph_bgd = ngh_finder.find_k_hop(2, dst_l_fake, ts_l_cut, num_neighbors=num_neighbors, e_idx_l=None) 
        walks_src = ngh_finder.find_k_walks(degree, src_l_cut, num_neighbors=1, subgraph_src=subgraph_src)
        walks_tgt = ngh_finder.find_k_walks(degree, dst_l_cut, num_neighbors=1, subgraph_src=subgraph_tgt)
        walks_bgd = ngh_finder.find_k_walks(degree, dst_l_fake, num_neighbors=1, subgraph_src=subgraph_bgd)
        _, eidx_records_src, _, out_anony_src = walks_src
        _, eidx_records_tgt, _, out_anony_tgt = walks_tgt
        _, eidx_records_bgd, _, out_anony_bgd = walks_bgd
        sat = statistic(out_anony_src, sat,strint_rep)
        sat = statistic(out_anony_tgt, sat, strint_rep)
        sat = statistic(out_anony_bgd, sat, strint_rep)
    for key, value in sat.items():
        sat[key] = value / (total_sample*3*degree)
    return sat


def get_null_distribution(data_name):
    num_neighbors = degree_dict.get(data_name, 20)
    rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l, finder = load_data_shuffle(mode="test", data=data_name)
    num_distribution = pre_processing(finder, rand_sampler, test_src_l, test_dst_l, test_ts_l, test_e_idx_l,num_neighbors)
    return num_distribution
