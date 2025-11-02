import numpy as np
import random
from tqdm import tqdm
import os.path as osp
import pandas as pd
import numpy as np
from ..graph import NeighborFinder
from ...utils.batch_loader import RandEdgeSampler

degree_dict = {"wikipedia": 20, "reddit": 20, "uci": 30, "mooc": 60, "enron": 30, "canparl": 30, "uslegis": 30}


def load_data_shuffle(mode, data):
    g_df = pd.read_csv(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'processed/ml_{}.csv'.format(data)))
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
    mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])),
                                      int(0.1 * num_total_unique_nodes)))
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
    num_neighbors = degree_dict[data_name]
    rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l, finder = load_data_shuffle(mode="test", data=data_name)
    num_distribution = pre_processing(finder, rand_sampler, test_src_l, test_dst_l, test_ts_l, test_e_idx_l,num_neighbors)
    return num_distribution
