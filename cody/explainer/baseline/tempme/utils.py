import math
import random
from typing import Literal

import h5py
import numpy as np
from numba import jit
from tqdm import tqdm

from cody.data import ContinuousTimeDynamicGraphDataset
from cody.implementations.tgn import TGNWrapper
from cody.constants import COL_NODE_I, COL_NODE_U
from submodules.tgn.TGN.utils.utils import RandEdgeSampler, NeighborFinder


class TempMeNeighborFinder(NeighborFinder):
    def __init__(self, adj_list, bias=0, ts_precision=5, use_cache=False, sample_method='multinomial', device=None):
        """
        Params
        ------
        node_idx_l: List[int]
        node_ts_l: List[int]
        off_set_l: List[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i]
        """
        super().__init__(adj_list)
        self.bias = bias  # the "alpha" hyperparameter
        node_idx_l, node_ts_l, edge_idx_l, binary_prob_l, off_set_l, self.nodeedge2idx = self.init_off_set(adj_list)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l
        self.binary_prob_l = binary_prob_l
        self.off_set_l = off_set_l
        self.use_cache = use_cache
        self.cache = {}
        self.ts_precision = ts_precision
        self.sample_method = sample_method
        self.device = device

    def init_off_set(self, adj_list):
        """
        Params
        ------
        adj_list: List[List[int]]

        """
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        binary_prob_l = []
        off_set_l = [0]
        nodeedge2idx = {}
        for i in range(len(adj_list)):
            curr = adj_list[i]
            curr = sorted(curr, key=lambda x: x[2])  # neighbors sorted by time
            n_idx_l.extend([x[0] for x in curr])
            e_idx_l.extend([x[1] for x in curr])
            ts_l = [x[2] for x in curr]
            n_ts_l.extend(ts_l)
            binary_prob_l.append(self.compute_binary_prob(np.array(ts_l)))
            off_set_l.append(len(n_idx_l))
            # nodeedge2idx[i] = {x[1]: i for i, x in enumerate(curr)}
            nodeedge2idx[i] = self.get_ts2idx(curr)
        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        binary_prob_l = np.concatenate(binary_prob_l)
        off_set_l = np.array(off_set_l)

        assert (len(n_idx_l) == len(n_ts_l))
        assert (off_set_l[-1] == len(n_ts_l))

        return n_idx_l, n_ts_l, e_idx_l, binary_prob_l, off_set_l, nodeedge2idx

    def compute_binary_prob(self, ts_l):
        if len(ts_l) == 0:
            return np.array([])
        ts_l = ts_l - np.max(ts_l)
        exp_ts_l = np.exp(self.bias * ts_l)
        exp_ts_l /= np.cumsum(exp_ts_l)
        #         print( exp_ts_l_cumsum, exp_ts_l, ts_l, exp_ts_l)
        return exp_ts_l

    def get_ts2idx(self, sorted_triples):
        ts2idx = {}
        if len(sorted_triples) == 0:
            return ts2idx
        tie_ts_e_indices = []
        last_ts = -1
        last_e_idx = -1
        for i, (n_idx, e_idx, ts_idx) in enumerate(sorted_triples):
            ts2idx[e_idx] = i

            if ts_idx == last_ts:
                if len(tie_ts_e_indices) == 0:
                    tie_ts_e_indices = [last_e_idx, e_idx]
                else:
                    tie_ts_e_indices.append(e_idx)

            if (not (ts_idx == last_ts)) and (len(tie_ts_e_indices) > 0):
                tie_len = len(tie_ts_e_indices)
                for j, tie_ts_e_idx in enumerate(tie_ts_e_indices):
                    # ts2idx[tie_ts_e_idx] += tie_len - j
                    ts2idx[tie_ts_e_idx] -= j  # very crucial to exempt ties
                tie_ts_e_indices = []  # reset the temporary index list
            last_ts = ts_idx
            last_e_idx = e_idx
        return ts2idx

    def find_before(self, src_idx, cut_time, e_idx=None, return_binary_prob=False):
        """
        Params
        ------
        src_idx: int
        cut_time: float
        (optional) e_idx: can be used to perform look up by e_idx
        """
        if self.use_cache:
            result = self.check_cache(src_idx, cut_time)
            if result is not None:
                return result[0], result[1], result[2]

        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        off_set_l = self.off_set_l
        binary_prob_l = self.binary_prob_l  # TODO: make it in preprocessing
        start = off_set_l[src_idx]
        end = off_set_l[src_idx + 1]
        neighbors_idx = node_idx_l[start: end]
        neighbors_ts = node_ts_l[start: end]
        neighbors_e_idx = edge_idx_l[start: end]

        assert (len(neighbors_idx) == len(neighbors_ts) and len(neighbors_idx) == len(neighbors_e_idx))  # check the next line validality
        if e_idx is None:
            cut_idx = bisect_left_adapt(neighbors_ts, cut_time)  # very crucial to exempt ties (so don't use bisect)
        else:
            # use quick index mapping to get node index and edge index
            # a problem though may happens when there is a tie of timestamps
            cut_idx = self.nodeedge2idx[src_idx].get(e_idx) if src_idx > 0 else 0
            if cut_idx is None:
                raise IndexError('e_idx {} not found in edge list of {}'.format(e_idx, src_idx))
        if not return_binary_prob:
            result = (neighbors_idx[:cut_idx], neighbors_e_idx[:cut_idx], neighbors_ts[:cut_idx], None)
        else:
            neighbors_binary_prob = binary_prob_l[start: end]
            result = (
            neighbors_idx[:cut_idx], neighbors_e_idx[:cut_idx], neighbors_ts[:cut_idx], neighbors_binary_prob[:cut_idx])

        if self.use_cache:
            self.update_cache(src_idx, cut_time, result)

        return result


    def find_before_walk(self, src_idx_list, cut_time, e_idx=None, return_binary_prob=False):
        """
        Params
        ------
        src_idx: int
        cut_time: float
        (optional) e_idx: can be used to perform look up by e_idx
        """
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        off_set_l = self.off_set_l
        binary_prob_l = self.binary_prob_l  # TODO: make it in preprocessing
        idx, e_idxs, ts, prob, sources = [], [], [], [], []
        for src_idx in src_idx_list:
            start = off_set_l[src_idx]
            end = off_set_l[src_idx + 1]
            neighbors_idx = node_idx_l[start: end]
            neighbors_ts = node_ts_l[start: end]
            neighbors_e_idx = edge_idx_l[start: end]

            assert (len(neighbors_idx) == len(neighbors_ts) and len(neighbors_idx) == len(neighbors_e_idx))  # check the next line validality
            if e_idx is None:
                cut_idx = bisect_left_adapt(neighbors_ts, cut_time)  # very crucial to exempt ties (so don't use bisect)
            else:
                cut_idx = self.nodeedge2idx[src_idx].get(e_idx) if src_idx > 0 else 0
                if cut_idx is None:
                    cut_idx = 0
            idx.append(neighbors_idx[:cut_idx])
            e_idxs.append(neighbors_e_idx[:cut_idx])
            ts.append(neighbors_ts[:cut_idx])
            source_ids = [src_idx] * len(neighbors_ts[:cut_idx])
            sources.extend(source_ids)
            if return_binary_prob:
                neighbors_binary_prob = binary_prob_l[start: end]
                prob.append(neighbors_binary_prob[:cut_idx])
        idx_array = np.concatenate(idx)   #[num possible targets]
        e_id_array = np.concatenate(e_idxs)
        ts_array = np.concatenate(ts)
        source_array = np.array(sources)
        if return_binary_prob:
            prob_array = np.concatenate(prob)
            result = (source_array, idx_array, e_id_array, ts_array, prob_array)
        else:
            result = (source_array, idx_array, e_id_array, ts_array, None)
        return result


    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbor, e_idx_l=None):
        """
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert (len(src_idx_l) == len(cut_time_l))

        out_ngh_node_batch = np.zeros((len(src_idx_l), num_neighbor)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbor)).astype(np.float32)
        out_ngh_eidx_batch = np.zeros((len(src_idx_l), num_neighbor)).astype(np.int32)

        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            ngh_idx, ngh_eidx, ngh_ts, ngh_binomial_prob = self.find_before(src_idx, cut_time, e_idx=e_idx_l[
                i] if e_idx_l is not None else None, return_binary_prob=(self.sample_method == 'binary'))
            if len(ngh_idx) == 0:  # no previous neighbors, return padding index
                continue
            if ngh_binomial_prob is None:  # self.sample_method is multinomial [ours!!!]
                if math.isclose(self.bias, 0):
                    sampled_idx = np.sort(np.random.randint(0, len(ngh_idx), num_neighbor))
                else:
                    time_delta = cut_time - ngh_ts
                    sampling_weight = np.exp(- self.bias * time_delta)
                    sampling_weight = sampling_weight / sampling_weight.sum()  # normalize
                    sampled_idx = np.sort(
                        np.random.choice(np.arange(len(ngh_idx)), num_neighbor, replace=True, p=sampling_weight))
            else:
                # get a bunch of sampled idx by using sequential binary comparison, may need to be written in C later on
                sampled_idx = seq_binary_sample(ngh_binomial_prob, num_neighbor)
            out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
            out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
            out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]
        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch

    def find_k_hop(self, k, src_idx_l, cut_time_l, num_neighbors, e_idx_l=None):
        if k == 0:
            return ([], [], [])
        batch = len(src_idx_l)
        layer_i = 0
        x, y, z = self.get_temporal_neighbor(src_idx_l, cut_time_l, num_neighbors, e_idx_l=e_idx_l)  #each: [batch, num_neighbors]
        node_records = [x]
        eidx_records = [y]
        t_records = [z]
        for layer_i in range(1, k):
            ngh_node_est, ngh_e_est, ngh_t_est = node_records[-1], eidx_records[-1], t_records[-1]
            ngh_node_est = ngh_node_est.flatten()
            ngh_e_est = ngh_e_est.flatten()  #[batch * num_neighbors]
            ngh_t_est = ngh_t_est.flatten()
            out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = self.get_temporal_neighbor(ngh_node_est,
                                                                                                 ngh_t_est,
                                                                                                 num_neighbors,
                                                                                                 e_idx_l=ngh_e_est)

            out_ngh_node_batch = out_ngh_node_batch.reshape(batch, -1) #[batch, num_neighbors* num_neighbors]
            out_ngh_eidx_batch = out_ngh_eidx_batch.reshape(batch, -1)
            out_ngh_t_batch = out_ngh_t_batch.reshape(batch, -1)

            node_records.append(out_ngh_node_batch)
            eidx_records.append(out_ngh_eidx_batch)
            t_records.append(out_ngh_t_batch)

        return (node_records, eidx_records, t_records)
        # each of them is a list of k numpy arrays,
        # first: (batch, num_neighbors), second: [batch, num_neighbors * num_neighbors]


    def find_k_walks(self, degree, src_idx_l, num_neighbors, subgraph_src):
        '''

        :param degree: degree
        :param src_idx_l: array(B, )
        :param cut_time_l: array(B, )
        :param num_neighbors: number of sampling at step2 => total 20 * num_neighbors walks
        :param subgraph_src:
        :param e_idx_l:
        :return: (n_id: [batch, N1 * N2, 6]
                  e_id: [batch, N1 * N2, 3]
                  t_id: [batch, N1 * N2, 3]
                  anony_id: [batch, N1 * N2, 3)
        '''
        node_records_sub, eidx_records_sub, t_records_sub = subgraph_src
        batch = len(src_idx_l)
        n_id_tgt_1, e_id_1, t_id_1 = node_records_sub[0], eidx_records_sub[0], t_records_sub[0]   #[B, N1]
        num_1 = degree
        n_id_src_1 = np.expand_dims(src_idx_l, axis=1).repeat(num_1 * num_neighbors, axis=1)  #[B, N1 * N2]
        ngh_node_est = n_id_tgt_1.flatten()
        ngh_e_est = e_id_1.flatten()  #[batch * N1]
        ngh_t_est = t_id_1.flatten()
        n_id_tgt_1 = n_id_tgt_1.repeat(num_neighbors, axis=1)  #[B, N1 * N2]
        e_id_1 = e_id_1.repeat(num_neighbors, axis=1)
        t_id_1 = t_id_1.repeat(num_neighbors, axis=1)
        n_id_src_2, n_id_tgt_2, e_id_2, t_id_2 = self.get_next_step(ngh_node_est, ngh_t_est, num_neighbors, degree, e_idx_l=ngh_e_est, source_id=src_idx_l)
        #each: [B*N1, N2]
        n_id_src_2 = n_id_src_2.reshape(batch, -1)
        n_id_tgt_2 = n_id_tgt_2.reshape(batch, -1)   #[batch, N1 * N2]
        e_id_2 = e_id_2.reshape(batch, -1)
        t_id_2 = t_id_2.reshape(batch, -1)
        n_id_src_3, n_id_tgt_3, e_id_3, t_id_3, out_anony = self.get_final_step(n_id_src_1, n_id_tgt_1, n_id_src_2, n_id_tgt_2, e_id_1, e_id_2,t_id_1, t_id_2)
        # each: [B*N1*N2, ], out_anony: [B*N1*N2, 3]
        n_id_src_3 = n_id_src_3.reshape(batch, -1)
        n_id_tgt_3 = n_id_tgt_3.reshape(batch, -1)   #[batch, N1 * N2]
        e_id_3 = e_id_3.reshape(batch, -1)
        t_id_3 = t_id_3.reshape(batch, -1)
        out_anony = out_anony.reshape((batch, n_id_src_3.shape[1], 3))
        node_records = np.stack([n_id_src_3, n_id_tgt_3, n_id_src_2, n_id_tgt_2, n_id_src_1, n_id_tgt_1], axis=2)
        eidx_records = np.stack([e_id_3, e_id_2, e_id_1], axis=2)
        t_records = np.stack([t_id_3, t_id_2, t_id_1], axis=2)
        return (node_records, eidx_records, t_records, out_anony)

    def get_next_step(self, src_idx_l, cut_time_l, num_neighbor, degree, e_idx_l=None, source_id=None):
        """
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        source_id = np.expand_dims(source_id, axis=1).repeat(degree, axis=1).flatten()   #[B*N1]
        assert len(src_idx_l) == len(cut_time_l) == len(source_id)
        out_ngh_node_batch = np.zeros((len(src_idx_l), num_neighbor)).astype(np.int32)  #[B*N1, N2]
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbor)).astype(np.float32)
        out_ngh_eidx_batch = np.zeros((len(src_idx_l), num_neighbor)).astype(np.int32)
        out_src_node_batch = np.zeros((len(src_idx_l), num_neighbor)).astype(np.int32)

        for i, (src_id, cut_time, source) in enumerate(zip(src_idx_l, cut_time_l, source_id)):
            src_idx_list = [source, src_id]
            src_idx, ngh_idx, ngh_eidx, ngh_ts, ngh_binomial_prob = self.find_before_walk(src_idx_list, cut_time, e_idx=e_idx_l[i] if e_idx_l is not None else None, return_binary_prob=(self.sample_method == 'binary'))
            if len(ngh_idx) == 0:  # no previous neighbors, return padding index
                continue
            sampled_idx = np.sort(np.random.randint(0, len(ngh_idx), num_neighbor))
            out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
            out_src_node_batch[i, :] = src_idx[sampled_idx]
            out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
            out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]
        return out_src_node_batch, out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch

    def get_final_step(self, n_id_src_1, n_id_tgt_1, n_id_src_2, n_id_tgt_2, e_id_1, e_id_2, t_id_1, t_id_2):
        n_id_src_1 = n_id_src_1.flatten()  #[B*N1*N2]
        n_id_tgt_1 = n_id_tgt_1.flatten()
        n_id_src_2 = n_id_src_2.flatten()
        n_id_tgt_2 = n_id_tgt_2.flatten()
        e_id_1 = e_id_1.flatten()
        e_id_2 = e_id_2.flatten()
        t_id_1 = t_id_1.flatten()
        t_id_2 = t_id_2.flatten()
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        off_set_l = self.off_set_l
        out_ngh_node_batch = np.zeros(len(n_id_src_1)).astype(np.int32)  #[B*N1*N2, ]
        out_ngh_t_batch = np.zeros(len(t_id_1)).astype(np.float32)
        out_ngh_eidx_batch = np.zeros(len(e_id_1)).astype(np.int32)
        out_src_node_batch = np.zeros(len(n_id_src_1)).astype(np.int32)
        out_anony = np.zeros((len(n_id_src_1), 3)).astype(np.int32)  #[B*N1*N2, 3]
        for i, (src_id_1, tgt_id_1, src_id_2, tgt_id_2, e_1, e_2, t_1, t_2) in enumerate(zip(n_id_src_1, n_id_tgt_1, n_id_src_2, n_id_tgt_2, e_id_1, e_id_2,t_id_1, t_id_2)):
            t = 0
            if src_id_1 == src_id_2 and tgt_id_1 != tgt_id_2:
                start_src, end_src = off_set_l[src_id_1], off_set_l[src_id_1+1]
                cut_idx = self.nodeedge2idx[src_id_1].get(e_2) if src_id_1 > 0 else 0
                neighbors_idx = node_idx_l[start_src: end_src][:cut_idx]
                selected_id = np.logical_or((neighbors_idx == tgt_id_1), (neighbors_idx == tgt_id_2))
                neighbors_idx = neighbors_idx[selected_id]
                neighbors_ts = node_ts_l[start_src: end_src][:cut_idx][selected_id]
                neighbors_e_idx = edge_idx_l[start_src: end_src][:cut_idx][selected_id]
                source_idx = np.array([src_id_1] * len(neighbors_idx))

                start_tgt2, end_tgt2 = off_set_l[tgt_id_2], off_set_l[tgt_id_2+1]
                cut_idx = self.nodeedge2idx[tgt_id_2].get(e_2) if tgt_id_2 > 0 else 0
                neighbors_tgt_idx = node_idx_l[start_tgt2: end_tgt2][:cut_idx]
                selected_id = neighbors_tgt_idx == tgt_id_1
                neighbors_idx_2 = neighbors_tgt_idx[selected_id]
                neighbors_ts_2 = node_ts_l[start_tgt2: end_tgt2][:cut_idx][selected_id]
                neighbors_e_idx_2 = edge_idx_l[start_tgt2: end_tgt2][:cut_idx][selected_id]
                source_idx_2 = np.array([tgt_id_2] * len(neighbors_idx_2))

                ngb_node = np.concatenate([neighbors_idx, neighbors_idx_2])
                src_node = np.concatenate([source_idx, source_idx_2])
                ts = np.concatenate([neighbors_ts, neighbors_ts_2])
                es = np.concatenate([neighbors_e_idx, neighbors_e_idx_2])
                assert len(ngb_node) == len(src_node) == len(ts) == len(es)
                if len(ngb_node) != 0:
                    sampled_idx = np.sort(np.random.randint(0, len(ngb_node), 1))

                    out_ngh_node_batch[i] = ngb_node[sampled_idx]
                    out_ngh_t_batch[i] = ts[sampled_idx]
                    out_ngh_eidx_batch[i] = es[sampled_idx]
                    out_src_node_batch[i] = src_node[sampled_idx]
                    if src_node[sampled_idx] == src_id_1 and ngb_node[sampled_idx] == tgt_id_1:
                        t = 1
                    elif src_node[sampled_idx] == src_id_1 and ngb_node[sampled_idx] == tgt_id_2:
                        t = 2
                    elif src_node[sampled_idx] == tgt_id_1 and ngb_node[sampled_idx] == tgt_id_2:
                        t = 3
                    else:
                        t = 0
                out_anony[i, :] = np.array([1,2,t])
            elif tgt_id_1 == src_id_2 and src_id_1 != tgt_id_2:
                start_src, end_src = off_set_l[tgt_id_1], off_set_l[tgt_id_1 + 1]
                cut_idx = self.nodeedge2idx[tgt_id_1].get(e_2) if tgt_id_1 > 0 else 0
                neighbors_idx = node_idx_l[start_src: end_src][:cut_idx]
                selected_id = np.logical_or((neighbors_idx == src_id_1), (neighbors_idx == tgt_id_2))
                neighbors_idx = neighbors_idx[selected_id]
                neighbors_ts = node_ts_l[start_src: end_src][:cut_idx][selected_id]
                neighbors_e_idx = edge_idx_l[start_src: end_src][:cut_idx][selected_id]
                source_idx = np.array([tgt_id_1] * len(neighbors_idx))

                start_tgt2, end_tgt2 = off_set_l[tgt_id_2], off_set_l[tgt_id_2 + 1]
                cut_idx = self.nodeedge2idx[tgt_id_2].get(e_2) if tgt_id_2 > 0 else 0
                neighbors_tgt_idx = node_idx_l[start_tgt2: end_tgt2][:cut_idx]
                selected_id = neighbors_tgt_idx == src_id_1
                neighbors_idx_2 = neighbors_tgt_idx[selected_id]
                neighbors_ts_2 = node_ts_l[start_tgt2: end_tgt2][:cut_idx][selected_id]
                neighbors_e_idx_2 = edge_idx_l[start_tgt2: end_tgt2][:cut_idx][selected_id]
                source_idx_2 = np.array([tgt_id_2] * len(neighbors_idx_2))

                ngb_node = np.concatenate([neighbors_idx, neighbors_idx_2])
                src_node = np.concatenate([source_idx, source_idx_2])
                ts = np.concatenate([neighbors_ts, neighbors_ts_2])
                es = np.concatenate([neighbors_e_idx, neighbors_e_idx_2])
                assert len(ngb_node) == len(src_node) == len(ts) == len(es)
                if len(ngb_node) != 0:
                    sampled_idx = np.sort(np.random.randint(0, len(ngb_node), 1))

                    out_ngh_node_batch[i] = ngb_node[sampled_idx]
                    out_ngh_t_batch[i] = ts[sampled_idx]
                    out_ngh_eidx_batch[i] = es[sampled_idx]
                    out_src_node_batch[i] = src_node[sampled_idx]

                    if src_node[sampled_idx] == tgt_id_1 and ngb_node[sampled_idx] == src_id_1:
                        t = 1
                    elif src_node[sampled_idx] == tgt_id_1 and ngb_node[sampled_idx] == tgt_id_2:
                        t = 3
                    elif src_node[sampled_idx] == tgt_id_2 and ngb_node[sampled_idx] == src_id_1:
                        t = 2
                    else:
                        t = 0
                out_anony[i, :] = np.array([1,3,t])
            else:
                start_src, end_src = off_set_l[tgt_id_1], off_set_l[tgt_id_1 + 1]
                cut_idx = self.nodeedge2idx[tgt_id_1].get(e_2) if tgt_id_1 > 0 else 0
                neighbors_idx = node_idx_l[start_src: end_src][:cut_idx]
                neighbors_ts = node_ts_l[start_src: end_src][:cut_idx]
                neighbors_e_idx = edge_idx_l[start_src: end_src][:cut_idx]
                source_idx = np.array([tgt_id_1] * len(neighbors_idx))

                start_tgt2, end_tgt2 = off_set_l[tgt_id_2], off_set_l[tgt_id_2 + 1]
                cut_idx = self.nodeedge2idx[tgt_id_2].get(e_2) if tgt_id_2 > 0 else 0
                neighbors_idx_2 = node_idx_l[start_tgt2: end_tgt2][:cut_idx]
                neighbors_ts_2 = node_ts_l[start_tgt2: end_tgt2][:cut_idx]
                neighbors_e_idx_2 = edge_idx_l[start_tgt2: end_tgt2][:cut_idx]
                source_idx_2 = np.array([tgt_id_2] * len(neighbors_idx_2))

                ngb_node = np.concatenate([neighbors_idx, neighbors_idx_2])
                src_node = np.concatenate([source_idx, source_idx_2])
                ts = np.concatenate([neighbors_ts, neighbors_ts_2])
                es = np.concatenate([neighbors_e_idx, neighbors_e_idx_2])
                assert len(ngb_node) == len(src_node) == len(ts) == len(es)
                if len(ngb_node) != 0:
                    sampled_idx = np.sort(np.random.randint(0, len(ngb_node), 1))

                    out_ngh_node_batch[i] = ngb_node[sampled_idx]
                    out_ngh_t_batch[i] = ts[sampled_idx]
                    out_ngh_eidx_batch[i] = es[sampled_idx]
                    out_src_node_batch[i] = src_node[sampled_idx]

                    if src_node[sampled_idx] == src_id_1 and ngb_node[sampled_idx] != tgt_id_1:
                        t = 3
                    elif src_node[sampled_idx] == tgt_id_1 and ngb_node[sampled_idx] != src_id_1:
                        t = 2
                    elif src_node[sampled_idx] == src_id_1 and ngb_node[sampled_idx] == tgt_id_1:
                        t = 1
                    elif src_node[sampled_idx] == tgt_id_1 and ngb_node[sampled_idx] == src_id_1:
                        t = 1
                    else:
                        t = 0
                out_anony[i, :] = np.array([1,1,t])

        return (out_src_node_batch, out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch, out_anony)


def get_tempme_neighbor_finder(data: ContinuousTimeDynamicGraphDataset, max_node_idx=None):
  max_node_idx = max(data.source_node_ids.max(), data.target_node_ids.max()) if max_node_idx is None else max_node_idx
  adj_list = [[] for _ in range(max_node_idx + 1)]
  for source, destination, edge_idx, timestamp in zip(data.source_node_ids, data.target_node_ids,
                                                      data.edge_ids,
                                                      data.timestamps):
    adj_list[source].append((destination, edge_idx, timestamp))
    adj_list[destination].append((source, edge_idx, timestamp))

  return TempMeNeighborFinder(adj_list)


@jit(nopython=True)
def seq_binary_sample(ngh_binomial_prob, num_neighbor):
    sampled_idx = []
    for j in range(num_neighbor):
        idx = seq_binary_sample_one(ngh_binomial_prob)
        sampled_idx.append(idx)
    sampled_idx = np.array(sampled_idx)  # not necessary but just for type alignment with the other branch
    return sampled_idx


@jit(nopython=True)
def seq_binary_sample_one(ngh_binomial_prob):
    seg_len = 10
    a_l_seg = np.random.random((seg_len,))
    seg_idx = 0
    for idx in range(len(ngh_binomial_prob)-1, -1, -1):
        a = a_l_seg[seg_idx]
        seg_idx += 1 # move one step forward
        if seg_idx >= seg_len:
            a_l_seg = np.random.random((seg_len,))  # regenerate a batch of new random values
            seg_idx = 0  # and reset the seg_idx
        if a < ngh_binomial_prob[idx]:
            # print('=' * 50)
            # print(a, len(ngh_binomial_prob) - idx, len(ngh_binomial_prob),
            #       (len(ngh_binomial_prob) - idx) / len(ngh_binomial_prob), ngh_binomial_prob)
            return idx
    return 0  # very extreme case due to float rounding error


@jit(nopython=True)
def bisect_left_adapt(a, x):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(x) will
    insert just before the leftmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """
    lo = 0
    hi = len(a)
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if a[mid] < x: lo = mid+1
        else: hi = mid
    return lo

def load_data(tgn: TGNWrapper, mode: str = 'test') -> (RandEdgeSampler, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                       np.ndarray, TempMeNeighborFinder):
    dataset = tgn.dataset

    node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
        new_node_test_data = tgn.get_training_data(randomize_features=False,
                                                   validation_fraction=dataset.parameters.validation_end - dataset.parameters.training_end,
                                                    test_fraction=1 - dataset.parameters.validation_end,
                                                   new_test_nodes_fraction=0.1,
                                                    different_new_nodes_between_val_and_test=False)


    max_node_idx = max(full_data.sources.max(), full_data.destinations.max())
    adj_list = [[] for _ in range(max_node_idx + 1)]
    for source, destination, edge_idx, timestamp in zip(train_data.sources, train_data.destinations,
                                                        train_data.edge_idxs,
                                                        train_data.timestamps):
        adj_list[source].append((destination, edge_idx, timestamp))
        adj_list[destination].append((source, edge_idx, timestamp))
    train_ngh_finder = TempMeNeighborFinder(adj_list)
    full_adj_list = [[] for _ in range(max_node_idx + 1)]
    for source, destination, edge_idx, timestamp in zip(full_data.sources, full_data.destinations,
                                                        full_data.edge_idxs,
                                                        full_data.timestamps):
        full_adj_list[source].append((destination, edge_idx, timestamp))
        full_adj_list[destination].append((source, edge_idx, timestamp))
    full_ngh_finder = TempMeNeighborFinder(full_adj_list)

    if mode == 'test':
        return (RandEdgeSampler(test_data.sources, test_data.destinations), test_data.sources, test_data.destinations,
                test_data.timestamps, test_data.labels, test_data.edge_idxs, full_ngh_finder)
    else:
        return (RandEdgeSampler(train_data.sources, train_data.destinations), train_data.sources, train_data.destinations,
                train_data.timestamps, train_data.labels, train_data.edge_idxs, train_ngh_finder)

def get_item(input_pack, batch_id):
    subgraph_src, subgraph_tgt, subgraph_bgd, walks_src, walks_tgt, walks_bgd, dst_fake = input_pack
    node_records, eidx_records, t_records = subgraph_src
    node_records = [i[batch_id] for i in node_records]
    eidx_records = [i[batch_id] for i in eidx_records]
    t_records = [i[batch_id] for i in t_records]
    subgraph_src = (node_records, eidx_records, t_records)

    node_records, eidx_records, t_records = subgraph_tgt
    node_records = [i[batch_id] for i in node_records]
    eidx_records = [i[batch_id] for i in eidx_records]
    t_records = [i[batch_id] for i in t_records]
    subgraph_tgt = (node_records, eidx_records, t_records)

    node_records, eidx_records, t_records = subgraph_bgd
    node_records = [i[batch_id] for i in node_records]
    eidx_records = [i[batch_id] for i in eidx_records]
    t_records = [i[batch_id] for i in t_records]
    subgraph_bgd = (node_records, eidx_records, t_records)

    walks_src = [item[batch_id] for item in walks_src]
    walks_src = (walks_src[0], walks_src[1], walks_src[2], walks_src[3], walks_src[4])

    walks_tgt = [item[batch_id] for item in walks_tgt]
    walks_tgt = (walks_tgt[0], walks_tgt[1], walks_tgt[2], walks_tgt[3], walks_tgt[4])

    walks_bgd = [item[batch_id] for item in walks_bgd]
    walks_bgd = (walks_bgd[0], walks_bgd[1], walks_bgd[2], walks_bgd[3], walks_bgd[4])

    dst_fake = dst_fake[batch_id]
    return subgraph_src, subgraph_tgt, subgraph_bgd, walks_src, walks_tgt, walks_bgd, dst_fake

def get_item_edge(edge_features, batch_id):
    edge_features = edge_features[:, batch_id, :, :, :]  # [3,bsz, n_walks,length, length]
    src_edge = edge_features[0]
    tgt_edge = edge_features[1]
    bgd_edge = edge_features[2]
    return src_edge, tgt_edge, bgd_edge

def pre_processing(filepath: str, full_ngh_finder: TempMeNeighborFinder, sampler: RandEdgeSampler, src: np.ndarray,
                   dst: np.ndarray, ts: np.ndarray, val_e_idx_l: np.ndarray, num_neighbors: int = 20):
    load_dict = {}
    save_dict = {}
    for item in ["subgraph_src_0", "subgraph_src_1", "subgraph_tgt_0", "subgraph_tgt_1",  "subgraph_bgd_0", "subgraph_bgd_1", "walks_src", "walks_tgt", "walks_bgd", "dst_fake"]:
        load_dict[item] = []
    num_test_instance = len(src)
    print("start extracting subgraph")
    for k in tqdm(range(num_test_instance-1)):
        src_l_cut = src[k:k+1]
        dst_l_cut = dst[k:k+1]
        ts_l_cut = ts[k:k+1]
        e_l_cut = val_e_idx_l[k:k+1] if (val_e_idx_l is not None) else None
        size = len(src_l_cut)
        src_l_fake, dst_l_fake = sampler.sample(size)
        load_dict["dst_fake"].append(dst_l_fake)
        subgraph_src = full_ngh_finder.find_k_hop(2, src_l_cut, ts_l_cut, e_idx_l=e_l_cut, num_neighbors=num_neighbors)  #first: (batch, num_neighbors), second: [batch, num_neighbors * num_neighbors]
        node_records, eidx_records, t_records = subgraph_src
        load_dict["subgraph_src_0"].append(np.concatenate([node_records[0], eidx_records[0], t_records[0]], axis=-1))  #append([1, num_neighbors * 3]
        load_dict["subgraph_src_1"].append(np.concatenate([node_records[1], eidx_records[1], t_records[1]], axis=-1))    #append([1, num_neighbors**2 * 3]
        subgraph_tgt = full_ngh_finder.find_k_hop(2, dst_l_cut, ts_l_cut, e_idx_l=e_l_cut, num_neighbors=num_neighbors)
        node_records, eidx_records, t_records = subgraph_tgt
        load_dict["subgraph_tgt_0"].append(np.concatenate([node_records[0], eidx_records[0], t_records[0]], axis=-1))  #append([1, num_neighbors * 3]
        load_dict["subgraph_tgt_1"].append(np.concatenate([node_records[1], eidx_records[1], t_records[1]], axis=-1))    #append([1, num_neighbors**2 * 3]
        subgraph_bgd = full_ngh_finder.find_k_hop(2, dst_l_fake, ts_l_cut, num_neighbors=num_neighbors, e_idx_l=None)
        node_records, eidx_records, t_records = subgraph_bgd
        load_dict["subgraph_bgd_0"].append(np.concatenate([node_records[0], eidx_records[0], t_records[0]], axis=-1))  #append([1, num_neighbors * 3]
        load_dict["subgraph_bgd_1"].append(np.concatenate([node_records[1], eidx_records[1], t_records[1]], axis=-1))    #append([1, num_neighbors**2 * 3]
        walks_src = full_ngh_finder.find_k_walks(num_neighbors, src_l_cut, num_neighbors=3, subgraph_src=subgraph_src)
        walks_tgt = full_ngh_finder.find_k_walks(num_neighbors, dst_l_cut, num_neighbors=3, subgraph_src=subgraph_tgt)
        walks_bgd = full_ngh_finder.find_k_walks(num_neighbors, dst_l_fake, num_neighbors=3, subgraph_src=subgraph_bgd)
        node_records, eidx_records, t_records, out_anony = walks_src
        load_dict["walks_src"].append(np.concatenate([node_records, eidx_records, t_records, out_anony], axis=-1))  #append([1, num_walks, 6+3+3+3])
        node_records, eidx_records, t_records, out_anony = walks_tgt
        load_dict["walks_tgt"].append(np.concatenate([node_records, eidx_records, t_records, out_anony], axis=-1))
        node_records, eidx_records, t_records, out_anony = walks_bgd
        load_dict["walks_bgd"].append(np.concatenate([node_records, eidx_records, t_records, out_anony], axis=-1))
    for item in ["subgraph_src_0", "subgraph_src_1", "subgraph_tgt_0", "subgraph_tgt_1", "subgraph_bgd_0",
                 "subgraph_bgd_1", "walks_src", "walks_tgt", "walks_bgd", "dst_fake"]:
        save_dict[item] = np.concatenate(load_dict[item], axis=0)

    hf = h5py.File(filepath, "w")
    for item in ["subgraph_src_0", "subgraph_src_1", "subgraph_tgt_0", "subgraph_tgt_1", "subgraph_bgd_0",
                 "subgraph_bgd_1", "walks_src", "walks_tgt", "walks_bgd","dst_fake"]:
        hf.create_dataset(item, data=save_dict[item])
    hf.close()
    print("done")
    return

def load_subgraph_margin(n_degree: int, file):
    ####### subgraph_src
    subgraph_src_0 = file["subgraph_src_0"][:]
    x0, y0, z0 = (subgraph_src_0[:, 0:n_degree], subgraph_src_0[:, n_degree: 2 * n_degree],
                  subgraph_src_0[:, 2 * n_degree: 3 * n_degree])
    node_records, eidx_records, t_records = [x0], [y0], [z0]
    subgraph_src_1 = file["subgraph_src_1"][:]
    x1, y1, z1 = (subgraph_src_1[:, 0:n_degree ** 2], subgraph_src_1[:,n_degree ** 2: 2 * n_degree ** 2],
                  subgraph_src_1[:, 2 * n_degree ** 2: 3 * n_degree ** 2])
    node_records.append(x1)
    eidx_records.append(y1)
    t_records.append(z1)
    subgraph_src = (node_records, eidx_records, t_records)

    ####### subgraph_tgt
    subgraph_tgt_0 = file["subgraph_tgt_0"][:]
    x0, y0, z0 = (subgraph_tgt_0[:, 0:n_degree], subgraph_tgt_0[:, n_degree: 2 * n_degree],
                  subgraph_tgt_0[:, 2 * n_degree: 3 * n_degree])
    node_records, eidx_records, t_records = [x0], [y0], [z0]
    subgraph_tgt_1 = file["subgraph_tgt_1"][:]
    x1, y1, z1 = (subgraph_tgt_1[:, 0:n_degree ** 2], subgraph_tgt_1[:, n_degree ** 2: 2 * n_degree ** 2],
                  subgraph_tgt_1[:, 2 * n_degree ** 2: 3 * n_degree ** 2])
    node_records.append(x1)
    eidx_records.append(y1)
    t_records.append(z1)
    subgraph_tgt = (node_records, eidx_records, t_records)

    ### subgraph_bgd
    subgraph_bgd_0 = file["subgraph_bgd_0"][:]
    x0, y0, z0 = subgraph_bgd_0[:, 0:n_degree], subgraph_bgd_0[:,
                                                     n_degree: 2 * n_degree], subgraph_bgd_0[:,
                                                                                        2 * n_degree: 3 * n_degree]
    node_records, eidx_records, t_records = [x0], [y0], [z0]
    subgraph_bgd_1 = file["subgraph_bgd_1"][:]
    x1, y1, z1 = (subgraph_bgd_1[:, 0:n_degree ** 2], subgraph_bgd_1[:, n_degree ** 2: 2 * n_degree ** 2],
                  subgraph_bgd_1[:, 2 * n_degree ** 2: 3 * n_degree ** 2])
    node_records.append(x1)
    eidx_records.append(y1)
    t_records.append(z1)
    subgraph_bgd = (node_records, eidx_records, t_records)

    walks_src = file["walks_src_new"][:]
    node_records, eidx_records, t_records, cat_feat, marginal = (walks_src[:, :, :6], walks_src[:, :, 6:9],
                                                                 walks_src[:, :, 9:12], walks_src[:, :, 12:13],
                                                                 walks_src[:, :, 13:14])
    walks_src = (node_records.astype(int), eidx_records.astype(int), t_records, cat_feat.astype(int), marginal)

    walks_tgt = file["walks_tgt_new"][:]
    node_records, eidx_records, t_records, cat_feat, marginal = (walks_tgt[:, :, :6], walks_tgt[:, :, 6:9],
                                                                 walks_tgt[:, :, 9:12], walks_tgt[:, :, 12:13],
                                                                 walks_tgt[:, :, 13:14])
    walks_tgt = (node_records.astype(int), eidx_records.astype(int), t_records, cat_feat.astype(int), marginal)

    walks_bgd = file["walks_bgd_new"][:]
    node_records, eidx_records, t_records, cat_feat, marginal = (walks_bgd[:, :, :6], walks_bgd[:, :, 6:9],
                                                                 walks_bgd[:, :, 9:12], walks_bgd[:, :, 12:13],
                                                                 walks_bgd[:, :, 13:14])
    walks_bgd = (node_records.astype(int), eidx_records.astype(int), t_records, cat_feat.astype(int), marginal)

    dst_fake = file["dst_fake"][:]
    pack = (subgraph_src, subgraph_tgt, subgraph_bgd, walks_src, walks_tgt, walks_bgd, dst_fake)
    return pack

def get_null_distribution(data: ContinuousTimeDynamicGraphDataset, num_neighbors: int = 20):
    rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l, finder = load_data_shuffle(mode="test", data=data)
    num_distribution = pre_processing_null_distr(finder, rand_sampler, test_src_l, test_dst_l, test_ts_l, test_e_idx_l,num_neighbors)
    return num_distribution

def statistic(out_anony, sat, strint_rep):
    batch = out_anony.shape[0]
    for i in range(batch):
        samples = out_anony[i]  #[N, 3]
        for t in range(samples.shape[0]):
            anony_string = np.array2string(samples[t])
            sat[strint_rep[anony_string]] += 1
    return sat

def pre_processing_null_distr(ngh_finder, sampler, src, dst, ts, val_e_idx_l, num_neighbors):
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

def load_data_shuffle(mode: Literal["train", "test"], data: ContinuousTimeDynamicGraphDataset):
    val_time, test_time = list(np.quantile(data.timestamps, [0.70, 0.85]))

    src_l = data.source_node_ids
    dst_l = data.target_node_ids  # g_df.i.values
    e_idx_l = data.edge_ids  # g_df.idx.values
    label_l = data.labels
    ts_l = data.timestamps
    length = len(ts_l)
    permutation = np.random.permutation(length)

    src_l = np.array(src_l)[permutation]
    dst_l = np.array(dst_l)[permutation]
    label_l = np.array(label_l)[permutation]

    max_src_index = src_l.max()
    max_idx = max(src_l.max(), dst_l.max())
    random.seed(2023)
    total_node_set = set(np.unique(np.hstack([data.source_node_ids, data.target_node_ids])))
    num_total_unique_nodes = len(total_node_set)
    mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])),
                                      int(0.1 * num_total_unique_nodes)))
    mask_src_flag = data.events[COL_NODE_U].map(lambda x: x in mask_node_set).values
    mask_dst_flag = data.events[COL_NODE_I].map(lambda x: x in mask_node_set).values
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
    train_ngh_finder = TempMeNeighborFinder(adj_list)
    # full graph with all the data for the test and validation purpose
    full_adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
        full_adj_list[src].append((dst, eidx, ts))
        full_adj_list[dst].append((src, eidx, ts))
    full_ngh_finder = TempMeNeighborFinder(full_adj_list)
    train_rand_sampler = RandEdgeSampler((train_src_l,), (train_dst_l,))
    # val_rand_sampler = RandEdgeSampler((train_src_l, val_src_l), (train_dst_l, val_dst_l))
    test_rand_sampler = RandEdgeSampler((train_src_l, val_src_l, test_src_l), (train_dst_l, val_dst_l, test_dst_l))
    if mode == "test":
        return test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l, full_ngh_finder
    else:
        return train_rand_sampler, train_src_l, train_dst_l, train_ts_l, train_label_l, train_e_idx_l, train_ngh_finder
