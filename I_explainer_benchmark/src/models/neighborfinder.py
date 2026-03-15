import numpy as np
import random
import torch
import math
import logging

try:
    from numba import jit
except ImportError:  # pragma: no cover - fallback for environments without numba
    def jit(*args, **kwargs):
        """
        Lightweight stand-in for numba.jit when numba is unavailable.
        Returns the original function without modification.
        """
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)
PRECISION = 5


class NeighborFinder:
    def __init__(self, adj_list, bias=0, ts_precision=PRECISION, use_cache=False, sample_method='multinomial', device=None):
        """
        Params
        ------
        node_idx_l: List[int]
        node_ts_l: List[int]
        off_set_l: List[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i]
        """
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
        binary_prob_l = self.binary_prob_l
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
        binary_prob_l = self.binary_prob_l 
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




