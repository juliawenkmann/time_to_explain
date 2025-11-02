import numpy as np
import torch

class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class MLP(torch.nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 80)
        self.fc_2 = torch.nn.Linear(80, 10)
        self.fc_3 = torch.nn.Linear(10, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1

        return self.num_round >= self.max_round


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list, seed=None):
        self.seed = None
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def sample(self, size):
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list), size)
            dst_index = np.random.randint(0, len(self.dst_list), size)
        else:

            src_index = self.random_state.randint(0, len(self.src_list), size)
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


def get_neighbor_finder(data, uniform, max_node_idx=None):
    max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
    adj_list = [[] for _ in range(max_node_idx + 1)]
    for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                        data.edge_idxs,
                                                        data.timestamps):
        adj_list[source].append((destination, edge_idx, timestamp))
        adj_list[destination].append((source, edge_idx, timestamp))

    # return NeighborFinder(adj_list, uniform=uniform)
    return NeighborFinder(adj_list, uniform=False)


class NeighborFinder:
    def __init__(self, adj_list, uniform=False):
        """
        Params
        ------
        node_idx_l: List[int]
        node_ts_l: List[int]
        off_set_l: List[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i]
        """ 
        self.adj_list = adj_list
        node_idx_l, node_ts_l, edge_idx_l, off_set_l = self.init_off_set(adj_list)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l
        
        self.off_set_l = off_set_l
        
        self.uniform = uniform

    def init_off_set(self, adj_list):
        """
        Params
        ------
        adj_list: List[List[int]]
        
        """
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        off_set_l = [0,]
        
        for i in range(len(adj_list)): # adj_list: [[], [(...), ...], ...]
            curr = adj_list[i]
            # curr = sorted(curr, key=lambda x: x[1])
            curr = sorted(curr, key=lambda x: x[2]) # sort according to time
            n_idx_l.extend([x[0] for x in curr])
            e_idx_l.extend([x[1] for x in curr])
            n_ts_l.extend([x[2] for x in curr])
           
            
            off_set_l.append(len(n_idx_l)) # the end index of this node's temporal interactions
            # import ipdb; ipdb.set_trace()
        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        off_set_l = np.array(off_set_l)

        assert(len(n_idx_l) == len(n_ts_l))
        assert(off_set_l[-1] == len(n_ts_l))
        
        return n_idx_l, n_ts_l, e_idx_l, off_set_l
        
    def find_before(self, src_idx, cut_time):
        """
    
        Params
        ------
        src_idx: int
        cut_time: float
        """
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        off_set_l = self.off_set_l
        
        neighbors_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]] # one node's neighbors
        neighbors_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_e_idx = edge_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]

        # import ipdb; ipdb.set_trace()
        
        if len(neighbors_idx) == 0 or len(neighbors_ts) == 0:
            return neighbors_idx, neighbors_ts, neighbors_e_idx

        # left = 0
        # right = len(neighbors_idx) - 1

        
        # while left + 1 < right: # ! binary search, include cut_time
        #     mid = (left + right) // 2
        #     curr_t = neighbors_ts[mid]
        #     if curr_t <= cut_time:
        #         left = mid
        #     else:
        #         right = mid
            
        # if neighbors_ts[right] <= cut_time:
        #     end_point = right + 1
        # elif neighbors_ts[left] <= cut_time:
        #     end_point = left + 1
        # else:
        #     end_point = left

        
        # indices = neighbors_ts <= cut_time
        indices = neighbors_ts < cut_time # NOTE: important?

        # import ipdb; ipdb.set_trace()

        
        # return neighbors_idx[:end_point], neighbors_e_idx[:end_point], neighbors_ts[:end_point]
        # return neighbors_idx[:end_point], neighbors_e_idx[:end_point], neighbors_ts[:end_point]
        return neighbors_idx[indices], neighbors_e_idx[indices], neighbors_ts[indices]

        # if neighbors_ts[right] < cut_time: # https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs/issues/8
        #     return neighbors_idx[:right], neighbors_e_idx[:right], neighbors_ts[:right]
        # else:
        #     return neighbors_idx[:left], neighbors_e_idx[:left], neighbors_ts[:left]

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20, edge_idx_preserve_list=None):
        """
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        """
        assert(len(src_idx_l) == len(cut_time_l))
        
        out_ngh_node_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.float32)
        out_ngh_eidx_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        
        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            ngh_idx, ngh_eidx, ngh_ts = self.find_before(src_idx, cut_time)
            # import ipdb; ipdb.set_trace()
            
            # if i == 1:
            #     import ipdb; ipdb.set_trace()

            if len(ngh_idx) > 0: #! only found neighbors list is not empty, otherwise all zeros
                if self.uniform:
                    raise NotImplementedError('Should not use this scheme')
                    sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors) # sample 'num_neighbors' neighbors in the ngh_idx.
                    
                    out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
                    out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]
                    
                    # resort based on time
                    pos = out_ngh_t_batch[i, :].argsort()
                    out_ngh_node_batch[i, :] = out_ngh_node_batch[i, :][pos]
                    out_ngh_t_batch[i, :] = out_ngh_t_batch[i, :][pos]
                    out_ngh_eidx_batch[i, :] = out_ngh_eidx_batch[i, :][pos]
                else: # we can use this setting to restrict the number of previous events observed.

                    # ngh_ts = ngh_ts[:num_neighbors]
                    # ngh_idx = ngh_idx[:num_neighbors]
                    # ngh_eidx = ngh_eidx[:num_neighbors]
                    

                    # get recent temporal edges
                    ngh_ts = ngh_ts[-num_neighbors:]
                    ngh_idx = ngh_idx[-num_neighbors:]
                    ngh_eidx = ngh_eidx[-num_neighbors:]

                    # mask out discarded edge_idxs, these should not be seen.
                    if edge_idx_preserve_list is not None:
                        mask = np.isin(ngh_eidx, edge_idx_preserve_list)
                        ngh_ts = ngh_ts[mask]
                        ngh_idx = ngh_idx[mask]
                        ngh_eidx = ngh_eidx[mask]
                    
                    assert(len(ngh_idx) <= num_neighbors)
                    assert(len(ngh_ts) <= num_neighbors)
                    assert(len(ngh_eidx) <= num_neighbors)
                    
                    # out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
                    # out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
                    # out_ngh_eidx_batch[i,  num_neighbors - len(ngh_eidx):] = ngh_eidx

                    # end positions already have been 0.
                    out_ngh_node_batch[i, :len(ngh_idx)] = ngh_idx
                    out_ngh_t_batch[i, :len(ngh_ts)] = ngh_ts
                    out_ngh_eidx_batch[i, :len(ngh_eidx)] = ngh_eidx
                    
        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch

    

            


# class NeighborFinder:
# 	def __init__(self, adj_list, uniform=False, seed=None):
# 		self.node_to_neighbors = []
# 		self.node_to_edge_idxs = []
# 		self.node_to_edge_timestamps = []

# 		for neighbors in adj_list:
# 			# Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
# 			# We sort the list based on timestamp
# 			sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
# 			self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
# 			self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
# 			self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

# 		self.uniform = uniform

# 		if seed is not None:
# 			self.seed = seed
# 			self.random_state = np.random.RandomState(self.seed)

# 	def find_before(self, src_idx, cut_time):
# 		"""
# 		Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

# 		Returns 3 lists: neighbors, edge_idxs, timestamps

# 		"""
# 		i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

# 		return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

# 	def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
# 		"""
# 		Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

# 		Params
# 		------
# 		src_idx_l: List[int]
# 		cut_time_l: List[float],
# 		num_neighbors: int
# 		"""
# 		assert (len(source_nodes) == len(timestamps))

# 		tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
# 		# NB! All interactions described in these matrices are sorted in each row by time
# 		neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
# 			np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
# 		edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
# 			np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
# 		edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
# 			np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

# 		for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
# 			source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
# 																									 timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

# 			if len(source_neighbors) > 0 and n_neighbors > 0:
# 				if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
# 					sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

# 					neighbors[i, :] = source_neighbors[sampled_idx]
# 					edge_times[i, :] = source_edge_times[sampled_idx]
# 					edge_idxs[i, :] = source_edge_idxs[sampled_idx]

# 					# re-sort based on time
# 					pos = edge_times[i, :].argsort()
# 					neighbors[i, :] = neighbors[i, :][pos]
# 					edge_times[i, :] = edge_times[i, :][pos]
# 					edge_idxs[i, :] = edge_idxs[i, :][pos]
# 				else:
# 					# Take most recent interactions
# 					source_edge_times = source_edge_times[-n_neighbors:]
# 					source_neighbors = source_neighbors[-n_neighbors:]
# 					source_edge_idxs = source_edge_idxs[-n_neighbors:]

# 					assert (len(source_neighbors) <= n_neighbors)
# 					assert (len(source_edge_times) <= n_neighbors)
# 					assert (len(source_edge_idxs) <= n_neighbors)

# 					neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
# 					edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
# 					edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

# 		return neighbors, edge_idxs, edge_times