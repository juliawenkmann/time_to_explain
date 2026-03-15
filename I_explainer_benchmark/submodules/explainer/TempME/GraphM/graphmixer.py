import numpy as np
import torch
import torch.nn as nn

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


class TimeEncoder(nn.Module):

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeEncoder, self).__init__()

        self.time_dim = time_dim
        # trainable parameters for time encoding
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        timestamps = timestamps.unsqueeze(dim=2)
        output = torch.cos(self.w(timestamps))

        return output



class GraphMixer(nn.Module):

    def __init__(self, n_feat, e_feat, n_neighbors, device,
                  num_tokens, num_layers=2, token_dim_expansion_factor=0.5,
                 channel_dim_expansion_factor=4.0, dropout=0.1):
        """
        TCL model.
        :param n_feat: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param e_feat: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param n_neighbors: neighbor number
        :param num_tokens: int, number of tokens
        :param num_layers: int, number of transformer layers
        :param token_dim_expansion_factor: float, dimension expansion factor for tokens
        :param channel_dim_expansion_factor: float, dimension expansion factor for channels
        :param dropout: float, dropout rate
        :param device: str, device
        """
        super(GraphMixer, self).__init__()
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)), requires_grad=False)
        self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)), requires_grad=False)

        self.node_raw_features = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
        self.edge_raw_features = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)
        self.num_neighbors = n_neighbors
        self.node_feat_dim = self.n_feat_th.shape[1]
        self.edge_feat_dim = self.e_feat_th.shape[1]
        self.time_feat_dim = self.node_feat_dim
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.token_dim_expansion_factor = token_dim_expansion_factor
        self.channel_dim_expansion_factor = channel_dim_expansion_factor
        self.dropout = dropout
        self.device = device

        self.num_channels = self.edge_feat_dim
        # in GraphMixer, the time encoding function is not trainable
        self.time_encoder = TimeEncoder(time_dim=self.time_feat_dim, parameter_requires_grad=False)
        self.projection_layer = nn.Linear(self.edge_feat_dim + self.time_feat_dim, self.num_channels)

        self.mlp_mixers = nn.ModuleList([
            MLPMixer(num_tokens=self.num_tokens, num_channels=self.num_channels,
                     token_dim_expansion_factor=self.token_dim_expansion_factor,
                     channel_dim_expansion_factor=self.channel_dim_expansion_factor, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(in_features=self.num_channels + self.node_feat_dim, out_features=self.node_feat_dim, bias=True)

        self.affinity_score = MergeLayer(self.node_feat_dim, self.node_feat_dim,
                                         self.node_feat_dim,
                                         1)

    def get_node_emb(self, src_idx, tgt_idx, bgd_idx, cut_time, e_idx,
                     subgraph_src, subgraph_tgt, subgraph_bgd, explain_weights=None, edge_attr=None, time_gap=2000):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param num_neighbors: int, number of neighbors to sample for each node
        :param time_gap: int, time gap for neighbors to compute node features
        :param explain_weights: list of tensor, shape (3*batch_size, num_neighbors)
        :return:
        """
        batch_size = len(src_idx)
        if explain_weights is not None:
            explain_weight0 = explain_weights[0]
            src_exp, tgt_exp, bgd_exp = explain_weight0[:batch_size], explain_weight0[batch_size: 2*batch_size], explain_weight0[2*batch_size:]
        else:
            src_exp, tgt_exp, bgd_exp = None, None, None
        if edge_attr is not None:
            src_edge_attr, tgt_edge_attr, bgd_edge_attr = edge_attr[:batch_size], edge_attr[batch_size: 2*batch_size], edge_attr[2*batch_size:]
        else:
            src_edge_attr, tgt_edge_attr, bgd_edge_attr = None, None, None
        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.compute_node_temporal_embeddings(node_ids=src_idx, node_interact_times=cut_time, subgraph=subgraph_src,
                                                                    num_neighbors=self.num_neighbors, time_gap=time_gap,
                                                                    exp_src=src_exp, edge_attr=src_edge_attr)
        # Tensor, shape (batch_size, node_feat_dim)
        tgt_node_embeddings = self.compute_node_temporal_embeddings(node_ids=tgt_idx, node_interact_times=cut_time, subgraph=subgraph_tgt,
                                                                    num_neighbors=self.num_neighbors, time_gap=time_gap,
                                                                    exp_src=tgt_exp, edge_attr=tgt_edge_attr)

        bgd_node_embeddings = self.compute_node_temporal_embeddings(node_ids=bgd_idx, node_interact_times=cut_time, subgraph=subgraph_bgd,
                                                                    num_neighbors=self.num_neighbors, time_gap=time_gap,
                                                                    exp_src=bgd_exp, edge_attr=bgd_edge_attr)

        return src_node_embeddings, tgt_node_embeddings, bgd_node_embeddings

    def compute_node_temporal_embeddings(self, node_ids, node_interact_times, subgraph,
                                         num_neighbors, time_gap, exp_src, edge_attr):
        """
        given node ids node_ids, and the corresponding time node_interact_times, return the temporal embeddings of nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :param time_gap: int, time gap for neighbors to compute node features
        :param explain_weights, [bsz, num_neighbors]
        :return:
        """
        node_record, edge_record, time_record = subgraph
        neighbor_node_ids, neighbor_edge_ids, neighbor_times = node_record[0], edge_record[0], time_record[0]
        nidx_records_th = torch.from_numpy(neighbor_node_ids).long().to(self.device)
        mask = (nidx_records_th != 0).long().to(self.device)  # [bsz, n_neighbors]
        if exp_src is not None:
            exp_src = exp_src * mask
        nodes_edge_raw_features = self.edge_raw_features(torch.from_numpy(neighbor_edge_ids).long().to(self.device)) if edge_attr is None else edge_attr  #[bsz, n_neighbors, e_feat_dim]
        nodes_neighbor_time_features = self.time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - neighbor_times).float().to(self.device))
        nodes_neighbor_time_features[torch.from_numpy(neighbor_node_ids == 0)] = 0.0
        # if explain_weights is not None:
        #     nodes_neighbor_time_features = nodes_neighbor_time_features * explain_weights.unsqueeze(-1)
        if edge_attr is None:
            nodes_edge_raw_features[torch.from_numpy(neighbor_node_ids == 0)] = 0.0
        combined_features = torch.cat([nodes_edge_raw_features, nodes_neighbor_time_features], dim=-1)
        combined_features = self.projection_layer(combined_features)     #(batch_size, num_neighbors, d_feat_dim + t_feat_dim)
        for mlp_mixer in self.mlp_mixers:
            # Tensor, shape (batch_size, num_neighbors, num_channels)
            combined_features = mlp_mixer(input_tensor=combined_features, explain_weights=exp_src)  #(batch_size, num_neighbors, n_channel)
        # num_nonzero = torch.from_numpy(neighbor_node_ids != 0).sum(-1).unsqueeze(-1).float().to(self.device)  #[bsz, 1]
        # # Tensor, shape (batch_size, num_channels)
        # if explain_weights is not None:
        #     num_nonzero = explain_weights.sum(-1).unsqueeze(-1).float().to(self.device)  #[bsz, 1]
        # combined_features = torch.sum(combined_features, dim=1) / (num_nonzero + 1e-10)
        combined_features[torch.from_numpy(neighbor_node_ids == 0)] = 0.0
        if exp_src is not None:
            combined_features = combined_features * exp_src.unsqueeze(-1)
        combined_features = torch.mean(combined_features, dim=1)

        # time_gap_neighbor_node_ids, _, _ = self.neighbor_sampler.get_temporal_neighbor(node_ids, node_interact_times, num_neighbor=time_gap)
        nodes_time_gap_neighbor_node_raw_features = self.node_raw_features(torch.from_numpy(neighbor_node_ids).long().to(self.device))
        valid_time_gap_neighbor_node_ids_mask = torch.from_numpy((neighbor_node_ids > 0).astype(np.float32))
        valid_time_gap_neighbor_node_ids_mask[valid_time_gap_neighbor_node_ids_mask == 0] = -1e10
        scores = torch.softmax(valid_time_gap_neighbor_node_ids_mask, dim=1).to(self.device)
        if exp_src is not None:
            scores = scores * exp_src
        nodes_time_gap_neighbor_node_agg_features = torch.mean(nodes_time_gap_neighbor_node_raw_features * scores.unsqueeze(dim=-1), dim=1)
        # (bsz, node_feat_dim)
        output_node_features = nodes_time_gap_neighbor_node_agg_features + self.node_raw_features(torch.from_numpy(node_ids).long().to(self.device))
        #
        #
        # output_node_features = self.node_raw_features(torch.from_numpy(node_ids).long().to(self.device))
        node_embeddings = self.output_layer(torch.cat([combined_features, output_node_features], dim=1))

        return node_embeddings


    def retrieve_edge_features(self, subgraph_src,subgraph_tgt,subgraph_bgd):
        src_edge_attr = self.edge_raw_features(torch.from_numpy(subgraph_src[1][0]).long().to(self.device))
        tgt_edge_attr = self.edge_raw_features(torch.from_numpy(subgraph_tgt[1][0]).long().to(self.device))
        bgd_edge_attr = self.edge_raw_features(torch.from_numpy(subgraph_bgd[1][0]).long().to(self.device))  #[bsz, n]
        edge_features = torch.cat([src_edge_attr, tgt_edge_attr, bgd_edge_attr], dim=0)
        return edge_features



    def contrast(self,src_idx, tgt_idx, bgd_idx, cut_time, e_idx,
                     subgraph_src, subgraph_tgt, subgraph_bgd, explain_weights=None, edge_attr=None, time_gap=2000):
        n_samples = len(src_idx)
        source_node_embedding, destination_node_embedding, negative_node_embedding = \
            self.get_node_emb(src_idx, tgt_idx, bgd_idx, cut_time, e_idx,
                              subgraph_src, subgraph_tgt, subgraph_bgd, explain_weights, edge_attr, time_gap=time_gap)

        score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                    torch.cat([destination_node_embedding,
                                               negative_node_embedding])).squeeze(dim=0)
        pos_score = score[:n_samples]
        neg_score = score[n_samples:]

        return pos_score, neg_score

    def set_neighbor_sampler(self, neighbor_sampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler

    def grab_subgraph(self, src_idx_l, cut_time_l):
        subgraph = self.neighbor_sampler.find_k_hop(2, src_idx_l, cut_time_l, num_neighbors=self.num_neighbors, e_idx_l=None)
        return subgraph


class FeedForwardNet(nn.Module):

    def __init__(self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0):
        """
        two-layered MLP with GELU activation function.
        :param input_dim: int, dimension of input
        :param dim_expansion_factor: float, dimension expansion factor
        :param dropout: float, dropout rate
        """
        super(FeedForwardNet, self).__init__()

        self.input_dim = input_dim
        self.dim_expansion_factor = dim_expansion_factor
        self.dropout = dropout

        self.ffn = nn.Sequential(nn.Linear(in_features=input_dim, out_features=int(dim_expansion_factor * input_dim)),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(in_features=int(dim_expansion_factor * input_dim), out_features=input_dim),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor):
        """
        feed forward net forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
        """
        return self.ffn(x)


class MLPMixer(nn.Module):

    def __init__(self, num_tokens: int, num_channels: int, token_dim_expansion_factor: float = 0.5,
                 channel_dim_expansion_factor: float = 4.0, dropout: float = 0.0):
        """
        MLP Mixer.
        :param num_tokens: int, number of tokens
        :param num_channels: int, number of channels
        :param token_dim_expansion_factor: float, dimension expansion factor for tokens
        :param channel_dim_expansion_factor: float, dimension expansion factor for channels
        :param dropout: float, dropout rate
        """
        super(MLPMixer, self).__init__()

        self.token_norm = nn.LayerNorm(num_tokens)
        self.token_feedforward = FeedForwardNet(input_dim=num_tokens, dim_expansion_factor=token_dim_expansion_factor,
                                                dropout=dropout)

        self.channel_norm = nn.LayerNorm(num_channels)
        self.channel_feedforward = FeedForwardNet(input_dim=num_channels, dim_expansion_factor=channel_dim_expansion_factor,
                                                  dropout=dropout)

    def forward(self, input_tensor, explain_weights=None):
        """
        mlp mixer to compute over tokens and channels
        :param input_tensor: Tensor, shape (batch_size, num_tokens, num_channels)
        :return:
        """
        # mix tokens
        # Tensor, shape (batch_size, num_channels, num_tokens)
        if explain_weights is not None:
            input_tensor = input_tensor * explain_weights.unsqueeze(-1)
        hidden_tensor = self.token_norm(input_tensor.permute(0, 2, 1))
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.token_feedforward(hidden_tensor).permute(0, 2, 1)
        if explain_weights is not None:
            hidden_tensor = hidden_tensor * explain_weights.unsqueeze(-1)
        # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
        output_tensor = hidden_tensor + input_tensor

        # mix channels
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_norm(output_tensor)
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_feedforward(hidden_tensor)
        # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
        if explain_weights is not None:
            hidden_tensor = hidden_tensor * explain_weights.unsqueeze(-1)
        output_tensor = hidden_tensor + output_tensor
        return output_tensor