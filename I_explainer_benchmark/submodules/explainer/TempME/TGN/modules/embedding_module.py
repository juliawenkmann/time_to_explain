from turtle import position
import torch
from torch import nn
import numpy as np
import math

class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, explain_weight, mask=None):
        # q: [B*N_src*n_head, 1, d_k]; k: [B*N_src*n_head, num_neighbors, d_k]
        # v: [B*N_src*n_head, num_neighbors, d_v], mask: [B*N_src*n_head, 1, num_neighbors]
        # explain_weight: [B*N_src*n_head, 1, num_neighbors]
        attn = torch.bmm(q, k.transpose(-1, -2))  # [B*N_src*n_head, 1, num_neighbors]
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_v, d]
        if explain_weight is not None:
            attn = attn * explain_weight  #if exp ==0 => masked!
        output = torch.bmm(attn, v)  # [B*N_src*n_head, 1, d_v]

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_emb, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_emb, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_k, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_v, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_emb + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_emb + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_emb + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.fc = nn.Linear(n_head * d_v, d_emb)
        nn.init.xavier_normal_(self.fc.weight)
        self.layer_norm = nn.LayerNorm(d_emb)
        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None, explain_weight=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        B, N_src, _ = q.size() # [B, 1, model_dim]
        B, N_ngh, _ = k.size() # [B, N_ngh, model_dim]
        B, N_ngh, _ = v.size() # [B, N_ngh, model_dim]
        assert(N_ngh % N_src == 0)
        num_neighbors = int(N_ngh / N_src)
        residual = q

        q = self.w_qs(q).view(B, N_src, 1, n_head, d_k)  # [B, N_src, 1, n_head, d_k]
        k = self.w_ks(k).view(B, N_src, num_neighbors, n_head, d_k)  # [B, N_src, num_neighbors, n_head, d_k]
        v = self.w_vs(v).view(B, N_src, num_neighbors, n_head, d_v)  # [B, N_src, num_neighbors, n_head, d_k]

        q = q.transpose(2, 3).contiguous().view(B*N_src*n_head, 1, d_k)  # [B*N_src*n_head, 1, d_k]
        k = k.transpose(2, 3).contiguous().view(B*N_src*n_head, num_neighbors, d_k)  # [B*N_src*n_head, num_neighbors, d_k]
        v = v.transpose(2, 3).contiguous().view(B*N_src*n_head, num_neighbors, d_v)  # [B*N_src*n_head, num_neighbors, d_v]
        # mask = mask.view(B*N_src, 1, num_neighbors).repeat(n_head, 1, 1) # [B*N_src*n_head, 1, num_neighbors]
        if explain_weight is not None:
            explain_weight = explain_weight.view(B*N_src, 1, num_neighbors).repeat(n_head, 1, 1)  # [B*N_src*n_head, 1, num_neighbors]
        output, attn_map = self.attention(q, k, v, explain_weight=explain_weight, mask=mask)
        # output: [B*N_src*n_head, 1, d_v], attn_map: [B*N_src*n_head, 1, num_neighbors]

        output = output.view(B, N_src, n_head*d_v)  # [B, N_src, n_head*d_v]
        output = self.dropout(self.fc(output))  # [B, N_src, model_dim]
        output = self.layer_norm(output + residual)  # [B, N_src, model_dim]
        attn_map = attn_map.view(B, N_src, n_head, num_neighbors)
        return output, attn_map


class TimeEncode(torch.nn.Module):
    def __init__(self, dimension):
        super(TimeEncode, self).__init__()
        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)
        self.w.weight = torch.nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension))).float().reshape(dimension, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

    def forward(self, t):
        t = t.unsqueeze(dim=2)
        output = torch.cos(self.w(t))
        return output


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

class TemporalAttentionLayer(torch.nn.Module):
    """
    Temporal attention layer. Return the temporal embedding of a node given the node itself,
     its neighbors and the edge timestamps.
    """

    def __init__(self, n_node_features, n_neighbors_features, n_edge_features, time_dim,
                 output_dimension, n_head=2,
                 dropout=0.1):
        super(TemporalAttentionLayer, self).__init__()

        self.n_head = n_head

        self.feat_dim = n_node_features
        self.time_dim = time_dim

        self.query_dim = n_node_features + time_dim
        self.key_dim = n_neighbors_features + time_dim + n_edge_features

        self.merger = MergeLayer(self.query_dim, n_node_features, n_node_features, output_dimension)

        self.multi_head_target = MultiHeadAttention(n_head=self.n_head,
                                                    d_emb=self.query_dim,
                                                    d_k=self.key_dim,
                                                    d_v=self.key_dim,
                                                    dropout=dropout)

    def forward(self, src_node_features, src_time_features, neighbors_features,
                neighbors_time_features, edge_features, neighbors_padding_mask, explain_weight):
        """
        "Temporal attention model
        :param src_node_features: float Tensor of shape [batch_size, n_node_features]
        :param src_time_features: float Tensor of shape [batch_size, 1, time_dim]
        :param neighbors_features: float Tensor of shape [batch_size, n_neighbors, n_node_features]
        :param neighbors_time_features: float Tensor of shape [batch_size, n_neighbors, time_dim]
        :param edge_features: float Tensor of shape [batch_size, n_neighbors, n_edge_features]
        :param neighbors_padding_mask: float Tensor of shape [batch_size, n_neighbors]
        :param explain_weight: float Tensor of shape [batch_size, n_neighbors]
        :return:
        attn_output: float Tensor of shape [1, batch_size, n_node_features]  TODO: make it output [bsz, n_node_features]
        attn_output_weights: [batch_size, 1, n_neighbors]
        """

        src_node_features_unrolled = torch.unsqueeze(src_node_features, dim=1)  #bsz, 1, num_of_features]

        query = torch.cat([src_node_features_unrolled, src_time_features], dim=2)  #[bsz, 1, d_q]
        key = torch.cat([neighbors_features, edge_features, neighbors_time_features], dim=2)   #[bsz, n_neighbors, d_k]

        # query = query.permute([1, 0, 2])  # [1, batch_size, num_of_features]
        # key = key.permute([1, 0, 2])  # [n_neighbors, batch_size, num_of_features]


        attn_mask = neighbors_padding_mask
        attn_mask = attn_mask.unsqueeze(1).repeat(self.n_head, 1, 1)  #[bsz*n_head, 1, n_neighbors]
        attn_output, attn_output_weights = self.multi_head_target(q=query, k=key, v=key,
                                                                  mask=attn_mask, explain_weight=explain_weight)
        # [B, N_src, query_dim]

        attn_output = attn_output.squeeze(1)
        attn_output_weights = attn_output_weights.squeeze(1)
        attn_output = self.merger(attn_output, src_node_features)  #[B, node_feature]

        return attn_output, attn_output_weights

class EmbeddingModule(nn.Module):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                             n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                             dropout):
        super(EmbeddingModule, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        # self.memory = memory
        self.neighbor_finder = neighbor_finder
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_time_features = n_time_features
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension
        self.device = device

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                                                use_time_proj=True):
        pass


class IdentityEmbedding(EmbeddingModule):
    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                                                use_time_proj=True):
        return memory[source_nodes, :], None


class TimeEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                             n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                             n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1):
        super(TimeEmbedding, self).__init__(node_features, edge_features, memory,
                                                                                neighbor_finder, time_encoder, n_layers,
                                                                                n_node_features, n_edge_features, n_time_features,
                                                                                embedding_dimension, device, dropout)

        class NormalLinear(nn.Linear):
            # From Jodie code
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.embedding_layer = NormalLinear(1, self.n_node_features)

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                                                use_time_proj=True):
        source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))

        return source_embeddings, None


class GraphAttentionEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, num_neighbor, time_encoder, n_layers,
                             n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                             n_heads=2, dropout=0.1, use_memory=True):
        super(GraphAttentionEmbedding, self).__init__(node_features, edge_features, memory,
                                             neighbor_finder, time_encoder, n_layers,
                                             n_node_features, n_edge_features, n_time_features,
                                             embedding_dimension, device, dropout)
        self.num_neighbor = num_neighbor
        self.use_memory = use_memory
        self.device = device
        self.atten_weights_list = []
        self.n_heads = n_heads
        self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
            n_node_features=n_node_features,
            n_neighbors_features=n_node_features,
            n_edge_features=n_edge_features,
            time_dim=n_time_features,
            n_head=n_heads,
            dropout=dropout,
            output_dimension=n_node_features)
            for _ in range(n_layers)])

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings, edge_time_embeddings, edge_features, mask, explain_weight):
        attention_model = self.attention_models[n_layer]

        source_embedding, atten_weights = attention_model(source_node_features,
                                                          source_nodes_time_embedding,
                                                          neighbor_embeddings,
                                                          edge_time_embeddings,
                                                          edge_features,
                                                          mask,
                                                          explain_weight)

        return source_embedding, atten_weights

    def __aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                                neighbor_embeddings,
                                edge_time_embeddings, edge_features, mask):
        neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features],
                                                                     dim=2)
        neighbor_embeddings = self.linear_1[n_layer - 1](neighbors_features)
        neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings, dim=1))

        source_features = torch.cat([source_node_features,
                                                                 source_nodes_time_embedding.squeeze()], dim=1)
        source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
        source_embedding = self.linear_2[n_layer - 1](source_embedding)

        return source_embedding, None


    def init_hidden_embeddings(self, node_list):
        hidden_embeddings, masks = [], []
        for i in range(len(node_list)):
            batch_node_idx = torch.from_numpy(node_list[i]).long().to(self.device)
            hidden_embeddings.append(self.node_features(batch_node_idx))
            masks.append(batch_node_idx == 0)
        return hidden_embeddings, masks

    def retrieve_time_features(self, cut_time, time_list):
        cut_time = np.concatenate([cut_time, cut_time, cut_time])
        batch = len(cut_time)
        first_time_stamp = np.expand_dims(cut_time, 1)  #[3*bsz, 1]
        # time_features = [self.time_encoder(torch.from_numpy(np.zeros_like(first_time_stamp)).float().to(self.device))]
        time_features = []
        standard_timestamps = np.expand_dims(first_time_stamp, 2)
        for layer_i in range(len(time_list)):
            t_record = time_list[layer_i]
            time_delta = standard_timestamps - t_record.reshape(batch, -1, self.num_neighbor)
            time_delta = time_delta.reshape(batch, -1)
            time_delta = torch.from_numpy(time_delta).float().to(self.device)
            time_features.append(self.time_encoder(time_delta))
            standard_timestamps = np.expand_dims(t_record, 2)
        return time_features

    def retrieve_edge_features(self, edge_list):
        edge_features = []
        for i in range(len(edge_list)):
            batch_edge_idx = torch.from_numpy(edge_list[i]).long().to(self.device)
            edge_features.append(self.edge_features(batch_edge_idx))
        return edge_features

    def embedding_update(self, memory, node_list, edge_list, time_list, cut_time, n_layers, explain_weights=None):
        """Recursive implementation of curr_layers temporal graph attention layers.

        src_idx_l [batch_size]: users / items input ids.
        cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
        curr_layers [scalar]: number of temporal convolutional layers to stack.
        num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
        """

        assert (n_layers >= 0)
        node_features, mask_list = self.init_hidden_embeddings(node_list)
        edge_features = self.retrieve_edge_features(edge_list)
        time_features =self.retrieve_time_features(cut_time, time_list)

        source_embedding = self.embedding_update_layer(memory, node_list, node_features, edge_features, time_features,
                                                      mask_list, explain_weights)  #[3*bs, node_features]

        return source_embedding


    def embedding_update_attr(self, memory, node_list, edge_list, time_list, cut_time, n_layers, edge_features, explain_weights=None):
        assert (n_layers >= 0)
        node_features, mask_list = self.init_hidden_embeddings(node_list)
        # edge_features = self.retrieve_edge_features(edge_list)   #list of [3bz, n, d]; [3bz, n**2, d]
        time_features =self.retrieve_time_features(cut_time, time_list)  #list of [3bz, n, d]; [3bz, n**2, d]

        source_embedding = self.embedding_update_layer(memory, node_list, node_features, edge_features, time_features,
                                                      mask_list, explain_weights)  #[3*bs, node_features]

        return source_embedding



    def embedding_update_layer(self, memory, node_list, node_features, edge_features, time_features, mask_list, explain_weights=None):
        num_layers = len(node_list)
        ## initial neighboring node feature
        neighbor_node = node_list[-1].flatten()
        neighbor_node_feature = node_features[-1].view(-1, self.n_node_features)
        if self.use_memory:
            neighbor_node_feature = memory[neighbor_node, :] + neighbor_node_feature
        else:
            neighbor_node_feature = neighbor_node_feature

        for i in range(num_layers-1):  #i=0, 1
            t = num_layers-1-i  #2, 1
            source_node = node_list[t-1].flatten()
            source_node_feature = node_features[t-1].view(-1, self.n_node_features)  #[bsz, feature_dim]
            batch_layer = source_node_feature.shape[0]
            if self.use_memory:
                source_node_feature = memory[source_node, :] + source_node_feature
            else:
                source_node_feature = source_node_feature
            source_nodes_time_embedding = self.time_encoder(torch.zeros((batch_layer, 1)).to(self.device))  #[bsz, 1, time_dim]
            neighbor_node_feature = neighbor_node_feature.view(batch_layer, self.num_neighbor,-1)  #[bsz, n_neighbor, feature_dim]
            assert neighbor_node_feature.shape[-1] == source_node_feature.shape[-1]
            edge_time_embeddings = time_features[t-1].view(batch_layer, self.num_neighbor, -1)
            edgh_feature = edge_features[t-1].view(batch_layer, self.num_neighbor, -1)
            mask = mask_list[t].view(batch_layer, -1)
            if explain_weights is not None:
                explain_weight = explain_weights[t - 1].view(batch_layer, -1)
            else:
                explain_weight = None

            assert mask.shape[-1] == self.num_neighbor

            updated_source_node_feature, _ = self.aggregate(i, source_node_feature, source_nodes_time_embedding,
                                                         neighbor_node_feature, edge_time_embeddings, edgh_feature, mask, explain_weight)
            # [bsz, n_node_features]
            neighbor_node_feature = updated_source_node_feature

        return updated_source_node_feature   #[true bsz, feature_dim]


def get_embedding_module(module_type, node_features, edge_features, memory, neighbor_finder, num_neighbor,
                                                 time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                                                 embedding_dimension, device,
                                                 n_heads=2, dropout=0.1,
                                                 use_memory=True):
    if module_type == "graph_attention":
        return GraphAttentionEmbedding(node_features=node_features,
                                    edge_features=edge_features,
                                    memory=memory,
                                    neighbor_finder=neighbor_finder,
                                    num_neighbor=num_neighbor,
                                    time_encoder=time_encoder,
                                    n_layers=n_layers,
                                    n_node_features=n_node_features,
                                    n_edge_features=n_edge_features,
                                    n_time_features=n_time_features,
                                    embedding_dimension=embedding_dimension,
                                    device=device,
                                    n_heads=n_heads, dropout=dropout, use_memory=use_memory)

    elif module_type == "identity":
        return IdentityEmbedding(node_features=node_features,
                                edge_features=edge_features,
                                memory=memory,
                                neighbor_finder=neighbor_finder,
                                time_encoder=time_encoder,
                                n_layers=n_layers,
                                n_node_features=n_node_features,
                                n_edge_features=n_edge_features,
                                n_time_features=n_time_features,
                                embedding_dimension=embedding_dimension,
                                device=device,
                                dropout=dropout)
    elif module_type == "time":
        return TimeEmbedding(node_features=node_features,
                            edge_features=edge_features,
                            memory=memory,
                            neighbor_finder=neighbor_finder,
                            time_encoder=time_encoder,
                            n_layers=n_layers,
                            n_node_features=n_node_features,
                            n_edge_features=n_edge_features,
                            n_time_features=n_time_features,
                            embedding_dimension=embedding_dimension,
                            device=device,
                            dropout=dropout)
    else:
        raise ValueError("Embedding Module {} not supported".format(module_type))


