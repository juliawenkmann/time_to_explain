import logging
import time
import numpy as np
import torch
import torch.nn as nn
PRECISION = 5
POS_DIM_ALTER = 100

class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4, non_linear=True):
        super().__init__()
        #self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc11 = torch.nn.Linear(dim1, dim3)
        self.fc12 = torch.nn.Linear(dim2, dim3)
        self.fc21 = torch.nn.Linear(dim3, dim4)
        self.fc22 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc11.weight)
        torch.nn.init.xavier_normal_(self.fc21.weight)
        torch.nn.init.xavier_normal_(self.fc12.weight)
        torch.nn.init.xavier_normal_(self.fc22.weight)

    def forward(self, x1, x2, explain_weight=None):  #[x1: B,N,D1], x2: [B, N,D2]
        x11 = self.fc11(x1)
        x21 = self.fc21(self.act(x11))
        x12 = self.fc12(x2)
        x22 = self.fc22(self.act(x12))
        if explain_weight is not None:
            explain_weight = explain_weight.unsqueeze(-1).repeat(1, 1, x21.size(-1))
            out = x22 + x21 * explain_weight
        else:
            out = x22 + x21
        return out


class MergeLayer_final(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        # self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        # x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        return self.fc2(h)


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

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None, explain_weight=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        B, N_src, _ = q.size() # [B, N_src, model_dim]
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
        mask = mask.view(B*N_src, 1, num_neighbors).repeat(n_head, 1, 1) # [B*N_src*n_head, 1, num_neighbors]
        if explain_weight is not None:
            explain_weight = explain_weight.view(B*N_src, 1, num_neighbors).repeat(n_head, 1, 1)  # [B*N_src*n_head, 1, num_neighbors]
        output, attn_map = self.attention(q, k, v, explain_weight=explain_weight, mask=mask) # output: [B*N_src*n_head, 1, d_v], attn_map: [B*N_src*n_head, 1, num_neighbors]

        output = output.view(B, N_src, n_head*d_v)  # [B, N_src, n_head*d_v]
        output = self.dropout(self.fc(output))  # [B, N_src, model_dim]
        output = self.layer_norm(output + residual)  # [B, N_src, model_dim]
        attn_map = attn_map.view(B, N_src, n_head, num_neighbors)
        return output, attn_map


class MapBasedMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.wq_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wk_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wv_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.weight_map = nn.Linear(2 * d_k, 1, bias=False)

        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()

        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.wq_node_transform(q).view(sz_b, len_q, n_head, d_k)

        k = self.wk_node_transform(k).view(sz_b, len_k, n_head, d_k)

        v = self.wv_node_transform(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        q = torch.unsqueeze(q, dim=2) # [(n*b), lq, 1, dk]
        q = q.expand(q.shape[0], q.shape[1], len_k, q.shape[3]) # [(n*b), lq, lk, dk]

        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        k = torch.unsqueeze(k, dim=1) # [(n*b), 1, lk, dk]
        k = k.expand(k.shape[0], len_q, k.shape[2], k.shape[3]) # [(n*b), lq, lk, dk]

        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x lq x lk

        # Map based Attention
        # output, attn = self.attention(q, k, v, mask=mask)
        q_k = torch.cat([q, k], dim=3) # [(n*b), lq, lk, dk * 2]
        attn = self.weight_map(q_k).squeeze(dim=3) # [(n*b), lq, lk]

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_q, l_k]

        # [n * b, l_q, l_k] * [n * b, l_v, d_v] >> [n * b, l_q, d_v]
        output = torch.bmm(attn, v)

        output = output.view(n_head, sz_b, len_q, d_v)

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.act(self.fc(output)))
        output = self.layer_norm(output + residual)

        return output, attn

class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()

        self.time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())


    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)

        return harmonic #self.dense(harmonic)


class PosEncode(torch.nn.Module):
    def __init__(self, expand_dim, seq_len):
        super().__init__()

        self.pos_embeddings = nn.Embedding(num_embeddings=seq_len, embedding_dim=expand_dim)

    def forward(self, ts):
        # ts: [N, L]
        order = ts.argsort()
        ts_emb = self.pos_embeddings(order)
        return ts_emb


class EmptyEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super().__init__()
        self.expand_dim = expand_dim

    def forward(self, ts):
        out = torch.zeros_like(ts).float()
        out = torch.unsqueeze(out, dim=-1)
        out = out.expand(out.shape[0], out.shape[1], self.expand_dim)
        return out


class LSTMPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim, time_dim):
        super(LSTMPool, self).__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.edge_dim = edge_dim

        self.att_dim = feat_dim + edge_dim + time_dim

        self.act = torch.nn.ReLU()

        self.lstm = torch.nn.LSTM(input_size=self.att_dim,
                                  hidden_size=self.feat_dim,
                                  num_layers=1,
                                  batch_first=True)
        self.merger = MergeLayer(feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        seq_x = torch.cat([seq, seq_e, seq_t], dim=2)

        _, (hn, _) = self.lstm(seq_x)

        hn = hn[-1, :, :] #hn.squeeze(dim=0)

        out = self.merger.forward(hn, src)
        return out, None


class MeanPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim):
        super(MeanPool, self).__init__()
        self.edge_dim = edge_dim
        self.feat_dim = feat_dim
        self.act = torch.nn.ReLU()
        self.merger = MergeLayer(edge_dim + feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        src_x = src
        seq_x = torch.cat([seq, seq_e], dim=2) #[B, N, De + D]
        hn = seq_x.mean(dim=1) #[B, De + D]
        output = self.merger(hn, src_x)
        return output, None


class AttnModel(torch.nn.Module):
    """Attention based temporal layers
    """
    def __init__(self, feat_dim, edge_dim, time_dim, model_dim,
                 attn_mode='prod', n_head=2, drop_out=0.1):
        """
        args:
          feat_dim: dim for the node features
          edge_dim: dim for the temporal edge features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super(AttnModel, self).__init__()

        self.feat_dim = feat_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        self.model_dim = model_dim

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

        assert(self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode

        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(n_head,
                                             d_model=self.model_dim,
                                             d_k=self.model_dim // n_head,
                                             d_v=self.model_dim // n_head,
                                             dropout=drop_out)
            self.logger.info('Using scaled prod attention')

        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(n_head,
                                             d_model=self.model_dim,
                                             d_k=self.model_dim // n_head,
                                             d_v=self.model_dim // n_head,
                                             dropout=drop_out)
            self.logger.info('Using map based attention')
        else:
            raise ValueError('attn_mode can only be prod or map')

    def forward(self, src, src_t, seq, seq_t, seq_e, mask, explain_weight=None):
        """"Attention based temporal attention forward pass
        args:
          src: source embedding [B, N_src, D]
          src_t: source time emb[B, N_src, Dt], Dt == D
          seq: neighbor embedding [B, N_ngh, D]
          seq_t: neighbor time emb[B, N_ngh, Dt]
          seq_e: neighbor edges [B, N_ngh, De], De == D
          mask: boolean Tensor of shape [B, N_ngh], where the true value indicate a null value in the sequence.

        returns:
          output, weight
          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """
        batch, N_src, _ = src.shape
        N_ngh = seq.shape[1]
        device = src.device
        src_e = torch.zeros((batch, N_src, self.edge_dim)).float().to(device)
        q = torch.cat([src, src_e, src_t], dim=2) # [B, N_src, D + De + Dt] -> [B, N_src, D]
        k = torch.cat([seq, seq_e, seq_t], dim=2) # [B, N_ngh, D + De + Dt] -> [B, N_ngh, D]
        output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask, explain_weight=explain_weight)
        # output: [B, N_src, D + De + Dt], attn: [B, N_src, n_head, num_neighbors]
        output = self.merger(output, src)
        return output, attn


class TGAT(torch.nn.Module):
    def __init__(self, n_feat, e_feat, num_neighbors, attn_mode='prod', use_time='time', attn_agg_method='attn',
                 num_layers=3, n_head=4, drop_out=0.1,verbosity=1):
        super(TGAT, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.verbosity = verbosity
        # subgraph extraction hyper-parameters
        self.num_neighbors, self.num_layers = num_neighbors, num_layers
        self.ngh_finder = None
        # features
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)), requires_grad=False)
        self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)), requires_grad=False)

        # dimensions of 4 elements: node, edge, time
        self.feat_dim = self.n_feat_th.shape[1]  # node feature dimension
        self.e_feat_dim = self.e_feat_th.shape[1]  # edge feature dimension
        self.time_dim = self.feat_dim  # default to be time feature dimension
        self.model_dim = self.feat_dim + self.e_feat_dim + self.time_dim
        print('neighbors: {}, node dim: {}, edge dim: {}, edge dim: {}'.format(self.num_neighbors, self.feat_dim, self.e_feat_dim,  self.time_dim))

        # dropout for both tree and walk based model
        self.dropout_p = drop_out

        # embedding layers and encoders
        self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
        self.time_encoder = self.init_time_encoder(use_time, seq_len=self.num_neighbors)

        # attention model
        self.attn_model_list = self.init_attn_model_list(attn_agg_method, attn_mode, n_head, drop_out)

        # final projection layer
        self.affinity_score = MergeLayer_final(self.feat_dim, self.feat_dim, self.feat_dim, 1)
        self.flag_for_cur_edge = True  # flagging whether the current edge under computation is real edges, for data analysis
        self.common_node_percentages = {'pos': [], 'neg': []}

    def init_attn_model_list(self, attn_agg_method, attn_mode, n_head, drop_out):
        if attn_agg_method == 'attn':
            self.logger.info('Aggregation uses attention model')
            attn_model_list = torch.nn.ModuleList([AttnModel(self.feat_dim, self.e_feat_dim, self.time_dim,
                                                             self.model_dim, attn_mode=attn_mode, n_head=n_head, drop_out=drop_out)
                                                   for _ in range(self.num_layers)])
        elif attn_agg_method == 'lstm':
            self.logger.info('Aggregation uses LSTM model')
            attn_model_list = torch.nn.ModuleList([LSTMPool(self.feat_dim,
                                                                 self.feat_dim,
                                                                 self.feat_dim) for _ in range(self.num_layers)])
        elif attn_agg_method == 'mean':
            self.logger.info('Aggregation uses constant mean model')
            attn_model_list = torch.nn.ModuleList([MeanPool(self.feat_dim,
                                                                 self.feat_dim) for _ in range(self.num_layers)])
        else:
            raise NotImplementedError('invalid agg_method value, use attn or lstm')
        return attn_model_list

    def init_time_encoder(self, use_time, seq_len):
        if use_time == 'time':
            self.logger.info('Using time encoding')
            time_encoder = TimeEncode(expand_dim=self.time_dim)
        elif use_time == 'pos':
            assert(seq_len is not None)
            self.logger.info('Using positional encoding')
            time_encoder = PosEncode(expand_dim=self.time_dim, seq_len=seq_len)
        elif use_time == 'empty':
            self.logger.info('Using empty encoding')
            time_encoder = EmptyEncode(expand_dim=self.time_dim)
        else:
            raise ValueError('invalid time option!')
        return time_encoder



    def contrast(self, src_idx_l, tgt_idx_l, bgd_idx_l, cut_time_l, e_idx_l, subgraph_src, subgraph_tgt, subgraph_bgd,
                 test=False, if_explain=False, exp_weights=None):
        '''
        1. grab subgraph for src, tgt, bgd
        2. add positional encoding for src & tgt nodes
        3. forward propagate to get src embeddings and tgt embeddings (and finally pos_score (shape: [batch, ]))
        4. forward propagate to get src embeddings and bgd embeddings (and finally neg_score (shape: [batch, ]))
        '''
        start = time.time()
        end = time.time()
        if self.verbosity > 1:
            self.logger.info('grab subgraph for the minibatch, time eclipsed: {} seconds'.format(str(end-start)))
        self.flag_for_cur_edge = True
        if if_explain == False:
            exp_tgt = exp_bgd = None
        else:
            exp_tgt, exp_bgd = exp_weights[0], exp_weights[1]
        pos_score = self.forward(src_idx_l, tgt_idx_l, cut_time_l, (subgraph_src, subgraph_tgt), test=test, explain_weights=exp_tgt)
        self.flag_for_cur_edge = False
        neg_score1 = self.forward(src_idx_l, bgd_idx_l, cut_time_l, (subgraph_src, subgraph_bgd), test=test, explain_weights=exp_bgd)
        return pos_score, neg_score1 #[B,N]

    def get_attn_map(self, src_idx_l, tgt_idx_l, bgd_idx_l, cut_time_l, e_idx_l, subgraph_src, subgraph_tgt, subgraph_bgd,
                 test=False, if_explain=False, exp_weights=None):
        '''
        1. grab subgraph for src, tgt, bgd
        2. add positional encoding for src & tgt nodes
        3. forward propagate to get src embeddings and tgt embeddings (and finally pos_score (shape: [batch, ]))
        4. forward propagate to get src embeddings and bgd embeddings (and finally neg_score (shape: [batch, ]))
        '''
        start = time.time()
        end = time.time()
        if self.verbosity > 1:
            self.logger.info('grab subgraph for the minibatch, time eclipsed: {} seconds'.format(str(end-start)))
        self.flag_for_cur_edge = True
        if if_explain == False:
            exp_tgt = exp_bgd = None
        else:
            exp_tgt, exp_bgd = exp_weights[0], exp_weights[1]
        pos_score, attn_pos = self.forward_attn(src_idx_l, tgt_idx_l, cut_time_l, (subgraph_src, subgraph_tgt), test=test, explain_weights=exp_tgt)
        self.flag_for_cur_edge = False
        neg_score1, attn_neg = self.forward_attn(src_idx_l, bgd_idx_l, cut_time_l, (subgraph_src, subgraph_bgd), test=test, explain_weights=exp_bgd)

        return attn_pos, attn_neg #[B,N]


    def get_node_emb(self, src_idx_l, tgt_idx_l, bgd_idx_l, cut_time_l, e_idx_l, subgraph_src, subgraph_tgt, subgraph_bgd,
                 test=False):
        '''
        1. grab subgraph for src, tgt, bgd
        2. add positional encoding for src & tgt nodes
        3. forward propagate to get src embeddings and tgt embeddings (and finally pos_score (shape: [batch, ]))
        4. forward propagate to get src embeddings and bgd embeddings (and finally neg_score (shape: [batch, ]))
        '''
        src_embed = self.forward_msg(src_idx_l, cut_time_l, subgraph_src, test=test)
        tgt_emded = self.forward_msg(tgt_idx_l, cut_time_l, subgraph_tgt, test=test)
        bgd_emded = self.forward_msg(bgd_idx_l, cut_time_l, subgraph_bgd, test=test)

        return src_embed, tgt_emded, bgd_emded #[B,N]



    def contrast_attr(self, src_idx_l, tgt_idx_l, bgd_idx_l, cut_time_l, e_idx_l, subgraph_src, subgraph_tgt, subgraph_bgd,
                      src_edge_attr, tgt_edge_attr, bgd_edge_attr, test=False, if_explain=False, exp_weights=None):
        start = time.time()
        end = time.time()
        if self.verbosity > 1:
            self.logger.info('grab subgraph for the minibatch, time eclipsed: {} seconds'.format(str(end-start)))
        self.flag_for_cur_edge = True
        if if_explain == False:
            exp_tgt = exp_bgd = None
        else:
            exp_tgt, exp_bgd = exp_weights[0], exp_weights[1]
        pos_score = self.forward_attr(src_idx_l, tgt_idx_l, cut_time_l, src_edge_attr, tgt_edge_attr, (subgraph_src, subgraph_tgt), test=test, explain_weights=exp_tgt)
        self.flag_for_cur_edge = False
        neg_score1 = self.forward_attr(src_idx_l, bgd_idx_l, cut_time_l, src_edge_attr, bgd_edge_attr, (subgraph_src, subgraph_bgd), test=test, explain_weights=exp_bgd)
        # neg_score2 = self.forward(tgt_idx_l, bgd_idx_l, cut_time_l, (subgraph_tgt, subgraph_bgd))
        # return pos_score.sigmoid(), (neg_score1.sigmoid() + neg_score2.sigmoid())/2.0
        return pos_score, neg_score1 #[B,N]

    def forward(self, src_idx_l, tgt_idx_l, cut_time_l, subgraphs=None, test=False, explain_weights=None):
        subgraph_src, subgraph_tgt = subgraphs
        if explain_weights is not None:
            exp_src, exp_tgt = explain_weights[0],explain_weights[1]
        else:
            exp_src = exp_tgt = None
        src_embed = self.forward_msg(src_idx_l, cut_time_l, subgraph_src, test=test, explain_weights=exp_src)
        tgt_embed = self.forward_msg(tgt_idx_l, cut_time_l, subgraph_tgt, test=test, explain_weights=exp_tgt)
        score = self.affinity_score(src_embed, tgt_embed) # score shape: [B,1]
        # score = score.squeeze(-1)
        return score  #[B,N]

    def forward_attn(self, src_idx_l, tgt_idx_l, cut_time_l, subgraphs=None, test=False, explain_weights=None):
        subgraph_src, subgraph_tgt = subgraphs
        if explain_weights is not None:
            exp_src, exp_tgt = explain_weights[0],explain_weights[1]
        else:
            exp_src = exp_tgt = None
        src_embed, attn_src = self.forward_msg_attn(src_idx_l, cut_time_l, subgraph_src, test=test, explain_weights=exp_src)
        tgt_embed, attn_tgt = self.forward_msg_attn(tgt_idx_l, cut_time_l, subgraph_tgt, test=test, explain_weights=exp_tgt)
        score = self.affinity_score(src_embed, tgt_embed) # score shape: [B,1]
        # score = score.squeeze(-1)
        attn_for_explanation = [attn_src, attn_tgt]
        return score, attn_for_explanation


    def forward_attr(self, src_idx_l, tgt_idx_l, cut_time_l, src_edge_attr, tgt_edge_attr, subgraphs=None, test=False, explain_weights=None):
        subgraph_src, subgraph_tgt = subgraphs
        if explain_weights is not None:
            exp_src, exp_tgt = explain_weights[0], explain_weights[1]
        else:
            exp_src = exp_tgt = None
        ### for src subgraph:
        node_records, eidx_records, t_records = subgraph_src
        hidden_embeddings, masks = self.init_hidden_embeddings(src_idx_l, node_records)  # length self.num_layers+1
        time_features = self.retrieve_time_features(cut_time_l, t_records)  # length self.num_layers+1
        n_layer = self.num_layers
        for layer in range(n_layer):
            hidden_embeddings = self.forward_msg_layer(hidden_embeddings, time_features[:n_layer + 1 - layer],
                                                       src_edge_attr[:n_layer - layer],
                                                       masks[:n_layer - layer], self.attn_model_list[layer],
                                                       explain_weights=exp_src)
        src_embed = hidden_embeddings[0].squeeze(1)

        ### for tgt subgrah:
        node_records, eidx_records, t_records = subgraph_tgt
        hidden_embeddings, masks = self.init_hidden_embeddings(tgt_idx_l, node_records)  # length self.num_layers+1
        time_features = self.retrieve_time_features(cut_time_l, t_records)  # length self.num_layers+1
        n_layer = self.num_layers
        for layer in range(n_layer):
            hidden_embeddings = self.forward_msg_layer(hidden_embeddings, time_features[:n_layer + 1 - layer],
                                                       tgt_edge_attr[:n_layer - layer],
                                                       masks[:n_layer - layer], self.attn_model_list[layer],
                                                       explain_weights=exp_tgt)
        tgt_embed = hidden_embeddings[0].squeeze(1)
        score = self.affinity_score(src_embed, tgt_embed) # score shape: [B,1]
        # score = score.squeeze(-1)
        return score  #[B,1]

    def set_neighbor_sampler(self, neighbor_sampler):
        self.ngh_finder = neighbor_sampler

    def grab_subgraph(self, src_idx_l, cut_time_l, e_idx_l=None):
        subgraph = self.ngh_finder.find_k_hop(self.num_layers, src_idx_l, cut_time_l, num_neighbors=self.num_neighbors, e_idx_l=e_idx_l)
        return subgraph

    def forward_msg(self, src_idx_l, cut_time_l, subgraph_src, test=False, explain_weights=None):
        node_records, eidx_records, t_records = subgraph_src
        hidden_embeddings, masks = self.init_hidden_embeddings(src_idx_l, node_records)  # length self.num_layers+1
        time_features = self.retrieve_time_features(cut_time_l, t_records)  # length self.num_layers+1
        edge_features = self.retrieve_edge_features(eidx_records)  # length self.num_layers
        n_layer = self.num_layers
        for layer in range(n_layer):
            hidden_embeddings = self.forward_msg_layer(hidden_embeddings, time_features[:n_layer+1-layer],
                                                           edge_features[:n_layer-layer],
                                                       masks[:n_layer-layer], self.attn_model_list[layer],
                                                       explain_weights=explain_weights)
        final_node_embeddings = hidden_embeddings[0].squeeze(1)
        return final_node_embeddings



    def forward_msg_attn(self, src_idx_l, cut_time_l, subgraph_src, test=False, explain_weights=None):
        node_records, eidx_records, t_records = subgraph_src
        hidden_embeddings, masks = self.init_hidden_embeddings(src_idx_l, node_records)  # length self.num_layers+1
        time_features = self.retrieve_time_features(cut_time_l, t_records)  # length self.num_layers+1
        edge_features = self.retrieve_edge_features(eidx_records)  # length self.num_layers
        n_layer = self.num_layers
        attn_map_list = []
        for layer in range(n_layer):
            hidden_embeddings, attn_map = self.retrieve_attn_map_layer(hidden_embeddings, time_features[:n_layer+1-layer],
                                                           edge_features[:n_layer-layer],
                                                       masks[:n_layer-layer], self.attn_model_list[layer],
                                                       explain_weights=explain_weights)
            attn_map_list.append(attn_map)
        final_node_embeddings = hidden_embeddings[0].squeeze(1)
        return final_node_embeddings, attn_map_list


    def tune_msg(self, src_embed, tgt_embed):
        return self.random_walk_attn_model.mutual_query(src_embed, tgt_embed)

    def init_hidden_embeddings(self, src_idx_l, node_records):
        device = self.n_feat_th.device
        hidden_embeddings, masks = [], []
        hidden_embeddings.append(self.node_raw_embed(torch.from_numpy(np.expand_dims(src_idx_l, 1)).long().to(device)))
        for i in range(len(node_records)):
            batch_node_idx = torch.from_numpy(node_records[i]).long().to(device)
            hidden_embeddings.append(self.node_raw_embed(batch_node_idx))
            masks.append(batch_node_idx == 0)
        return hidden_embeddings, masks

    def retrieve_time_features(self, cut_time_l, t_records):
        device = self.n_feat_th.device
        batch = len(cut_time_l)
        first_time_stamp = np.expand_dims(cut_time_l, 1)
        time_features = [self.time_encoder(torch.from_numpy(np.zeros_like(first_time_stamp)).float().to(device))]
        standard_timestamps = np.expand_dims(first_time_stamp, 2)
        for layer_i in range(len(t_records)):
            t_record = t_records[layer_i]
            time_delta = standard_timestamps - t_record.reshape(batch, -1, self.num_neighbors)
            time_delta = time_delta.reshape(batch, -1)
            time_delta = torch.from_numpy(time_delta).float().to(device)
            time_features.append(self.time_encoder(time_delta))
            standard_timestamps = np.expand_dims(t_record, 2)
        return time_features

    def retrieve_edge_features(self, eidx_records):
        # Notice that if subgraph is tree, then len(eidx_records) is just the number of hops, excluding the src node
        # but if subgraph is walk, then eidx_records contains the random walks of length len_walk+1, including the src node
        device = self.n_feat_th.device
        edge_features = []
        for i in range(len(eidx_records)):
            batch_edge_idx = torch.from_numpy(eidx_records[i]).long().to(device)
            edge_features.append(self.edge_raw_embed(batch_edge_idx))
        return edge_features

    def forward_msg_layer(self, hidden_embeddings, time_features, edge_features, masks, attn_m, explain_weights=None):
        assert(len(hidden_embeddings) == len(time_features))
        assert(len(hidden_embeddings) == (len(edge_features) + 1))
        assert(len(masks) == len(edge_features))
        new_src_embeddings = []
        for i in range(len(edge_features)):  #num_layer
            src_embedding = hidden_embeddings[i]
            src_time_feature = time_features[i]
            ngh_embedding = hidden_embeddings[i+1]
            ngh_time_feature = time_features[i+1]
            ngh_edge_feature = edge_features[i]
            ngh_mask = masks[i]
            if explain_weights is not None:
                explain_weight = explain_weights[i]
            else:
                explain_weight = None
            # NOTE: n_neighbor_support = n_source_support * num_neighbor this layer
            # new_src_embedding shape: [batch, n_source_support, feat_dim]
            # attn_map shape: [batch, n_source_support, n_head, num_neighbors]
            new_src_embedding, attn_map = attn_m(src_embedding,  # shape [batch, n_source_support, feat_dim]
                                                 src_time_feature,  # shape [batch, n_source_support, time_feat_dim]
                                                 ngh_embedding,  # shape [batch, n_neighbor_support, feat_dim]
                                                 ngh_time_feature,  # shape [batch, n_neighbor_support, time_feat_dim]
                                                 ngh_edge_feature,  # shape [batch, n_neighbor_support, edge_feat_dim]
                                                 ngh_mask,
                                                 explain_weight=explain_weight)  # shape [batch, n_neighbor_support]

            new_src_embeddings.append(new_src_embedding)
        return new_src_embeddings



    def retrieve_attn_map_layer(self, hidden_embeddings, time_features, edge_features, masks, attn_m, explain_weights=None):
        assert(len(hidden_embeddings) == len(time_features))
        assert(len(hidden_embeddings) == (len(edge_features) + 1))
        assert(len(masks) == len(edge_features))
        attn_map_list = []
        new_src_embeddings = []
        for i in range(len(edge_features)):  #num_layer
            src_embedding = hidden_embeddings[i]
            src_time_feature = time_features[i]
            ngh_embedding = hidden_embeddings[i+1]
            ngh_time_feature = time_features[i+1]
            ngh_edge_feature = edge_features[i]
            ngh_mask = masks[i]
            if explain_weights is not None:
                explain_weight = explain_weights[i]
            else:
                explain_weight = None
            # NOTE: n_neighbor_support = n_source_support * num_neighbor this layer
            # new_src_embedding shape: [batch, n_source_support, feat_dim]
            # attn_map shape: [batch, n_source_support, n_head, num_neighbors]
            new_src_embedding, attn_map = attn_m(src_embedding,  # shape [batch, n_source_support, feat_dim]
                                                 src_time_feature,  # shape [batch, n_source_support, time_feat_dim]
                                                 ngh_embedding,  # shape [batch, n_neighbor_support, feat_dim]
                                                 ngh_time_feature,  # shape [batch, n_neighbor_support, time_feat_dim]
                                                 ngh_edge_feature,  # shape [batch, n_neighbor_support, edge_feat_dim]
                                                 ngh_mask,
                                                 explain_weight=explain_weight)  # shape [batch, n_neighbor_support]
            new_src_embeddings.append(new_src_embedding)
            attn_map_list.append(attn_map)
        return new_src_embeddings, attn_map_list
