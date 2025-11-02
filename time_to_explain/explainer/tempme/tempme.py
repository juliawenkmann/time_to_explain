import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_scatter import scatter
from utils import get_null_distribution

class Attention(nn.Module) :
    def __init__(self, input_dim, hid_dim):
        super(Attention, self).__init__()
        self.hidden_size = hid_dim
        self.W1 = nn.Linear(input_dim, input_dim)
        self.W2 = nn.Linear(input_dim, input_dim)
        self.MLP = nn.Sequential(nn.Linear(input_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim))
        self.init_weights()

    def init_weights(self) :
        nn.init.xavier_uniform_(self.W2.weight.data)
        self.W2.bias.data.fill_(0.1)

    def forward(self, src_feature):
        '''
        :param src: [bsz, n_walks, length, input_dim]
        :return: updated src features with attention: [bsz, n_walks, input_dim]
        '''
        bsz, n_walks = src_feature.shape[0], src_feature.shape[1]
        src = src_feature[:,:, 2, :].unsqueeze(2)  #[bsz, n_walks, 1, input_dim]
        tgt = src_feature[:,:,[0,1],:] #[bsz, n_walks, 2, input_dim]
        src = src.view(bsz*n_walks, 1, -1).contiguous()
        tgt = tgt.view(bsz*n_walks, 2, -1).contiguous()
        Wp = self.W1(src)    # [bsz , 1, emd]
        Wq = self.W2(tgt)   # [bsz, m,emd]
        scores = torch.bmm(Wp, Wq.transpose(2, 1))     #[bsz,1,m]
        alpha = F.softmax(scores, dim=-1)
        output = torch.bmm(alpha, Wq)  # [bsz,1,emd]
        output = src + output.sum(-2).unsqueeze(-2)
        output = self.MLP(output)  #[bsz,1,hid_dim]
        output = output.view(bsz, n_walks, 1, -1).squeeze(2)
        return output

class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super(TimeEncode, self).__init__()
        self.time_dim = expand_dim
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())
    def forward(self, ts):
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)
        harmonic = torch.cos(map_ts)
        return harmonic


class _MergeLayer(torch.nn.Module):
    def __init__(self, input_dim, hid_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(2 * input_dim, hid_dim)
        self.fc2 = torch.nn.Linear(hid_dim, 1)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        self.act = torch.nn.ReLU()

    def forward(self, x1, x2):
        #x1, x2: [bsz, input_dim]
        x = torch.cat([x1, x2], dim=-1)   #[bsz, 2*input_dim]
        h = self.act(self.fc1(x))
        z = self.fc2(h)
        return z


class event_gcn(torch.nn.Module):
    def __init__(self, event_dim, node_dim, hid_dim):
        super().__init__()
        self.lin_event = nn.Linear(event_dim, node_dim)
        self.relu = nn.ReLU()
        self.MLP = nn.Sequential(nn.Linear(node_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim))
    def forward(self, event_feature, src_features, tgt_features):
        '''
        similar to GINEConv
        :param event_feature: [bsz, n_walks, length, event_dim]
        :param src_features:  [bsz, n_walks, length, node_dim]
        :param tgt_features: [bsz, n_walks, length, node_dim]
        :return: MLP(src + ReLU(tgt+ edge info)): [bsz, n_walks, length, hid_dim]
        '''
        event = self.lin_event(event_feature)
        msg = self.relu(tgt_features + event)
        output = self.MLP(src_features + msg)
        return output


class TempME(nn.Module):
    '''
    two modules: gru + tranformer-self-attention
    '''
    def __init__(self, base, base_model_type, data, out_dim, hid_dim, prior="empirical", temp=0.07,
                 if_cat_feature=True, dropout_p=0.1, device=None):
        super(TempME, self).__init__()
        self.node_dim = base.n_feat_th.shape[1]  # node feature dimension
        self.edge_dim = base.e_feat_th.shape[1]  # edge feature dimension
        self.time_dim = self.node_dim  # default to be time feature dimension
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.base_type = base_model_type
        self.dropout_p = dropout_p
        self.temp = temp
        self.prior = prior
        self.if_cat = if_cat_feature
        self.dropout = nn.Dropout(dropout_p)
        self.device = device
        self.event_dim = self.edge_dim + self.time_dim + 3
        self.event_conv = event_gcn(event_dim=self.event_dim, node_dim=self.node_dim, hid_dim=self.hid_dim)
        self.attention = Attention(2 * self.hid_dim, self.hid_dim)
        self.mlp_dim = self.hid_dim + 12 if self.if_cat else self.hid_dim
        self.MLP = nn.Sequential(nn.Linear(self.mlp_dim, self.mlp_dim),
                                 nn.ReLU(), nn.Dropout(self.dropout_p), nn.Linear(self.mlp_dim, self.hid_dim), nn.ReLU(),
                                 nn.Linear(self.hid_dim, 1))
        self.final_linear = nn.Linear(2 * self.hid_dim, self.hid_dim)
        self.node_emd_dim = self.hid_dim + 12 + self.node_dim if self.if_cat else self.hid_dim + self.node_dim
        self.affinity_score = _MergeLayer(self.node_emd_dim, self.node_emd_dim)
        self.edge_raw_embed = base.edge_raw_features
        self.node_raw_embed = base.node_raw_features
        self.time_encoder = TimeEncode(expand_dim=self.time_dim)
        self.null_model = get_null_distribution(data_name=data)


    def forward(self, walks, cut_time_l, edge_identify):
        node_idx, edge_idx, time_idx, cat_feat, _ = walks  # [bsz, n_walk, len_walk]
        edge_features, _ = self.retrieve_edge_features(edge_idx)  # [bsz, n_walk, len_walk, edge_dim]
        edge_count = torch.from_numpy(edge_identify).float().to(self.device)
        time_features = self.retrieve_time_features(cut_time_l, time_idx)
        event_features = torch.cat([edge_features, edge_count, time_features], dim=-1)
        assert event_features.shape[-1] == self.event_dim
        src_features, tgt_features = self.retrieve_node_features(node_idx)  # [bsz, n_walk, len_walk, node_dim]
        updated_src_feature = self.event_conv(event_features, src_features,
                                              tgt_features)  # [bsz, n_walks, length, hid_dim]
        updated_tgt_feature = self.event_conv(event_features, tgt_features, src_features)
        updated_feature = torch.cat([updated_src_feature, updated_tgt_feature],
                                    dim=-1)  # [bsz, n_walks, length, hid_dim*2]
        src_feature = self.attention(updated_feature)  # [bsz, n_walks, hid_dim]
        if self.if_cat:
            event_cat_f = self.compute_catogory_feautres(cat_feat, level="event")  #[bsz, n_walks, 12]
            src_feature = torch.cat([src_feature, event_cat_f], dim=-1)
        else:
            src_feature = src_feature
        out = self.MLP(src_feature).sigmoid()
        return out  # [bsz, n_walks, 1]

    def enhance_predict_agg(self, ts_l_cut, walks_src , walks_tgt, walks_bgd, edge_id_info, src_gat, tgt_gat, bgd_gat):
        src_edge, tgt_edge, bgd_edge = edge_id_info
        src_emb, tgt_emb = self.enhance_predict_pairs(walks_src, walks_tgt, ts_l_cut, src_edge, tgt_edge)
        src_emb = torch.cat([src_emb, src_gat], dim=-1)
        tgt_emb = torch.cat([tgt_emb, tgt_gat], dim=-1)
        pos_score = self.affinity_score(src_emb, tgt_emb)  #[bsz, 1]
        src_emb, bgd_emb = self.enhance_predict_pairs(walks_src, walks_bgd, ts_l_cut, src_edge, bgd_edge)
        src_emb = torch.cat([src_emb, src_gat], dim=-1)
        bgd_emb = torch.cat([bgd_emb, bgd_gat], dim=-1)
        neg_score = self.affinity_score(src_emb, bgd_emb)  #[bsz, 1]
        return pos_score, neg_score

    def enhance_predict_pairs(self, walks_src, walks_tgt, cut_time_l, src_edge, tgt_edge):
        src_walk_emb = self.enhance_predict_walks(walks_src, cut_time_l, src_edge)
        tgt_walk_emb = self.enhance_predict_walks(walks_tgt, cut_time_l, tgt_edge)
        return src_walk_emb, tgt_walk_emb  #[bsz, hid_dim]


    def enhance_predict_walks(self, walks, cut_time_l, edge_identify):
        node_idx, edge_idx, time_idx, cat_feat, _ = walks  # [bsz, n_walk, len_walk]
        edge_features, _ = self.retrieve_edge_features(edge_idx)  # [bsz, n_walk, len_walk, edge_dim]
        edge_count = torch.from_numpy(edge_identify).float().to(self.device)
        time_features = self.retrieve_time_features(cut_time_l, time_idx)
        event_features = torch.cat([edge_features, edge_count, time_features], dim=-1)
        assert event_features.shape[-1] == self.event_dim
        src_features, tgt_features = self.retrieve_node_features(node_idx)  # [bsz, n_walk, len_walk, node_dim]
        updated_src_feature = self.event_conv(event_features, src_features,
                                              tgt_features)  # [bsz, n_walks, length, hid_dim]
        updated_tgt_feature = self.event_conv(event_features, tgt_features, src_features)
        updated_feature = torch.cat([updated_src_feature, updated_tgt_feature],
                                    dim=-1)  # [bsz, n_walks, length, hid_dim*2]
        src_features = self.attention(updated_feature)  # [bsz, n_walks, hid_dim]
        src_features = src_features.sum(1)  # [bsz, hid_dim]
        if self.if_cat:
            node_cat_f = self.compute_catogory_feautres(cat_feat, level="node")
            src_features = torch.cat([src_features, node_cat_f], dim=-1)  # [bsz, hid_dim+12]
        else:
            src_features = src_features
        return src_features

    def compute_catogory_feautres(self, cat_feat, level="node"):
        cat_feat = torch.from_numpy(cat_feat).long().to(self.device).squeeze(-1)  # [bsz, n_walks]
        cat_feat = torch.nn.functional.one_hot(cat_feat, num_classes=12).to(self.device)  #[bsz, n_walks, 12]
        node_cat_feat = torch.sum(cat_feat, dim=1)  #[bsz, 12]
        if level == "node":
            return node_cat_feat
        else:
            return cat_feat


    def retrieve_time_features(self, cut_time_l, t_records):
        '''
        :param cut_time_l: [bsz, ]
        :param t_records: [bsz, n_walk, len_walk] along the time direction
        :return: tensor shape [bsz, n_walk, len_walk, time_dim]
        '''
        batch = len(cut_time_l)
        t_records_th = torch.from_numpy(t_records).float().to(self.device)
        t_records_th = t_records_th.select(dim=-1, index=-1).unsqueeze(dim=2) - t_records_th
        n_walk, len_walk = t_records_th.size(1), t_records_th.size(2)
        time_features = self.time_encoder(t_records_th.view(batch, -1))
        time_features = time_features.view(batch, n_walk, len_walk, self.time_encoder.time_dim)
        return time_features

    def retrieve_edge_features(self, eidx_records):
        '''
        :param eidx_records: [bsz, n_walk, len_walk] along the time direction
        :return: tensor shape [bsz, n_walk, len_walk, edge_dim]
        '''
        eidx_records_th = torch.from_numpy(eidx_records).long().to(self.device)
        edge_features = self.edge_raw_embed(eidx_records_th)  # shape [batch, n_walk, len_walk+1, edge_dim]
        masks = (eidx_records_th == 0).long().to(self.device)  #[bsz, n_walk] the number of null edges in each ealk
        masks = masks.unsqueeze(-1)
        return edge_features, masks

    def retrieve_node_features(self,n_id):
        '''
        :param n_id: [bsz, n_walk, len_walk *2] along the time direction
        :return: tensor shape [bsz, n_walk, len_walk, node_dim]
        '''
        src_node = torch.from_numpy(n_id[:,:,[0,2,4]]).long().to(self.device)
        tgt_node = torch.from_numpy(n_id[:,:,[1,3,5]]).long().to(self.device)
        src_features = self.node_raw_embed(src_node)  #[bsz, n_walk, len_walk, node_dim]
        tgt_features = self.node_raw_embed(tgt_node)
        return src_features, tgt_features

    def retrieve_edge_imp_node(self, subgraph, graphlet_imp, walks, training=True):
        '''
        :param subgraph:
        :param graphlet_imp: #[bsz, n_walk, 1]
        :param walks: (n_id: [batch, n_walk, 6]
                  e_id: [batch, n_walk, 3]
                  t_id: [batch, n_walk, 3]
                  anony_id: [batch, n_walk, 3)
        :return: edge_imp_0: [batch, 20]
                 edge_imp_1: [batch, 20 * 20]
        '''
        node_record, eidx_record, t_record = subgraph
        # each of them is a list of k numpy arrays,  first: (batch, n_degree), second: [batch, n_degree*n_degree]
        edge_idx_0, edge_idx_1 = eidx_record[0], eidx_record[1]
        index_tensor_0 = torch.from_numpy(edge_idx_0).long().to(self.device)
        index_tensor_1 = torch.from_numpy(edge_idx_1).long().to(self.device)
        edge_walk = walks[1]
        num_edges = int(max(np.max(edge_idx_0), np.max(edge_idx_1), np.max(edge_walk)) + 1)
        edge_walk = edge_walk.reshape(edge_walk.shape[0], -1)   #[bsz, n_walk * 3]
        edge_walk = torch.from_numpy(edge_walk).long().to(self.device)
        walk_imp = graphlet_imp.repeat(1,1,3).view(edge_walk.shape[0], -1)  #[bsz, n_walk * 3]
        edge_imp = scatter(walk_imp, edge_walk, dim=-1, dim_size=num_edges, reduce="max")  #[bsz, num_edges]
        edge_imp_0 = torch.gather(edge_imp, dim=-1, index=index_tensor_0)
        edge_imp_1 = torch.gather(edge_imp, dim=-1, index=index_tensor_1)
        edge_imp_0 = self.concrete_bern(edge_imp_0, training)
        edge_imp_1 = self.concrete_bern(edge_imp_1, training)
        batch_node_idx0 = torch.from_numpy(node_record[0]).long().to(self.device)
        mask0 = batch_node_idx0 == 0
        edge_imp_0 = edge_imp_0.masked_fill(mask0, 0)
        batch_node_idx1 = torch.from_numpy(node_record[1]).long().to(self.device)
        mask1 = batch_node_idx1 == 0
        edge_imp_1 = edge_imp_1.masked_fill(mask1, 0)
        return edge_imp_0, edge_imp_1

    def retrieve_explanation(self, subgraph_src, graphlet_imp_src, walks_src,
                             subgraph_tgt, graphlet_imp_tgt, walks_tgt,
                             subgraph_bgd, graphlet_imp_bgd, walks_bgd, training=True):
        src_0, src_1 = self.retrieve_edge_imp_node(subgraph_src, graphlet_imp_src, walks_src, training=training)
        tgt_0, tgt_1 = self.retrieve_edge_imp_node(subgraph_tgt, graphlet_imp_tgt, walks_tgt, training=training)
        bgd_0, bgd_1 = self.retrieve_edge_imp_node(subgraph_bgd, graphlet_imp_bgd, walks_bgd, training=training)
        if self.base_type == "tgn":
            edge_imp = [torch.cat([src_0, tgt_0, bgd_0], dim=0), torch.cat([src_1, tgt_1, bgd_1], dim=0)]
        else:
            edge_imp = [torch.cat([src_0, tgt_0, bgd_0], dim=0)]
        return edge_imp

    def concrete_bern(self, prob, training):
        temp = self.temp
        if training:
            random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(self.device)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
            prob_bern = ((prob + random_noise) / temp).sigmoid()  # close
        else:
            prob_bern = prob
        return prob_bern


    def kl_loss(self, prob, walks, ratio=1, target=0.3):
        '''
        :param prob: [bsz, n_walks, 1]
        :return: KL loss: scalar
        '''
        _, _, _, cat_feat, _ = walks
        # prob = self.concrete_bern(prob, training)
        if self.prior == "empirical":
            s = torch.mean(prob, dim=1)
            null_distribution = torch.tensor(list(self.null_model.values())).to(self.device)
            num_cat = len(self.null_model.keys())
            cat_feat = torch.tensor(cat_feat).to(self.device)
            empirical_distribution = scatter(prob, index = cat_feat, reduce="mean", dim=1, dim_size=num_cat).to(self.device)
            empirical_distribution = s * empirical_distribution.reshape(-1, num_cat)
            null_distribution = target * null_distribution.reshape(-1, num_cat)
            kl_loss = ((1-s) * torch.log((1-s)/(1-target+1e-6) + 1e-6) + empirical_distribution * torch.log(empirical_distribution/(null_distribution + 1e-6)+1e-6)).mean()
        else:
            kl_loss = (prob * torch.log(prob/target + 1e-6) +
                    (1-prob) * torch.log((1-prob)/(1-target+1e-6) + 1e-6)).mean()
        return kl_loss
    
    
    

class MergeLayer_final(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        # self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        # self.fc2 = torch.nn.Linear(dim3, dim4)
        # self.act = torch.nn.ReLU()

        # torch.nn.init.xavier_normal_(self.fc1.weight)
        # torch.nn.init.xavier_normal_(self.fc2.weight)
        self.fc = torch.nn.Linear(dim1, 1)
        torch.nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x1, x2):
        #x1, x2: [bsz, n_walks, n_feat]
        x = torch.cat([x1, x2], dim=1)   #[bsz, 2M, n_feat]
        z_walk = self.fc(x).squeeze(-1)  #[bsz, 2M]
        z_final = z_walk.sum(dim=-1, keepdim=True)  #[bsz, 1]
        return z_final

class TempME_TGAT(nn.Module):
    def __init__(self, base, data, out_dim, hid_dim, temp, prior="empirical",  if_attn=True, n_head=8, dropout_p=0.1, device=None):
        super(TempME_TGAT, self).__init__()
        # self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)), requires_grad=False)
        # self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)), requires_grad=False)
        self.node_dim = base.n_feat_th.shape[1]  # node feature dimension
        self.edge_dim = base.e_feat_th.shape[1]  # edge feature dimension
        self.time_dim = self.node_dim  # default to be time feature dimension
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.if_attn = if_attn
        self.n_head = n_head
        self.dropout_p = dropout_p
        self.temp = temp
        self.dropout = nn.Dropout(dropout_p)
        self.device = device
        self.gru_dim = self.edge_dim + self.time_dim + self.node_dim * 2
        self.MLP = nn.Sequential(nn.Linear(self.out_dim + self.node_dim * 2, self.hid_dim),
                                       nn.ReLU(), nn.Dropout(self.dropout_p), nn.Linear(self.hid_dim, 1))
        self.MLP_attn = nn.Sequential(nn.Linear(self.gru_dim, self.hid_dim),
                                 nn.ReLU(), nn.Dropout(self.dropout_p), nn.Linear(self.hid_dim, self.out_dim))
        self.self_attention = TransformerEncoderLayer(d_model=self.out_dim, nhead=self.n_head,
                                                      dim_feedforward=32 * self.out_dim, dropout=self.dropout_p, batch_first=True, activation='relu')

        self.feat_dim = self.out_dim + 12
        self.affinity_score = MergeLayer_final(self.feat_dim, self.feat_dim, self.feat_dim, 1)
        self.self_attention_cat = TransformerEncoderLayer(d_model=self.gru_dim, nhead=self.n_head,
                                                      dim_feedforward=32 * self.out_dim, dropout=self.dropout_p,
                                                      batch_first=True, activation='relu')
        self.edge_raw_embed = base.edge_raw_embed
        self.node_raw_embed = base.node_raw_embed
        self.time_encoder = TimeEncode(expand_dim=self.time_dim)
        self.null_model = get_null_distribution(data_name=data)
        self.prior = prior
        
        

    def forward(self, walks, src_idx_l, cut_time_l, tgt_idx_l):
        '''
        walks: (n_id: [batch, N1 * N2, 6]
                  e_id: [batch, N1 * N2, 3]
                  t_id: [batch, N1 * N2, 3]
                  anony_id: [batch, N1 * N2, 3)
        subgraph:
        src_id: array(B, )
        tgt_id: array(B, )
        Return shape [batch,  N1 * N2, 1]
        '''
        node_idx, edge_idx, time_idx, _, _ = walks
        edge_features, masks = self.retrieve_edge_features(edge_idx)
        time_features = self.retrieve_time_features(cut_time_l, time_idx)
        node_features = self.retrieve_node_features(node_idx)  #[bsz, n_walk, len_walk, node_dim * 2]
        combined_features = torch.cat([edge_features, time_features, node_features], dim=-1).to(self.device)  #[bsz, n_walk, len_walk, gru_dim]
        n_walk = combined_features.size(1)
        src_emb = self.node_raw_embed(torch.from_numpy(np.expand_dims(src_idx_l, 1)).long().to(self.device))  #[bsz, 1, node_dim]
        tgt_emb = self.node_raw_embed(torch.from_numpy(np.expand_dims(tgt_idx_l, 1)).long().to(self.device))  # [bsz, 1, node_dim]
        src_emb = src_emb.repeat(1, n_walk, 1)
        tgt_emb = tgt_emb.repeat(1, n_walk, 1)
        assert combined_features.size(-1) == self.gru_dim
        if self.if_attn:
            graphlet_emb = self.self_attention(graphlet_emb)  #[bsz, n_walk, out_dim]
        graphlet_features = torch.cat((graphlet_emb, src_emb, tgt_emb), dim=-1)
        out = self.MLP(graphlet_features)
        return out.sigmoid()  #[bsz, n_walk, 1]

    def enhance_predict_walks(self, walks, src_idx_l, cut_time_l, tgt_idx_l):
        node_idx, edge_idx, time_idx, cat_feat, _ = walks
        cat_feat = torch.from_numpy(cat_feat).long().to(self.device).squeeze(-1) #[bsz, n_walks]
        cat_feat = torch.nn.functional.one_hot(cat_feat, num_classes=12).to(self.device)
        #[bsz, n_walks, 12]
        edge_features, masks = self.retrieve_edge_features(edge_idx)
        time_features = self.retrieve_time_features(cut_time_l, time_idx)
        node_features = self.retrieve_node_features(node_idx)  # [bsz, n_walk, len_walk, node_dim * 2]
        combined_features = torch.cat([edge_features, time_features, node_features], dim=-1).to(
            self.device)  # [bsz, n_walk, len_walk, gru_dim]
        n_walk = combined_features.size(1)
        assert combined_features.size(-1) == self.gru_dim
        graphlet_emb = self.attention_encode(combined_features)  # [bsz, n_walk, out_dim]
        graphlet_emb = torch.cat([graphlet_emb, cat_feat], dim=-1)
        if self.if_attn:
            graphlet_emb = self.self_attention_cat(graphlet_emb)  # [bsz, n_walk, out_dim+12]
        return graphlet_emb

    def enhance_predict_pairs(self, walks_src, walks_tgt, src_idx_l, cut_time_l, tgt_idx_l):
        src_walk_emb = self.enhance_predict_walks(walks_src, src_idx_l, cut_time_l, tgt_idx_l)
        tgt_walk_emb = self.enhance_predict_walks(walks_tgt, tgt_idx_l, cut_time_l, src_idx_l)
        return src_walk_emb, tgt_walk_emb  #[bsz, n_walk, n_feat]


    def enhance_predict_agg(self, src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, walks_src , walks_tgt, walks_bgd):
        src_emb, tgt_emb = self.enhance_predict_pairs(walks_src, walks_tgt, src_l_cut, ts_l_cut, dst_l_cut)
        pos_score = self.affinity_score(src_emb, tgt_emb)  #[bsz, 1]
        src_emb, bgd_emb = self.enhance_predict_pairs(walks_src, walks_bgd, src_l_cut, ts_l_cut, dst_l_fake)
        neg_score = self.affinity_score(src_emb, bgd_emb)  #[bsz, 1]
        return pos_score, neg_score


    def retrieve_time_features(self, cut_time_l, t_records):
        '''
        :param cut_time_l: [bsz, ]
        :param t_records: [bsz, n_walk, len_walk] along the time direction
        :return: tensor shape [bsz, n_walk, len_walk, time_dim]
        '''
        batch = len(cut_time_l)
        t_records_th = torch.from_numpy(t_records).float().to(self.device)
        t_records_th = t_records_th.select(dim=-1, index=-1).unsqueeze(dim=2) - t_records_th
        n_walk, len_walk = t_records_th.size(1), t_records_th.size(2)
        time_features = self.time_encoder(t_records_th.view(batch, -1))
        time_features = time_features.view(batch, n_walk, len_walk, self.time_encoder.time_dim)
        return time_features

    def retrieve_edge_features(self, eidx_records):
        '''
        :param eidx_records: [bsz, n_walk, len_walk] along the time direction
        :return: tensor shape [bsz, n_walk, len_walk, edge_dim]
        '''
        eidx_records_th = torch.from_numpy(eidx_records).long().to(self.device)
        edge_features = self.edge_raw_embed(eidx_records_th)  # shape [batch, n_walk, len_walk+1, edge_dim]
        masks = (eidx_records_th == 0).sum(dim=-1).long().to(self.device)  #[bsz, n_walk] the number of null edges in each ealk
        return edge_features, masks

    def retrieve_node_features(self,n_id):
        '''
        :param n_id: [bsz, n_walk, len_walk *2] along the time direction
        :return: tensor shape [bsz, n_walk, len_walk, node_dim * 2]
        '''
        src_node = torch.from_numpy(n_id[:,:,[0,2,4]]).long().to(self.device)
        tgt_node = torch.from_numpy(n_id[:,:,[1,3,5]]).long().to(self.device)
        src_features = self.node_raw_embed(src_node)  #[bsz, n_walk, len_walk, node_dim]
        tgt_features = self.node_raw_embed(tgt_node)
        node_features = torch.cat([src_features, tgt_features], dim=-1)
        return node_features

    def attention_encode(self, X, mask=None):
        '''
        :param X: [bsz, n_walk, len_walk, gru_dim]
        :param mask: [bsz, n_walk]
        :return: graphlet_emb: [bsz, n_walk, out_dim]
        '''
        batch, n_walk, len_walk, gru_dim = X.shape
        X = X.view(batch*n_walk, len_walk, gru_dim)

        if mask is not None:
            lengths = mask.view(batch*n_walk)
            X = pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=False)
        encoded_features = self.self_attention_cat(X)  #[bsz*n_walks, len_walks, out_dim]
        encoded_features = encoded_features.mean(1).view(batch, n_walk, gru_dim)
        if mask is not None:
            encoded_features, lengths = pad_packed_sequence(encoded_features, batch_first=True)
        encoded_features = self.MLP_attn(encoded_features)
        encoded_features = self.dropout(encoded_features)
        return encoded_features


    def retrieve_edge_imp(self, subgraph, graphlet_imp, walks, training=True):
        '''
        :param subgraph:
        :param graphlet_imp: #[bsz, n_walk, 1]
        :param walks: (n_id: [batch, n_walk, 6]
                  e_id: [batch, n_walk, 3]
                  t_id: [batch, n_walk, 3]
                  anony_id: [batch, n_walk, 3)
        :return: edge_imp_0: [batch, 20]
                 edge_imp_1: [batch, 20 * 20]
        '''
        node_record, eidx_record, t_record = subgraph
        # each of them is a list of k numpy arrays,  first: (batch, 20), second: [batch, 20 * 20]
        edge_idx_0, edge_idx_1 = eidx_record[0], eidx_record[1]
        index_tensor_0 = torch.from_numpy(edge_idx_0).long().to(self.device)
        index_tensor_1 = torch.from_numpy(edge_idx_1).long().to(self.device)
        node_walk, edge_walk, time_walk, _, _ = walks
        num_edges = int(max(np.max(edge_idx_0), np.max(edge_idx_1), np.max(edge_walk)) + 1)
        edge_walk = edge_walk.reshape(edge_walk.shape[0], -1)   #[bsz, n_walk * 3]
        edge_walk = torch.from_numpy(edge_walk).long().to(self.device)
        walk_imp = graphlet_imp.repeat(1,1,3).view(edge_walk.shape[0], -1)  #[bsz, n_walk * 3]
        edge_imp = scatter(walk_imp, edge_walk, dim=-1, dim_size=num_edges, reduce="max")  #[bsz, num_edges]
        edge_imp_0 = torch.gather(edge_imp, dim=-1, index=index_tensor_0)
        edge_imp_1 = torch.gather(edge_imp, dim=-1, index=index_tensor_1)
        edge_imp_0 = self.concrete_bern(edge_imp_0, training)
        edge_imp_1 = self.concrete_bern(edge_imp_1, training)
        batch_node_idx0 = torch.from_numpy(node_record[0]).long().to(self.device)
        mask0 = batch_node_idx0 == 0
        edge_imp_0 = edge_imp_0.masked_fill(mask0, 0)
        batch_node_idx1 = torch.from_numpy(node_record[1]).long().to(self.device)
        mask1 = batch_node_idx1 == 0
        edge_imp_1 = edge_imp_1.masked_fill(mask1, 0)
        edge_src = [edge_imp_0, edge_imp_1]
        return edge_src



    def concrete_bern(self, prob, training):
        temp = self.temp
        if training:
            random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(self.device)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
            prob_bern = ((prob + random_noise) / temp).sigmoid()
        else:
            prob_bern = prob
        return prob_bern


    def kl_loss(self, prob, walks, ratio=1, target=0.3):
        '''
        :param prob: [bsz, n_walks, 1]
        :return: KL loss: scalar
        '''
        _, _, _, cat_feat, _ = walks
        # prob = self.concrete_bern(prob, training)
        if self.prior == "empirical":
            s = torch.mean(prob, dim=1)
            null_distribution = torch.tensor(list(self.null_model.values())).to(self.device)
            num_cat = len(self.null_model.keys())
            cat_feat = torch.tensor(cat_feat).to(self.device)
            empirical_distribution = scatter(prob, index = cat_feat, reduce="mean", dim=1, dim_size=num_cat).to(self.device)
            empirical_distribution = s * empirical_distribution.reshape(-1, num_cat)
            null_distribution = target * null_distribution.reshape(-1, num_cat)
            kl_loss = ((1-s) * torch.log((1-s)/(1-target+1e-6) + 1e-6) + empirical_distribution * torch.log(empirical_distribution/(null_distribution+1e-6) + 1e-6)).mean()
        else:
            kl_loss = (prob * torch.log(prob/target + 1e-6) +
                    (1-prob) * torch.log((1-prob)/(1-target+1e-6) + 1e-6)).mean()
        return kl_loss
    