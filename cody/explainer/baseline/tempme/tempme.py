import math
import os.path
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from cody.data import ContinuousTimeDynamicGraphDataset
from cody.explainer.baseline.tempme.utils import load_data, get_item, get_item_edge, get_null_distribution
from cody.implementations.tgn import TGNWrapper
from torch_scatter import scatter
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

from cody.explainer.base import Explainer
from cody.implementations.connector import TGNNWrapper


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
        """
        :param src: [bsz, n_walks, length, input_dim]
        :return: updated src features with attention: [bsz, n_walks, input_dim]
        """
        bsz, n_walks = src_feature.shape[0], src_feature.shape[1]
        src = src_feature[:,:, 2, :].unsqueeze(2)  #[bsz, n_walks, 1, input_dim]
        tgt = src_feature[:,:,[0,1],:] #[bsz, n_walks, 2, input_dim]
        src = src.view(bsz*n_walks, 1, -1).contiguous()
        tgt = tgt.view(bsz*n_walks, 2, -1).contiguous()
        Wp = self.W1(src)    # [bsz , 1, emd]
        Wq = self.W2(tgt)   # [bsz, m,emd]
        scores = torch.bmm(Wp, Wq.transpose(2, 1))     #[bsz,1,m]
        alpha = nn.functional.softmax(scores, dim=-1)
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


class EventGCN(torch.nn.Module):
    def __init__(self, event_dim, node_dim, hid_dim):
        super().__init__()
        self.lin_event = nn.Linear(event_dim, node_dim)
        self.relu = nn.ReLU()
        self.MLP = nn.Sequential(nn.Linear(node_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim))
    def forward(self, event_feature, src_features, tgt_features):
        """
        similar to GINEConv
        :param event_feature: [bsz, n_walks, length, event_dim]
        :param src_features:  [bsz, n_walks, length, node_dim]
        :param tgt_features: [bsz, n_walks, length, node_dim]
        :return: MLP(src + ReLU(tgt+ edge info)): [bsz, n_walks, length, hid_dim]
        """
        event = self.lin_event(event_feature)
        msg = self.relu(tgt_features + event)
        output = self.MLP(src_features + msg)
        return output


class TempME(nn.Module):
    """
    two modules: gru + transformer-self-attention
    """
    def __init__(self, base: TGNNWrapper, out_dim: int, hid_dim: int,  prior="empirical", temp=0.07,
                 if_cat_feature=True, dropout_p=0.1, device=None):
        super(TempME, self).__init__()
        self.node_dim = base.dataset.node_features.shape[1]  # node feature dimension
        self.edge_dim = base.dataset.edge_features.shape[1]  # edge feature dimension
        self.time_dim = self.node_dim  # default to be time feature dimension
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.dropout_p = dropout_p
        self.temp = temp
        self.prior = prior
        self.if_cat = if_cat_feature
        self.dropout = nn.Dropout(dropout_p)
        self.device = device
        self.event_dim = self.edge_dim + self.time_dim + 3
        self.event_conv = EventGCN(event_dim=self.event_dim, node_dim=self.node_dim, hid_dim=self.hid_dim)
        self.attention = Attention(2 * self.hid_dim, self.hid_dim)
        self.mlp_dim = self.hid_dim + 12 if self.if_cat else self.hid_dim
        self.MLP = nn.Sequential(nn.Linear(self.mlp_dim, self.mlp_dim),
                                 nn.ReLU(), nn.Dropout(self.dropout_p), nn.Linear(self.mlp_dim, self.hid_dim), nn.ReLU(),
                                 nn.Linear(self.hid_dim, 1))
        self.final_linear = nn.Linear(2 * self.hid_dim, self.hid_dim)
        self.node_emd_dim = self.hid_dim + 12 + self.node_dim if self.if_cat else self.hid_dim + self.node_dim
        self.affinity_score = _MergeLayer(self.node_emd_dim, self.node_emd_dim)
        self.edge_raw_embed = base.dataset.edge_features
        self.node_raw_embed = base.dataset.node_features
        self.time_encoder = TimeEncode(expand_dim=self.time_dim)
        self.null_model = get_null_distribution(data=base.dataset)


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
        """
        :param cut_time_l: [bsz, ]
        :param t_records: [bsz, n_walk, len_walk] along the time direction
        :return: tensor shape [bsz, n_walk, len_walk, time_dim]
        """
        batch = len(cut_time_l)
        t_records_th = torch.from_numpy(t_records).float().to(self.device)
        t_records_th = t_records_th.select(dim=-1, index=-1).unsqueeze(dim=2) - t_records_th
        n_walk, len_walk = t_records_th.size(1), t_records_th.size(2)
        time_features = self.time_encoder(t_records_th.view(batch, -1))
        time_features = time_features.view(batch, n_walk, len_walk, self.time_encoder.time_dim)
        return time_features

    def retrieve_edge_features(self, eidx_records):
        """
        :param eidx_records: [bsz, n_walk, len_walk] along the time direction
        :return: tensor shape [bsz, n_walk, len_walk, edge_dim]
        """
        eidx_records_th = torch.from_numpy(eidx_records).long().to(self.device)
        edge_features = self.edge_raw_embed[eidx_records_th]  # shape [batch, n_walk, len_walk+1, edge_dim]
        masks = (eidx_records_th == 0).long().to(self.device)  #[bsz, n_walk] the number of null edges in each ealk
        masks = masks.unsqueeze(-1)
        return edge_features, masks

    def retrieve_node_features(self,n_id):
        """
        :param n_id: [bsz, n_walk, len_walk *2] along the time direction
        :return: tensor shape [bsz, n_walk, len_walk, node_dim]
        """
        src_node = torch.from_numpy(n_id[:,:,[0,2,4]]).long().to(self.device)
        tgt_node = torch.from_numpy(n_id[:,:,[1,3,5]]).long().to(self.device)
        src_features = self.node_raw_embed[src_node]  #[bsz, n_walk, len_walk, node_dim]
        tgt_features = self.node_raw_embed[tgt_node]
        return src_features, tgt_features

    def retrieve_edge_imp_node(self, subgraph, graphlet_imp, walks, training=True):
        """
        :param subgraph:
        :param graphlet_imp: #[bsz, n_walk, 1]
        :param walks: (n_id: [batch, n_walk, 6]
                  e_id: [batch, n_walk, 3]
                  t_id: [batch, n_walk, 3]
                  anony_id: [batch, n_walk, 3)
        :return: edge_imp_0: [batch, 20]
                 edge_imp_1: [batch, 20 * 20]
        """
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
        edge_imp = [torch.cat([src_0, tgt_0, bgd_0], dim=0), torch.cat([src_1, tgt_1, bgd_1], dim=0)]
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
        """
        :param prob: [bsz, n_walks, 1]
        :return: KL loss: scalar
        """
        _, _, _, cat_feat, _ = walks
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


class TempMeExplainer(Explainer):

    def __init__(self, tgnn_wrapper: TGNWrapper, device: str = 'cpu', dropout: float = 0.1, out_dim: int = 40,
                 hid_dim: int = 64, temp: float = 0.07):
        super().__init__(tgnn_wrapper=tgnn_wrapper)
        self.tgnn = tgnn_wrapper
        self.explainer = TempME(self.tgnn, out_dim, hid_dim, temp=temp, dropout_p=dropout, device=device)
        self.device = device

        self.explainer = self.explainer.to(self.device)




    def train(self, train_pack, test_pack, train_edge, test_edge, save_directory: str, learning_rate: float = 1e-3,
              learning_rate_decay: float = 0.999, beta: float = 0.5, weight_decay: float = 0, batch_size: int = 32,
              epochs: int = 150, prior_p: float = 0.3, use_bernoulli: bool = True):
        optimizer = torch.optim.Adam(self.explainer.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
                                     weight_decay=weight_decay)
        criterion = torch.nn.BCEWithLogitsLoss()

        rand_sampler, src_l, dst_l, ts_l, label_l, e_idx_l, ngh_finder = load_data(self.tgnn, mode="training")
        test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l, full_ngh_finder = load_data(
            self.tgnn, mode="test")

        num_instance = len(src_l) - 1
        num_batch = math.ceil(num_instance / batch_size)
        best_acc = 0
        print('num of training instances: {}'.format(num_instance))
        print('num of batches per epoch: {}'.format(num_batch))
        idx_list = np.arange(num_instance)
        np.random.shuffle(idx_list)

        for epoch in range(epochs):
            self.tgnn.model.set_neighbor_finder(ngh_finder)
            train_aps = []
            train_auc = []
            train_acc = []
            train_fid_prob = []
            train_fid_logit = []
            train_loss = []
            train_pred_loss = []
            train_kl_loss = []
            np.random.shuffle(idx_list)
            self.explainer.train()
            for k in tqdm(range(num_batch)):
                s_idx = k * batch_size
                e_idx = min(num_instance - 1, s_idx + batch_size)
                if s_idx == e_idx:
                    continue
                batch_idx = idx_list[s_idx:e_idx]
                src_l_cut, dst_l_cut = src_l[batch_idx], dst_l[batch_idx]
                ts_l_cut = ts_l[batch_idx]
                e_l_cut = e_idx_l[batch_idx]
                subgraph_src, subgraph_tgt, subgraph_bgd, walks_src, walks_tgt, walks_bgd, dst_l_fake = get_item(
                    train_pack, batch_idx)
                src_edge, tgt_edge, bgd_edge = get_item_edge(train_edge, batch_idx)
                with torch.no_grad():
                    pos_out_ori, neg_out_ori = self.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut,
                                                                   e_l_cut,
                                                                   subgraph_src, subgraph_tgt,
                                                                   subgraph_bgd)  # [B, 1]

                    y_pred = torch.cat([pos_out_ori, neg_out_ori], dim=0).sigmoid()  # [B*2, 1]
                    y_ori = torch.where(y_pred > 0.5, 1., 0.).view(y_pred.size(0), 1)
                optimizer.zero_grad()
                graphlet_imp_src = TempME(self.tgnn, walks_src, ts_l_cut, src_edge)
                graphlet_imp_tgt = TempME(self.tgnn, walks_tgt, ts_l_cut, tgt_edge)
                graphlet_imp_bgd = TempME(self.tgnn, walks_bgd, ts_l_cut, bgd_edge)

                explanation = self.explainer.retrieve_explanation(subgraph_src, graphlet_imp_src, walks_src,
                                                             subgraph_tgt, graphlet_imp_tgt, walks_tgt,
                                                             subgraph_bgd, graphlet_imp_bgd, walks_bgd,
                                                             training=use_bernoulli)
                pos_logit, neg_logit = self.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                                           subgraph_src, subgraph_tgt, subgraph_bgd,
                                                           explain_weights=explanation)

                pred = torch.cat([pos_logit, neg_logit], dim=0).to(self.device)
                pred_loss = criterion(pred, y_ori)
                kl_loss = self.explainer.kl_loss(graphlet_imp_src, walks_src, target=prior_p) + \
                          self.explainer.kl_loss(graphlet_imp_tgt, walks_tgt, target=prior_p) + \
                          self.explainer.kl_loss(graphlet_imp_bgd, walks_bgd, target=prior_p)
                loss = pred_loss + beta * kl_loss
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    y_pred = torch.cat([pos_logit, neg_logit], dim=0).sigmoid()
                    pred_label = torch.where(y_pred > 0.5, 1., 0.).view(y_pred.size(0), 1)
                    fid_prob_batch = torch.cat(
                        [pos_logit.sigmoid() - pos_out_ori.sigmoid(), neg_out_ori.sigmoid() - neg_logit.sigmoid()],
                        dim=0)
                    fid_prob = torch.mean(fid_prob_batch, dim=0)
                    fid_logit_batch = torch.cat([pos_logit - pos_out_ori, neg_out_ori - neg_logit], dim=0)
                    fid_logit = torch.mean(fid_logit_batch, dim=0)
                    train_fid_prob.append(fid_prob.item())
                    train_fid_logit.append(fid_logit.item())
                    train_aps.append(average_precision_score(y_ori.cpu(), y_pred.cpu()))
                    train_auc.append(roc_auc_score(y_ori.cpu(), y_pred.cpu()))
                    train_acc.append((pred_label.cpu() == y_ori.cpu()).float().mean())
                    train_loss.append(loss.item())
                    train_pred_loss.append(pred_loss.item())
                    train_kl_loss.append(kl_loss.item())

            aps_epoch = np.mean(train_aps)
            auc_epoch = np.mean(train_auc)
            acc_epoch = np.mean(train_acc)
            fid_prob_epoch = np.mean(train_fid_prob)
            fid_logit_epoch = np.mean(train_fid_logit)
            loss_epoch = np.mean(train_loss)
            print((f'Training Epoch: {epoch} | '
                   f'Training loss: {loss_epoch} | '
                   f'Training Aps: {aps_epoch} | '
                   f'Training Auc: {auc_epoch} | '
                   f'Training Acc: {acc_epoch} | '
                   f'Training Fidelity Prob: {fid_prob_epoch} | '
                   f'Training Fidelity Logit: {fid_logit_epoch} | '))

            ### evaluation:
            best_acc = self.eval_one_epoch(full_ngh_finder, test_src_l, test_dst_l, test_ts_l, test_e_idx_l, epoch,
                                           best_acc, test_pack, test_edge, beta, batch_size, prior_p, use_bernoulli)

            Path(save_directory).mkdir(parents=True, exist_ok=True)
            checkpoint_path = os.path.join(save_directory, f'tempME_checkpoint_e{epoch}.pth')
            self._save_explainer(checkpoint_path)

        print("Finished training.")
        save_path = os.path.join(save_directory, 'tempME_final.pth')
        self._save_explainer(save_path)


    def contrast(self, src_idx, tgt_idx, bgd_idx, cut_time, e_idx,
                 subgraph_src, subgraph_tgt, subgraph_bgd,
                 explain_weights=None, edge_attr=None):

        if hasattr(self.tgnn.model.embedding_module, 'atten_weights_list'):  # ! avoid cuda memory leakage
            self.tgnn.model.embedding_module.atten_weights_list = []

        n_samples = len(src_idx)
        source_node_embedding, destination_node_embedding, negative_node_embedding = \
            self.tgnn.model.get_node_emb(src_idx, tgt_idx, bgd_idx, cut_time, e_idx,
                              subgraph_src, subgraph_tgt, subgraph_bgd, explain_weights, edge_attr)

        score = self.tgnn.model.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                    torch.cat([destination_node_embedding,
                                               negative_node_embedding])).squeeze(dim=0)
        pos_score = score[:n_samples]
        neg_score = score[n_samples:]

        return pos_score, neg_score

    def eval_one_epoch(self, explainer, full_ngh_finder, src, dst, ts, val_e_idx_l, epoch, best_accuracy,
                       test_pack, test_edge, beta: float = 0.5, batch_size: int = 32, prior_p: float = 0.3,
                       use_bernoulli: bool = True):
        test_aps = []
        test_auc = []
        test_acc = []
        test_fid_prob = []
        test_fid_logit = []
        test_loss = []
        test_pred_loss = []
        test_kl_loss = []
        ratio_AUC_aps, ratio_AUC_auc, ratio_AUC_acc, ratio_AUC_prob, ratio_AUC_logit = [], [], [], [], []
        base_model = self.tgnn.model.eval()
        num_test_instance = len(src) - 1
        num_test_batch = math.ceil(num_test_instance / batch_size) - 1
        idx_list = np.arange(num_test_instance)
        criterion = torch.nn.BCEWithLogitsLoss()
        base_model.set_neighbor_sampler(full_ngh_finder)
        for k in tqdm(range(num_test_batch)):
            s_idx = k * batch_size
            e_idx = min(num_test_instance - 1, s_idx + batch_size)
            if s_idx == e_idx:
                continue
            batch_idx = idx_list[s_idx:e_idx]
            src_l_cut = src[batch_idx]
            dst_l_cut = dst[batch_idx]
            ts_l_cut = ts[batch_idx]
            e_l_cut = val_e_idx_l[batch_idx] if (val_e_idx_l is not None) else None
            subgraph_src, subgraph_tgt, subgraph_bgd, walks_src, walks_tgt, walks_bgd, dst_l_fake = get_item(test_pack,
                                                                                                             batch_idx)
            src_edge, tgt_edge, bgd_edge = get_item_edge(test_edge, batch_idx)
            with torch.no_grad():
                pos_out_ori, neg_out_ori = base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                                               subgraph_src, subgraph_tgt, subgraph_bgd)  # [B, 1]
                y_pred = torch.cat([pos_out_ori, neg_out_ori], dim=0).sigmoid()  # [B*2, 1]
                y_ori = torch.where(y_pred > 0.5, 1., 0.).view(y_pred.size(0), 1)  # [2 * B, 1]

            explainer.eval()
            graphlet_imp_src = explainer(walks_src, ts_l_cut, src_edge)
            graphlet_imp_tgt = explainer(walks_tgt, ts_l_cut, tgt_edge)
            graphlet_imp_bgd = explainer(walks_bgd, ts_l_cut, bgd_edge)
            explanation = explainer.retrieve_explanation(subgraph_src, graphlet_imp_src, walks_src,
                                                         subgraph_tgt, graphlet_imp_tgt, walks_tgt,
                                                         subgraph_bgd, graphlet_imp_bgd, walks_bgd,
                                                         training=use_bernoulli)
            pos_logit, neg_logit = base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                                       subgraph_src, subgraph_tgt, subgraph_bgd,
                                                       explain_weights=explanation)
            pred = torch.cat([pos_logit, neg_logit], dim=0).to(self.device)
            pred_loss = criterion(pred, y_ori)
            kl_loss = explainer.kl_loss(graphlet_imp_src, walks_src, target=prior_p) + \
                      explainer.kl_loss(graphlet_imp_tgt, walks_tgt, target=prior_p) + \
                      explainer.kl_loss(graphlet_imp_bgd, walks_bgd, target=prior_p)
            loss = pred_loss + beta * kl_loss
            with torch.no_grad():
                y_pred = torch.cat([pos_logit, neg_logit], dim=0).sigmoid()
                pred_label = torch.where(y_pred > 0.5, 1., 0.).view(y_pred.size(0), 1)
                fid_prob_batch = torch.cat(
                    [pos_logit.sigmoid() - pos_out_ori.sigmoid(), neg_out_ori.sigmoid() - neg_logit.sigmoid()], dim=0)
                fid_prob = torch.mean(fid_prob_batch, dim=0)
                fid_logit_batch = torch.cat([pos_logit - pos_out_ori, neg_out_ori - neg_logit], dim=0)
                fid_logit = torch.mean(fid_logit_batch, dim=0)
                test_fid_prob.append(fid_prob.item())
                test_fid_logit.append(fid_logit.item())
                test_aps.append(average_precision_score(y_ori.cpu(), y_pred.cpu()))
                test_auc.append(roc_auc_score(y_ori.cpu(), y_pred.cpu()))
                test_acc.append((pred_label.cpu() == y_ori.cpu()).float().mean())
                test_loss.append(loss.item())
                test_pred_loss.append(pred_loss.item())
                test_kl_loss.append(kl_loss.item())

                explanation = explainer.retrieve_explanation(subgraph_src, graphlet_imp_src, walks_src,
                                                             subgraph_tgt, graphlet_imp_tgt, walks_tgt,
                                                             subgraph_bgd, graphlet_imp_bgd, walks_bgd,
                                                             training=False)
                aps_AUC, auc_AUC, acc_AUC, fid_prob_AUC, fid_logit_AUC = self.threshold_test(explanation,
                                                                                        src_l_cut, dst_l_cut,
                                                                                        dst_l_fake, ts_l_cut,
                                                                                        e_l_cut,
                                                                                        pos_out_ori, neg_out_ori,
                                                                                        y_ori,
                                                                                        subgraph_src, subgraph_tgt,
                                                                                        subgraph_bgd)
                ratio_AUC_aps.append(aps_AUC)
                ratio_AUC_auc.append(auc_AUC)
                ratio_AUC_acc.append(acc_AUC)
                ratio_AUC_prob.append(fid_prob_AUC)
                ratio_AUC_logit.append(fid_logit_AUC)

        aps_ratios_AUC = np.mean(ratio_AUC_aps) if len(ratio_AUC_aps) != 0 else 0
        auc_ratios_AUC = np.mean(ratio_AUC_auc) if len(ratio_AUC_auc) != 0 else 0
        acc_ratios_AUC = np.mean(ratio_AUC_acc) if len(ratio_AUC_acc) != 0 else 0
        prob_ratios_AUC = np.mean(ratio_AUC_prob) if len(ratio_AUC_prob) != 0 else 0
        logit_ratios_AUC = np.mean(ratio_AUC_logit) if len(ratio_AUC_logit) != 0 else 0
        aps_epoch = np.mean(test_aps)
        auc_epoch = np.mean(test_auc)
        acc_epoch = np.mean(test_acc)
        fid_prob_epoch = np.mean(test_fid_prob)
        fid_logit_epoch = np.mean(test_fid_logit)
        loss_epoch = np.mean(test_loss)
        print((f'Testing Epoch: {epoch} | '
               f'Testing loss: {loss_epoch} | '
               f'Testing Aps: {aps_epoch} | '
               f'Testing Auc: {auc_epoch} | '
               f'Testing Acc: {acc_epoch} | '
               f'Testing Fidelity Prob: {fid_prob_epoch} | '
               f'Testing Fidelity Logit: {fid_logit_epoch} | '
               f'Ratio APS: {aps_ratios_AUC} | '
               f'Ratio AUC: {auc_ratios_AUC} | '
               f'Ratio ACC: {acc_ratios_AUC} | '
               f'Ratio Prob: {prob_ratios_AUC} | '
               f'Ratio Logit: {logit_ratios_AUC} | '))

        if aps_ratios_AUC > best_accuracy:
            return aps_ratios_AUC
        else:
            return best_accuracy

    def threshold_test(self, explanation, src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                       pos_out_ori, neg_out_ori, y_ori, subgraph_src, subgraph_tgt, subgraph_bgd, batch_size: int = 32,
                       n_degree: int = 20):
        '''
        calculate the AUC over ratios in [0~0.3]
        '''
        AUC_aps, AUC_acc, AUC_auc, AUC_fid_logit, AUC_fid_prob = [], [], [], [], []
        for ratio in np.append(0.01, np.arange(0.02, 0.31, 0.02)):

            num_edge = n_degree + n_degree * n_degree
            topk = min(max(math.ceil(ratio * num_edge), 1), num_edge)
            edge_imp_src = torch.cat([explanation[0][:batch_size], explanation[1][:batch_size]],
                                     dim=1)  # first: (batch, num_neighbors), second: (batch, num_neighbors * num_neighbors)
            edge_imp_tgt = torch.cat([explanation[0][batch_size:2 * batch_size], explanation[1][batch_size:2 * batch_size]],
                                     dim=1)
            edge_imp_bgd = torch.cat([explanation[0][2 * batch_size:], explanation[1][2 * batch_size:]], dim=1)
            selected_src = torch.topk(edge_imp_src, k=num_edge - topk, dim=-1, largest=False).indices
            selected_tgt = torch.topk(edge_imp_tgt, k=num_edge - topk, dim=-1, largest=False).indices
            selected_bgd = torch.topk(edge_imp_bgd, k=num_edge - topk, dim=-1, largest=False).indices

            node_records_src, eidx_records_src, t_records_src = subgraph_src
            node_records_src_cat = np.concatenate(node_records_src, axis=-1)  # [B, NUM + NUM**2]
            np.put_along_axis(node_records_src_cat, selected_src.cpu().numpy(), 0, axis=-1)
            node_records_src = np.split(node_records_src_cat, [n_degree], axis=1)
            subgraph_src_sub = node_records_src, eidx_records_src, t_records_src

            node_records_tgt, eidx_records_tgt, t_records_tgt = subgraph_tgt
            node_records_tgt_cat = np.concatenate(node_records_tgt, axis=-1)  # [B, NUM + NUM**2]
            np.put_along_axis(node_records_tgt_cat, selected_tgt.cpu().numpy(), 0, axis=-1)
            node_records_tgt = np.split(node_records_tgt_cat, [n_degree], axis=1)
            subgraph_tgt_sub = node_records_tgt, eidx_records_tgt, t_records_tgt

            node_records_bgd, eidx_records_bgd, t_records_bgd = subgraph_bgd
            node_records_bgd_cat = np.concatenate(node_records_bgd, axis=-1)  # [B, NUM + NUM**2]
            np.put_along_axis(node_records_bgd_cat, selected_bgd.cpu().numpy(), 0, axis=-1)
            node_records_bgd = np.split(node_records_bgd_cat, [n_degree], axis=1)
            subgraph_bgd_sub = node_records_bgd, eidx_records_bgd, t_records_bgd


            with torch.no_grad():
                pos_logit, neg_logit = self.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                                           subgraph_src_sub, subgraph_tgt_sub, subgraph_bgd_sub)
                y_pred = torch.cat([pos_logit, neg_logit], dim=0).sigmoid()
                pred_label = torch.where(y_pred > 0.5, 1., 0.).view(y_pred.size(0), 1)
                fid_prob_batch = torch.cat(
                    [pos_logit.sigmoid() - pos_out_ori.sigmoid(), neg_out_ori.sigmoid() - neg_logit.sigmoid()],
                    dim=0)
                fid_prob = torch.mean(fid_prob_batch, dim=0)
                fid_logit_batch = torch.cat([pos_logit - pos_out_ori, neg_out_ori - neg_logit], dim=0)
                fid_logit = torch.mean(fid_logit_batch, dim=0)
                AUC_fid_prob.append(fid_prob.item())
                AUC_fid_logit.append(fid_logit.item())
                AUC_aps.append(average_precision_score(y_ori.cpu(), y_pred.cpu()))
                AUC_auc.append(roc_auc_score(y_ori.cpu(), y_pred.cpu()))
                AUC_acc.append((pred_label.cpu() == y_ori.cpu()).float().mean())
        aps_AUC = np.mean(AUC_aps)
        auc_AUC = np.mean(AUC_auc)
        acc_AUC = np.mean(AUC_acc)
        fid_prob_AUC = np.mean(AUC_fid_prob)
        fid_logit_AUC = np.mean(AUC_fid_logit)
        return aps_AUC, auc_AUC, acc_AUC, fid_prob_AUC, fid_logit_AUC

    def _save_explainer(self, path: str):
        torch.save(self.explainer, path)
