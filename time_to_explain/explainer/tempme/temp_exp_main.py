"""Unified interface to all dynamic graph model experiments"""
import math
import random
import sys
from tqdm import tqdm
import argparse
import os.path as osp
import h5py
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from utils import RandEdgeSampler, load_subgraph, load_subgraph_margin, get_item, get_item_edge, NeighborFinder
from models import *
from GraphM import GraphMixer
from submodules.models.tgn.model.tgn import TGN


degree_dict = {"wikipedia": 20, "reddit": 20, "uci": 30, "mooc": 60, "enron": 30, "canparl": 30, "uslegis": 30}

parser = argparse.ArgumentParser('Interface for temporal explanation')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument("--base_type", type=str, default="tgn", help="tgn or graphmixer or tgat")
parser.add_argument('--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--bs', type=int, default=500, help='batch_size')
parser.add_argument('--test_bs', type=int, default=500, help='test batch_size')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=4, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=150, help='number of epochs')
parser.add_argument('--out_dim', type=int, default=40, help='number of attention dim')
parser.add_argument('--hid_dim', type=int, default=64, help='number of hidden dim')
parser.add_argument('--temp', type=float, default=0.07, help='temperature')
parser.add_argument('--prior_p', type=float, default=0.3, help='prior belief of the sparsity')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--if_attn', type=bool, default=True, help='use dot product attention or mapping based')
parser.add_argument('--if_bern', type=bool, default=True, help='use bernoulli')
parser.add_argument('--save_model', type=bool, default=True, help='if save model')
parser.add_argument('--test_threshold', type=bool, default=True, help='if test threshold in the evaluation')
parser.add_argument('--verbose', type=int, default=1, help='use dot product attention or mapping based')
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--lr_decay', type=float, default=0.999)
parser.add_argument('--task_type', type=str, default="temporal explanation")


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)



def norm_imp(imp):
    imp[imp < 0] = 0
    imp += 1e-16
    return imp / imp.sum()


### Load data and train val test split
def load_data(mode):
    g_df = pd.read_csv(osp.join(osp.dirname(osp.realpath(__file__)), 'processed/ml_{}.csv'.format(args.data)))
    val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))
    src_l = g_df.u.values
    dst_l = g_df.i.values
    e_idx_l = g_df.idx.values
    label_l = g_df.label.values
    ts_l = g_df.ts.values
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
    full_adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
        full_adj_list[src].append((dst, eidx, ts))
        full_adj_list[dst].append((src, eidx, ts))
    full_ngh_finder = NeighborFinder(full_adj_list)
    train_rand_sampler = RandEdgeSampler((train_src_l,), (train_dst_l,))
    test_rand_sampler = RandEdgeSampler((train_src_l, val_src_l, test_src_l), (train_dst_l, val_dst_l, test_dst_l))
    if mode == "test":
        return test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l, full_ngh_finder
    else:
        return train_rand_sampler, train_src_l, train_dst_l, train_ts_l, train_label_l, train_e_idx_l, train_ngh_finder


def threshold_test(args, explanation, base_model, src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                   pos_out_ori, neg_out_ori, y_ori, subgraph_src, subgraph_tgt, subgraph_bgd):
    '''
    calculate the AUC over ratios in [0~0.3]
    '''
    AUC_aps, AUC_acc, AUC_auc, AUC_fid_logit, AUC_fid_prob = [], [], [], [], []
    for ratio in args.ratios:
        if args.base_type == "tgn":
            num_edge = args.n_degree + args.n_degree * args.n_degree
            topk = min(max(math.ceil(ratio * num_edge), 1), num_edge)
            edge_imp_src = torch.cat([explanation[0][:args.bs], explanation[1][:args.bs]],
                                     dim=1)  # first: (batch, num_neighbors), second: (batch, num_neighbors * num_neighbors)
            edge_imp_tgt = torch.cat([explanation[0][args.bs:2 * args.bs], explanation[1][args.bs:2 * args.bs]],
                                     dim=1)
            edge_imp_bgd = torch.cat([explanation[0][2 * args.bs:], explanation[1][2 * args.bs:]], dim=1)
            selected_src = torch.topk(edge_imp_src, k=num_edge - topk, dim=-1, largest=False).indices
            selected_tgt = torch.topk(edge_imp_tgt, k=num_edge - topk, dim=-1, largest=False).indices
            selected_bgd = torch.topk(edge_imp_bgd, k=num_edge - topk, dim=-1, largest=False).indices

            node_records_src, eidx_records_src, t_records_src = subgraph_src
            node_records_src_cat = np.concatenate(node_records_src, axis=-1)  # [B, NUM + NUM**2]
            np.put_along_axis(node_records_src_cat, selected_src.cpu().numpy(), 0, axis=-1)
            node_records_src = np.split(node_records_src_cat, [args.n_degree], axis=1)
            subgraph_src_sub = node_records_src, eidx_records_src, t_records_src

            node_records_tgt, eidx_records_tgt, t_records_tgt = subgraph_tgt
            node_records_tgt_cat = np.concatenate(node_records_tgt, axis=-1)  # [B, NUM + NUM**2]
            np.put_along_axis(node_records_tgt_cat, selected_tgt.cpu().numpy(), 0, axis=-1)
            node_records_tgt = np.split(node_records_tgt_cat, [args.n_degree], axis=1)
            subgraph_tgt_sub = node_records_tgt, eidx_records_tgt, t_records_tgt

            node_records_bgd, eidx_records_bgd, t_records_bgd = subgraph_bgd
            node_records_bgd_cat = np.concatenate(node_records_bgd, axis=-1)  # [B, NUM + NUM**2]
            np.put_along_axis(node_records_bgd_cat, selected_bgd.cpu().numpy(), 0, axis=-1)
            node_records_bgd = np.split(node_records_bgd_cat, [args.n_degree], axis=1)
            subgraph_bgd_sub = node_records_bgd, eidx_records_bgd, t_records_bgd

        elif args.base_type == "graphmixer":
            num_edge = args.n_degree
            topk = min(max(math.ceil(ratio * num_edge), 1), num_edge)
            edge_imp_src, edge_imp_tgt, edge_imp_bgd = explanation[0][:args.bs], \
                                                       explanation[0][args.bs:2 * args.bs], \
                                                       explanation[0][2 * args.bs:]
            selected_src = torch.topk(edge_imp_src, k=num_edge - topk, dim=-1, largest=False).indices
            selected_tgt = torch.topk(edge_imp_tgt, k=num_edge - topk, dim=-1, largest=False).indices
            selected_bgd = torch.topk(edge_imp_bgd, k=num_edge - topk, dim=-1, largest=False).indices
            node_records_src, eidx_records_src, t_records_src = subgraph_src
            node_src_0 = node_records_src[0].copy()
            np.put_along_axis(node_src_0, selected_src.cpu().numpy(), 0, axis=-1)
            node_records_src_sub = [node_src_0, node_records_src[1]]
            subgraph_src_sub = node_records_src_sub, eidx_records_src, t_records_src

            node_records_tgt, eidx_records_tgt, t_records_tgt = subgraph_tgt
            node_tgt_0 = node_records_tgt[0].copy()
            np.put_along_axis(node_tgt_0, selected_tgt.cpu().numpy(), 0, axis=-1)
            node_records_tgt_sub = [node_tgt_0, node_records_tgt[1]]
            subgraph_tgt_sub = node_records_tgt_sub, eidx_records_tgt, t_records_tgt

            node_records_bgd, eidx_records_bgd, t_records_bgd = subgraph_bgd
            node_bgd_0 = node_records_bgd[0].copy()
            np.put_along_axis(node_bgd_0, selected_bgd.cpu().numpy(), 0, axis=-1)
            node_records_bgd_sub = [node_bgd_0, node_records_bgd[1]]
            subgraph_bgd_sub = node_records_bgd_sub, eidx_records_bgd, t_records_bgd
        elif args.base_type == "tgat":
            num_edge = args.n_degree + args.n_degree * args.n_degree
            topk = min(max(math.ceil(ratio * num_edge), 1), num_edge)
            edge_imp_src = torch.cat([explanation[0], explanation[1]], dim=1)  # first: (batch, num_neighbors), second: [batch, num_neighbors * num_neighbors]
            edge_imp_tgt = torch.cat([explanation[2], explanation[3]], dim=1)
            edge_imp_bgd = torch.cat([explanation[4], explanation[5]], dim=1)
            selected_src = torch.topk(edge_imp_src, k=num_edge - topk, dim=-1, largest=False).indices
            selected_tgt = torch.topk(edge_imp_tgt, k=num_edge - topk, dim=-1, largest=False).indices
            selected_bgd = torch.topk(edge_imp_bgd, k=num_edge - topk, dim=-1, largest=False).indices
            node_records_src, eidx_records_src, t_records_src = subgraph_src
            node_records_src_cat = np.concatenate(node_records_src, axis=-1)  # [B, NUM + NUM**2]
            np.put_along_axis(node_records_src_cat, selected_src.cpu().numpy(), 0, axis=-1)
            node_records_src = np.split(node_records_src_cat, [args.n_degree], axis=1)
            subgraph_src_sub = node_records_src, eidx_records_src, t_records_src

            node_records_tgt, eidx_records_tgt, t_records_tgt = subgraph_tgt
            node_records_tgt_cat = np.concatenate(node_records_tgt, axis=-1)  # [B, NUM + NUM**2]
            np.put_along_axis(node_records_tgt_cat, selected_tgt.cpu().numpy(), 0, axis=-1)
            node_records_tgt = np.split(node_records_tgt_cat, [args.n_degree], axis=1)
            subgraph_tgt_sub = node_records_tgt, eidx_records_tgt, t_records_tgt

            node_records_bgd, eidx_records_bgd, t_records_bgd = subgraph_bgd
            node_records_bgd_cat = np.concatenate(node_records_bgd, axis=-1)  # [B, NUM + NUM**2]
            np.put_along_axis(node_records_bgd_cat, selected_bgd.cpu().numpy(), 0, axis=-1)
            node_records_bgd = np.split(node_records_bgd_cat, [args.n_degree], axis=1)
            subgraph_bgd_sub = node_records_bgd, eidx_records_bgd, t_records_bgd
            
        else:
            raise ValueError(f"Wrong value for base_type {args.base_type}!")

        with torch.no_grad():
            if args.base_type == "tgat":
                pos_logit, neg_logit = base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                                 subgraph_src_sub, subgraph_tgt_sub, subgraph_bgd_sub, test=True,
                                                 if_explain=False)
            else:
                pos_logit, neg_logit = base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
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



def eval_one_epoch_tgat(args, base_model, explainer, full_ngh_finder, sampler, src, dst, ts, val_e_idx_l, epoch, best_accuracy, test_pack, test_edge):
    test_aps = []
    test_auc = []
    test_acc = []
    test_fid_prob = []
    test_fid_logit = []
    test_loss = []
    test_pred_loss = []
    test_kl_loss = []
    ratio_AUC_aps, ratio_AUC_auc, ratio_AUC_acc, ratio_AUC_prob, ratio_AUC_logit  = [],[],[],[],[]
    base_model = base_model.eval()
    num_test_instance = len(src) - 1
    num_test_batch = math.ceil(num_test_instance / args.test_bs)-1
    idx_list = np.arange(num_test_instance)
    criterion = torch.nn.BCEWithLogitsLoss()
    base_model.ngh_finder = full_ngh_finder
    for k in tqdm(range(num_test_batch)):
        s_idx = k * args.test_bs
        e_idx = min(num_test_instance - 1, s_idx + args.test_bs)
        if s_idx == e_idx:
            continue
        batch_idx = idx_list[s_idx:e_idx]
        src_l_cut = src[batch_idx]
        dst_l_cut = dst[batch_idx]
        ts_l_cut = ts[batch_idx]
        e_l_cut = val_e_idx_l[batch_idx] if (val_e_idx_l is not None) else None
        subgraph_src, subgraph_tgt, subgraph_bgd, walks_src, walks_tgt, walks_bgd, dst_l_fake = get_item(test_pack, batch_idx)
        edge_idfeature = get_item_edge(test_edge, batch_idx)
        src_edge, tgt_edge, bgd_edge = edge_idfeature
        with torch.no_grad():
            pos_out_ori, neg_out_ori = base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                                     subgraph_src, subgraph_tgt, subgraph_bgd,
                                                     test=True, if_explain=False)  # [B, 1]
            y_pred = torch.cat([pos_out_ori, neg_out_ori], dim=0).sigmoid()  # [B*2, 1]
            y_ori = torch.where(y_pred > 0.5, 1., 0.).view(y_pred.size(0), 1)  # [2 * B, 1]

        explainer.eval()
        graphlet_imp_src = explainer(walks_src, ts_l_cut, src_edge)
        edge_imp_src = explainer.retrieve_edge_imp(subgraph_src, graphlet_imp_src, walks_src, training=args.if_bern)
        graphlet_imp_tgt = explainer(walks_tgt, ts_l_cut, tgt_edge)
        edge_imp_tgt = explainer.retrieve_edge_imp(subgraph_tgt, graphlet_imp_tgt, walks_tgt, training=args.if_bern)
        graphlet_imp_bgd = explainer(walks_bgd, ts_l_cut, bgd_edge)
        edge_imp_bgd = explainer.retrieve_edge_imp(subgraph_bgd, graphlet_imp_bgd, walks_bgd, training=args.if_bern)
        explain_weight = [[edge_imp_src, edge_imp_tgt], [edge_imp_src, edge_imp_bgd]]
        pos_logit, neg_logit = base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                             subgraph_src, subgraph_tgt, subgraph_bgd, test=True,
                                             if_explain=True,
                                             exp_weights=explain_weight)
        pred = torch.cat([pos_logit, neg_logit], dim=0).to(args.device)
        pred_loss = criterion(pred, y_ori)
        kl_loss = explainer.kl_loss(graphlet_imp_src, walks_src, target=args.prior_p) + \
                    explainer.kl_loss(graphlet_imp_tgt, walks_tgt, target=args.prior_p) + \
                    explainer.kl_loss(graphlet_imp_bgd, walks_bgd, target=args.prior_p)
        loss = pred_loss + args.beta * kl_loss
        with torch.no_grad():
            y_pred = torch.cat([pos_logit, neg_logit], dim=0).sigmoid()
            pred_label = torch.where(y_pred > 0.5, 1., 0.).view(y_pred.size(0), 1)
            fid_prob_batch = torch.cat([pos_logit.sigmoid() - pos_out_ori.sigmoid(), neg_out_ori.sigmoid() - neg_logit.sigmoid()], dim=0)
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
            if args.test_threshold:
                node_records, eidx_records, t_records = subgraph_src
                for i in range(len(node_records)):
                    batch_node_idx = torch.from_numpy(node_records[i]).long().to(args.device)
                    mask = batch_node_idx == 0
                    edge_imp_src[i] = edge_imp_src[i].masked_fill(mask, -1e10)
                node_records, eidx_records, t_records = subgraph_tgt
                for i in range(len(node_records)):
                    batch_node_idx = torch.from_numpy(node_records[i]).long().to(args.device)
                    mask = batch_node_idx == 0
                    edge_imp_tgt[i] = edge_imp_tgt[i].masked_fill(mask, -1e10)
                node_records, eidx_records, t_records = subgraph_bgd
                for i in range(len(node_records)):
                    batch_node_idx = torch.from_numpy(node_records[i]).long().to(args.device)
                    mask = batch_node_idx == 0
                    edge_imp_bgd[i] = edge_imp_bgd[i].masked_fill(mask, -1e10)
                edge_imps = edge_imp_src + edge_imp_tgt + edge_imp_bgd
                aps_AUC, auc_AUC, acc_AUC, fid_prob_AUC, fid_logit_AUC = threshold_test(args, edge_imps, base_model, src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                     pos_out_ori, neg_out_ori, y_ori, subgraph_src, subgraph_tgt, subgraph_bgd)
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
    pred_loss_epoch = np.mean(test_pred_loss)
    kl_loss_epoch = np.mean(test_kl_loss)

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
        if args.save_model:
            model_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'params', 'explainer/tgat/')
            if not osp.exists(model_path):
                os.makedirs(model_path)
            save_path = f"{args.data}.pt"
            torch.save(explainer, osp.join(model_path, save_path))
            print(f"Save model to {osp.join(model_path, save_path)}")
        return aps_ratios_AUC
    else:
        return best_accuracy



def eval_one_epoch(args, base_model, explainer, full_ngh_finder, src, dst, ts, val_e_idx_l, epoch, best_accuracy,
                   test_pack, test_edge):
    test_aps = []
    test_auc = []
    test_acc = []
    test_fid_prob = []
    test_fid_logit = []
    test_loss = []
    test_pred_loss = []
    test_kl_loss = []
    ratio_AUC_aps, ratio_AUC_auc, ratio_AUC_acc, ratio_AUC_prob, ratio_AUC_logit = [], [], [], [], []
    base_model = base_model.eval()
    num_test_instance = len(src) - 1
    num_test_batch = math.ceil(num_test_instance / args.test_bs) - 1
    idx_list = np.arange(num_test_instance)
    criterion = torch.nn.BCEWithLogitsLoss()
    base_model.set_neighbor_sampler(full_ngh_finder)
    for k in tqdm(range(num_test_batch)):
        s_idx = k * args.test_bs
        e_idx = min(num_test_instance - 1, s_idx + args.test_bs)
        if s_idx == e_idx:
            continue
        batch_idx = idx_list[s_idx:e_idx]
        src_l_cut = src[batch_idx]
        dst_l_cut = dst[batch_idx]
        ts_l_cut = ts[batch_idx]
        e_l_cut = val_e_idx_l[batch_idx] if (val_e_idx_l is not None) else None
        subgraph_src, subgraph_tgt, subgraph_bgd, walks_src, walks_tgt, walks_bgd, dst_l_fake = get_item(test_pack,
                                                                                                         batch_idx)
        src_edge, tgt_edge, bgd_edge  = get_item_edge(test_edge, batch_idx)
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
                                                     training=args.if_bern)
        pos_logit, neg_logit = base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                                   subgraph_src, subgraph_tgt, subgraph_bgd,
                                                   explain_weights=explanation)
        pred = torch.cat([pos_logit, neg_logit], dim=0).to(args.device)
        pred_loss = criterion(pred, y_ori)
        kl_loss = explainer.kl_loss(graphlet_imp_src, walks_src, target=args.prior_p) + \
                    explainer.kl_loss(graphlet_imp_tgt, walks_tgt, target=args.prior_p) + \
                    explainer.kl_loss(graphlet_imp_bgd, walks_bgd, target=args.prior_p)
        loss = pred_loss + args.beta * kl_loss
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
            if args.test_threshold:
                explanation = explainer.retrieve_explanation(subgraph_src, graphlet_imp_src, walks_src,
                                                             subgraph_tgt, graphlet_imp_tgt, walks_tgt,
                                                             subgraph_bgd, graphlet_imp_bgd, walks_bgd,
                                                             training=False)
                aps_AUC, auc_AUC, acc_AUC, fid_prob_AUC, fid_logit_AUC = threshold_test(args, explanation, base_model,
                                                                                        src_l_cut, dst_l_cut,
                                                                                        dst_l_fake, ts_l_cut, e_l_cut,
                                                                                        pos_out_ori, neg_out_ori, y_ori,
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
    pred_loss_epoch = np.mean(test_pred_loss)
    kl_loss_epoch = np.mean(test_kl_loss)
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
        if args.save_model:
            model_path = osp.join(osp.dirname(osp.realpath(__file__)), 'params', f'explainer/{args.base_type}/')
            if not osp.exists(model_path):
                os.makedirs(model_path)
            save_path = f"{args.data}.pt"
            torch.save(explainer, osp.join(model_path, save_path))
            print(f"Save model to {osp.join(model_path, save_path)}")
        return aps_ratios_AUC
    else:
        return best_accuracy

def train(args, base_model, train_pack, test_pack, train_edge, test_edge):
    if args.base_type == "tgat":
        Explainer = TempME_TGAT(base_model, data=args.data, out_dim=args.out_dim, hid_dim=args.hid_dim, temp=args.temp,
                                dropout_p=args.drop_out, device=args.device)
    else:
        Explainer = TempME(base_model, base_model_type=args.base_type, data=args.data, out_dim=args.out_dim, hid_dim=args.hid_dim,
                                temp=args.temp, if_cat_feature=True,
                                dropout_p=args.drop_out, device=args.device)
    Explainer = Explainer.to(args.device)
    optimizer = torch.optim.Adam(Explainer.parameters(),
                                 lr=args.lr,
                                 betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    rand_sampler, src_l, dst_l, ts_l, label_l, e_idx_l, ngh_finder = load_data(mode="training")
    test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l, full_ngh_finder = load_data(
        mode="test")
    num_instance = len(src_l) - 1
    num_batch = math.ceil(num_instance / args.bs)
    best_acc = 0
    print('num of training instances: {}'.format(num_instance))
    print('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)
    np.random.shuffle(idx_list)

    for epoch in range(args.n_epoch):
        base_model.set_neighbor_sampler(ngh_finder)
        train_aps = []
        train_auc = []
        train_acc = []
        train_fid_prob = []
        train_fid_logit = []
        train_loss = []
        train_pred_loss = []
        train_kl_loss = []
        np.random.shuffle(idx_list)
        Explainer.train()
        for k in tqdm(range(num_batch)):
            s_idx = k * args.bs
            e_idx = min(num_instance - 1, s_idx + args.bs)
            if s_idx == e_idx:
                continue
            batch_idx = idx_list[s_idx:e_idx]
            src_l_cut, dst_l_cut = src_l[batch_idx], dst_l[batch_idx]
            ts_l_cut = ts_l[batch_idx]
            e_l_cut = e_idx_l[batch_idx]
            subgraph_src, subgraph_tgt, subgraph_bgd, walks_src, walks_tgt, walks_bgd, dst_l_fake = get_item(train_pack,
                                                                                                             batch_idx)
            src_edge, tgt_edge, bgd_edge = get_item_edge(train_edge, batch_idx)
            with torch.no_grad():
                if args.base_type == "tgat":
                    pos_out_ori, neg_out_ori = base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                                         subgraph_src, subgraph_tgt, subgraph_bgd, test=True,
                                                         if_explain=False)  #[B, 1]
                else:
                    pos_out_ori, neg_out_ori = base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                                            subgraph_src, subgraph_tgt, subgraph_bgd)  # [B, 1]
                y_pred = torch.cat([pos_out_ori, neg_out_ori], dim=0).sigmoid()  # [B*2, 1]
                y_ori = torch.where(y_pred > 0.5, 1., 0.).view(y_pred.size(0), 1)
            optimizer.zero_grad()
            graphlet_imp_src = Explainer(walks_src, ts_l_cut, src_edge)
            graphlet_imp_tgt = Explainer(walks_tgt, ts_l_cut, tgt_edge)
            graphlet_imp_bgd = Explainer(walks_bgd, ts_l_cut, bgd_edge)
            if args.base_type == "tgat":
                edge_imp_src = Explainer.retrieve_edge_imp(subgraph_src, graphlet_imp_src, walks_src, training=args.if_bern)
                edge_imp_tgt = Explainer.retrieve_edge_imp(subgraph_tgt, graphlet_imp_tgt, walks_tgt, training=args.if_bern)
                edge_imp_bgd = Explainer.retrieve_edge_imp(subgraph_bgd, graphlet_imp_bgd, walks_bgd, training=args.if_bern)
                explain_weight = [[edge_imp_src, edge_imp_tgt], [edge_imp_src, edge_imp_bgd]]
                pos_logit, neg_logit = base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                                    subgraph_src, subgraph_tgt, subgraph_bgd, test=True,
                                                    if_explain=True, exp_weights=explain_weight)
            else:
                explanation = Explainer.retrieve_explanation(subgraph_src, graphlet_imp_src, walks_src,
                                                            subgraph_tgt, graphlet_imp_tgt, walks_tgt,
                                                            subgraph_bgd, graphlet_imp_bgd, walks_bgd,
                                                            training=args.if_bern)
                pos_logit, neg_logit = base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                                    subgraph_src, subgraph_tgt, subgraph_bgd, explain_weights=explanation)
                
            pred = torch.cat([pos_logit, neg_logit], dim=0).to(args.device)
            pred_loss = criterion(pred, y_ori)
            kl_loss = Explainer.kl_loss(graphlet_imp_src, walks_src, target=args.prior_p) + \
                      Explainer.kl_loss(graphlet_imp_tgt, walks_tgt, target=args.prior_p) + \
                      Explainer.kl_loss(graphlet_imp_bgd, walks_bgd, target=args.prior_p)
            loss = pred_loss + args.beta * kl_loss
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                y_pred = torch.cat([pos_logit, neg_logit], dim=0).sigmoid()
                pred_label = torch.where(y_pred > 0.5, 1., 0.).view(y_pred.size(0), 1)
                fid_prob_batch = torch.cat(
                    [pos_logit.sigmoid() - pos_out_ori.sigmoid(), neg_out_ori.sigmoid() - neg_logit.sigmoid()], dim=0)
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
        if (epoch + 1) % args.verbose == 0:
            best_acc = eval_one_epoch(args, base_model, Explainer, full_ngh_finder, test_src_l,
                                      test_dst_l, test_ts_l, test_e_idx_l, epoch, best_acc, test_pack, test_edge)

if __name__ == '__main__':
    args.device = torch.device('cuda:{}'.format(args.gpu))
    args.n_degree = degree_dict[args.data]
    args.ratios = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.30]
    gnn_model_path = osp.join(osp.dirname(osp.realpath(__file__)), 'params', 'tgnn',
                              f'{args.base_type}_{args.data}.pt')
    base_model = torch.load(gnn_model_path).to(args.device)
    if args.base_type == "tgn":
        base_model.forbidden_memory_update = True
    pre_load_train = h5py.File(osp.join(osp.dirname(osp.realpath(__file__)),  'processed', f'{args.data}_train_cat.h5'), 'r')
    pre_load_test = h5py.File(osp.join(osp.dirname(osp.realpath(__file__)),  'processed', f'{args.data}_test_cat.h5'), 'r')
    e_feat = np.load(osp.join(osp.dirname(osp.realpath(__file__)), 'processed', f'ml_{args.data}.npy'))
    n_feat = np.load(osp.join(osp.dirname(osp.realpath(__file__)), 'processed', f'ml_{args.data}_node.npy'))

    train_pack = load_subgraph_margin(args, pre_load_train)
    test_pack = load_subgraph_margin(args, pre_load_test)

    train_edge = np.load(osp.join(osp.dirname(osp.realpath(__file__)), 'processed', f'{args.data}_train_edge.npy'))
    test_edge = np.load(osp.join(osp.dirname(osp.realpath(__file__)),  'processed', f'{args.data}_test_edge.npy'))

    train(args, base_model, train_pack=train_pack, test_pack=test_pack, train_edge=train_edge, test_edge=test_edge)

