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
import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

from utils import EarlyStopMonitor, RandEdgeSampler, load_subgraph_margin, get_item, get_item_edge, NeighborFinder
from models import *
from GraphM import GraphMixer
from TGN.tgn import TGN


degree_dict = {"wikipedia":20, "reddit":20 ,"uci":30 ,"mooc":60, "enron": 30, "canparl": 30, "uslegis": 30}
### Argument and global variables
parser = argparse.ArgumentParser('Motif Enhancement Verification')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument("--base_type", type=str, default="tgn", help="tgn or graphmixer or tgat")
parser.add_argument('--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--bs', type=int, default=1000, help='batch_size')
parser.add_argument('--test_bs', type=int, default=1000, help='test batch_size')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=3, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=100, help='number of epochs')
parser.add_argument('--out_dim', type=int, default=32, help='number of attention dim')
parser.add_argument('--hid_dim', type=int, default=32, help='number of hidden dim')
parser.add_argument('--temp', type=float, default=0.07, help='temperature')
parser.add_argument('--num_layers', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--if_bern', type=bool, default=True, help='use bernoulli')
parser.add_argument('--save_model', type=bool, default=False, help='if save model')
parser.add_argument('--verbose', type=int, default=3, help='use dot product attention or mapping based')
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--lr_decay', type=float, default=0.999)
parser.add_argument('--task_type', type=str, default="motif-enhanced prediction")
parser.add_argument('--wandb_sync', type=str, default="disabled", help='online  or disabled')

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



def eval_one_epoch(args, base_model, predictor, full_ngh_finder, src, dst, ts, val_e_idx_l, epoch, best_accuracy,
                   test_pack, test_edge):
    test_aps = []
    test_auc = []
    test_acc = []
    test_loss = []
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
        subgraph_src, subgraph_tgt, subgraph_bgd, walks_src, walks_tgt, walks_bgd, dst_l_fake = get_item(test_pack, batch_idx)
        edge_id_feature = get_item_edge(test_edge, batch_idx)
        predictor.eval()
        with torch.no_grad():
            #########################
            subgraph_src = base_model.grab_subgraph(src_l_cut, ts_l_cut)
            subgraph_tgt = base_model.grab_subgraph(dst_l_cut, ts_l_cut)
            subgraph_bgd = base_model.grab_subgraph(dst_l_fake, ts_l_cut)
            #########################
            src_emb, tgt_emb, bgd_emb = base_model.get_node_emb(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                               subgraph_src, subgraph_tgt, subgraph_bgd)
            pos_logit, neg_logit = predictor.enhance_predict_agg(ts_l_cut, walks_src, walks_tgt, walks_bgd,
                                                               edge_id_feature, src_emb, tgt_emb, bgd_emb)
            size = len(src_l_cut)
            pos_label = torch.ones((size, 1), dtype=torch.float, device=args.device, requires_grad=False)
            neg_label = torch.zeros((size, 1), dtype=torch.float, device=args.device, requires_grad=False)
            loss = criterion(pos_logit, pos_label) + criterion(neg_logit, neg_label)
            pos_prob = pos_logit.sigmoid().squeeze(-1)
            neg_prob = neg_logit.sigmoid().squeeze(-1)
            pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
            pred_label = pred_score > 0.5

            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            test_aps.append(average_precision_score(true_label, pred_score))
            test_auc.append(roc_auc_score(true_label, pred_score))
            test_acc.append((pred_label == true_label).mean())
            test_loss.append(loss.item())

    aps_epoch = np.mean(test_aps)
    auc_epoch = np.mean(test_auc)
    acc_epoch = np.mean(test_acc)
    loss_epoch = np.mean(test_loss)
    wandb_dict = {'Test Loss': loss_epoch, "Test Aps": aps_epoch, "Test Auc": auc_epoch, 'Test Acc': acc_epoch}
    wandb.log(wandb_dict)
    print((f'Testing Epoch: {epoch} | '
           f'Testing loss: {loss_epoch} | '
           f'Testing Aps: {aps_epoch} | '
           f'Testing Auc: {auc_epoch} | '
           f'Testing Acc: {acc_epoch} | '))

    if aps_epoch > best_accuracy:
        if args.save_model:
            model_path = osp.join(osp.dirname(osp.realpath(__file__)), 'params', f'predictors/{args.base_type}/')
            if not osp.exists(model_path):
                os.makedirs(model_path)
            save_path = f"{args.data}.pt"
            torch.save(predictor, osp.join(model_path, save_path))
            print(f"Save model to {osp.join(model_path, save_path)}")
        return aps_epoch
    else:
        return best_accuracy


def train(args, base_model, train_pack, test_pack, train_edge, test_edge):
    if args.base_type == "tgat":
        predictor = TempME_TGAT(base_model, data=args.data, out_dim=args.out_dim, hid_dim=args.hid_dim, temp=args.temp,
                                dropout_p=args.drop_out, device=args.device)
    else:
        predictor = TempME(base_model, base_model_type=args.base_type, data=args.data, out_dim=args.out_dim, hid_dim=args.hid_dim,
                                temp=args.temp, if_cat_feature=True, dropout_p=args.drop_out, device=args.device)
    predictor = predictor.to(args.device)
    optimizer = torch.optim.Adam(list(predictor.parameters()) + list(base_model.parameters()),
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
    print(f"start training: {args.data}")
    print('num of training instances: {}'.format(num_instance))
    print('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)
    np.random.shuffle(idx_list)
    for epoch in range(args.n_epoch):
        base_model.set_neighbor_sampler(ngh_finder)
        train_aps = []
        train_auc = []
        train_acc = []
        train_loss = []
        np.random.shuffle(idx_list)
        predictor.train()
        base_model.train()
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
            edge_id_feature = get_item_edge(train_edge, batch_idx)
            optimizer.zero_grad()
            #########################
            subgraph_src = base_model.grab_subgraph(src_l_cut, ts_l_cut)
            subgraph_tgt = base_model.grab_subgraph(dst_l_cut, ts_l_cut)
            subgraph_bgd = base_model.grab_subgraph(dst_l_fake, ts_l_cut)
            #########################
            src_emb, tgt_emb, bgd_emb = base_model.get_node_emb(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                                                subgraph_src, subgraph_tgt, subgraph_bgd)
            pos_logit, neg_logit = predictor.enhance_predict_agg(ts_l_cut, walks_src, walks_tgt, walks_bgd,
                                                                 edge_id_feature, src_emb, tgt_emb, bgd_emb)
            size = len(src_l_cut)
            pos_label = torch.ones((size, 1), dtype=torch.float, device=args.device, requires_grad=False)
            neg_label = torch.zeros((size, 1), dtype=torch.float, device=args.device, requires_grad=False)
            loss = criterion(pos_logit, pos_label) + criterion(neg_logit, neg_label)
            loss.backward()
            optimizer.step()

            if args.base_type == "tgn":
                base_model.memory.detach_memory()

            with torch.no_grad():
                pos_prob = pos_logit.sigmoid().squeeze(-1)
                neg_prob = neg_logit.sigmoid().squeeze(-1)
                pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
                pred_label = pred_score > 0.5
                true_label = np.concatenate([np.ones(size), np.zeros(size)])
                train_aps.append(average_precision_score(true_label, pred_score))
                train_auc.append(roc_auc_score(true_label, pred_score))
                train_acc.append((pred_label == true_label).mean())
                train_loss.append(loss.item())

        aps_epoch = np.mean(train_aps)
        auc_epoch = np.mean(train_auc)
        acc_epoch = np.mean(train_acc)
        loss_epoch = np.mean(train_loss)
        wandb_dict = {'Train Loss': loss_epoch, "Train Aps": aps_epoch, "Train Auc": auc_epoch, 'Train Acc': acc_epoch}
        wandb.log(wandb_dict)
        print((f'Training Epoch: {epoch} | '
               f'Training loss: {loss_epoch} | '
               f'Training Aps: {aps_epoch} | '
               f'Training Auc: {auc_epoch} | '
               f'Training Acc: {acc_epoch} | '))

        ### evaluation:
        if (epoch + 1) % args.verbose == 0:
            if args.base_type == "tgn":
                train_memory_backup = base_model.memory.backup_memory()
            best_acc = eval_one_epoch(args, base_model, predictor, full_ngh_finder, test_src_l,
                                      test_dst_l, test_ts_l, test_e_idx_l, epoch, best_acc, test_pack, test_edge)
            if args.base_type == "tgn":
                base_model.memory.restore_memory(train_memory_backup)

if __name__ == '__main__':
    args.device = torch.device('cuda:{}'.format(args.gpu))
    args.n_degree = degree_dict[args.data]
    gnn_model_path = osp.join(osp.dirname(osp.realpath(__file__)), 'params', 'tgnn',
                              f'{args.base_type}_{args.data}.pt')
    base_model = torch.load(gnn_model_path).to(args.device)
    pre_load_train = h5py.File(osp.join(osp.dirname(osp.realpath(__file__)), 'processed', f'{args.data}_train_cat.h5'),
                               'r')
    pre_load_test = h5py.File(osp.join(osp.dirname(osp.realpath(__file__)), 'processed', f'{args.data}_test_cat.h5'),
                              'r')
    e_feat = np.load(osp.join(osp.dirname(osp.realpath(__file__)), 'processed', f'ml_{args.data}.npy'))
    n_feat = np.load(osp.join(osp.dirname(osp.realpath(__file__)), 'processed', f'ml_{args.data}_node.npy'))

    train_pack = load_subgraph_margin(args, pre_load_train)
    test_pack = load_subgraph_margin(args, pre_load_test)

    train_edge = np.load(osp.join(osp.dirname(osp.realpath(__file__)), 'processed', f'{args.data}_train_edge.npy'))
    test_edge = np.load(osp.join(osp.dirname(osp.realpath(__file__)), 'processed', f'{args.data}_test_edge.npy'))

    train(args, base_model, train_pack=train_pack, test_pack=test_pack, train_edge=train_edge, test_edge=test_edge)






