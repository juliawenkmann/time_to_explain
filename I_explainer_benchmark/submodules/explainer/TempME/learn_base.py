
import math
import random
import sys
import argparse
import os.path as osp
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
# import numba

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

from GraphM import GraphMixer
from TGN.tgn import TGN
from TGAT import TGAT
from utils import NeighborFinder, EarlyStopMonitor, RandEdgeSampler

degree_dict = {
    "wikipedia": 20,
    "reddit": 20,
    "simulate_v1": 20,
    "simulate_v2": 20,
    "uci": 30,
    "ucim": 30,
    "mooc": 60,
    "enron": 30,
    "canparl": 30,
    "uslegis": 30,
}

### Argument and global variables
parser = argparse.ArgumentParser('Interface for temporal GNN on future link prediction')
parser.add_argument("--base_type", type=str, default="graphmixer", help="tgn or graphmixer or tgat")
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--bs', type=int, default=512, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=100, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=3, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.5, help='dropout probability')
parser.add_argument('--node_dim', type=int, default=100, help='Dimentions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimentions of the time embedding')
parser.add_argument(
    '--strict_feature_check',
    type=int,
    default=0,
    help='if 1, abort when node/edge feature arrays are near-constant (e.g., all zeros)',
)


def eval_one_epoch(args, base_model, sampler, src, dst, ts, label, val_e_idx_l=None):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    with torch.no_grad():
        base_model = base_model.eval()
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / args.bs)
        for k in tqdm(range(num_test_batch)):
            s_idx = k * args.bs
            e_idx = min(num_test_instance, s_idx + args.bs)
            if s_idx >= e_idx:
                continue
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            e_l_cut = val_e_idx_l[s_idx:e_idx] if (val_e_idx_l is not None) else None
            size = len(src_l_cut)
            src_l_fake, dst_l_fake = sampler.sample(size)
            subgraph_src = base_model.grab_subgraph(src_l_cut, ts_l_cut)
            subgraph_tgt = base_model.grab_subgraph(dst_l_cut, ts_l_cut)
            subgraph_bgd = base_model.grab_subgraph(dst_l_fake, ts_l_cut)
            if args.base_type == "tgat":
                pos_prob, neg_prob = base_model.contrast(
                    src_l_cut,
                    dst_l_cut,
                    dst_l_fake,
                    ts_l_cut,
                    e_l_cut,
                    subgraph_src,
                    subgraph_tgt,
                    subgraph_bgd,
                    test=True,
                    if_explain=False,
                )
            else:
                pos_prob, neg_prob = base_model.contrast(
                    src_l_cut,
                    dst_l_cut,
                    dst_l_fake,
                    ts_l_cut,
                    e_l_cut,
                    subgraph_src,
                    subgraph_tgt,
                    subgraph_bgd,
                    explain_weights=None,
                    edge_attr=None,
                )
            pos_prob = pos_prob.sigmoid().squeeze(-1)
            neg_prob = neg_prob.sigmoid().squeeze(-1)
            pred_score = np.concatenate([(pos_prob).cpu().numpy(), neg_prob.cpu().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))
    return np.mean(val_acc), np.mean(val_ap), None, np.mean(val_auc)


for i in range(1):
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    args.n_degree = degree_dict[args.data]

    ### Load data and train val test split
    path = osp.join(osp.dirname(osp.realpath(__file__)),  'processed')
    g_df = pd.read_csv(osp.join(osp.dirname(osp.realpath(__file__)), 'processed/ml_{}.csv'.format(args.data)))
    e_feat = np.load(osp.join(osp.dirname(osp.realpath(__file__)),  'processed/ml_{}.npy'.format(args.data)))
    n_feat = np.load(osp.join(osp.dirname(osp.realpath(__file__)),  'processed/ml_{}_node.npy'.format(args.data)))
    edge_feat_std = float(np.std(e_feat))
    node_feat_std = float(np.std(n_feat))
    print(f"edge feature std: {edge_feat_std:.6g}, node feature std: {node_feat_std:.6g}")
    if edge_feat_std <= 1e-12 or node_feat_std <= 1e-12:
        msg = (
            "Detected near-constant feature arrays. TGAT training can collapse to random-guess metrics "
            "(~0.5 acc/ap/auc). Regenerate non-zero features for ml_*.npy."
        )
        if args.strict_feature_check == 1:
            raise ValueError(msg)
        warnings.warn(msg)

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

    candidate_nodes = sorted(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])))
    sample_size = min(int(0.1 * num_total_unique_nodes), len(candidate_nodes))
    mask_node_set = set(random.sample(candidate_nodes, sample_size))
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
    val_ts_l = ts_l[valid_val_flag]
    val_e_idx_l = e_idx_l[valid_val_flag]
    val_label_l = label_l[valid_val_flag]

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

    ### Model initialize
    if args.gpu < 0 or not torch.cuda.is_available():
        args.device = torch.device("cpu")
    else:
        args.device = torch.device('cuda:{}'.format(args.gpu))
    if args.base_type == "tgn":
        base_model = TGN(n_feat, e_feat, n_neighbors=args.n_degree, device=args.device, n_layers=args.n_layer,
                         n_heads=args.n_head, dropout=args.drop_out)
    elif args.base_type == "graphmixer":
        base_model = GraphMixer(n_feat, e_feat, n_neighbors=args.n_degree, device=args.device,
                                num_tokens=args.n_degree, num_layers=args.n_layer,
                                dropout=args.drop_out)
    elif args.base_type == "tgat":
        base_model = TGAT(n_feat, e_feat, num_layers=args.n_layer, num_neighbors=args.n_degree,
            n_head=args.n_head, drop_out=args.drop_out)
    else:
        raise ValueError(f"Wrong value for base_type {args.base_type}!")

    base_model = base_model.to(args.device)
    optimizer = torch.optim.Adam(base_model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()


    num_instance = len(train_src_l)
    num_batch = math.ceil(num_instance / args.bs)
    print(f"dataset:{args.data}, base_type model:{args.base_type}")
    print('num of training instances: {}'.format(num_instance))
    print('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)
    np.random.shuffle(idx_list)
    best_aps = 0
    early_stopper = EarlyStopMonitor(max_round=5)
    for epoch in range(args.n_epoch):
        base_model.train()
        base_model.set_neighbor_sampler(train_ngh_finder)
        acc, ap, f1, auc, m_loss = [], [], [], [], []
        np.random.shuffle(idx_list)
        print('start {} epoch'.format(epoch))
        for k in tqdm(range(num_batch)):
            # percent = 100 * k / num_batch
            # if k % int(0.2 * num_batch) == 0:
            #     logger.info('progress: {0:10.4f}'.format(percent))

            s_idx = k * args.bs
            e_idx = min(num_instance, s_idx + args.bs)
            if s_idx >= e_idx:
                continue
            batch_idx = idx_list[s_idx:e_idx]
            src_l_cut, dst_l_cut = train_src_l[batch_idx], train_dst_l[batch_idx]
            ts_l_cut = train_ts_l[batch_idx]
            e_l_cut = train_e_idx_l[batch_idx]
            label_l_cut = train_label_l[batch_idx]  # currently useless since we are not predicting edge labels
            size = len(src_l_cut)
            src_l_fake, dst_l_fake = train_rand_sampler.sample(size)

            # feed in the data and learn from error
            optimizer.zero_grad()
            subgraph_src = base_model.grab_subgraph(src_l_cut, ts_l_cut)
            subgraph_tgt = base_model.grab_subgraph(dst_l_cut, ts_l_cut)
            subgraph_bgd = base_model.grab_subgraph(dst_l_fake, ts_l_cut)
            if args.base_type == "tgat":
                pos_prob, neg_prob = base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                    subgraph_src, subgraph_tgt, subgraph_bgd, if_explain=False)
            else:
                pos_prob, neg_prob = base_model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut,
                                                subgraph_src, subgraph_tgt, subgraph_bgd, explain_weights=None, edge_attr=None)
            pos_label = torch.ones((size, 1), dtype=torch.float, device=args.device, requires_grad=False)
            neg_label = torch.zeros((size, 1), dtype=torch.float, device=args.device, requires_grad=False)
            loss = criterion(pos_prob, pos_label) + criterion(neg_prob, neg_label)
            loss.backward()
            optimizer.step()

            if args.base_type == "tgn":
                base_model.memory.detach_memory()

            # get training results
            with torch.no_grad():
                pos_prob = pos_prob.sigmoid().squeeze(-1)
                neg_prob = neg_prob.sigmoid().squeeze(-1)
                pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
                pred_label = pred_score > 0.5
                true_label = np.concatenate([np.ones(size), np.zeros(size)])
                acc.append((pred_label == true_label).mean())
                ap.append(average_precision_score(true_label, pred_score))
                m_loss.append(loss.item())
                auc.append(roc_auc_score(true_label, pred_score))


        print(f"training epoch{epoch}")
        print('train acc: {}, train ap: {}, train auc:{}'.format(np.mean(acc), np.mean(ap), np.mean(auc)))
        if (epoch+1) % 1 == 0:
            if args.base_type == "tgn":
                train_memory_backup = base_model.memory.backup_memory()
            base_model.set_neighbor_sampler(full_ngh_finder)
            test_acc, test_ap, test_f1, test_auc = eval_one_epoch(args, base_model, test_rand_sampler,
                                                                  test_src_l, test_dst_l, test_ts_l, test_label_l,
                                                                  test_e_idx_l)
            if args.base_type == "tgn":
                base_model.memory.restore_memory(train_memory_backup)
            print("--------------------------")
            print('train acc: {}, test acc: {}'.format(np.mean(acc), test_acc))
            print('train ap: {}, test ap: {}'.format(np.mean(ap), test_ap))
            print('train auc: {}, test auc: {}'.format(np.mean(auc), test_auc))
            if test_ap > best_aps:
                model_path = osp.join(osp.dirname(osp.realpath(__file__)), 'params', 'tgnn')
                if not osp.exists(model_path):
                    os.makedirs(model_path)
                save_path = f"{args.base_type}_{args.data}.pt"
                torch.save(base_model, osp.join(model_path, save_path))
                print(f"save model to {osp.join(model_path, save_path)}")
                best_aps = test_ap
            if early_stopper.early_stop_check(test_ap):
                break
