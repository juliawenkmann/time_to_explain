import numpy as np

### Utility function and class
#TODO same as in utils.utils
class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        self.epoch_count += 1

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
        return self.num_round >= self.max_round


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list):
        src_list = np.concatenate(src_list)
        dst_list = np.concatenate(dst_list)
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

    def sample(self, size):
        src_index = np.random.randint(0, len(self.src_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]
#TODO : end same

def load_subgraph(args, file, batch_id):
    ####### subgraph_src
    subgraph_src_0 = file["subgraph_src_0"][:]
    subgraph_src_0 = subgraph_src_0[batch_id]  # array [bsz, degree * 3]
    x0, y0, z0 = subgraph_src_0[:, 0:args.n_degree], subgraph_src_0[:,
                                                     args.n_degree: 2 * args.n_degree], subgraph_src_0[:,
                                                                                        2 * args.n_degree: 3 * args.n_degree]
    node_records, eidx_records, t_records = [x0], [y0], [z0]
    subgraph_src_1 = file["subgraph_src_1"][:]
    subgraph_src_1 = subgraph_src_1[batch_id]  # array [bsz, degree * 3]
    x1, y1, z1 = subgraph_src_1[:, 0:args.n_degree ** 2], subgraph_src_1[:,
                                                          args.n_degree ** 2: 2 * args.n_degree ** 2], subgraph_src_1[:,
                                                                                                       2 * args.n_degree ** 2: 3 * args.n_degree ** 2]
    node_records.append(x1)
    eidx_records.append(y1)
    t_records.append(z1)
    subgraph_src = (node_records, eidx_records, t_records)

    ####### subgraph_tgt
    subgraph_tgt_0 = file["subgraph_tgt_0"][:]
    subgraph_tgt_0 = subgraph_tgt_0[batch_id]  # array [bsz, degree * 3]
    x0, y0, z0 = subgraph_tgt_0[:, 0:args.n_degree], subgraph_tgt_0[:,
                                                     args.n_degree: 2 * args.n_degree], subgraph_tgt_0[:,
                                                                                        2 * args.n_degree: 3 * args.n_degree]
    node_records, eidx_records, t_records = [x0], [y0], [z0]
    subgraph_tgt_1 = file["subgraph_tgt_1"][:]
    subgraph_tgt_1 = subgraph_tgt_1[batch_id]  # array [bsz, degree * 3]
    x1, y1, z1 = subgraph_tgt_1[:, 0:args.n_degree ** 2], subgraph_tgt_1[:,
                                                          args.n_degree ** 2: 2 * args.n_degree ** 2], subgraph_tgt_1[:,
                                                                                                       2 * args.n_degree ** 2: 3 * args.n_degree ** 2]
    node_records.append(x1)
    eidx_records.append(y1)
    t_records.append(z1)
    subgraph_tgt = (node_records, eidx_records, t_records)

    ### subgraph_bgd
    subgraph_bgd_0 = file["subgraph_bgd_0"][:]
    subgraph_bgd_0 = subgraph_bgd_0[batch_id]  # array [bsz, degree * 3]
    x0, y0, z0 = subgraph_bgd_0[:, 0:args.n_degree], subgraph_bgd_0[:,
                                                     args.n_degree: 2 * args.n_degree], subgraph_bgd_0[:,
                                                                                        2 * args.n_degree: 3 * args.n_degree]
    node_records, eidx_records, t_records = [x0], [y0], [z0]
    subgraph_bgd_1 = file["subgraph_bgd_1"][:]
    subgraph_bgd_1 = subgraph_bgd_1[batch_id]  # array [bsz, degree * 3]
    x1, y1, z1 = subgraph_bgd_1[:, 0:args.n_degree ** 2], subgraph_bgd_1[:,
                                                          args.n_degree ** 2: 2 * args.n_degree ** 2], subgraph_bgd_1[:,
                                                                                                       2 * args.n_degree ** 2: 3 * args.n_degree ** 2]
    node_records.append(x1)
    eidx_records.append(y1)
    t_records.append(z1)
    subgraph_bgd = (node_records, eidx_records, t_records)

    walks_src = file["walks_src"][:]
    walks_src = walks_src[batch_id]
    node_records, eidx_records, t_records, out_anony = walks_src[:, :, :6], walks_src[:, :, 6:9], walks_src[:, :,
                                                                                                  9:12], walks_src[:, :,
                                                                                                         12:15]
    walks_src = (node_records.astype(int), eidx_records.astype(int), t_records, out_anony.astype(int))

    walks_tgt = file["walks_tgt"][:]
    walks_tgt = walks_tgt[batch_id]
    node_records, eidx_records, t_records, out_anony = walks_tgt[:, :, :6], walks_tgt[:, :, 6:9], walks_tgt[:, :,
                                                                                                  9:12], walks_tgt[:, :,
                                                                                                         12:15]
    walks_tgt = (node_records.astype(int), eidx_records.astype(int), t_records, out_anony.astype(int))

    walks_bgd = file["walks_bgd"][:]
    walks_bgd = walks_bgd[batch_id]
    node_records, eidx_records, t_records, out_anony = walks_bgd[:, :, :6], walks_bgd[:, :, 6:9], walks_bgd[:, :,
                                                                                                  9:12], walks_bgd[:, :,
                                                                                                         12:15]
    walks_bgd = (node_records.astype(int), eidx_records.astype(int), t_records, out_anony.astype(int))
    return subgraph_src, subgraph_tgt, subgraph_bgd, walks_src, walks_tgt, walks_bgd


def load_subgraph_margin(args, file):
    ####### subgraph_src
    subgraph_src_0 = file["subgraph_src_0"][:]
    x0, y0, z0 = subgraph_src_0[:, 0:args.n_degree], subgraph_src_0[:,
                                                     args.n_degree: 2 * args.n_degree], subgraph_src_0[:,
                                                                                        2 * args.n_degree: 3 * args.n_degree]
    node_records, eidx_records, t_records = [x0], [y0], [z0]
    subgraph_src_1 = file["subgraph_src_1"][:]
    x1, y1, z1 = subgraph_src_1[:, 0:args.n_degree ** 2], subgraph_src_1[:,
                                                          args.n_degree ** 2: 2 * args.n_degree ** 2], subgraph_src_1[:,
                                                                                                       2 * args.n_degree ** 2: 3 * args.n_degree ** 2]
    node_records.append(x1)
    eidx_records.append(y1)
    t_records.append(z1)
    subgraph_src = (node_records, eidx_records, t_records)

    ####### subgraph_tgt
    subgraph_tgt_0 = file["subgraph_tgt_0"][:]
    x0, y0, z0 = subgraph_tgt_0[:, 0:args.n_degree], subgraph_tgt_0[:,
                                                     args.n_degree: 2 * args.n_degree], subgraph_tgt_0[:,
                                                                                        2 * args.n_degree: 3 * args.n_degree]
    node_records, eidx_records, t_records = [x0], [y0], [z0]
    subgraph_tgt_1 = file["subgraph_tgt_1"][:]
    x1, y1, z1 = subgraph_tgt_1[:, 0:args.n_degree ** 2], subgraph_tgt_1[:,
                                                          args.n_degree ** 2: 2 * args.n_degree ** 2], subgraph_tgt_1[:,
                                                                                                       2 * args.n_degree ** 2: 3 * args.n_degree ** 2]
    node_records.append(x1)
    eidx_records.append(y1)
    t_records.append(z1)
    subgraph_tgt = (node_records, eidx_records, t_records)

    ### subgraph_bgd
    subgraph_bgd_0 = file["subgraph_bgd_0"][:]
    x0, y0, z0 = subgraph_bgd_0[:, 0:args.n_degree], subgraph_bgd_0[:,
                                                     args.n_degree: 2 * args.n_degree], subgraph_bgd_0[:,
                                                                                        2 * args.n_degree: 3 * args.n_degree]
    node_records, eidx_records, t_records = [x0], [y0], [z0]
    subgraph_bgd_1 = file["subgraph_bgd_1"][:]
    x1, y1, z1 = subgraph_bgd_1[:, 0:args.n_degree ** 2], subgraph_bgd_1[:,
                                                          args.n_degree ** 2: 2 * args.n_degree ** 2], subgraph_bgd_1[:,
                                                                                                       2 * args.n_degree ** 2: 3 * args.n_degree ** 2]
    node_records.append(x1)
    eidx_records.append(y1)
    t_records.append(z1)
    subgraph_bgd = (node_records, eidx_records, t_records)

    walks_src = file["walks_src_new"][:]
    node_records, eidx_records, t_records, cat_feat, marginal = walks_src[:, :, :6], walks_src[:, :, 6:9], walks_src[:,
                                                                                                           :,
                                                                                                           9:12], walks_src[
                                                                                                                  :, :,
                                                                                                                  12:13], walks_src[
                                                                                                                          :,
                                                                                                                          :,
                                                                                                                          13:14]
    walks_src = (node_records.astype(int), eidx_records.astype(int), t_records, cat_feat.astype(int), marginal)

    walks_tgt = file["walks_tgt_new"][:]
    node_records, eidx_records, t_records, cat_feat, marginal = walks_tgt[:, :, :6], walks_tgt[:, :, 6:9], walks_tgt[:,
                                                                                                           :,
                                                                                                           9:12], walks_tgt[
                                                                                                                  :, :,
                                                                                                                  12:13], walks_tgt[
                                                                                                                          :,
                                                                                                                          :,
                                                                                                                          13:14]
    walks_tgt = (node_records.astype(int), eidx_records.astype(int), t_records, cat_feat.astype(int), marginal)

    walks_bgd = file["walks_bgd_new"][:]
    node_records, eidx_records, t_records, cat_feat, marginal = walks_bgd[:, :, :6], walks_bgd[:, :, 6:9], walks_bgd[:,
                                                                                                           :,
                                                                                                           9:12], walks_bgd[
                                                                                                                  :, :,
                                                                                                                  12:13], walks_bgd[
                                                                                                                          :,
                                                                                                                          :,
                                                                                                                          13:14]
    walks_bgd = (node_records.astype(int), eidx_records.astype(int), t_records, cat_feat.astype(int), marginal)

    dst_fake = file["dst_fake"][:]
    pack = (subgraph_src, subgraph_tgt, subgraph_bgd, walks_src, walks_tgt, walks_bgd, dst_fake)
    return pack


def get_item(input_pack, batch_id):
    subgraph_src, subgraph_tgt, subgraph_bgd, walks_src, walks_tgt, walks_bgd, dst_fake = input_pack
    node_records, eidx_records, t_records = subgraph_src
    node_records = [i[batch_id] for i in node_records]
    eidx_records = [i[batch_id] for i in eidx_records]
    t_records = [i[batch_id] for i in t_records]
    subgraph_src = (node_records, eidx_records, t_records)

    node_records, eidx_records, t_records = subgraph_tgt
    node_records = [i[batch_id] for i in node_records]
    eidx_records = [i[batch_id] for i in eidx_records]
    t_records = [i[batch_id] for i in t_records]
    subgraph_tgt = (node_records, eidx_records, t_records)

    node_records, eidx_records, t_records = subgraph_bgd
    node_records = [i[batch_id] for i in node_records]
    eidx_records = [i[batch_id] for i in eidx_records]
    t_records = [i[batch_id] for i in t_records]
    subgraph_bgd = (node_records, eidx_records, t_records)

    walks_src = [item[batch_id] for item in walks_src]
    walks_src = (walks_src[0], walks_src[1], walks_src[2], walks_src[3], walks_src[4])

    walks_tgt = [item[batch_id] for item in walks_tgt]
    walks_tgt = (walks_tgt[0], walks_tgt[1], walks_tgt[2], walks_tgt[3], walks_tgt[4])

    walks_bgd = [item[batch_id] for item in walks_bgd]
    walks_bgd = (walks_bgd[0], walks_bgd[1], walks_bgd[2], walks_bgd[3], walks_bgd[4])

    dst_fake = dst_fake[batch_id]
    return subgraph_src, subgraph_tgt, subgraph_bgd, walks_src, walks_tgt, walks_bgd, dst_fake


def get_item_edge(edge_features, batch_id):
    edge_features = edge_features[:, batch_id, :, :, :]  # [3,bsz, n_walks,length, length]
    src_edge = edge_features[0]
    tgt_edge = edge_features[1]
    bgd_edge = edge_features[2]
    return src_edge, tgt_edge, bgd_edge