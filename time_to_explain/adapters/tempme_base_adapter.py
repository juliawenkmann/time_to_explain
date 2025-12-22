# time_to_explain/adapters/tempme_base_adapter.py
from __future__ import annotations
from typing import Any, Tuple
import numpy as np
import torch


class TempMEBaseAdapter(torch.nn.Module):
    """
    Compatibility wrapper that makes the canonical TGN/TGAT backbone look like
    TempME's expected base model API (attributes + helper methods).

    Exposes:
      - n_feat_th / e_feat_th (frozen embeddings)
      - node_raw_features / edge_raw_features (Embedding modules)
      - num_neighbors
      - grab_subgraph: 2-hop neighbors via ngh_finder
      - contrast: positive/negative scores using backbone.get_prob
    """

    def __init__(self, backbone: Any):
        super().__init__()
        self.backbone = backbone
        self.device = getattr(backbone, "device", torch.device("cpu"))
        self._tempme_path_logged = False

        node_feats = getattr(backbone, "node_raw_features", None)
        if node_feats is None:
            node_feats = getattr(backbone, "node_raw_embed", None)
        edge_feats = getattr(backbone, "edge_raw_features", None)
        if edge_feats is None:
            edge_feats = getattr(backbone, "edge_raw_embed", None)
        if node_feats is None or edge_feats is None:
            raise ValueError("Backbone must expose node_raw_features/edge_raw_features (or node_raw_embed/edge_raw_embed).")

        # If tensors, wrap in embeddings; if already embeddings, keep as-is.
        if isinstance(node_feats, torch.Tensor):
            self.n_feat_th = torch.nn.Parameter(node_feats.detach().clone(), requires_grad=False)
            self.node_raw_features = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
        else:
            self.node_raw_features = node_feats
            self.n_feat_th = torch.nn.Parameter(node_feats.weight.detach().clone(), requires_grad=False)  # type: ignore

        if isinstance(edge_feats, torch.Tensor):
            self.e_feat_th = torch.nn.Parameter(edge_feats.detach().clone(), requires_grad=False)
            self.edge_raw_features = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)
        else:
            self.edge_raw_features = edge_feats
            self.e_feat_th = torch.nn.Parameter(edge_feats.weight.detach().clone(), requires_grad=False)  # type: ignore

        self.num_neighbors = getattr(backbone, "num_neighbors", 20)
        self.ngh_finder = getattr(backbone, "ngh_finder", None)
        if self.ngh_finder is None:
            raise ValueError("Backbone must expose an ngh_finder for temporal neighbors.")
        if hasattr(self.backbone, "forbidden_memory_update"):
            # Avoid memory side effects when calling get_prob repeatedly during explainer training.
            self.backbone.forbidden_memory_update = True

    def grab_subgraph(self, src_idx_l, cut_time_l):
        """
        Build a 2-hop subgraph in TempME layout:
          ([hop1_nodes, hop2_nodes], [hop1_eidx, hop2_eidx], [hop1_ts, hop2_ts])
        """
        # hop-1
        hop1_nodes, hop1_eidx, hop1_ts = self.ngh_finder.get_temporal_neighbor(
            src_idx_l, cut_time_l, num_neighbors=self.num_neighbors
        )
        hop1_nodes = hop1_nodes.squeeze(1)
        hop1_eidx = hop1_eidx.squeeze(1)
        hop1_ts = hop1_ts.squeeze(1)

        # hop-2
        hop1_nodes_list = hop1_nodes.flatten()
        hop1_ts_list = hop1_ts.flatten()
        mask = hop1_nodes_list != 0
        hop1_nodes_list = hop1_nodes_list[mask]
        hop1_ts_list = hop1_ts_list[mask]
        if hop1_nodes_list.numel() == 0:
            hop2_nodes = torch.zeros_like(hop1_nodes)
            hop2_eidx = torch.zeros_like(hop1_eidx)
            hop2_ts = torch.zeros_like(hop1_ts)
        else:
            hop2_nodes, hop2_eidx, hop2_ts = self.ngh_finder.get_temporal_neighbor(
                hop1_nodes_list, hop1_ts_list, num_neighbors=self.num_neighbors
            )
            # reshape back to [batch, num_neighbors]
            hop2_nodes = hop2_nodes.view(hop1_nodes.shape[0], self.num_neighbors, self.num_neighbors)
            hop2_eidx = hop2_eidx.view(hop1_eidx.shape[0], self.num_neighbors, self.num_neighbors)
            hop2_ts = hop2_ts.view(hop1_ts.shape[0], self.num_neighbors, self.num_neighbors)

        node_records = [hop1_nodes.cpu().numpy(), hop2_nodes.cpu().numpy()]
        eidx_records = [hop1_eidx.cpu().numpy(), hop2_eidx.cpu().numpy()]
        t_records = [hop1_ts.cpu().numpy(), hop2_ts.cpu().numpy()]
        return (node_records, eidx_records, t_records)

    def contrast(self, src_idx, tgt_idx, bgd_idx, cut_time, e_idx,
                 subgraph_src=None, subgraph_tgt=None, subgraph_bgd=None,
                 explain_weights=None, edge_attr=None):
        """
        TempME-style contrast: return pos/neg logits using backbone.get_prob.
        """
        src = np.asarray(src_idx)
        tgt = np.asarray(tgt_idx)
        bgd = np.asarray(bgd_idx)
        ts = np.asarray(cut_time)

        emb_module = getattr(self.backbone, "embedding_module", None)
        if (
            subgraph_src is not None
            and subgraph_tgt is not None
            and subgraph_bgd is not None
            and hasattr(self.backbone, "contrast")
            and emb_module is not None
            and hasattr(emb_module, "embedding_update")
        ):
            if not self._tempme_path_logged:
                print("[TempME] Using embedding_update path for explain weights.")
                self._tempme_path_logged = True
            return self.backbone.contrast(
                src,
                tgt,
                bgd,
                ts,
                e_idx,
                subgraph_src,
                subgraph_tgt,
                subgraph_bgd,
                explain_weights=explain_weights,
                edge_attr=edge_attr,
            )

        if explain_weights is None or subgraph_src is None or subgraph_tgt is None or subgraph_bgd is None:
            return self._score_batch(src, tgt, bgd, ts, e_idx)

        edge_imp_0, edge_imp_1 = _coerce_explain_weights(explain_weights)
        batch = len(src)
        if edge_imp_0 is None or edge_imp_1 is None or edge_imp_0.shape[0] != 3 * batch:
            pos = self.backbone.get_prob(src, tgt, ts, logit=True)
            neg = self.backbone.get_prob(src, bgd, ts, logit=True)
            return torch.as_tensor(pos, device=self.device), torch.as_tensor(neg, device=self.device)

        pos_scores = []
        neg_scores = []
        for idx in range(batch):
            cand_pos = _build_candidate_weights(
                idx,
                batch,
                parts=((subgraph_src, 0), (subgraph_tgt, 1)),
                edge_imp_0=edge_imp_0,
                edge_imp_1=edge_imp_1,
            )
            cand_neg = _build_candidate_weights(
                idx,
                batch,
                parts=((subgraph_src, 0), (subgraph_bgd, 2)),
                edge_imp_0=edge_imp_0,
                edge_imp_1=edge_imp_1,
            )
            src_i = src[idx:idx + 1]
            tgt_i = tgt[idx:idx + 1]
            bgd_i = bgd[idx:idx + 1]
            ts_i = ts[idx:idx + 1]
            pos = self.backbone.get_prob(src_i, tgt_i, ts_i, logit=True, candidate_weights_dict=cand_pos)
            neg = self.backbone.get_prob(src_i, bgd_i, ts_i, logit=True, candidate_weights_dict=cand_neg)
            pos_scores.append(torch.as_tensor(pos, device=self.device))
            neg_scores.append(torch.as_tensor(neg, device=self.device))

        pos_scores = torch.cat([p.view(-1) for p in pos_scores], dim=0)
        neg_scores = torch.cat([n.view(-1) for n in neg_scores], dim=0)
        return pos_scores, neg_scores

    def _score_batch(self, src, tgt, bgd, ts, e_idx):
        edge_idxs = None if e_idx is None else np.asarray(e_idx)
        source_emb, dest_emb, neg_emb = self.backbone.compute_temporal_embeddings(
            src,
            tgt,
            bgd,
            ts,
            edge_idxs,
            self.num_neighbors,
        )
        score = self.backbone.affinity_score(
            torch.cat([source_emb, source_emb], dim=0),
            torch.cat([dest_emb, neg_emb]),
        ).squeeze(dim=0)
        n_samples = len(src)
        pos_score = score[:n_samples]
        neg_score = score[n_samples:]
        return pos_score, neg_score

    def set_neighbor_sampler(self, neighbor_sampler):
        if hasattr(self.backbone, "set_neighbor_sampler"):
            self.backbone.set_neighbor_sampler(neighbor_sampler)
        elif hasattr(self.backbone, "set_neighbor_finder"):
            self.backbone.set_neighbor_finder(neighbor_sampler)
        else:
            self.ngh_finder = neighbor_sampler

    def __getattr__(self, item: str):
        # Delegate anything else to the underlying backbone
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.backbone, item)


def _coerce_explain_weights(explain_weights):
    if not isinstance(explain_weights, (list, tuple)) or len(explain_weights) < 2:
        return None, None
    edge_imp_0, edge_imp_1 = explain_weights[:2]
    if isinstance(edge_imp_0, torch.Tensor):
        edge_imp_0 = edge_imp_0.detach().cpu().numpy()
    else:
        edge_imp_0 = np.asarray(edge_imp_0)
    if isinstance(edge_imp_1, torch.Tensor):
        edge_imp_1 = edge_imp_1.detach().cpu().numpy()
    else:
        edge_imp_1 = np.asarray(edge_imp_1)
    return edge_imp_0, edge_imp_1


def _build_candidate_weights(
    idx: int,
    batch: int,
    *,
    parts,
    edge_imp_0: np.ndarray,
    edge_imp_1: np.ndarray,
):
    edge_ids = []
    edge_weights = []
    for subgraph, offset_block in parts:
        offset = offset_block * batch + idx
        e0 = np.asarray(subgraph[1][0][idx]).reshape(-1)
        e1 = np.asarray(subgraph[1][1][idx]).reshape(-1)
        w0 = np.asarray(edge_imp_0[offset]).reshape(-1)
        w1 = np.asarray(edge_imp_1[offset]).reshape(-1)
        edge_ids.append(e0)
        edge_ids.append(e1)
        edge_weights.append(w0)
        edge_weights.append(w1)

    if not edge_ids:
        return None

    edge_ids = np.concatenate(edge_ids)
    edge_weights = np.concatenate(edge_weights)

    valid = edge_ids > 0
    if not np.any(valid):
        return None

    edge_ids = edge_ids[valid].astype(np.int64, copy=False)
    edge_weights = edge_weights[valid].astype(np.float32, copy=False)

    agg = {}
    for eid, weight in zip(edge_ids, edge_weights):
        prev = agg.get(int(eid))
        if prev is None or weight > prev:
            agg[int(eid)] = float(weight)

    if not agg:
        return None

    ordered = sorted(agg.keys())
    return {
        "candidate_events": np.array(ordered, dtype=np.int64),
        "edge_weights": np.array([agg[eid] for eid in ordered], dtype=np.float32),
        # TempME-style: multiply attention weights after softmax.
        "weight_mode": "mult",
    }
