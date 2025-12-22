# time_to_explain/extractors/tempme_extractor.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

from time_to_explain.core.types import Subgraph
from time_to_explain.core.registry import register_extractor


@register_extractor("tempme_minimal")
class TempMEExtractor:
    """
    Build a minimal TempME-compatible payload on the fly from events + ngh_finder.
    This avoids reliance on TempME's precomputed HDF5 files.

    Output Subgraph.payload keys:
      - event_idx (1-based)
      - subgraph_src/subgraph_tgt/subgraph_bgd: (node_records, eidx_records, t_records)
      - walks_src/walks_tgt/walks_bgd: (node_idx, edge_idx, time_idx, cat_feat, extra)
      - edge_src/edge_tgt/edge_bgd: trivial edge id arrays
      - u/i/ts of the anchor
    """
    name = "tempme_minimal"

    def __init__(self, *, model: Any, events, num_neighbors: int = 20):
        self.model = model
        self.events = events
        self.num_neighbors = num_neighbors

    def extract(self, dataset: Any, anchor: Dict[str, Any], *, k_hop: int,
                num_neighbors: int, window: Optional[Tuple[float, float]] = None) -> Subgraph:
        eidx = anchor.get("event_idx") or anchor.get("index") or anchor.get("idx")
        if eidx is None:
            raise ValueError("TempMEExtractor requires anchor['event_idx'|'index'|'idx'].")
        eidx = int(eidx)

        row = self.events.iloc[eidx - 1]
        u = int(row[0]); v = int(row[1]); ts = float(row[2])

        n_nodes = int(max(self.events.iloc[:, 0].max(), self.events.iloc[:, 1].max())) + 1
        bgd = v if n_nodes <= 1 else (v + 1) % n_nodes

        sub_src = self._grab_subgraph([u], [ts])
        sub_tgt = self._grab_subgraph([v], [ts])
        sub_bgd = self._grab_subgraph([bgd], [ts])

        walks_src, edge_src = self._make_walk(u, v)
        walks_tgt, edge_tgt = self._make_walk(v, v)
        walks_bgd, edge_bgd = self._make_walk(bgd, v)

        payload = {
            "event_idx": eidx,
            "u": u,
            "i": v,
            "ts": ts,
            "bgd": bgd,
            "subgraph_src": sub_src,
            "subgraph_tgt": sub_tgt,
            "subgraph_bgd": sub_bgd,
            "walks_src": walks_src,
            "walks_tgt": walks_tgt,
            "walks_bgd": walks_bgd,
            "edge_src": edge_src,
            "edge_tgt": edge_tgt,
            "edge_bgd": edge_bgd,
            "tempme_subgraph_src": sub_src,
            "tempme_subgraph_tgt": sub_tgt,
            "tempme_subgraph_bgd": sub_bgd,
            "tempme_walks_src": walks_src,
            "tempme_walks_tgt": walks_tgt,
            "tempme_walks_bgd": walks_bgd,
            "tempme_edge_src": edge_src,
            "tempme_edge_tgt": edge_tgt,
            "tempme_edge_bgd": edge_bgd,
        }
        return Subgraph(node_ids=[], edge_index=[], payload=payload)

    def _grab_subgraph(self, src_idx_l, cut_time_l):
        """
        2-hop neighbor expansion using model.ngh_finder, returned in TempME layout:
        ([hop1_nodes, hop2_nodes], [hop1_eidx, hop2_eidx], [hop1_ts, hop2_ts])
        """
        ngh = self.model.ngh_finder
        K = int(getattr(self.model, "num_neighbors", self.num_neighbors))

        hop1_nodes, hop1_eidx, hop1_ts = ngh.get_temporal_neighbor(src_idx_l, cut_time_l, num_neighbors=K)
        hop1_nodes = np.asarray(hop1_nodes)
        hop1_eidx = np.asarray(hop1_eidx)
        hop1_ts = np.asarray(hop1_ts)
        if hop1_nodes.ndim > 2 and hop1_nodes.shape[1] == 1:
            hop1_nodes = np.squeeze(hop1_nodes, axis=1)
            hop1_eidx = np.squeeze(hop1_eidx, axis=1)
            hop1_ts = np.squeeze(hop1_ts, axis=1)

        hop1_nodes_list = hop1_nodes.flatten()
        hop1_ts_list = hop1_ts.flatten()
        mask = hop1_nodes_list != 0
        hop1_nodes_list = hop1_nodes_list[mask]
        hop1_ts_list = hop1_ts_list[mask]

        if hop1_nodes_list.size == 0:
            hop2_nodes = np.zeros_like(hop1_nodes)
            hop2_eidx = np.zeros_like(hop1_eidx)
            hop2_ts = np.zeros_like(hop1_ts)
        else:
            hop2_nodes, hop2_eidx, hop2_ts = ngh.get_temporal_neighbor(
                hop1_nodes_list, hop1_ts_list, num_neighbors=K
            )
            hop2_nodes = np.asarray(hop2_nodes)
            hop2_eidx = np.asarray(hop2_eidx)
            hop2_ts = np.asarray(hop2_ts)
            try:
                hop2_nodes = hop2_nodes.reshape(hop1_nodes.shape[0], K, K)
                hop2_eidx = hop2_eidx.reshape(hop1_eidx.shape[0], K, K)
                hop2_ts = hop2_ts.reshape(hop1_ts.shape[0], K, K)
            except Exception:
                hop2_nodes = np.zeros_like(hop1_nodes)
                hop2_eidx = np.zeros_like(hop1_eidx)
                hop2_ts = np.zeros_like(hop1_ts)

        node_records = [hop1_nodes, hop2_nodes]
        eidx_records = [hop1_eidx, hop2_eidx]
        t_records = [hop1_ts, hop2_ts]
        return (node_records, eidx_records, t_records)

    def _make_walk(self, src_id: int, dst_id: int):
        """
        Minimal walk tensor: shapes [1,1,6] for node_idx, [1,1,3] for edge_idx/time_idx, [1,1,1] cat_feat, [1,1,3] extras.
        """
        node_idx = np.array([[[src_id, dst_id, 0, 0, 0, 0]]], dtype=np.int64)
        edge_idx = np.zeros((1, 1, 3), dtype=np.int64)
        time_idx = np.zeros((1, 1, 3), dtype=np.float32)
        cat_feat = np.zeros((1, 1, 1), dtype=np.int64)
        out_anony = np.zeros((1, 1, 3), dtype=np.int64)
        walks = (node_idx, edge_idx, time_idx, cat_feat, out_anony)
        edge_identity = self._edge_identity(node_idx).astype(np.float32, copy=False)
        return walks, edge_identity

    @staticmethod
    def _edge_identity(node_records: np.ndarray) -> np.ndarray:
        edges = np.stack(
            [
                node_records[:, :, 0:2],
                node_records[:, :, 2:4],
                node_records[:, :, 4:6],
            ],
            axis=2,
        )
        edges = np.sort(edges, axis=-1)
        eq = (edges[:, :, :, None, :] == edges[:, :, None, :, :]).all(-1).astype(np.float32)
        mask = (edges[..., 0] == 0) & (edges[..., 1] == 0)
        if mask.any():
            eq[mask] = 0.0
        return eq
