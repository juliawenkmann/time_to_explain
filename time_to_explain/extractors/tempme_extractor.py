from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from time_to_explain.core.registry import register_extractor
from time_to_explain.core.types import Subgraph

from .common import EventIndex, anchor_event_idx


@register_extractor("tempme")
@register_extractor("tempme_minimal")
class TempMEExtractor:
    """Build a minimal TempME-compatible payload on the fly.

    This avoids reliance on TempME's precomputed HDF5 files by generating the
    required walk/subgraph tensors from `events` and `model.ngh_finder`.

    Payload keys (both with and without the `tempme_` prefix):
      - subgraph_src/subgraph_tgt/subgraph_bgd
      - walks_src/walks_tgt/walks_bgd
      - edge_src/edge_tgt/edge_bgd
      - event_idx, u, i, ts, bgd
    """

    name = "tempme_minimal"

    def __init__(self, *, model: Any, events, num_neighbors: int = 20):
        self.model = model
        self.num_neighbors = int(num_neighbors)
        self._idx = EventIndex.from_events(events)

    def extract(
        self,
        dataset: Any,
        anchor: Dict[str, Any],
        *,
        k_hop: int,
        num_neighbors: int,
        window: Optional[Tuple[float, float]] = None,
    ) -> Subgraph:
        del dataset, k_hop, window

        eidx = anchor_event_idx(anchor)
        row = self._idx.resolve_row(eidx)
        meta = self._idx.event_meta(row)
        u, v, ts = meta["u"], meta["i"], meta["ts"]

        n_nodes = int(max(int(self._idx.src.max()), int(self._idx.dst.max()))) + 1
        bgd = v if n_nodes <= 1 else (v + 1) % n_nodes

        sub_src = self._grab_subgraph([u], [ts], num_neighbors=num_neighbors)
        sub_tgt = self._grab_subgraph([v], [ts], num_neighbors=num_neighbors)
        sub_bgd = self._grab_subgraph([bgd], [ts], num_neighbors=num_neighbors)

        walks_src, edge_src = self._make_walk(u, v)
        walks_tgt, edge_tgt = self._make_walk(v, v)
        walks_bgd, edge_bgd = self._make_walk(bgd, v)

        payload: Dict[str, Any] = {
            "event_idx": int(eidx),
            "u": u,
            "i": v,
            "ts": ts,
            "bgd": int(bgd),
        }

        for prefix in ("", "tempme_"):
            payload[f"{prefix}subgraph_src"] = sub_src
            payload[f"{prefix}subgraph_tgt"] = sub_tgt
            payload[f"{prefix}subgraph_bgd"] = sub_bgd
            payload[f"{prefix}walks_src"] = walks_src
            payload[f"{prefix}walks_tgt"] = walks_tgt
            payload[f"{prefix}walks_bgd"] = walks_bgd
            payload[f"{prefix}edge_src"] = edge_src
            payload[f"{prefix}edge_tgt"] = edge_tgt
            payload[f"{prefix}edge_bgd"] = edge_bgd

        return Subgraph(node_ids=[], edge_index=[], payload=payload)

    def _grab_subgraph(self, src_idx_l, cut_time_l, *, num_neighbors: int):
        """Two-hop temporal neighbor expansion returned in TempME layout.

        Returns: (node_records, eidx_records, t_records) where each record is a
        list [hop1, hop2] and hop2 is flattened to shape (batch, K*K).
        """
        ngh = getattr(self.model, "ngh_finder", None)
        if ngh is None:
            raise ValueError("Model must expose .ngh_finder with get_temporal_neighbor().")

        K = int(getattr(self.model, "num_neighbors", num_neighbors))

        hop1_nodes, hop1_eidx, hop1_ts = ngh.get_temporal_neighbor(src_idx_l, cut_time_l, num_neighbors=K)
        hop1_nodes = np.asarray(hop1_nodes)
        hop1_eidx = np.asarray(hop1_eidx)
        hop1_ts = np.asarray(hop1_ts)

        # Some neighbor finders return (batch, 1, K)
        if hop1_nodes.ndim == 3 and hop1_nodes.shape[1] == 1:
            hop1_nodes = hop1_nodes[:, 0, :]
            hop1_eidx = hop1_eidx[:, 0, :]
            hop1_ts = hop1_ts[:, 0, :]

        batch = int(hop1_nodes.shape[0]) if hop1_nodes.ndim >= 2 else 1

        # hop2: query neighbors for each hop1 node and flatten to (batch, K*K)
        hop1_nodes_list = hop1_nodes.reshape(-1)
        hop1_ts_list = hop1_ts.reshape(-1)

        def _reshape_hop2(arr: np.ndarray) -> np.ndarray:
            if arr.ndim == 3 and arr.shape == (batch, K, K):
                return arr.reshape(batch, K * K)
            if arr.ndim == 2:
                if arr.shape == (batch * K, K):
                    return arr.reshape(batch, K * K)
                if arr.shape == (batch, K * K):
                    return arr
            return np.zeros((batch, K * K), dtype=arr.dtype)

        if hop1_nodes_list.size == 0:
            hop2_nodes = np.zeros((batch, K * K), dtype=hop1_nodes.dtype)
            hop2_eidx = np.zeros((batch, K * K), dtype=hop1_eidx.dtype)
            hop2_ts = np.zeros((batch, K * K), dtype=hop1_ts.dtype)
        else:
            hop2_nodes, hop2_eidx, hop2_ts = ngh.get_temporal_neighbor(hop1_nodes_list, hop1_ts_list, num_neighbors=K)
            hop2_nodes = _reshape_hop2(np.asarray(hop2_nodes))
            hop2_eidx = _reshape_hop2(np.asarray(hop2_eidx))
            hop2_ts = _reshape_hop2(np.asarray(hop2_ts))

        return ([hop1_nodes, hop2_nodes], [hop1_eidx, hop2_eidx], [hop1_ts, hop2_ts])

    def _make_walk(self, src_id: int, dst_id: int):
        """Minimal walk tensor matching TempME's expected shapes."""
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
