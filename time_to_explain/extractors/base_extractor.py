from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from time_to_explain.core.registry import register_extractor
from time_to_explain.core.types import Subgraph

from .common import EventIndex, anchor_event_idx


@register_extractor("tg_event_candidates")
class TGEventCandidatesExtractor:
    """Deterministic candidate edge list using model.ngh_finder.

    For L layers, queries `ngh_finder.get_temporal_neighbor()` starting from the
    anchor endpoints and collects unique edge ids.

    Payload keys:
      - event_idx
      - candidate_eidx
      - (optional) candidate_edge_times, candidate_edge_index, edge_index
      - (optional) u, i, ts
    """

    name = "tg_event_candidates"

    def __init__(
        self,
        *,
        model: Any,
        events,
        threshold_num: int = 500_000,
        keep_order: str = "last-N-then-sort",
        attach_event_meta: bool = True,
        attach_candidate_meta: bool = True,
        max_node_mask_size: int = 5_000_000,
    ) -> None:
        self.model = model
        self.threshold_num = int(threshold_num)
        self.keep_order = str(keep_order)
        self.attach_event_meta = bool(attach_event_meta)
        self.attach_candidate_meta = bool(attach_candidate_meta)

        self._idx = EventIndex.from_events(events, max_node_mask_size=max_node_mask_size)

    def extract(
        self,
        dataset: Any,
        anchor: Dict[str, Any],
        *,
        k_hop: int,
        num_neighbors: int,
        window: Optional[Tuple[float, float]] = None,
    ) -> Subgraph:
        # window is currently unused; preserved for extractor interface compatibility
        del dataset, k_hop, window

        eidx = anchor_event_idx(anchor)
        base_row = self._idx.resolve_row(eidx)
        meta = self._idx.event_meta(base_row)
        u, v, ts = meta["u"], meta["i"], meta["ts"]

        ngh = getattr(self.model, "ngh_finder", None)
        if ngh is None:
            raise ValueError("Model must expose .ngh_finder with get_temporal_neighbor().")

        L = int(getattr(self.model, "num_layers", 1))
        K = int(getattr(self.model, "num_neighbors", num_neighbors))

        candidates = self._collect_candidate_eids(ngh, u=u, v=v, ts=ts, layers=L, num_neighbors=K)

        payload: Dict[str, Any] = {"event_idx": int(eidx), "candidate_eidx": candidates}

        if self.attach_candidate_meta and candidates:
            rows = self._idx.rows_for_eids(candidates)
            edge_pairs = self._idx.edge_pairs(rows)
            payload.update(
                {
                    "candidate_edge_times": self._idx.edge_times(rows),
                    "candidate_edge_index": edge_pairs,
                    "edge_index": edge_pairs,
                }
            )

        if self.attach_event_meta:
            payload.update(meta)

        return Subgraph(node_ids=[], edge_index=payload.get("edge_index", []), payload=payload)

    def _collect_candidate_eids(self, ngh: Any, *, u: int, v: int, ts: float, layers: int, num_neighbors: int) -> List[int]:
        nodes = np.asarray([u, v], dtype=np.int64)
        times = np.asarray([ts, ts], dtype=np.float64)

        collected: List[np.ndarray] = []
        for _ in range(int(layers)):
            out_nodes, out_eids, out_ts = ngh.get_temporal_neighbor(nodes.tolist(), times.tolist(), num_neighbors=int(num_neighbors))
            out_nodes = np.asarray(out_nodes).reshape(-1)
            out_eids = np.asarray(out_eids).reshape(-1)
            out_ts = np.asarray(out_ts).reshape(-1)

            mask = out_nodes != 0
            if not np.any(mask):
                break

            nodes = out_nodes[mask]
            times = out_ts[mask]
            collected.append(out_eids[mask])

        if not collected:
            return []

        unique = np.unique(np.concatenate(collected))
        unique = unique[unique > 0]  # remove padding
        candidates = unique.astype(np.int64).tolist()

        if self.threshold_num > 0 and len(candidates) > self.threshold_num:
            candidates = candidates[-self.threshold_num :]

        if self.keep_order == "last-N-then-sort":
            candidates = sorted(candidates)
        elif self.keep_order in {"chronological", "as-is"}:
            # np.unique is sorted already; keep for compatibility
            pass
        else:
            raise ValueError(f"Unknown keep_order='{self.keep_order}'")

        return candidates


# Backwards-compat alias (older code may import BaseExtractor from this module)
BaseExtractor = TGEventCandidatesExtractor
