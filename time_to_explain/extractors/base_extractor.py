# time_to_explain/extractors/tg_event_candidates_extractor.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
from itertools import chain

from time_to_explain.core.types import Subgraph
from time_to_explain.core.registry import register_extractor


@register_extractor("tg_event_candidates")
class BaseExtractor:
    """
    Freeze the anchor event and compute a deterministic candidate edge list.

    Args:
        model: trained TGN/TGAT with attributes .ngh_finder, .num_layers, .num_neighbors
        events: DataFrame with at least columns [u, i, ts] in that order
        threshold_num: cap the candidate list (like your SubgraphX‑TG config)
        keep_order: "last-N-then-sort" (matches your pattern) or "chronological" or "as-is"

    Returns:
        Subgraph with payload:
          - event_idx (1-based)
          - candidate_eidx (stable order for vector alignment)
          - (optional) u, i, ts for convenience
    """
    name = "tg_event_candidates"

    def __init__(self, *, model: Any, events, threshold_num: int = 500000, keep_order: str = "last-N-then-sort",
                 attach_event_meta: bool = True) -> None:
        self.model = model
        self.events = events
        self.threshold_num = int(threshold_num)
        self.keep_order = keep_order
        self.attach_event_meta = attach_event_meta

    def extract(self, dataset: Any, anchor: Dict[str, Any], *, k_hop: int,
                num_neighbors: int, window: Optional[Tuple[float, float]] = None) -> Subgraph:

        eidx = anchor.get("event_idx") or anchor.get("index") or anchor.get("idx")
        if eidx is None:
            raise ValueError("TGEventCandidatesExtractor requires anchor['event_idx'|'index'|'idx'].")

        eidx = int(eidx)  # 1-based
        ngh = self.model.ngh_finder
        L   = int(getattr(self.model, "num_layers", 1))
        K   = int(getattr(self.model, "num_neighbors", num_neighbors))  # prefer model setting

        # events.iloc[eidx-1] matches your 1-based e_idx convention
        u = int(self.events.iloc[eidx - 1, 0])
        i = int(self.events.iloc[eidx - 1, 1])
        ts = float(self.events.iloc[eidx - 1, 2])

        accu_nodes: List[List[int]] = [[u, i]]
        accu_ts:    List[List[float]] = [[ts, ts]]
        accu_eidx:  List[List[int]] = []

        for _ in range(L):
            last_nodes = accu_nodes[-1]
            last_ts    = accu_ts[-1]
            out_nodes, out_eidx, out_t = ngh.get_temporal_neighbor(
                last_nodes, last_ts, num_neighbors=K
            )
            out_nodes = out_nodes.flatten()
            out_eidx  = out_eidx.flatten()
            out_t     = out_t.flatten()
            mask = out_nodes != 0
            accu_nodes.append(out_nodes[mask].tolist())
            accu_ts.append(out_t[mask].tolist())
            accu_eidx.append(out_eidx[mask].tolist())

        unique_e = np.array(list(chain.from_iterable(accu_eidx)))
        unique_e = unique_e[unique_e != 0]          # remove paddings
        unique_e = np.unique(unique_e).tolist()

        # enforce a bounded, stable order (align with your SubgraphX‑TG)
        candidates = unique_e
        if self.threshold_num and len(candidates) > self.threshold_num:
            # common pattern in your code: take the last-N then sort to stabilize
            candidates = candidates[-self.threshold_num:]
        if self.keep_order == "last-N-then-sort":
            candidates = sorted(candidates)
        elif self.keep_order == "chronological":
            # Keep original chronological order by event id (assuming e_idx correlates with time)
            pass
        # else "as-is"

        payload: Dict[str, Any] = {
            "event_idx": eidx,
            "candidate_eidx": candidates,
        }

        def _column_values(df, rows, preferred: str, fallback_idx: int):
            if hasattr(df, "columns") and preferred in df.columns:
                return rows[preferred].to_numpy()
            return rows.iloc[:, fallback_idx].to_numpy()

        if candidates:
            idx = [max(0, int(c) - 1) for c in candidates]
            rows = self.events.iloc[idx]
            src_vals = _column_values(self.events, rows, "u", 0).astype(np.int64)
            dst_vals = _column_values(self.events, rows, "i", 1).astype(np.int64)
            time_vals = _column_values(self.events, rows, "ts", 2).astype(float)
            edge_pairs = np.stack([src_vals, dst_vals], axis=1).tolist()
            payload.update({
                "candidate_edge_times": time_vals.tolist(),
                "candidate_edge_index": edge_pairs,
                "edge_index": edge_pairs,
            })

        if self.attach_event_meta:
            payload.update({"u": u, "i": i, "ts": ts})

        return Subgraph(node_ids=[], edge_index=[], payload=payload)
