from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ...core.registry import register_extractor
from ...core.types import Subgraph


def anchor_event_idx(anchor: Dict[str, Any]) -> int:
    """Extract an event id from a generic anchor dict."""
    eidx = anchor.get("event_idx") or anchor.get("index") or anchor.get("idx")
    if eidx is None:
        raise ValueError("Anchor must contain one of: 'event_idx', 'index', 'idx'.")
    return int(eidx)


def col_or_fallback(df: Any, name: str, fallback_idx: int) -> np.ndarray:
    """Return df[name] if present, else df.iloc[:, fallback_idx]."""
    if hasattr(df, "columns") and name in getattr(df, "columns"):
        return df[name].to_numpy()
    return df.iloc[:, fallback_idx].to_numpy()


@dataclass(slots=True)
class EventIndex:
    """Lightweight index over an events table."""

    events: Any
    src: np.ndarray
    dst: np.ndarray
    ts: np.ndarray
    model_eids: np.ndarray
    row_by_eid: Dict[int, int]
    src0: np.ndarray
    dst0: np.ndarray
    node_shift: int
    node_mask: Optional[np.ndarray]

    @classmethod
    def from_events(
        cls,
        events: Any,
        *,
        max_node_mask_size: int = 5_000_000,
    ) -> "EventIndex":
        src = col_or_fallback(events, "u", 0).astype(np.int64)
        dst = col_or_fallback(events, "i", 1).astype(np.int64)
        ts = col_or_fallback(events, "ts", 2).astype(np.float64)

        if src.size == 0:
            raise ValueError("Events table is empty.")

        if hasattr(events, "columns") and "e_idx" in events.columns:
            model_eids = events["e_idx"].to_numpy(dtype=np.int64)
        elif hasattr(events, "columns") and "idx" in events.columns:
            model_eids = events["idx"].to_numpy(dtype=np.int64)
        else:
            model_eids = np.arange(1, len(events) + 1, dtype=np.int64)

        row_by_eid = {int(e): int(r) for r, e in enumerate(model_eids.tolist())}

        node_min = int(min(int(src.min()), int(dst.min())))
        node_shift = node_min
        src0 = src - node_shift
        dst0 = dst - node_shift
        max_node0 = int(max(int(src0.max()), int(dst0.max())))

        node_mask: Optional[np.ndarray]
        if (max_node0 + 1) <= int(max_node_mask_size):
            node_mask = np.zeros((max_node0 + 1,), dtype=bool)
        else:
            node_mask = None

        return cls(
            events=events,
            src=src,
            dst=dst,
            ts=ts,
            model_eids=model_eids,
            row_by_eid=row_by_eid,
            src0=src0,
            dst0=dst0,
            node_shift=node_shift,
            node_mask=node_mask,
        )

    def __len__(self) -> int:  # pragma: no cover
        return int(self.model_eids.shape[0])

    def resolve_row(self, event_idx: int) -> int:
        if event_idx in self.row_by_eid:
            return int(self.row_by_eid[event_idx])

        n = len(self)
        if 0 <= event_idx < n:
            return int(event_idx)
        if 1 <= event_idx <= n:
            return int(event_idx - 1)

        raise ValueError(f"event_idx={event_idx} not found (N={n}).")

    def rows_for_eids(self, eids: Sequence[int]) -> np.ndarray:
        n = len(self)
        out = np.empty((len(eids),), dtype=np.int64)
        for j, eid in enumerate(eids):
            eid_i = int(eid)
            r = self.row_by_eid.get(eid_i)
            if r is None:
                r = eid_i - 1
            if not (0 <= int(r) < n):
                raise ValueError(f"Cannot map candidate eid={eid_i} to a valid row (N={n}).")
            out[j] = int(r)
        return out

    def event_meta(self, row: int) -> Dict[str, Any]:
        return {"u": int(self.src[row]), "i": int(self.dst[row]), "ts": float(self.ts[row])}

    def edge_pairs(self, rows: np.ndarray) -> list[tuple[int, int]]:
        if rows.size == 0:
            return []
        return list(zip(self.src[rows].astype(np.int64).tolist(), self.dst[rows].astype(np.int64).tolist()))

    def edge_times(self, rows: np.ndarray) -> list[float]:
        if rows.size == 0:
            return []
        return self.ts[rows].astype(np.float64).tolist()

    def khop_edge_rows(
        self,
        *,
        base_row: int,
        k: int,
        directed: bool = False,
        exclude_base: bool = True,
    ) -> np.ndarray:
        end = int(base_row) + 1
        src0 = self.src0[:end]
        dst0 = self.dst0[:end]

        center = np.array([self.src0[base_row], self.dst0[base_row]], dtype=np.int64)
        reached: list[np.ndarray] = [np.unique(center)]

        for _ in range(int(k)):
            frontier = reached[-1]
            if frontier.size == 0:
                reached.append(frontier)
                continue

            if self.node_mask is None:
                m_src = np.isin(src0, frontier)
                new_nodes = dst0[m_src]
                if not directed:
                    m_dst = np.isin(dst0, frontier)
                    new_nodes = np.concatenate([new_nodes, src0[m_dst]])
                reached.append(np.unique(new_nodes))
            else:
                mask = self.node_mask
                mask.fill(False)
                mask[frontier] = True
                new_nodes = dst0[mask[src0]]
                if not directed:
                    new_nodes = np.concatenate([new_nodes, src0[mask[dst0]]])
                reached.append(np.unique(new_nodes))

        neighboring_nodes = np.unique(np.concatenate(reached))

        if self.node_mask is None:
            edge_mask = np.isin(src0, neighboring_nodes) & np.isin(dst0, neighboring_nodes)
        else:
            mask = self.node_mask
            mask.fill(False)
            mask[neighboring_nodes] = True
            edge_mask = mask[src0] & mask[dst0]

        rows = np.nonzero(edge_mask)[0].astype(np.int64)
        if exclude_base and rows.size > 0:
            rows = rows[rows != int(base_row)]
        return rows

    def rows_to_model_eids(self, rows: np.ndarray) -> np.ndarray:
        if rows.size == 0:
            return np.array([], dtype=np.int64)
        return self.model_eids[rows].astype(np.int64)

    @staticmethod
    def tail(rows: np.ndarray, mmax: int) -> np.ndarray:
        if int(mmax) > 0 and rows.size > int(mmax):
            return rows[-int(mmax) :]
        return rows


@register_extractor("khop_candidates")
class KHopCandidatesExtractor:
    """CoDy/Greedy-style k-hop candidate set."""

    name = "khop_candidates"

    def __init__(
        self,
        *,
        model: Any,
        events,
        candidates_size: int = 75,
        num_hops: Optional[int] = None,
        directed: bool = False,
        attach_event_meta: bool = True,
        attach_candidate_meta: bool = True,
        max_node_mask_size: int = 5_000_000,
    ) -> None:
        self.model = model
        self.candidates_size = int(candidates_size)
        self.num_hops = int(num_hops) if num_hops is not None else None
        self.directed = bool(directed)
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
        del dataset, num_neighbors, window

        eidx = anchor_event_idx(anchor)

        k = self.num_hops
        if k is None:
            k = int(k_hop) if k_hop is not None else int(getattr(self.model, "num_layers", 1))
        if int(k) <= 0:
            k = int(getattr(self.model, "num_layers", 1))

        base_row = self._idx.resolve_row(eidx)

        support_rows = self._idx.khop_edge_rows(base_row=base_row, k=int(k), directed=self.directed, exclude_base=False)
        tail_rows = EventIndex.tail(support_rows, self.candidates_size)
        rows = tail_rows
        if rows.size:
            rows = rows[rows != int(base_row)]

        support_rows_no_anchor = support_rows
        if support_rows_no_anchor.size:
            support_rows_no_anchor = support_rows_no_anchor[support_rows_no_anchor != int(base_row)]

        base_rows = support_rows_no_anchor
        if support_rows_no_anchor.size and rows.size:
            base_rows = support_rows_no_anchor[~np.isin(support_rows_no_anchor, rows)]

        cand_eidx = self._idx.rows_to_model_eids(rows)
        support_eidx = self._idx.rows_to_model_eids(support_rows_no_anchor)
        base_eidx = self._idx.rows_to_model_eids(base_rows)

        payload: Dict[str, Any] = {
            "event_idx": int(eidx),
            "candidate_eidx": cand_eidx.tolist(),
            "candidate_row_idx": rows.astype(np.int64).tolist(),
            "support_eidx": support_eidx.astype(np.int64).tolist(),
            "base_eidx": base_eidx.astype(np.int64).tolist(),
            "k_hop": int(k),
            "mmax": int(self.candidates_size),
        }

        if self.attach_candidate_meta and rows.size:
            edge_pairs = self._idx.edge_pairs(rows)
            payload.update(
                {
                    "candidate_edge_times": self._idx.edge_times(rows),
                    "candidate_edge_index": edge_pairs,
                    "edge_index": edge_pairs,
                }
            )

        if self.attach_event_meta:
            payload.update(self._idx.event_meta(base_row))

        return Subgraph(node_ids=[], edge_index=payload.get("edge_index", []), payload=payload)


__all__ = ["EventIndex", "KHopCandidatesExtractor", "anchor_event_idx", "col_or_fallback"]
