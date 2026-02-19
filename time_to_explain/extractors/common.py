from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


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
    """Lightweight index over an events table.

    Assumptions:
      - events is DataFrame-like and supports .iloc
      - columns are either named (u/i/ts/e_idx) or positional (0/1/2)

    Provides:
      - consistent mapping from model event id -> row index
      - fast k-hop neighborhood filtering for CoDy-style candidate pools
      - utilities to attach (u,v,ts) metadata for a set of rows
    """

    events: Any
    src: np.ndarray
    dst: np.ndarray
    ts: np.ndarray
    model_eids: np.ndarray
    row_by_eid: Dict[int, int]

    # normalized node ids for fast k-hop operations
    src0: np.ndarray
    dst0: np.ndarray
    node_shift: int

    # reusable boolean mask for membership tests (None => fallback to np.isin)
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

        # Model edge ids (what most TG backbones expect in preserve-lists)
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

    # --------- id/row utilities ---------

    def resolve_row(self, event_idx: int) -> int:
        """Resolve an input event id to a 0-based row index."""
        if event_idx in self.row_by_eid:
            return int(self.row_by_eid[event_idx])

        n = len(self)
        # common fallbacks: treat input as 0-based row, or 1-based row
        if 0 <= event_idx < n:
            return int(event_idx)
        if 1 <= event_idx <= n:
            return int(event_idx - 1)

        raise ValueError(f"event_idx={event_idx} not found (N={n}).")

    def rows_for_eids(self, eids: Sequence[int]) -> np.ndarray:
        """Map event ids to rows, with a safe fallback to (eid-1) indexing."""
        n = len(self)
        out = np.empty((len(eids),), dtype=np.int64)
        for j, eid in enumerate(eids):
            eid_i = int(eid)
            r = self.row_by_eid.get(eid_i)
            if r is None:
                r = eid_i - 1  # common TGN convention
            if not (0 <= int(r) < n):
                raise ValueError(f"Cannot map candidate eid={eid_i} to a valid row (N={n}).")
            out[j] = int(r)
        return out

    # --------- metadata helpers ---------

    def event_meta(self, row: int) -> Dict[str, Any]:
        return {"u": int(self.src[row]), "i": int(self.dst[row]), "ts": float(self.ts[row])}

    def edge_pairs(self, rows: np.ndarray) -> List[Tuple[int, int]]:
        if rows.size == 0:
            return []
        return list(zip(self.src[rows].astype(np.int64).tolist(), self.dst[rows].astype(np.int64).tolist()))

    def edge_times(self, rows: np.ndarray) -> List[float]:
        if rows.size == 0:
            return []
        return self.ts[rows].astype(np.float64).tolist()

    # --------- k-hop neighborhood helpers ---------

    def khop_edge_rows(
        self,
        *,
        base_row: int,
        k: int,
        directed: bool = False,
        exclude_base: bool = True,
    ) -> np.ndarray:
        """Rows <= base_row whose endpoints are in the k-hop neighborhood of the anchor endpoints."""
        end = int(base_row) + 1
        src0 = self.src0[:end]
        dst0 = self.dst0[:end]

        center = np.array([self.src0[base_row], self.dst0[base_row]], dtype=np.int64)
        reached: List[np.ndarray] = [np.unique(center)]

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
        """Keep only the most recent mmax rows (rows are assumed chronological)."""
        if int(mmax) > 0 and rows.size > int(mmax):
            return rows[-int(mmax) :]
        return rows
