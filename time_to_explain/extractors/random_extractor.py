from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Literal

import numpy as np

from time_to_explain.core.registry import register_extractor
from time_to_explain.core.types import Subgraph

from .common import EventIndex, anchor_event_idx


@register_extractor("random_event_candidates")
class RandomExtractor:
    """Random candidate extractor for temporal event-level explainers.

    Modes:
      - pool="history": all past events (rows < anchor)
      - pool="cody_khop": CoDy k-hop pool (events whose endpoints fall in k-hop neighborhood)

      - random_mode="shuffle_tail": keep tail(mmax) then shuffle (paper-like baseline)
      - random_mode="sample": uniform sample without replacement
    """

    name = "random_event_candidates"

    def __init__(
        self,
        *,
        model: Any,
        events,
        candidates_size: int = 64,
        pool: Literal["history", "cody_khop"] = "cody_khop",
        random_mode: Literal["shuffle_tail", "sample"] = "sample",
        num_hops: Optional[int] = None,
        directed: bool = False,
        seed: Optional[int] = 0,
        per_anchor_seed: bool = True,
        keep_sorted_by_time: bool = False,
        attach_event_meta: bool = True,
        attach_candidate_meta: bool = True,
        drop_nonpositive_eids: bool = True,
        max_node_mask_size: int = 5_000_000,
    ) -> None:
        self.model = model
        self.candidates_size = int(candidates_size)
        self.pool = pool
        self.random_mode = random_mode
        self.num_hops = int(num_hops) if num_hops is not None else None
        self.directed = bool(directed)
        self.seed = seed
        self.per_anchor_seed = bool(per_anchor_seed)
        self.keep_sorted_by_time = bool(keep_sorted_by_time)
        self.attach_event_meta = bool(attach_event_meta)
        self.attach_candidate_meta = bool(attach_candidate_meta)
        self.drop_nonpositive_eids = bool(drop_nonpositive_eids)

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
        # This extractor ignores dataset, num_neighbors, window by design.
        del dataset, num_neighbors, window

        eidx = anchor_event_idx(anchor)
        base_row = self._idx.resolve_row(eidx)

        k = self.num_hops
        if k is None:
            k = int(getattr(self.model, "num_layers", k_hop))

        if self.pool == "history":
            pool_rows = np.arange(0, base_row, dtype=np.int64)
        elif self.pool == "cody_khop":
            pool_rows = self._idx.khop_edge_rows(base_row=base_row, k=int(k), directed=self.directed, exclude_base=True)
        else:
            raise ValueError(f"Unknown pool='{self.pool}'")

        if pool_rows.size == 0:
            payload: Dict[str, Any] = {"event_idx": int(eidx), "candidate_eidx": []}
            if self.attach_event_meta:
                payload.update(self._idx.event_meta(base_row))
            return Subgraph(node_ids=[], edge_index=[], payload=payload)

        rng = self._rng_for_anchor(int(eidx))
        cand_rows = self._pick_rows(pool_rows, rng=rng)

        cand_eidx = self._idx.rows_to_model_eids(cand_rows)
        if self.drop_nonpositive_eids and cand_eidx.size:
            mask = cand_eidx > 0
            cand_rows = cand_rows[mask]
            cand_eidx = cand_eidx[mask]

        payload = {
            "event_idx": int(eidx),
            "candidate_eidx": cand_eidx.astype(np.int64).tolist(),
            "candidate_row_idx": cand_rows.astype(np.int64).tolist(),
            "pool": self.pool,
            "random_mode": self.random_mode,
            "k_hop": int(k),
            "candidates_size": int(self.candidates_size),
            "seed": self.seed,
            "per_anchor_seed": self.per_anchor_seed,
        }

        if self.attach_candidate_meta and cand_rows.size:
            edge_pairs = self._idx.edge_pairs(cand_rows)
            payload.update(
                {
                    "candidate_edge_times": self._idx.edge_times(cand_rows),
                    "candidate_edge_index": edge_pairs,
                    "edge_index": edge_pairs,
                }
            )

        if self.attach_event_meta:
            payload.update(self._idx.event_meta(base_row))

        return Subgraph(node_ids=[], edge_index=payload.get("edge_index", []), payload=payload)

    # ---------------- helpers ----------------

    def _rng_for_anchor(self, anchor_eidx: int) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        if self.per_anchor_seed:
            return np.random.default_rng(int(self.seed) + int(anchor_eidx))
        return np.random.default_rng(int(self.seed))

    def _pick_rows(self, pool_rows: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
        pool_rows = pool_rows.astype(np.int64, copy=False)
        if pool_rows.size == 0:
            return pool_rows

        if self.random_mode == "shuffle_tail":
            rows = pool_rows
            if self.candidates_size > 0 and rows.size > self.candidates_size:
                rows = rows[-self.candidates_size :]
            rows = rows.copy()
            rng.shuffle(rows)
            return rows.astype(np.int64)

        if self.random_mode == "sample":
            n = pool_rows.size if self.candidates_size <= 0 else min(self.candidates_size, pool_rows.size)
            rows = rng.choice(pool_rows, size=int(n), replace=False).astype(np.int64)
            if self.keep_sorted_by_time:
                rows = np.sort(rows)
            return rows

        raise ValueError(f"Unknown random_mode='{self.random_mode}'")
