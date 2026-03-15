from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ...core.types import BaseExplainer, ExplanationContext, ExplanationResult
from ..extractors.common import EventIndex


@dataclass
class KHopCloserAdapterConfig:
    alias: str = "khop_closer"
    directed: bool = False
    max_hops: Optional[int] = None
    fallback_hop_penalty: int = 1
    score_mode: str = "reciprocal"  # reciprocal | linear
    max_node_mask_size: int = 5_000_000


class KHopCloserAdapter(BaseExplainer):
    """
    Deterministic baseline: importance is inverse k-hop distance to the anchor.

    For each candidate edge, we compute the smallest hop depth `h` such that the
    candidate edge row appears in `EventIndex.khop_edge_rows(base_row, k=h)`.
    Smaller `h` => higher score.
    """

    def __init__(self, cfg: Optional[KHopCloserAdapterConfig] = None) -> None:
        self.cfg = cfg or KHopCloserAdapterConfig()
        super().__init__(name="khop_closer_baseline", alias=self.cfg.alias)
        self._events: Optional[pd.DataFrame] = None
        self._idx: Optional[EventIndex] = None

    # ------------------------------------------------------------------ setup
    def prepare(self, *, model: Any, dataset: Any) -> None:
        super().prepare(model=model, dataset=dataset)
        events = dataset.get("events") if isinstance(dataset, dict) else dataset
        if events is None or not isinstance(events, pd.DataFrame):
            raise ValueError("KHopCloserAdapter expects dataset['events'] to be a pandas DataFrame.")
        self._events = events.reset_index(drop=True)
        self._idx = EventIndex.from_events(self._events, max_node_mask_size=int(self.cfg.max_node_mask_size))

    # ------------------------------------------------------------------ helpers
    def _event_idx(self, context: ExplanationContext) -> int:
        eidx = (
            context.target.get("event_idx")
            or context.target.get("index")
            or context.target.get("idx")
        )
        if eidx is None and context.subgraph and context.subgraph.payload:
            eidx = context.subgraph.payload.get("event_idx")
        if eidx is None:
            raise ValueError("KHopCloserAdapter expects event index in target or subgraph payload.")
        return int(eidx)

    def _candidate_eidx(self, context: ExplanationContext) -> list[int]:
        payload = getattr(getattr(context, "subgraph", None), "payload", None) or {}
        raw = payload.get("candidate_eidx") or []
        if isinstance(raw, np.ndarray):
            raw = raw.tolist()
        out: list[int] = []
        for value in raw:
            try:
                out.append(int(value))
            except Exception:
                continue
        return out

    def _resolve_max_hops(self, context: ExplanationContext, payload: Dict[str, Any]) -> int:
        if self.cfg.max_hops is not None:
            return max(0, int(self.cfg.max_hops))
        if payload.get("k_hop") is not None:
            return max(0, int(payload.get("k_hop")))
        if context.k_hop is not None:
            return max(0, int(context.k_hop))
        return max(0, int(getattr(getattr(self, "_model", None), "num_layers", 1)))

    def _rows_for_candidates(self, candidate_eidx: Sequence[int]) -> np.ndarray:
        assert self._idx is not None
        rows = np.full((len(candidate_eidx),), -1, dtype=np.int64)
        for j, eid in enumerate(candidate_eidx):
            try:
                rows[j] = int(self._idx.resolve_row(int(eid)))
            except Exception:
                rows[j] = -1
        return rows

    def _score_from_hops(self, hops: np.ndarray, *, max_hops: int) -> np.ndarray:
        h = hops.astype(float)
        if self.cfg.score_mode == "linear":
            # In-range hops map to (0,1], out-of-range fallback hops map to 0.
            denom = float(max(max_hops, 0) + 1)
            return np.clip((denom - h) / denom, 0.0, 1.0)
        # Default: reciprocal decay with hop distance.
        return 1.0 / (1.0 + h)

    # ---------------------------------------------------------------- explain
    def explain(self, context: ExplanationContext) -> ExplanationResult:
        if self._idx is None:
            raise RuntimeError("KHopCloserAdapter not prepared. Call prepare() first.")

        payload = getattr(getattr(context, "subgraph", None), "payload", None) or {}
        event_idx = self._event_idx(context)
        candidate_eidx = self._candidate_eidx(context)

        if not candidate_eidx:
            return ExplanationResult(
                run_id=context.run_id,
                explainer=self.alias,
                context_fp=context.fingerprint(),
                importance_edges=[],
                importance_nodes=None,
                importance_time=None,
                extras={
                    "baseline": "khop_closer",
                    "event_idx": event_idx,
                    "candidate_eidx": [],
                    "hop_levels": [],
                    "score_mode": self.cfg.score_mode,
                },
            )

        try:
            base_row = int(self._idx.resolve_row(event_idx))
        except Exception:
            base_row = -1

        max_hops = self._resolve_max_hops(context, payload)
        fallback_hop = int(max_hops + max(1, int(self.cfg.fallback_hop_penalty)))

        candidate_rows = self._rows_for_candidates(candidate_eidx)
        hop_levels = np.full((len(candidate_eidx),), fallback_hop, dtype=np.int64)

        if base_row >= 0:
            unresolved = candidate_rows >= 0
            for h in range(max_hops + 1):
                support_rows_h = self._idx.khop_edge_rows(
                    base_row=base_row,
                    k=h,
                    directed=bool(self.cfg.directed),
                    exclude_base=False,
                )
                if support_rows_h.size == 0:
                    continue
                matched = unresolved & np.isin(candidate_rows, support_rows_h)
                hop_levels[matched] = int(h)
                unresolved = unresolved & (~matched)
                if not np.any(unresolved):
                    break

        scores = self._score_from_hops(hop_levels, max_hops=max_hops)

        return ExplanationResult(
            run_id=context.run_id,
            explainer=self.alias,
            context_fp=context.fingerprint(),
            importance_edges=[float(v) for v in scores.tolist()],
            importance_nodes=None,
            importance_time=None,
            extras={
                "baseline": "khop_closer",
                "event_idx": event_idx,
                "candidate_eidx": [int(e) for e in candidate_eidx],
                "candidate_row_idx": [int(r) for r in candidate_rows.tolist()],
                "hop_levels": [int(h) for h in hop_levels.tolist()],
                "score_mode": self.cfg.score_mode,
                "max_hops": int(max_hops),
                "directed": bool(self.cfg.directed),
                "fallback_hop": int(fallback_hop),
            },
        )


__all__ = ["KHopCloserAdapter", "KHopCloserAdapterConfig"]
