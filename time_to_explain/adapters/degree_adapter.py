# time_to_explain/adapters/degree_baseline_adapter.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from time_to_explain.core.types import BaseExplainer, ExplanationContext, ExplanationResult


@dataclass
class DegreeAdapterConfig:
    alias: str = "degree"
    aggregation: str = "mean"          # "mean" or "sum"
    min_degree: float = 0.0
    source_target_columns: Optional[Tuple[str, str]] = None


class DegreeAdapter(BaseExplainer):
    """
    Deterministic baseline that ranks edges by the degree of their endpoints.
    It looks up (u, v) for each candidate event, computes global interaction
    counts per node, and assigns score = agg(deg(u), deg(v)).
    """

    _COLUMN_CANDIDATES: Sequence[Tuple[str, str]] = (
        ("u", "i"),
        ("src", "dst"),
        ("source", "target"),
        ("user", "item"),
        ("node_u", "node_v"),
        ("node_1", "node_2"),
    )

    def __init__(self, cfg: Optional[DegreeAdapterConfig] = None) -> None:
        self.cfg = cfg or DegreeAdapterConfig()
        super().__init__(name="degree_baseline", alias=self.cfg.alias)
        self._events: Optional[pd.DataFrame] = None
        self._src_col: Optional[str] = None
        self._dst_col: Optional[str] = None
        self._degrees: Dict[Any, float] = {}

    # ------------------------------------------------------------------ helpers
    def _resolve_columns(self, events: pd.DataFrame) -> Tuple[str, str]:
        if self.cfg.source_target_columns:
            src, dst = self.cfg.source_target_columns
            if src not in events or dst not in events:
                raise KeyError(f"Configured columns {(src, dst)} missing from events dataframe.")
            return src, dst
        for src, dst in self._COLUMN_CANDIDATES:
            if src in events.columns and dst in events.columns:
                return src, dst
        raise KeyError(
            f"Could not infer source/destination columns from events dataframe. "
            f"Tried: {self._COLUMN_CANDIDATES}"
        )

    def _compute_degrees(self, events: pd.DataFrame, src: str, dst: str) -> Dict[Any, float]:
        src_counts = events[src].value_counts()
        dst_counts = events[dst].value_counts()
        degrees = src_counts.add(dst_counts, fill_value=0.0)
        return degrees.to_dict()

    def _event_nodes(self, event_idx: int) -> Tuple[Any, Any]:
        assert self._events is not None and self._src_col and self._dst_col
        if not (1 <= event_idx <= len(self._events)):
            raise IndexError(f"event_idx {event_idx} out of bounds for events dataframe.")
        row = self._events.iloc[event_idx - 1]
        return row[self._src_col], row[self._dst_col]

    def _score_edge(self, src_node: Any, dst_node: Any) -> float:
        deg_u = float(self._degrees.get(src_node, self.cfg.min_degree))
        deg_v = float(self._degrees.get(dst_node, self.cfg.min_degree))
        if self.cfg.aggregation == "sum":
            return deg_u + deg_v
        # default mean
        return (deg_u + deg_v) / 2.0

    # ------------------------------------------------------------------ lifecycle
    def prepare(self, *, model: Any, dataset: Any) -> None:
        super().prepare(model=model, dataset=dataset)
        events = dataset.get("events") if isinstance(dataset, dict) else dataset
        if events is None or not isinstance(events, pd.DataFrame):
            raise ValueError("DegreeHeuristicAdapter expects dataset['events'] to be a pandas DataFrame.")
        events_df = events.reset_index(drop=True)
        self._src_col, self._dst_col = self._resolve_columns(events_df)
        self._events = events_df
        self._degrees = self._compute_degrees(events_df, self._src_col, self._dst_col)

    # ------------------------------------------------------------------ explain
    def explain(self, context: ExplanationContext) -> ExplanationResult:
        subgraph = getattr(context, "subgraph", None)
        payload = getattr(subgraph, "payload", None) or {}
        candidate = payload.get("candidate_eidx") or []
        if isinstance(candidate, np.ndarray):
            candidate = candidate.tolist()
        try:
            candidate_list = [int(e) for e in candidate]
        except Exception:
            candidate_list = list(candidate)

        scores: Sequence[float]
        if not candidate_list:
            scores = []
        else:
            scores = []
            for eidx in candidate_list:
                try:
                    nodes = self._event_nodes(eidx)
                    score = self._score_edge(*nodes)
                except Exception:
                    score = float(self.cfg.min_degree)
                scores.append(score)

        extras = {
            "baseline": "degree_heuristic",
            "aggregation": self.cfg.aggregation,
            "min_degree": self.cfg.min_degree,
            "source_column": self._src_col,
            "target_column": self._dst_col,
            "candidate_eidx": candidate_list,
        }

        return ExplanationResult(
            run_id=context.run_id,
            explainer=self.alias,
            context_fp=context.fingerprint(),
            importance_edges=list(scores),
            importance_nodes=None,
            importance_time=None,
            extras=extras,
        )


__all__ = ["DegreeAdapter", "DegreeAdapterConfig"]
