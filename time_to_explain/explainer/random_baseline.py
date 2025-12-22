# time_to_explain/explainer/random_baseline.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from time_to_explain.core.types import ExplanationContext


@dataclass
class RandomEdgeImportanceGenerator:
    """
    Lightweight helper that produces random importance scores aligned with
    the candidate edge order supplied in `ExplanationContext.subgraph.payload`.
    """

    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------ utils
    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reinitialize the RNG (call this from an adapter's `prepare` to keep
        notebook executions deterministic).
        """
        if seed is not None:
            self.seed = seed
        self._rng = np.random.default_rng(self.seed)

    def _extract_payload(self, context: ExplanationContext) -> tuple[Optional[List[int]], Optional[int]]:
        subgraph = getattr(context, "subgraph", None)
        if subgraph is None:
            return None, None
        payload = getattr(subgraph, "payload", None) or {}
        candidate = payload.get("candidate_eidx")
        event_idx = payload.get("event_idx")
        if candidate is not None:
            if isinstance(candidate, list):
                cand_list = candidate
            else:
                try:
                    cand_list = list(candidate)
                except TypeError:
                    cand_list = candidate
            return cand_list, event_idx
        return None, event_idx

    def _count_edges(self, context: ExplanationContext) -> int:
        candidate, _ = self._extract_payload(context)
        if candidate is not None:
            try:
                return len(candidate)
            except Exception:
                pass

        subgraph = getattr(context, "subgraph", None)
        edge_index = getattr(subgraph, "edge_index", None) if subgraph is not None else None
        if edge_index is None:
            return 0
        arr = np.asarray(edge_index)
        if arr.ndim == 2:
            if arr.shape[0] == 2:
                return int(arr.shape[1])
            if arr.shape[1] == 2:
                return int(arr.shape[0])
        try:
            return int(len(edge_index))
        except Exception:
            return 0

    # ----------------------------------------------------------------- public
    def generate(self, context: ExplanationContext) -> tuple[List[float], Dict[str, int | float | str]]:
        """
        Produce a list of random scores (Uniform[0,1)) aligned with the
        candidate edge order and accompanying metadata for logging.
        """
        candidate, event_idx = self._extract_payload(context)
        if candidate is None:
            print("No candidate set given")
        if candidate is not None:
            print(f"Candidate set given.{candidate}")
        n_edges = len(candidate) if candidate is not None else self._count_edges(context)
        if n_edges <= 0:
            extras = {
                "baseline": "random_uniform",
                "seed": self.seed,
                "n_edges": 0,
                "candidate_eidx": candidate,
                "event_idx": event_idx,
            }
            return [], extras

        scores = self._rng.random(n_edges).tolist()
        return scores, {
            "baseline": "random_uniform",
            "seed": self.seed,
            "n_edges": n_edges,
            "candidate_eidx": candidate,
            "event_idx": event_idx,
        }


__all__ = ["RandomEdgeImportanceGenerator"]
