from __future__ import annotations

from dataclasses import dataclass
import secrets
from typing import Any, Dict, Optional

import numpy as np

from ...core.types import BaseExplainer, ExplanationContext, ExplanationResult


@dataclass
class RandomEdgeImportanceGenerator:
    seed: Optional[int] = None
    reseed_each_call: bool = True

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def reset(self, seed: Optional[int] = None, reseed_each_call: Optional[bool] = None) -> None:
        if seed is not None:
            self.seed = seed
        if reseed_each_call is not None:
            self.reseed_each_call = bool(reseed_each_call)
        self._rng = np.random.default_rng(self.seed)

    def _draw_rng(self) -> np.random.Generator:
        """
        Build a per-call RNG when requested.

        - seed=None  + reseed_each_call=True: pull entropy from OS each call
          (fully random, non-reproducible baseline).
        - seed=<int> + reseed_each_call=True: deterministic child-stream per call.
        - reseed_each_call=False: use one persistent stream.
        """
        if not bool(self.reseed_each_call):
            return self._rng
        if self.seed is None:
            return np.random.default_rng(secrets.randbits(128))
        child_seed = int(self._rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
        return np.random.default_rng(child_seed)

    def _extract_event_idx(self, context: ExplanationContext) -> Optional[int]:
        subgraph = getattr(context, "subgraph", None)
        if subgraph is None:
            return None
        payload = getattr(subgraph, "payload", None) or {}
        event_idx = payload.get("event_idx")
        try:
            return int(event_idx) if event_idx is not None else None
        except Exception:
            return None

    def _count_edges(self, context: ExplanationContext) -> int:
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
        return int(len(edge_index))

    def generate(self, context: ExplanationContext) -> tuple[list[float], Dict[str, Any]]:
        event_idx = self._extract_event_idx(context)
        n_edges = self._count_edges(context)
        if n_edges <= 0:
            return [], {
                "baseline": "random_uniform",
                "random_mode": "fully_random" if self.seed is None and self.reseed_each_call else "seeded_random",
                "seed": self.seed,
                "reseed_each_call": bool(self.reseed_each_call),
                "n_edges": 0,
                "event_idx": event_idx,
            }

        rng = self._draw_rng()
        # Random ranking without ties: random permutation mapped to strictly
        # descending scores, then tiny jitter to keep scores unique in practice.
        perm = np.asarray(rng.permutation(n_edges), dtype=np.int64)
        rank = np.empty(n_edges, dtype=np.int64)
        rank[perm] = np.arange(n_edges, dtype=np.int64)
        scores = (float(n_edges) - rank.astype(float)) + rng.random(n_edges) * 1e-9

        return scores.tolist(), {
            "baseline": "random_uniform",
            "random_mode": "fully_random" if self.seed is None and self.reseed_each_call else "seeded_random",
            "seed": self.seed,
            "reseed_each_call": bool(self.reseed_each_call),
            "n_edges": n_edges,
            "event_idx": event_idx,
        }


@dataclass
class RandomAdapterConfig:
    alias: str = "random"
    seed: Optional[int] = None
    reseed_each_call: bool = True


class RandomAdapter(BaseExplainer):
    """
    Minimal adapter that exposes the random baseline through the shared
    `BaseExplainer` interface so it can be scheduled next to TGNNExplainer, TEMP-ME, etc.
    """

    def __init__(self, cfg: Optional[RandomAdapterConfig] = None) -> None:
        self.cfg = cfg or RandomAdapterConfig()
        super().__init__(name="random_baseline", alias=self.cfg.alias)
        self._generator = RandomEdgeImportanceGenerator(
            seed=self.cfg.seed,
            reseed_each_call=self.cfg.reseed_each_call,
        )

    # ------------------------------------------------------------------ setup
    def prepare(self, *, model: Any, dataset: Any) -> None:
        """
        No model/dataset dependency beyond tracking the RNG seed for reproducibility,
        but we keep the signature to align with the other explainers.
        """
        super().prepare(model=model, dataset=dataset)
        self._generator.reset(self.cfg.seed, reseed_each_call=self.cfg.reseed_each_call)

    # ---------------------------------------------------------------- explain
    def explain(self, context: ExplanationContext) -> ExplanationResult:
        scores, extras = self._generator.generate(context)
        return ExplanationResult(
            run_id=context.run_id,
            explainer=self.alias,
            context_fp=context.fingerprint(),
            importance_edges=scores,
            importance_nodes=None,
            importance_time=None,
            extras=extras,
        )


__all__ = ["RandomAdapter", "RandomAdapterConfig"]
