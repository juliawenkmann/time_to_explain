# time_to_explain/adapters/shap_tg_adapter.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

from time_to_explain.core.types import BaseExplainer, ExplanationContext, ExplanationResult


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ShapAdapterConfig:
    """Configuration for temporal SHAP adapter.

    SHAP is estimated by Monte Carlo sampling of random permutations
    of candidate events and computing marginal contributions to the
    model's score for a given target event.
    """

    model_name: str
    dataset_name: str
    explanation_level: str = "event"
    results_dir: Optional[str] = None
    debug_mode: bool = False

    # Caching / naming
    cache: bool = True
    alias: Optional[str] = None
    device: Optional[Union[str, torch.device]] = None

    # SHAP-specific
    num_samples: int = 100              # number of Monte Carlo permutations
    random_seed: Optional[int] = None   # for reproducibility (None => random)

    # Scoring hook: if None, you must subclass ShapAdapter and override _score().
    # Signature: score_fn(model, dataset, target_event_idx, active_event_ids) -> float
    score_fn: Optional[
        Callable[[Any, Any, int, Sequence[int]], float]
    ] = None


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------


class ShapExplainer(BaseExplainer):
    """SHAP-style temporal explainer.

    This adapter is model-agnostic: it treats the temporal GNN as a black box
    and estimates Shapley values for candidate events around a target event.

    The actual model scoring logic is delegated to either:

      * cfg.score_fn(model, dataset, target_event_idx, active_event_ids), or
      * a custom override of ShapAdapter._score(...)

    where `active_event_ids` is the subset of candidate events treated as "on".
    """

    def __init__(self, cfg: ShapAdapterConfig) -> None:
        super().__init__(name="shap", alias=cfg.alias or "shap")
        self.cfg = cfg

        self.device: Optional[torch.device] = (
            torch.device(cfg.device)
            if isinstance(cfg.device, (str, torch.device))
            else None
        )

        self._model: Any = None
        self._dataset: Any = None
        self._events: Any = None
        self._prepared: bool = False

        # cache: (target_event_idx, tuple(sorted(active_event_ids))) -> float
        self._score_cache: Dict[Tuple[int, Tuple[int, ...]], float] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def prepare(self, *, model: Any, dataset: Any) -> None:
        """Store model/dataset and infer device."""
        super().prepare(model=model, dataset=dataset)
        self._model = model
        self._dataset = dataset

        if self.device is None:
            try:
                p = next(model.parameters())
                self.device = p.device
            except Exception:
                self.device = torch.device(
                    "cuda:0" if torch.cuda.is_available() else "cpu"
                )

        # Keep the same convention as other adapters
        if isinstance(dataset, dict) and "events" in dataset:
            self._events = dataset["events"]
        else:
            self._events = dataset

        if isinstance(dataset, dict) and "dataset_name" in dataset:
            self.cfg.dataset_name = dataset["dataset_name"]

        self._prepared = True

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def explain(self, context: ExplanationContext) -> ExplanationResult:
        """Compute SHAP values for candidate events of a target event."""
        assert self._prepared, "Call .prepare() first."

        t0 = time.perf_counter()

        target_eidx = self._resolve_event_idx(context)
        candidate_eidx = self._resolve_candidate_events(context, target_eidx)

        shap_values = self._estimate_shapley(
            target_event_idx=target_eidx,
            candidate_eidx=candidate_eidx,
        )

        elapsed = time.perf_counter() - t0

        extras: Dict[str, Any] = {
            "event_idx": target_eidx,
            "candidate_eidx": candidate_eidx,
            "shap_values_raw": shap_values,
            "num_samples": self.cfg.num_samples,
        }

        return ExplanationResult(
            run_id=context.run_id,
            explainer=self.alias,
            context_fp=context.fingerprint(),
            importance_edges=shap_values,
            importance_nodes=None,
            importance_time=None,
            elapsed_sec=elapsed,
            extras=extras,
        )

    # ------------------------------------------------------------------
    # Internals: index & candidate resolution
    # ------------------------------------------------------------------
    def _resolve_event_idx(self, context: ExplanationContext) -> int:
        eidx = (
            context.target.get("event_idx")
            or context.target.get("index")
            or context.target.get("idx")
        )
        if eidx is None and context.subgraph and context.subgraph.payload:
            eidx = context.subgraph.payload.get("event_idx")
        if eidx is None:
            raise ValueError(
                "ShapAdapter expects an event index in "
                "target['event_idx'|'index'|'idx'] or "
                "subgraph.payload['event_idx']."
            )
        return int(eidx)

    def _resolve_candidate_events(
        self,
        context: ExplanationContext,
        target_event_idx: int,
    ) -> List[int]:
        """Return the list of candidate events to attribute.

        By default we expect a list of candidate event ids in:

            context.subgraph.payload['candidate_eidx']

        (or 'events' / 'event_ids' as aliases).

        This keeps the adapter generic and lets the caller decide how big
        the neighbourhood should be, how to slice in time, etc.

        You can subclass and override this method to implement your own
        neighbourhood selection logic.
        """
        if context.subgraph and context.subgraph.payload:
            payload = context.subgraph.payload
            candidate = (
                payload.get("candidate_eidx")
                or payload.get("events")
                or payload.get("event_ids")
            )
            if candidate is not None:
                return [int(e) for e in candidate]

        raise ValueError(
            "ShapAdapter expects `candidate_eidx` (or `events` / `event_ids`) "
            "in context.subgraph.payload. Override `_resolve_candidate_events` "
            "if you want a different behaviour."
        )

    # ------------------------------------------------------------------
    # Core SHAP logic
    # ------------------------------------------------------------------
    def _estimate_shapley(
        self,
        target_event_idx: int,
        candidate_eidx: List[int],
    ) -> List[float]:
        """Monte Carlo estimation of Shapley values over candidate events."""
        import numpy as np  # local import, only needed if you actually use SHAP

        n = len(candidate_eidx)
        if n == 0:
            return []

        num_samples = max(1, int(self.cfg.num_samples))

        # Reproducible RNG if requested
        rng = np.random.default_rng(self.cfg.random_seed)

        shap = np.zeros(n, dtype=np.float64)

        for _ in range(num_samples):
            # permutation over indices [0, ..., n-1]
            perm = rng.permutation(n)

            active: List[int] = []
            prev_score = self._score(target_event_idx, active)

            for pos in perm:
                e_id = candidate_eidx[pos]
                active.append(e_id)
                new_score = self._score(target_event_idx, active)
                contrib = new_score - prev_score
                shap[pos] += contrib
                prev_score = new_score

        shap /= float(num_samples)
        return shap.tolist()

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _score(self, target_event_idx: int, active_event_ids: Sequence[int]) -> float:
        """Score the model for a given target event and active event subset.

        Default behaviour:
            - If cfg.score_fn is provided, call that.
            - Otherwise, raise – you must either pass score_fn in the config
              or override this method in a subclass.

        The semantics of `active_event_ids` are up to you, but a common choice:
        the subset of candidate events *kept* in the temporal neighbourhood
        of `target_event_idx`.
        """
        key: Optional[Tuple[int, Tuple[int, ...]]] = None
        if self.cfg.cache:
            key = (int(target_event_idx), tuple(sorted(int(e) for e in active_event_ids)))
            if key in self._score_cache:
                return self._score_cache[key]

        if self.cfg.score_fn is None:
            raise RuntimeError(
                "ShapAdapter requires either `cfg.score_fn` or an override of `_score`."
            )

        score = float(
            self.cfg.score_fn(
                self._model,
                self._dataset,
                int(target_event_idx),
                [int(e) for e in active_event_ids],
            )
        )

        if self.cfg.cache and key is not None:
            self._score_cache[key] = score

        return score
