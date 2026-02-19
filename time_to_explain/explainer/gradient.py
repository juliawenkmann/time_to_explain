# time_to_explain/adapters/grad_tg_adapter.py
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
class GradientAdapterConfig:
    """Configuration for a gradient-based temporal explainer.

    This adapter is intentionally generic: it assumes that the model's score
    for a given target event can be written as a differentiable function of
    a continuous mask over the candidate events.

    The mask is a 1D tensor of shape [len(candidate_eidx)] with values in
    [0, 1]. The user must either:
      * provide `forward_fn` in the config, or
      * subclass :class:`GradientExplainer` and override `_forward(...)`.

    `forward_fn` must have the signature:

        forward_fn(
            model,
            dataset,
            target_event_idx: int,
            candidate_eidx: Sequence[int],
            mask: torch.Tensor,              # shape [num_candidates], requires_grad=True
        ) -> torch.Tensor                    # scalar score

    and is responsible for applying the mask to the temporal neighbourhood
    corresponding to `candidate_eidx` (e.g. gating edges / events).
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

    # Gradient-specific options
    method: str = "vanilla"          # "vanilla" or "integrated"
    ig_steps: int = 32               # number of steps for integrated gradients
    abs_attributions: bool = True    # take absolute value of gradients
    normalize: bool = False          # L1 normalize attributions to sum to 1

    # Forward hook. If None, you must override `_forward`.
    # See class docstring for the expected signature.
    forward_fn: Optional[
        Callable[[Any, Any, int, Sequence[int], torch.Tensor], torch.Tensor]
    ] = None


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------


class GradientExplainer(BaseExplainer):
    """Gradient-based temporal explainer.

    Conceptually, this is a "gradient×mask" style explainer over *events*
    (or edges) in a temporal neighbourhood of a target event.

    1. Resolve target event index from `ExplanationContext`.
    2. Resolve candidate event indices from `context.subgraph.payload`.
    3. Build a continuous mask over the candidates (length = num candidates).
    4. Backpropagate the model's score w.r.t. that mask.
    5. Convert mask gradients into per-event importances.

    The event-level importances are returned in the same order as
    `candidate_eidx` and exposed in `ExplanationResult.importance_edges`.
    """

    def __init__(self, cfg: GradientAdapterConfig) -> None:
        super().__init__(name="grad", alias=cfg.alias or "grad")
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

        # Cache: (target_event_idx, tuple(candidate_eidx)) -> List[float]
        self._attr_cache: Dict[Tuple[int, Tuple[int, ...]], List[float]] = {}

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
        """Compute gradient-based importances for candidate events."""
        assert self._prepared, "Call .prepare() first."

        t0 = time.perf_counter()

        target_eidx = self._resolve_event_idx(context)
        candidate_eidx = self._resolve_candidate_events(context, target_eidx)

        try:
            attributions = self._compute_attributions(
                target_event_idx=target_eidx,
                candidate_eidx=candidate_eidx,
            )
        except RuntimeError as exc:
            # Gracefully degrade if gradients cannot flow (e.g., backbone uses no-grad inference).
            if "No gradients found" in str(exc):
                attributions = [0.0 for _ in candidate_eidx]
                print(f"[GradientExplainer] gradients unavailable; returning zeros ({len(attributions)}).")
            else:
                raise

        elapsed = time.perf_counter() - t0

        extras: Dict[str, Any] = {
            "event_idx": target_eidx,
            "candidate_eidx": candidate_eidx,
            "grad_attributions_raw": attributions,
            "method": self.cfg.method,
            "ig_steps": self.cfg.ig_steps,
            "abs_attributions": self.cfg.abs_attributions,
            "normalize": self.cfg.normalize,
        }

        return ExplanationResult(
            run_id=context.run_id,
            explainer=self.alias,
            context_fp=context.fingerprint(),
            importance_edges=attributions,
            importance_nodes=None,
            importance_time=None,
            elapsed_sec=elapsed,
            extras=extras,
        )

    # ------------------------------------------------------------------
    # Internals: index & candidate resolution
    # (kept identical to ShapExplainer for drop-in replacement)
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
                "GradientExplainer expects an event index in "
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

        This mirrors the behaviour of :class:`ShapExplainer` and keeps
        the adapter generic – the caller decides how big the neighbourhood
        should be, how to slice in time, etc.

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
            "GradientExplainer expects `candidate_eidx` "
            "(or `events` / `event_ids`) in context.subgraph.payload. "
            "Override `_resolve_candidate_events` if you want a different "
            "behaviour."
        )

    # ------------------------------------------------------------------
    # Core gradient logic
    # ------------------------------------------------------------------
    def _compute_attributions(
        self,
        target_event_idx: int,
        candidate_eidx: List[int],
    ) -> List[float]:
        """Compute event-level attributions using gradients."""
        n = len(candidate_eidx)
        if n == 0:
            return []

        key: Optional[Tuple[int, Tuple[int, ...]]] = None
        if self.cfg.cache:
            key = (int(target_event_idx), tuple(int(e) for e in candidate_eidx))
            if key in self._attr_cache:
                return self._attr_cache[key]

        method = self.cfg.method.lower()
        if method == "integrated":
            try:
                attr = self._integrated_gradients(target_event_idx, candidate_eidx)
            except RuntimeError as exc:
                if "No gradients found" in str(exc):
                    # Automatic fallback to vanilla gradients if IG cannot flow.
                    attr = self._vanilla_gradients(target_event_idx, candidate_eidx)
                else:
                    raise
        else:
            # Fallback to vanilla gradients
            attr = self._vanilla_gradients(target_event_idx, candidate_eidx)

        # Post-processing: abs + normalization if requested
        assert self.device is not None, "Device not set – did you call .prepare()?"
        attr_tensor = torch.as_tensor(attr, dtype=torch.float32, device=self.device)

        if self.cfg.abs_attributions:
            attr_tensor = attr_tensor.abs()

        if self.cfg.normalize:
            denom = attr_tensor.sum().abs()
            if denom > 0:
                attr_tensor = attr_tensor / denom

        attr_list = attr_tensor.detach().cpu().tolist()

        if self.cfg.cache and key is not None:
            self._attr_cache[key] = attr_list

        return attr_list

    def _vanilla_gradients(
        self,
        target_event_idx: int,
        candidate_eidx: Sequence[int],
    ) -> List[float]:
        """Single backprop of score w.r.t. event mask."""
        assert self.device is not None, "Device not set – did you call .prepare()?"

        num_candidates = len(candidate_eidx)
        if num_candidates == 0:
            return []

        mask = torch.ones(num_candidates, device=self.device, requires_grad=True)

        model = self._model
        # Switch to eval mode during attribution
        was_training = getattr(model, "training", False)
        if hasattr(model, "eval"):
            model.eval()

        try:
            if hasattr(model, "zero_grad"):
                model.zero_grad()

            score = self._forward(target_event_idx, candidate_eidx, mask)
            # Ensure scalar
            score = score.squeeze()
            if score.ndim != 0:
                raise RuntimeError(
                    "GradientExplainer expected a scalar score from `_forward`, "
                    f"got tensor of shape {tuple(score.shape)} instead."
                )
            if not score.requires_grad:
                raise RuntimeError(
                    "No gradients found on score – did `_forward` use the mask "
                    "in a differentiable way?"
                )

            score.backward()

            if mask.grad is None:
                raise RuntimeError(
                    "No gradients found on mask – did `_forward` use the mask "
                    "in a differentiable way?"
                )

            grads = mask.grad.detach().clone()
        finally:
            if hasattr(model, "train"):
                model.train(was_training)

        return grads.cpu().tolist()

    def _integrated_gradients(
        self,
        target_event_idx: int,
        candidate_eidx: Sequence[int],
    ) -> List[float]:
        """Integrated gradients along a path in mask-space.

        We integrate the gradient of the score w.r.t. the event mask as the
        mask moves from an all-zero baseline (no events) to an all-one mask
        (all candidate events fully active).

        The final attribution is:

            (mask_final - mask_baseline) * mean_{steps} grad(score, mask_step)
        """
        assert self.device is not None, "Device not set – did you call .prepare()?"

        num_candidates = len(candidate_eidx)
        if num_candidates == 0:
            return []

        steps = max(1, int(self.cfg.ig_steps))
        model = self._model
        was_training = getattr(model, "training", False)
        if hasattr(model, "eval"):
            model.eval()

        baseline = torch.zeros(num_candidates, device=self.device)
        final_mask = torch.ones(num_candidates, device=self.device)
        path_delta = final_mask - baseline

        total_grad = torch.zeros(num_candidates, device=self.device)

        try:
            for s in range(1, steps + 1):
                alpha = float(s) / float(steps)
                mask = (baseline + alpha * path_delta)
                mask.requires_grad_(True)

                if hasattr(model, "zero_grad"):
                    model.zero_grad()

                score = self._forward(target_event_idx, candidate_eidx, mask)
                score = score.squeeze()
                if score.ndim != 0:
                    raise RuntimeError(
                        "GradientExplainer expected a scalar score from `_forward`, "
                        f"got tensor of shape {tuple(score.shape)} instead."
                    )
                if not score.requires_grad:
                    raise RuntimeError(
                        "No gradients found on score during integrated gradients – "
                        "did `_forward` use the mask in a differentiable way?"
                    )

                score.backward()

                if mask.grad is None:
                    raise RuntimeError(
                        "No gradients found on mask during integrated gradients – "
                        "did `_forward` use the mask in a differentiable way?"
                    )

                total_grad += mask.grad.detach()
        finally:
            if hasattr(model, "train"):
                model.train(was_training)

        avg_grad = total_grad / float(steps)
        attributions = path_delta * avg_grad

        return attributions.cpu().tolist()

    # ------------------------------------------------------------------
    # Forward hook
    # ------------------------------------------------------------------
    def _forward(
        self,
        target_event_idx: int,
        candidate_eidx: Sequence[int],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward hook: compute scalar score given an event mask.

        Default behaviour:
            - If cfg.forward_fn is provided, call that.
            - Otherwise, raise – you must either pass forward_fn in the config
              or override this method in a subclass.

        The semantics of `mask` are entirely up to the user, but the typical
        interpretation is that it gates the contribution of each candidate
        event (e.g. multiplicative mask on edge weights or event features).
        """
        if self.cfg.forward_fn is None:
            raise RuntimeError(
                "GradientExplainer requires either `cfg.forward_fn` or an "
                "override of `_forward`."
            )

        return self.cfg.forward_fn(
            self._model,
            self._dataset,
            int(target_event_idx),
            [int(e) for e in candidate_eidx],
            mask,
        )
