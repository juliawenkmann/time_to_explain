# time_to_explain/adapters/pg_and_pbone_tg_adapter.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

from time_to_explain.core.types import BaseExplainer, ExplanationContext, ExplanationResult

# Your existing temporal explainer (from submodules/tgnnexplainer)
from submodules.explainer.tgnnexplainer.tgnnexplainer.xgraph.method.other_baselines_tg import PBOneExplainerTG


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PerturbOneAdapterConfig:
    """Configuration for the PBOne (perturb-one) adapter.

    hierarchical:
        If True, build a coarse-to-fine hierarchy of event groups on top of
        the base per-event scores.
    lime_num_samples:
        If > 0, run a lightweight LIME-style local linear fit using random
        masks over events. By default this uses PBOne scores as the signal;
        you can later plug in true model calls if you want.
    """

    model_name: str                     # "tgn" | "tgat"
    dataset_name: str
    explanation_level: str = "event"
    results_dir: Optional[str] = None
    debug_mode: bool = False
    threshold_num: int = 25

    # Extras
    cache: bool = True
    alias: Optional[str] = None
    device: Optional[Union[str, torch.device]] = None

    # Hierarchical grouping on top of single-event scores
    hierarchical: bool = False
    hierarchy_levels: int = 3           # number of aggregation levels (incl. leaves)

    # LIME-style refinement over PBOne scores
    lime_num_samples: int = 0           # 0 => disabled
    lime_keep_prob: float = 0.8         # prob. that an event is kept in a sample


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class PerturbOneAdapter(BaseExplainer):
    """Wrap PBOneExplainerTG ("perturb-one") into the unified API.

    Base behaviour:
        - For a target event index e_idx, PBOneExplainerTG perturbs each
          candidate event around it and measures impact on the predictive score.
        - We expose the resulting dense importance vector over candidate events.

    Optional extras:
        - hierarchical=True => build a coarse-to-fine hierarchy of grouped
          event importances (e.g., temporal segments).
        - lime_num_samples>0 => run a simple LIME-style regression over
          PBOne scores using random event masks; this gives a smoothed,
          locally linear importance vector.
    """

    def __init__(self, cfg: PerturbOneAdapterConfig) -> None:
        super().__init__(name="perturb_one", alias=cfg.alias or "perturb_one")
        self.cfg = cfg

        self.device: Optional[torch.device] = (
            torch.device(cfg.device) if cfg.device is not None else None
        )
        self._explainer: Optional[PBOneExplainerTG] = None
        self._events: Any = None
        self._prepared: bool = False

        # event_idx -> raw result dict from PBOne
        self._cache: Dict[int, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def prepare(self, *, model: Any, dataset: Any) -> None:
        """Initialize underlying PBOneExplainerTG and infer device."""
        super().prepare(model=model, dataset=dataset)

        # Device inference
        if self.device is None:
            try:
                p = next(model.parameters())
                self.device = p.device
            except Exception:
                self.device = torch.device(
                    "cuda:0" if torch.cuda.is_available() else "cpu"
                )

        # Events and dataset name
        if isinstance(dataset, dict):
            self._events = dataset.get("events", dataset)
            if "dataset_name" in dataset:
                self.cfg.dataset_name = dataset["dataset_name"]
        else:
            self._events = dataset

        # Build underlying PBOneExplainerTG
        self._explainer = PBOneExplainerTG(
            model=model,
            model_name=self.cfg.model_name,
            explainer_name=self.alias,
            dataset_name=self.cfg.dataset_name,
            all_events=self._events,
            explanation_level=self.cfg.explanation_level,
            device=self.device,
            verbose=not self.cfg.debug_mode,
            results_dir=self.cfg.results_dir,
            debug_mode=self.cfg.debug_mode,
            threshold_num=self.cfg.threshold_num,
        )
        self._prepared = True

    # ------------------------------------------------------------------
    # Single explanation
    # ------------------------------------------------------------------

    def explain(self, context: ExplanationContext) -> ExplanationResult:
        assert self._prepared and self._explainer is not None, "Call .prepare() first."

        # 1) target event index
        eidx = (
            context.target.get("event_idx")
            or context.target.get("index")
            or context.target.get("idx")
        )
        if eidx is None and context.subgraph and context.subgraph.payload:
            eidx = context.subgraph.payload.get("event_idx")
        if eidx is None:
            raise ValueError(
                "PerturbOneAdapter expects an event index in "
                "target['event_idx'|'index'|'idx'] or subgraph.payload['event_idx']."
            )
        eidx = int(eidx)

        # 2) run or reuse PBOne for this event
        raw_pack = self._get_or_run_pbone(eidx)

        # 3) canonical candidate order
        candidate = self._get_candidate_order(context, raw_pack)

        # 4) aligned per-event importance vector from PBOne
        score_map = {
            int(e): float(s)
            for e, s in zip(raw_pack["candidate_eidx"], raw_pack["scores"])
        }
        pbone_scores_aligned = [score_map.get(e, 0.0) for e in candidate]

        extras: Dict[str, Any] = {
            "event_idx": eidx,
            "candidate_eidx": candidate,
            "pbone_scores_aligned": pbone_scores_aligned,
            "raw_candidate_eidx": raw_pack["candidate_eidx"],
            "raw_scores": raw_pack["scores"],
            "elapsed_sec_pbone": raw_pack["elapsed_sec"],
        }

        # 5) optional hierarchical aggregation
        if self.cfg.hierarchical:
            extras["hierarchy"] = self._build_hierarchy(
                candidate_eidx=candidate,
                scores=pbone_scores_aligned,
            )

        # 6) optional LIME-style refinement (over PBOne scores)
        importance_edges = pbone_scores_aligned
        total_elapsed = raw_pack["elapsed_sec"]

        if self.cfg.lime_num_samples > 0:
            t0 = time.perf_counter()
            lime_out = self._lime_refinement(
                base_scores=pbone_scores_aligned,
                num_samples=self.cfg.lime_num_samples,
                keep_prob=self.cfg.lime_keep_prob,
            )
            lime_elapsed = time.perf_counter() - t0
            total_elapsed += lime_elapsed

            # By default we expose the LIME coefs as the main importance,
            # but keep PBOne scores in extras for comparison.
            importance_edges = lime_out["importance"]
            extras["lime"] = lime_out["meta"]
            extras["lime"]["elapsed_sec"] = lime_elapsed

        return ExplanationResult(
            run_id=context.run_id,
            explainer=self.alias,
            context_fp=context.fingerprint(),
            importance_edges=importance_edges,
            importance_nodes=None,
            importance_time=None,
            elapsed_sec=total_elapsed,
            extras=extras,
        )

    # ------------------------------------------------------------------
    # Bulk API
    # ------------------------------------------------------------------

    def explain_many(self, event_idxs: List[int]) -> Dict[int, Dict[str, Any]]:
        """Return raw PBOne scores for multiple event indices.

        This is a thin convenience wrapper that bypasses ExplanationResult and
        does not run hierarchy or LIME. It also populates the cache.
        """
        assert self._prepared and self._explainer is not None, "Call .prepare() first."
        results: Dict[int, Dict[str, Any]] = {}

        out_list = self._explainer(event_idxs=event_idxs)
        for eidx, (keys, vals) in zip(event_idxs, out_list):
            candidate_eidx = [int(k) for k in keys]
            scores = [float(v) for v in vals]
            pack = {
                "candidate_eidx": candidate_eidx,
                "scores": scores,
                "elapsed_sec": None,
            }
            results[int(eidx)] = pack

            if self.cfg.cache:
                # preserve measured elapsed time if we had it
                if (
                    eidx in self._cache
                    and self._cache[eidx].get("elapsed_sec") is not None
                ):
                    pack["elapsed_sec"] = self._cache[eidx]["elapsed_sec"]
                self._cache[int(eidx)] = pack

        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_or_run_pbone(self, eidx: int) -> Dict[str, Any]:
        """Return raw PBOne output for a single event index, using cache if enabled."""
        if self.cfg.cache and eidx in self._cache:
            return self._cache[eidx]

        assert self._explainer is not None

        t0 = time.perf_counter()
        out_list = self._explainer(event_idxs=[eidx])  # [[keys, values]]
        elapsed = time.perf_counter() - t0

        keys, vals = out_list[0]
        candidate_eidx = [int(k) for k in keys]
        scores = [float(v) for v in vals]

        raw_pack = {
            "candidate_eidx": candidate_eidx,
            "scores": scores,
            "elapsed_sec": elapsed,
        }
        if self.cfg.cache:
            self._cache[eidx] = raw_pack
        return raw_pack

    def _get_candidate_order(
        self,
        context: ExplanationContext,
        raw_pack: Dict[str, Any],
    ) -> List[int]:
        """Resolve canonical candidate event ordering for the importance vector."""
        # 1) explicit candidate list in subgraph payload, if present
        if context.subgraph and context.subgraph.payload:
            candidate = context.subgraph.payload.get("candidate_eidx")
            if candidate is not None:
                return [int(e) for e in candidate]

        # 2) underlying explainer may expose its own candidate order
        if hasattr(self._explainer, "candidate_events"):
            candidate_attr = getattr(self._explainer, "candidate_events", None)
            if candidate_attr:
                return [int(e) for e in candidate_attr]

        # 3) fallback: use PBOne's own keys
        return [int(e) for e in raw_pack["candidate_eidx"]]

    # ------------------------------------------------------------------
    # Hierarchical aggregation
    # ------------------------------------------------------------------

    def _build_hierarchy(
        self,
        candidate_eidx: List[int],
        scores: List[float],
    ) -> Dict[str, Any]:
        """Build a simple balanced hierarchy over events.

        Current implementation:
            - Assumes `candidate_eidx` is already in a meaningful order
              (often temporal).
            - Level 0: one node per event.
            - Higher levels: merge adjacent pairs, summing scores.
        """
        n = len(candidate_eidx)
        if n == 0:
            return {"levels": []}

        assert n == len(scores)

        # level 0: leaves
        level0: List[Dict[str, Any]] = []
        for i, (e, s) in enumerate(zip(candidate_eidx, scores)):
            level0.append(
                {
                    "indices": [i],        # positions in the flat vector
                    "events": [int(e)],    # event ids
                    "score": float(s),
                }
            )

        levels: List[List[Dict[str, Any]]] = [level0]

        if self.cfg.hierarchy_levels <= 1:
            return {"levels": levels}

        current = level0
        while len(levels) < self.cfg.hierarchy_levels and len(current) > 1:
            next_level: List[Dict[str, Any]] = []
            for i in range(0, len(current), 2):
                group = current[i : i + 2]
                if len(group) == 1:
                    next_level.append(group[0])
                    continue

                indices: List[int] = []
                events: List[int] = []
                score = 0.0
                for g in group:
                    indices.extend(g["indices"])
                    events.extend(g["events"])
                    score += float(g["score"])

                next_level.append(
                    {
                        "indices": indices,
                        "events": events,
                        "score": float(score),
                    }
                )

            levels.append(next_level)
            current = next_level

        return {"levels": levels}

    # ------------------------------------------------------------------
    # LIME-style refinement on PBOne scores
    # ------------------------------------------------------------------

    def _lime_refinement(
        self,
        base_scores: List[float],
        num_samples: int,
        keep_prob: float,
    ) -> Dict[str, Any]:
        """Fit a simple local linear model on top of PBOne scores.

        This does NOT perform new model calls; instead it treats PBOne's
        per-event scores as an additive surrogate for the underlying model.
        The goal is to get a smoothed, locally linear importance profile.

        Returns:
            {
                "importance": [float] * num_events,  # LIME coefficients
                "meta": {
                    "used": bool,
                    "intercept": float,
                    "num_samples": int,
                    "keep_prob": float,
                }
            }
        """
        num_events = len(base_scores)
        if num_events == 0 or num_samples <= 0:
            return {
                "importance": list(base_scores),
                "meta": {"used": False},
            }

        device = self.device or torch.device("cpu")

        base = torch.tensor(base_scores, dtype=torch.float32, device=device)
        y0 = base.sum()

        # Sample random masks over events
        masks = (torch.rand(num_samples, num_events, device=device) < keep_prob).float()

        # Simulated prediction under each mask, assuming additive PBOne effects:
        #   y(mask) = y0 - sum_{j: mask_j == 0} base[j]
        dropped = (1.0 - masks) * base  # (num_samples, num_events)
        y = y0 - dropped.sum(dim=1)     # (num_samples,)

        # Design matrix: [bias | masks]
        ones = torch.ones(num_samples, 1, device=device)
        X = torch.cat([ones, masks], dim=1)  # (num_samples, 1 + num_events)

        # Ordinary least squares: beta = (X^T X)^-1 X^T y
        XtX = X.T @ X
        Xty = X.T @ y
        beta = torch.linalg.pinv(XtX) @ Xty

        intercept = float(beta[0].item())
        coefs = beta[1:].detach().cpu().tolist()

        return {
            "importance": coefs,
            "meta": {
                "used": True,
                "intercept": intercept,
                "num_samples": int(num_samples),
                "keep_prob": float(keep_prob),
            },
        }
