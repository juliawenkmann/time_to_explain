# time_to_explain/adapters/attn_adapter.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import sys
import time
import torch

from time_to_explain.core.types import BaseExplainer, ExplanationContext, ExplanationResult

# Ensure vendored tgnnexplainer is importable
_TGNN_VENDOR = Path(__file__).resolve().parents[2] / "submodules" / "explainer" / "tgnnexplainer"
if str(_TGNN_VENDOR) not in sys.path:
    sys.path.insert(0, str(_TGNN_VENDOR))

from tgnnexplainer.xgraph.method.attn_explainer_tg import AttnExplainerTG


@dataclass
class AttnAdapterConfig:
    model_name: str                     # "tgn" | "tgat"
    dataset_name: str
    explanation_level: str = "event"
    results_dir: Optional[str] = None
    debug_mode: bool = False
    threshold_num: int = 25
    cache: bool = True
    alias: Optional[str] = None
    device: Optional[Union[str, torch.device]] = None


class AttnAdapter(BaseExplainer):
    """
    Wraps the vendored AttnExplainerTG into the unified explainer interface.

    - Requires a 1-based `event_idx` in the context target.
    - If the subgraph payload supplies `candidate_eidx`, the importance vector
      is aligned to that ordering for metric compatibility.
    """

    def __init__(self, cfg: AttnAdapterConfig) -> None:
        super().__init__(name="attn", alias=cfg.alias or "attn")
        self.cfg = cfg
        self._explainer: Optional[AttnExplainerTG] = None
        self._events = None
        self.device = torch.device(cfg.device) if cfg.device is not None else None
        self._prepared = False
        self._cache: Dict[int, Dict[str, Any]] = {}

    def prepare(self, *, model: Any, dataset: Any) -> None:
        super().prepare(model=model, dataset=dataset)

        if self.device is None:
            try:
                p = next(model.parameters())
                self.device = p.device
            except Exception:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._events = dataset["events"] if isinstance(dataset, dict) and "events" in dataset else dataset
        if isinstance(dataset, dict) and "dataset_name" in dataset:
            self.cfg.dataset_name = dataset["dataset_name"]

        self._explainer = AttnExplainerTG(
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

    def explain(self, context: ExplanationContext) -> ExplanationResult:
        assert self._prepared and self._explainer is not None, "Call .prepare() first."

        eidx = (
            context.target.get("event_idx")
            or context.target.get("index")
            or context.target.get("idx")
        )
        if eidx is None and context.subgraph and context.subgraph.payload:
            eidx = context.subgraph.payload.get("event_idx")
        if eidx is None:
            raise ValueError("AttnAdapter expects an event index in target['event_idx'|'index'|'idx'].")

        eidx = int(eidx)

        if self.cfg.cache and eidx in self._cache:
            cached = self._cache[eidx]
            return self._pack_result(context, cached)

        # Optional candidate ordering from extractor
        cand_payload: Optional[Sequence[int]] = None
        if context.subgraph and context.subgraph.payload:
            cand_payload = context.subgraph.payload.get("candidate_eidx")
            cand_payload = [int(c) for c in cand_payload] if cand_payload is not None else None

        t0 = time.perf_counter()

        # Initialize internal state (build subgraph, base events, etc.)
        self._explainer._initialize(eidx)

        # If an external candidate order is provided, override and recompute baseline
        if cand_payload is not None:
            self._explainer.candidate_events = list(cand_payload)
            self._explainer.base_events = []
            self._explainer.tgnn_reward_wraper.compute_original_score(
                self._explainer.base_events + self._explainer.candidate_events,
                eidx,
            )

        weights_dict = self._explainer.explain(event_idx=eidx)
        elapsed = time.perf_counter() - t0

        # Determine candidate ordering for importance vector
        if cand_payload is not None:
            candidate = list(cand_payload)
        else:
            candidate = list(weights_dict.keys())

        # Align importance to candidate order
        pos = {int(k): float(v) for k, v in weights_dict.items()}
        importance = [pos.get(int(e), 0.0) for e in candidate]

        pack = {
            "event_idx": eidx,
            "candidate_eidx": candidate,
            "importance_edges": importance,
            "elapsed_sec": elapsed,
        }

        if self.cfg.cache:
            self._cache[eidx] = pack

        return self._pack_result(context, pack)

    def _pack_result(self, context: ExplanationContext, pack: Dict[str, Any]) -> ExplanationResult:
        extras = {
            "event_idx": pack["event_idx"],
            "candidate_eidx": pack["candidate_eidx"],
        }
        return ExplanationResult(
            run_id=context.run_id,
            explainer=self.alias,
            context_fp=context.fingerprint(),
            importance_edges=list(pack["importance_edges"]),
            importance_nodes=None,
            importance_time=None,
            elapsed_sec=float(pack.get("elapsed_sec", 0.0)),
            extras=extras,
        )
