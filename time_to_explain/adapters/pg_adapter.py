# time_to_explain/adapters/pg_and_pbone_tg_adapter.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import time
import torch
import sys
from pathlib import Path

from time_to_explain.core.types import BaseExplainer, ExplanationContext, ExplanationResult

# Ensure vendored tgnnexplainer (under submodules) is importable as `tgnnexplainer`
_TGNN_VENDOR = Path(__file__).resolve().parents[2] / "submodules" / "explainer" / "tgnnexplainer"
if str(_TGNN_VENDOR) not in sys.path:
    sys.path.insert(0, str(_TGNN_VENDOR))

# our existing temporal explainer
from tgnnexplainer.xgraph.method.other_baselines_tg import PGExplainerExt


# ---------------------------------------------------------------------------
# PGExplainerExt adapter
# ---------------------------------------------------------------------------

@dataclass
class PGAdapterConfig:
    model_name: str                     # "tgn" | "tgat"
    dataset_name: str
    explanation_level: str = "event"
    results_dir: Optional[str] = None
    debug_mode: bool = False
    threshold_num: int = 25

    # PGExplainerExt-specific
    train_epochs: int = 50
    explainer_ckpt_dir: Optional[str] = None   # REQUIRED in practice
    reg_coefs: Optional[List[float]] = None    # REQUIRED: [size_reg, entropy_reg]
    batch_size: int = 64
    lr: float = 1e-4

    # Extras
    cache: bool = True
    alias: Optional[str] = None
    device: Optional[Union[str, torch.device]] = None


class PGAdapter(BaseExplainer):
    """
    Wraps PGExplainerExt into the unified API.

    Produces a dense importance vector over candidate events (edges).
    `importance_edges` is aligned to `candidate_eidx`, which is taken from:
      1. context.subgraph.payload['candidate_eidx'] if present, else
      2. explainer.candidate_events, else
      3. PGExplainerExt's own returned order.
    """
 
    def __init__(self, cfg: PGAdapterConfig) -> None:
        super().__init__(name="pg_explainer",
                         alias=cfg.alias or f"pg_explainer")
        self.cfg = cfg
        self._explainer: Optional[PGExplainerExt] = None
        self._events = None
        self.device = torch.device(cfg.device) if cfg.device is not None else None
        self._prepared = False
        self._cache: Dict[int, Dict[str, Any]] = {}   # event_idx -> raw result dict

    # ----- lifecycle -----
    def prepare(self, *, model: Any, dataset: Any) -> None:
        super().prepare(model=model, dataset=dataset)

        # Device inference
        if self.device is None:
            try:
                p = next(model.parameters())
                self.device = p.device
            except Exception:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Events and dataset name
        self._events = dataset["events"] if isinstance(dataset, dict) and "events" in dataset else dataset
        if isinstance(dataset, dict) and "dataset_name" in dataset:
            self.cfg.dataset_name = dataset["dataset_name"]

        # Sanity checks for PG-specific config
        if self.cfg.explainer_ckpt_dir is None:
            raise ValueError("PGExplainerExtAdapter requires cfg.explainer_ckpt_dir.")
        if self.cfg.reg_coefs is None:
            raise ValueError("PGExplainerExtAdapter requires cfg.reg_coefs "
                             "(e.g. [size_reg, entropy_reg]).")

        # Build underlying PGExplainerExt
        self._explainer = PGExplainerExt(
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
            train_epochs=self.cfg.train_epochs,
            explainer_ckpt_dir=self.cfg.explainer_ckpt_dir,
            reg_coefs=self.cfg.reg_coefs,
            batch_size=self.cfg.batch_size,
            lr=self.cfg.lr,
        )
        self._prepared = True

    # ----- one explanation -----
    def explain(self, context: ExplanationContext) -> ExplanationResult:
        assert self._prepared and self._explainer is not None, "Call .prepare() first."

        # 1) event index
        eidx = (
            context.target.get("event_idx")
            or context.target.get("index")
            or context.target.get("idx")
        )
        if eidx is None and context.subgraph and context.subgraph.payload:
            eidx = context.subgraph.payload.get("event_idx")
        if eidx is None:
            raise ValueError("PGExplainerExtAdapter expects an event index: "
                             "target['event_idx'|'index'|'idx'].")

        eidx = int(eidx)

        # 2) cache
        if self.cfg.cache and eidx in self._cache:
            raw_pack = self._cache[eidx]
        else:
            t0 = time.perf_counter()
            out_list = self._explainer(event_idxs=[eidx])    # [[keys, values]]
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

        # 3) canonical candidate order
        candidate = None
        if context.subgraph and context.subgraph.payload:
            candidate = context.subgraph.payload.get("candidate_eidx")
        if candidate is None and hasattr(self._explainer, "candidate_events"):
            candidate = list(getattr(self._explainer, "candidate_events", []))
        if not candidate:
            candidate = list(raw_pack["candidate_eidx"])

        # 4) aligned importance vector
        score_map = {int(e): float(s)
                     for e, s in zip(raw_pack["candidate_eidx"], raw_pack["scores"])}
        imp_edges = [score_map.get(e, 0.0) for e in candidate]

        extras: Dict[str, Any] = {
            "event_idx": eidx,
            "candidate_eidx": candidate,
            "raw_candidate_eidx": raw_pack["candidate_eidx"],
            "raw_scores": raw_pack["scores"],
            "elapsed_sec_pgexpl": raw_pack["elapsed_sec"],
        }

        return ExplanationResult(
            run_id=context.run_id,
            explainer=self.alias,
            context_fp=context.fingerprint(),
            importance_edges=imp_edges,
            importance_nodes=None,
            importance_time=None,
            elapsed_sec=raw_pack["elapsed_sec"],
            extras=extras,
        )

    # Optional: bulk explain convenience
    def explain_many(self, event_idxs: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Convenience wrapper returning raw PG scores (no ExplanationResult).
        """
        assert self._prepared and self._explainer is not None
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
            results[eidx] = pack
            if self.cfg.cache:
                # don't overwrite an existing elapsed_sec if we have one
                if eidx in self._cache and self._cache[eidx].get("elapsed_sec") is not None:
                    pack["elapsed_sec"] = self._cache[eidx]["elapsed_sec"]
                self._cache[eidx] = pack

        return results
