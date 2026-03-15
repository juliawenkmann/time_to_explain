from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Union

import torch

from ...core.types import BaseExplainer, ExplanationContext, ExplanationResult
from ._tgnn_dataframe_compat import install_tgnn_dataframe_compat

# Ensure vendored tgnnexplainer (under submodules) is importable as `tgnnexplainer`.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SUBMODULES_ROOT = (_REPO_ROOT / "submodules").resolve()
_MOVED_SUBMODULES_ROOT = (_REPO_ROOT / "I_explainer_benchmark" / "submodules").resolve()
if not _SUBMODULES_ROOT.exists() and _MOVED_SUBMODULES_ROOT.exists():
    _SUBMODULES_ROOT = _MOVED_SUBMODULES_ROOT

_TGNN_VENDOR = (_SUBMODULES_ROOT / "explainer" / "tgnnexplainer").resolve()
if str(_TGNN_VENDOR) not in sys.path:
    sys.path.insert(0, str(_TGNN_VENDOR))

from tgnnexplainer.xgraph.method.other_baselines_tg import PGExplainerExt

_PG_IMPL_PATH = Path(inspect.getfile(PGExplainerExt)).resolve()
if _TGNN_VENDOR not in _PG_IMPL_PATH.parents:
    raise ImportError(
        "PGExplainerExt import did not resolve to submodules/explainer/tgnnexplainer. "
        f"Resolved path: {_PG_IMPL_PATH}"
    )


@dataclass
class PGAdapterConfig:
    model_name: str  # "tgn" | "tgat"
    dataset_name: str
    explainer_name: str = "pg_explainer_tg"
    explanation_level: str = "event"
    results_dir: Optional[str] = None
    debug_mode: bool = False
    threshold_num: int = 20

    # Official PGExplainerExt hyperparameters.
    train_epochs: int = 100
    explainer_ckpt_dir: Optional[str] = None
    reg_coefs: List[float] = field(default_factory=lambda: [0.5, 0.1])
    batch_size: int = 16
    lr: float = 1e-4

    # Extras.
    cache: bool = True
    alias: Optional[str] = None
    device: Optional[Union[str, torch.device]] = None


class PGAdapter(BaseExplainer):
    """Thin wrapper over official `PGExplainerExt`."""

    def __init__(self, cfg: PGAdapterConfig) -> None:
        super().__init__(name="pg_explainer", alias=cfg.alias or "pg")
        self.cfg = cfg
        self._explainer: Optional[PGExplainerExt] = None
        self._events = None
        self.device = torch.device(cfg.device) if cfg.device is not None else None
        self._prepared = False
        self._cache: Dict[int, Dict[str, Any]] = {}

    def prepare(self, *, model: Any, dataset: Any) -> None:
        super().prepare(model=model, dataset=dataset)

        if self.device is None:
            try:
                self.device = next(model.parameters()).device
            except Exception:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._events = dataset["events"] if isinstance(dataset, dict) and "events" in dataset else dataset
        if isinstance(dataset, dict) and "dataset_name" in dataset:
            self.cfg.dataset_name = dataset["dataset_name"]
        install_tgnn_dataframe_compat(events=self._events, dataset_name=self.cfg.dataset_name)

        if self.cfg.explainer_ckpt_dir is None:
            raise ValueError("PGAdapter requires cfg.explainer_ckpt_dir.")

        self._explainer = PGExplainerExt(
            model=model,
            model_name=self.cfg.model_name,
            explainer_name=self.cfg.explainer_name,
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
            reg_coefs=list(self.cfg.reg_coefs),
            batch_size=self.cfg.batch_size,
            lr=self.cfg.lr,
        )
        self._prepared = True

    def explain(self, context: ExplanationContext) -> ExplanationResult:
        assert self._prepared and self._explainer is not None, "Call .prepare() first."

        eidx = context.target.get("event_idx") or context.target.get("index") or context.target.get("idx")
        if eidx is None and context.subgraph and context.subgraph.payload:
            eidx = context.subgraph.payload.get("event_idx")
        if eidx is None:
            raise ValueError("PGAdapter expects target['event_idx'|'index'|'idx'].")
        eidx = int(eidx)

        if self.cfg.cache and eidx in self._cache:
            raw_pack = self._cache[eidx]
        else:
            t0 = time.perf_counter()
            out_list = self._explainer(event_idxs=[eidx])  # [[keys, values]]
            elapsed = time.perf_counter() - t0
            keys, vals = out_list[0]
            raw_pack = {
                "candidate_eidx": [int(k) for k in keys],
                "scores": [float(v) for v in vals],
                "elapsed_sec": float(elapsed),
            }
            if self.cfg.cache:
                self._cache[eidx] = raw_pack

        candidate = list(raw_pack["candidate_eidx"])
        imp_edges = list(raw_pack["scores"])

        extras: Dict[str, Any] = {
            "event_idx": eidx,
            "candidate_eidx": candidate,
            "raw_candidate_eidx": candidate,
            "raw_scores": imp_edges,
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

    def explain_many(self, event_idxs: List[int]) -> Dict[int, Dict[str, Any]]:
        assert self._prepared and self._explainer is not None, "Call .prepare() first."
        out: Dict[int, Dict[str, Any]] = {}
        out_list = self._explainer(event_idxs=event_idxs)
        for eidx, (keys, vals) in zip(event_idxs, out_list):
            pack = {
                "candidate_eidx": [int(k) for k in keys],
                "scores": [float(v) for v in vals],
                "elapsed_sec": self._cache.get(int(eidx), {}).get("elapsed_sec"),
            }
            out[int(eidx)] = pack
            if self.cfg.cache:
                self._cache[int(eidx)] = pack
        return out
