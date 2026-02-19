# time_to_explain/adapters/tgnnexplainer_adapter.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import time
import torch
import sys
from pathlib import Path

from time_to_explain.core.types import BaseExplainer, ExplanationContext, ExplanationResult

# Ensure vendored tgnnexplainer (under submodules) is importable as `tgnnexplainer`
_TGNN_VENDOR = Path(__file__).resolve().parents[2] / "submodules" / "explainer" / "tgnnexplainer"
if str(_TGNN_VENDOR) not in sys.path:
    sys.path.insert(0, str(_TGNN_VENDOR))

# TGNNExplainer + navigators
from tgnnexplainer.xgraph.method.subgraphx_tg import SubgraphXTG as TGNNExplainer
from tgnnexplainer.xgraph.method.navigators import PGNavigator, MLPNavigator, DotProductNavigator


@dataclass
class TGNNExplainerAdapterConfig:
    model_name: str                     # "tgn" | "tgat"
    dataset_name: str
    explanation_level: str = "event"
    results_dir: Optional[str] = None
    debug_mode: bool = False
    threshold_num: int = 25
    save_results: bool = False
    mcts_saved_dir: Optional[str] = None
    load_results: bool = False
    rollout: int = 20
    min_atoms: int = 1
    c_puct: float = 10.0
    # Navigator
    use_navigator: bool = False
    navigator_type: Optional[str] = None  # "pg" | "mlp" | "dot"
    pg_positive: bool = True
    navigator_params: Dict[str, Any] = field(default_factory=dict)
    # Extras
    cache: bool = True                   # cache explanations by event_idx
    alias: Optional[str] = None
    device: Optional[Union[str, torch.device]] = None


class TGNNExplainerAdapter(BaseExplainer):
    """
    Wraps your TGNNExplainer inside the unified API.

    context.target must carry a 1-based 'event_idx' (or 'index'/'idx').
    If context.subgraph.payload['candidate_eidx'] is present, we align the
    importance vector to that order for metric comparability.
    """

    def __init__(self, cfg: TGNNExplainerAdapterConfig) -> None:
        super().__init__(name="tgnnexplainer", alias=cfg.alias or f"tgnnexplainer")
        self.cfg = cfg
        self._explainer: Optional[TGNNExplainer] = None
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

        # Optional navigator
        navigator = None
        if self.cfg.use_navigator:
            if self.cfg.navigator_type == "pg":
                navigator = PGNavigator(
                    model, self.cfg.model_name, self.alias, self.cfg.dataset_name, self._events,
                    self.cfg.explanation_level,
                    device=self.device,
                    results_dir=self.cfg.results_dir,
                    debug_mode=self.cfg.debug_mode,
                    train_epochs=self.cfg.navigator_params.get("train_epochs", 0),
                    explainer_ckpt_dir=self.cfg.navigator_params.get("explainer_ckpt_dir"),
                    reg_coefs=self.cfg.navigator_params.get("reg_coefs"),
                    batch_size=self.cfg.navigator_params.get("batch_size", 32),
                    lr=self.cfg.navigator_params.get("lr", 1e-3),
                )
            elif self.cfg.navigator_type == "mlp":
                navigator = MLPNavigator(
                    model, self.cfg.model_name, self.alias, self.cfg.dataset_name, self._events,
                    self.cfg.explanation_level,
                    device=self.device,
                    results_dir=self.cfg.results_dir,
                    debug_mode=self.cfg.debug_mode,
                    train_epochs=self.cfg.navigator_params.get("train_epochs", 0),
                    explainer_ckpt_dir=self.cfg.navigator_params.get("explainer_ckpt_dir"),
                    reg_coefs=self.cfg.navigator_params.get("reg_coefs"),
                    batch_size=self.cfg.navigator_params.get("batch_size", 32),
                    lr=self.cfg.navigator_params.get("lr", 1e-3),
                )
            elif self.cfg.navigator_type == "dot":
                navigator = DotProductNavigator(
                    model=model,
                    model_name=self.cfg.model_name,
                    device=self.device,
                    all_events=self._events,
                )
            else:
                raise ValueError(f"Unknown navigator_type={self.cfg.navigator_type!r}")

        # Build your real TGNNExplainer‑TG (matches your Hydra pipeline)
        self._explainer = TGNNExplainer(
            model=model,
            model_name=self.cfg.model_name,
            explainer_name=self.alias,
            dataset_name=self.cfg.dataset_name,
            all_events=self._events,
            explanation_level=self.cfg.explanation_level,
            device=self.device,
            results_dir=self.cfg.results_dir,
            debug_mode=self.cfg.debug_mode,
            threshold_num=self.cfg.threshold_num,
            rollout=self.cfg.rollout,
            min_atoms=self.cfg.min_atoms,
            c_puct=self.cfg.c_puct,
            save_results=self.cfg.save_results,
            mcts_saved_dir=self.cfg.mcts_saved_dir,
            load_results=self.cfg.load_results,
            navigator=navigator if self.cfg.use_navigator else None,
            navigator_type=self.cfg.navigator_type,
            pg_positive=self.cfg.pg_positive,
        )
        self._prepared = True

    # ----- one explanation -----
    def explain(self, context: ExplanationContext) -> ExplanationResult:
        assert self._prepared and self._explainer is not None, "Call .prepare() first."

        # 1) event index (1‑based)
        eidx = (
            context.target.get("event_idx")
            or context.target.get("index")
            or context.target.get("idx")
        )
        if eidx is None and context.subgraph and context.subgraph.payload:
            eidx = context.subgraph.payload.get("event_idx")
        if eidx is None:
            raise ValueError("TGNNExplainerAdapter expects an event index: target['event_idx'|'index'|'idx'].")

        eidx = int(eidx)

        # 2) cache
        if self.cfg.cache and eidx in self._cache:
            raw_pack = self._cache[eidx]
        else:
            t0 = time.perf_counter()
            try:
                out_list = self._explainer(event_idxs=[eidx])               # typical path
                tree_nodes, best_node = out_list[0]
            except TypeError:
                # multiprocessing style with return_dict
                from multiprocessing import Manager
                rd = Manager().dict()
                _ = self._explainer(event_idxs=[eidx], return_dict=rd)
                tree_nodes, best_node = rd[eidx]
            elapsed = time.perf_counter() - t0

            raw_pack = {
                "tree_nodes": tree_nodes,
                "best_node": best_node,
                "elapsed_sec": elapsed,
            }
            # Pre-compute simple, serializable tree stats for downstream metrics
            candidate = context.subgraph.payload.get("candidate_eidx") if context.subgraph else None
            cand_len = len(candidate) if candidate else len(getattr(self._explainer, "candidate_events", []) or [])
            if cand_len:
                sparsity = []
                rewards = []
                for n in tree_nodes:
                    sparsity.append(len(getattr(n, "coalition", [])) / cand_len)
                    rewards.append(float(getattr(n, "P", 0.0)))
                raw_pack["tree_nodes_sparsity"] = sparsity
                raw_pack["tree_nodes_reward"] = rewards
            if self.cfg.cache:
                self._cache[eidx] = raw_pack

        # 3) stable candidate order (for vector metrics)
        candidate = None
        if context.subgraph and context.subgraph.payload:
            candidate = context.subgraph.payload.get("candidate_eidx")
        if candidate is None and hasattr(self._explainer, "candidate_events"):
            candidate = list(getattr(self._explainer, "candidate_events", []))

        # 4) map coalition -> importance vector aligned to `candidate_eidx`
        imp_edges: Optional[List[float]] = None
        coalition = list(getattr(raw_pack["best_node"], "coalition", [])) if raw_pack["best_node"] is not None else []
        if candidate:
            pos = {e: i for i, e in enumerate(candidate)}
            vec = [0.0] * len(candidate)
            for e in coalition:
                if e in pos: vec[pos[e]] = 1.0
            imp_edges = vec

        extras: Dict[str, Any] = {
            "event_idx": eidx,
            "candidate_eidx": candidate,
            "coalition_eidx": coalition,
            "best_node_score": float(getattr(raw_pack["best_node"], "P", 0.0)) if raw_pack["best_node"] is not None else 0.0,
            "elapsed_sec_tgnnexplainer": raw_pack["elapsed_sec"],
            "mcts_tree_nodes_sparsity": raw_pack.get("tree_nodes_sparsity"),
            "mcts_tree_nodes_reward": raw_pack.get("tree_nodes_reward"),
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

    # Optional convenience for bulk explain (e.g., precompute cache)
    def explain_many(self, event_idxs: List[int]) -> Dict[int, Dict[str, Any]]:
        assert self._prepared and self._explainer is not None
        results: Dict[int, Dict[str, Any]] = {}
        out_list = self._explainer(event_idxs=event_idxs)
        for eidx, out in zip(event_idxs, out_list):
            tree_nodes, best_node = out
            results[eidx] = {"tree_nodes": tree_nodes, "best_node": best_node}
            if self.cfg.cache:
                self._cache[eidx] = results[eidx]
        return results
