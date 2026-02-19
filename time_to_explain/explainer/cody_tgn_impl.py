# time_to_explain/explainer/cody_tgn_impl.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import time

import numpy as np
import torch

from time_to_explain.core.types import BaseExplainer, ExplanationContext, ExplanationResult

from submodules.cody_tgn_explainer import (
    CoDyExplainer,
    GreeDyExplainer,
    TemporalGraphIndex,
    TGNLinkPredictor,
)


@dataclass
class CoDyTGNImplAdapterConfig:
    # Required
    model_name: str
    dataset_name: str

    # Basic options
    explanation_level: str = "event"
    results_dir: Optional[str] = None
    debug_mode: bool = False
    alias: Optional[str] = None
    device: Optional[Union[str, torch.device]] = None
    seed: int = 0

    # Candidate extraction + MCTS params
    k_hops: Optional[int] = None
    m_max: int = 75
    it_max: int = 200
    alpha: float = 2 / 3
    policy: str = "temporal"
    per_node_cap: Optional[int] = None
    assume_undirected: bool = True

    # Greedy baseline (optional)
    use_greedy: bool = False
    greedy_l: int = 3
    greedy_it_max: int = 50

    # TGN scoring wrapper
    extra_neighbors: int = 50
    assume_model_returns_prob: Optional[bool] = None

    # Candidate alignment
    use_payload_candidates: bool = False

    # Cache per-event
    cache: bool = True


class CoDyTGNImplAdapter(BaseExplainer):
    """
    Adapter for submodules/cody_tgn_explainer.py (MCTS/Greedy CoDy for TGN).

    It builds a TemporalGraphIndex from the dataset events and uses the provided
    TGNLinkPredictor wrapper to score omitted-edge counterfactuals.
    """

    def __init__(self, cfg: CoDyTGNImplAdapterConfig) -> None:
        super().__init__(name="cody_tgn_impl", alias=cfg.alias or "cody_tgn_impl")
        self.cfg = cfg
        self.device: Optional[torch.device] = (
            torch.device(cfg.device) if cfg.device is not None else None
        )

        self._model: Any = None
        self._dataset: Any = None
        self._events: Any = None
        self._events_df: Any = None
        self._graph_index: Optional[TemporalGraphIndex] = None
        self._k_hops: int = 2
        self._m_max: int = 75
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._prepared: bool = False

    def prepare(self, *, model: Any, dataset: Any) -> None:
        super().prepare(model=model, dataset=dataset)
        self._model = model
        self._dataset = dataset

        if isinstance(dataset, dict):
            if "dataset_name" in dataset:
                self.cfg.dataset_name = dataset["dataset_name"]
            self._events = dataset.get("events")
        else:
            self._events = dataset

        if self._events is None:
            raise ValueError("CoDyTGNImplAdapter expects `dataset['events']` or a dataframe of events.")

        self._events_df = None
        try:
            import pandas as pd  # type: ignore

            if isinstance(self._events, pd.DataFrame):
                self._events_df = self._events
        except Exception:
            self._events_df = None

        if self.device is None:
            try:
                self.device = next(model.parameters()).device
            except Exception:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        sources, destinations, timestamps, edge_idxs = self._extract_event_arrays(self._events)
        self._graph_index = TemporalGraphIndex(
            sources=sources,
            destinations=destinations,
            timestamps=timestamps,
            edge_idxs=edge_idxs,
            assume_undirected=self.cfg.assume_undirected,
        )

        self._k_hops = int(self.cfg.k_hops) if self.cfg.k_hops is not None else int(
            getattr(model, "num_layers", 2) or 2
        )
        self._m_max = int(self.cfg.m_max)

        self._prepared = True

    def explain(self, context: ExplanationContext) -> ExplanationResult:
        if not self._prepared or self._graph_index is None:
            raise RuntimeError("CoDyTGNImplAdapter not prepared. Call prepare() first.")

        eidx = (
            context.target.get("event_idx")
            or context.target.get("index")
            or context.target.get("idx")
        )
        if eidx is None and context.subgraph and context.subgraph.payload:
            eidx = context.subgraph.payload.get("event_idx")
        if eidx is None:
            raise ValueError("CoDyTGNImplAdapter expects an event index in target or subgraph payload.")
        eidx = int(eidx)

        if self.cfg.cache and eidx in self._cache:
            pack = self._cache[eidx]
            return self._pack_to_result(context, pack)

        src, dst, ts = self._get_event_triplet(eidx)

        candidates, _ = self._graph_index.k_hop_candidates(
            src,
            dst,
            ts,
            k_hops=self._k_hops,
            m_max=self._m_max,
            per_node_cap=self.cfg.per_node_cap,
        )

        num_nodes = self._infer_num_nodes()
        predictor = TGNLinkPredictor(
            model=self._model,
            explained_src=src,
            explained_dst=dst,
            explained_ts=ts,
            explained_edge_idx=eidx,
            n_neighbors=int(getattr(self._model, "num_neighbors", 20) or 20),
            num_nodes=num_nodes,
            device=self.device,
            memory_snapshot=None,
            assume_model_returns_prob=self.cfg.assume_model_returns_prob,
            extra_neighbors=self.cfg.extra_neighbors,
        )

        score_fn = predictor.score

        if self.cfg.use_greedy:
            explainer = GreeDyExplainer(
                score_fn=score_fn,
                graph_index=self._graph_index,
                k_hops=self._k_hops,
                m_max=self._m_max,
                l=self.cfg.greedy_l,
                it_max=self.cfg.greedy_it_max,
                policy=self.cfg.policy,
                seed=self.cfg.seed,
            )
        else:
            explainer = CoDyExplainer(
                score_fn=score_fn,
                graph_index=self._graph_index,
                k_hops=self._k_hops,
                m_max=self._m_max,
                it_max=self.cfg.it_max,
                alpha=self.cfg.alpha,
                policy=self.cfg.policy,
                seed=self.cfg.seed,
            )

        t0 = time.perf_counter()
        result = explainer.explain(src, dst, ts, edge_idx=eidx, per_node_cap=self.cfg.per_node_cap)
        elapsed = time.perf_counter() - t0

        omitted = set(int(e) for e in result.omitted_edge_idxs)

        payload_candidate = None
        if context.subgraph and context.subgraph.payload:
            raw_candidate = context.subgraph.payload.get("candidate_eidx")
            if raw_candidate is not None:
                payload_candidate = [int(e) for e in raw_candidate]

        candidate = list(candidates) if candidates else sorted(omitted)
        if self.cfg.use_payload_candidates and payload_candidate:
            if not candidate:
                candidate = payload_candidate
            elif set(payload_candidate) == {int(e) for e in candidate}:
                candidate = payload_candidate

        if context.subgraph is not None:
            if context.subgraph.payload is None:
                context.subgraph.payload = {}
            if isinstance(context.subgraph.payload, dict):
                context.subgraph.payload["candidate_eidx"] = list(candidate)

        importance_edges = [1.0 if int(e) in omitted else 0.0 for e in candidate]

        pack = {
            "event_idx": eidx,
            "candidate_eidx": list(candidate),
            "importance_edges": importance_edges,
            "omitted_edge_idxs": sorted(omitted),
            "original_score": result.original_score,
            "perturbed_score": result.perturbed_score,
            "is_counterfactual": result.is_counterfactual,
            "iterations_run": result.iterations_run,
            "best_score": result.best_score,
            "policy": result.policy,
            "candidate_size": result.candidate_size,
            "elapsed_sec": elapsed,
            "use_greedy": self.cfg.use_greedy,
        }

        if self.cfg.cache:
            self._cache[eidx] = pack

        return self._pack_to_result(context, pack)

    def _pack_to_result(self, context: ExplanationContext, pack: Dict[str, Any]) -> ExplanationResult:
        extras = {
            "event_idx": pack.get("event_idx"),
            "candidate_eidx": list(pack.get("candidate_eidx", [])),
            "omitted_edge_idxs": list(pack.get("omitted_edge_idxs", [])),
            "original_score": pack.get("original_score"),
            "perturbed_score": pack.get("perturbed_score"),
            "is_counterfactual": pack.get("is_counterfactual"),
            "iterations_run": pack.get("iterations_run"),
            "best_score": pack.get("best_score"),
            "policy": pack.get("policy"),
            "candidate_size": pack.get("candidate_size"),
            "use_greedy": pack.get("use_greedy"),
        }
        return ExplanationResult(
            run_id=context.run_id,
            explainer=self.alias,
            context_fp=context.fingerprint(),
            importance_edges=pack.get("importance_edges"),
            importance_nodes=None,
            importance_time=None,
            elapsed_sec=pack.get("elapsed_sec", 0.0),
            extras=extras,
        )

    def _get_event_triplet(self, eidx: int) -> Tuple[int, int, float]:
        if self._events_df is not None:
            row = self._events_df.iloc[eidx - 1]
            u = int(row["u"] if "u" in self._events_df.columns else row.iloc[0])
            v = int(row["i"] if "i" in self._events_df.columns else row.iloc[1])
            ts = float(row["ts"] if "ts" in self._events_df.columns else row.iloc[2])
            return u, v, ts

        ev = self._events[eidx - 1]
        if isinstance(ev, dict):
            u = ev.get("u", ev.get("src", ev.get("source")))
            v = ev.get("i", ev.get("dst", ev.get("dest")))
            ts = ev.get("ts", ev.get("time"))
            return int(u), int(v), float(ts)

        if hasattr(ev, "u") and hasattr(ev, "i") and hasattr(ev, "ts"):
            return int(ev.u), int(ev.i), float(ev.ts)

        try:
            u, v, ts = ev[:3]
            return int(u), int(v), float(ts)
        except Exception as exc:
            raise ValueError("Unable to parse event triplet from dataset.") from exc

    def _extract_event_arrays(self, events: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        try:
            import pandas as pd  # type: ignore

            if isinstance(events, pd.DataFrame):
                df = events
                src = df["u"].to_numpy() if "u" in df.columns else df.iloc[:, 0].to_numpy()
                dst = df["i"].to_numpy() if "i" in df.columns else df.iloc[:, 1].to_numpy()
                ts = df["ts"].to_numpy() if "ts" in df.columns else df.iloc[:, 2].to_numpy()
                if "e_idx" in df.columns:
                    eidx = df["e_idx"].to_numpy()
                elif "idx" in df.columns:
                    eidx = df["idx"].to_numpy()
                else:
                    eidx = np.arange(1, len(df) + 1, dtype=np.int64)
                return (
                    src.astype(np.int64),
                    dst.astype(np.int64),
                    ts.astype(np.float64),
                    eidx.astype(np.int64),
                )
        except Exception:
            pass

        seq = list(events)
        src_list: List[int] = []
        dst_list: List[int] = []
        ts_list: List[float] = []
        for ev in seq:
            if isinstance(ev, dict):
                src = ev.get("u", ev.get("src", ev.get("source")))
                dst = ev.get("i", ev.get("dst", ev.get("dest")))
                ts = ev.get("ts", ev.get("time"))
            elif hasattr(ev, "u") and hasattr(ev, "i") and hasattr(ev, "ts"):
                src, dst, ts = ev.u, ev.i, ev.ts
            else:
                src, dst, ts = ev[:3]
            src_list.append(int(src))
            dst_list.append(int(dst))
            ts_list.append(float(ts))

        eidx = np.arange(1, len(src_list) + 1, dtype=np.int64)
        return (
            np.asarray(src_list, dtype=np.int64),
            np.asarray(dst_list, dtype=np.int64),
            np.asarray(ts_list, dtype=np.float64),
            eidx,
        )

    def _infer_num_nodes(self) -> int:
        if self._events_df is not None:
            if "u" in self._events_df.columns and "i" in self._events_df.columns:
                max_node = int(max(self._events_df["u"].max(), self._events_df["i"].max()))
                return max_node + 1
        try:
            n_nodes = int(getattr(self._model, "n_nodes"))
            if n_nodes > 0:
                return n_nodes
        except Exception:
            pass
        sources, destinations, _, _ = self._extract_event_arrays(self._events)
        if sources.size == 0 and destinations.size == 0:
            return 0
        max_src = int(sources.max()) if sources.size else 0
        max_dst = int(destinations.max()) if destinations.size else 0
        return max(max_src, max_dst) + 1
