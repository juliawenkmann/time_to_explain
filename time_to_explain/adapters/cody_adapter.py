from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import sys

import numpy as np
import pandas as pd
import torch

from time_to_explain.core.types import BaseExplainer, ExplanationContext, ExplanationResult

_TEMGX_VENDOR = Path(__file__).resolve().parents[2] / "submodules" / "explainer" / "TemGX" / "link"
if str(_TEMGX_VENDOR) not in sys.path:
    sys.path.insert(0, str(_TEMGX_VENDOR))

from temgxlib.data import ContinuousTimeDynamicGraphDataset
from time_to_explain.models.adapter.wrapper import TGNNWrapper

# Ensure vendored CoDy (under submodules) is importable as `cody`
_CODY_VENDOR = Path(__file__).resolve().parents[2] / "submodules" / "explainer" / "CoDy"
if str(_CODY_VENDOR) not in sys.path:
    sys.path.insert(0, str(_CODY_VENDOR))
if str(_CODY_VENDOR / "cody") not in sys.path:
    sys.path.insert(0, str(_CODY_VENDOR / "cody"))

from cody.explainer.cody import CoDy


@dataclass
class CoDyAdapterConfig:
    selection_policy: str = "temporal"
    candidates_size: int = 75
    max_steps: int = 200
    approximate_predictions: bool = True
    alpha: float = 2 / 3
    verbose: bool = False
    num_hops: Optional[int] = None
    batch_size: int = 256
    alias: Optional[str] = None
    device: Optional[Union[str, torch.device]] = None


def _column_or_fallback(df: pd.DataFrame, primary: str, fallback_idx: int) -> np.ndarray:
    if primary in df.columns:
        return df[primary].to_numpy()
    return df.iloc[:, fallback_idx].to_numpy()


def _normalize_events(events: pd.DataFrame) -> pd.DataFrame:
    if {"user_id", "item_id", "timestamp", "state_label", "idx"}.issubset(events.columns):
        return events.copy()

    u = _column_or_fallback(events, "u", 0)
    i = _column_or_fallback(events, "i", 1)
    ts = _column_or_fallback(events, "ts", 2)
    if "label" in events.columns:
        label = events["label"].to_numpy()
    else:
        label = np.zeros(len(events), dtype=float)

    return pd.DataFrame(
        {
            "user_id": u.astype(int),
            "item_id": i.astype(int),
            "timestamp": ts.astype(float),
            "state_label": label.astype(float),
            "idx": np.arange(len(events), dtype=int),
        }
    )


def _resolve_features(
    model: torch.nn.Module,
    dataset: Any,
) -> tuple[np.ndarray, np.ndarray]:
    edge_feats = None
    node_feats = None
    if isinstance(dataset, dict):
        edge_feats = dataset.get("edge_features")
        node_feats = dataset.get("node_features")

    if edge_feats is None:
        edge_feats = getattr(model, "edge_raw_features", None)
    if node_feats is None:
        node_feats = getattr(model, "node_raw_features", None)

    if isinstance(edge_feats, torch.Tensor):
        edge_feats = edge_feats.detach().cpu().numpy()
    if isinstance(node_feats, torch.Tensor):
        node_feats = node_feats.detach().cpu().numpy()

    if edge_feats is None or node_feats is None:
        raise ValueError("CoDyAdapter requires edge and node features from dataset or model.")

    return np.asarray(edge_feats), np.asarray(node_feats)


class CoDyAdapter(BaseExplainer):
    def __init__(self, cfg: CoDyAdapterConfig) -> None:
        super().__init__(name="cody", alias=cfg.alias or "cody")
        self.cfg = cfg
        self._explainer: Optional[CoDy] = None
        self._wrapper: Optional[TGNNWrapper] = None
        self._event_id_offset: int = 0

    def prepare(self, *, model: Any, dataset: Any) -> None:
        super().prepare(model=model, dataset=dataset)

        events = dataset["events"] if isinstance(dataset, dict) and "events" in dataset else dataset
        if not isinstance(events, pd.DataFrame):
            raise ValueError("CoDyAdapter expects a pandas DataFrame for events.")

        edge_feats, node_feats = _resolve_features(model, dataset)
        cody_events = _normalize_events(events)
        dataset_name = ""
        if isinstance(dataset, dict):
            dataset_name = str(dataset.get("dataset_name") or "")

        event_id_offset = 0
        if "e_idx" in events.columns:
            min_eidx = int(events["e_idx"].min())
            event_id_offset = min_eidx if min_eidx >= 0 else 0
        elif "idx" in events.columns:
            min_idx = int(events["idx"].min())
            event_id_offset = min_idx if min_idx >= 0 else 0

        if edge_feats.shape[0] == len(events) + event_id_offset and event_id_offset > 0:
            edge_feats = edge_feats[event_id_offset:]

        cody_dataset = ContinuousTimeDynamicGraphDataset(
            cody_events,
            edge_feats,
            node_feats,
            dataset_name or "cody_dataset",
            directed=False,
            bipartite=False,
        )

        num_hops = self.cfg.num_hops
        if num_hops is None:
            num_hops = int(getattr(model, "num_layers", 2))

        model_event_ids = None
        if "e_idx" in events.columns:
            model_event_ids = events["e_idx"].to_numpy(dtype=int)
        elif "idx" in events.columns:
            model_event_ids = events["idx"].to_numpy(dtype=int)
        else:
            model_event_ids = np.arange(1, len(events) + 1, dtype=int)

        device = self.cfg.device
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        wrapper = TGNNWrapper(
            model=model,
            dataset=cody_dataset,
            num_hops=num_hops,
            model_name=getattr(model, "__class__", type(model)).__name__,
            device=device,
            batch_size=self.cfg.batch_size,
            model_event_ids=model_event_ids,
        )
        self._event_id_offset = wrapper.event_id_offset
        self._wrapper = wrapper
        self._explainer = CoDy(
            tgnn_wrapper=wrapper,
            selection_policy=self.cfg.selection_policy,
            candidates_size=self.cfg.candidates_size,
            max_steps=self.cfg.max_steps,
            approximate_predictions=self.cfg.approximate_predictions,
            alpha=self.cfg.alpha,
            verbose=self.cfg.verbose,
        )

    def _resolve_candidate_eidx(self, internal_event_id: int) -> Optional[list[int]]:
        if self._explainer is None or self._wrapper is None:
            return None
        try:
            subgraph = self._explainer.subgraph_generator.get_fixed_size_k_hop_temporal_subgraph(
                num_hops=self._explainer.num_hops,
                base_event_id=internal_event_id,
                size=self._explainer.candidates_size,
            )
        except Exception:
            return None

        if subgraph is None or len(subgraph) == 0 or "idx" not in subgraph.columns:
            return None

        candidate_ids = subgraph["idx"].to_numpy(dtype=int)
        if candidate_ids.size == 0:
            return None

        try:
            model_ids = self._wrapper._to_model_event_ids(candidate_ids)
        except Exception:
            model_ids = candidate_ids + int(self._event_id_offset)

        return [int(e) for e in model_ids.tolist()]

    def explain(self, context: ExplanationContext) -> ExplanationResult:
        if self._explainer is None or self._wrapper is None:
            raise RuntimeError("CoDyAdapter not prepared. Call prepare() first.")

        eidx = (
            context.target.get("event_idx")
            or context.target.get("index")
            or context.target.get("idx")
        )
        if eidx is None and context.subgraph and context.subgraph.payload:
            eidx = context.subgraph.payload.get("event_idx")
        if eidx is None:
            raise ValueError("CoDyAdapter expects an event index in target or subgraph payload.")

        eidx = int(eidx)
        internal_event_id = eidx - self._event_id_offset
        if internal_event_id < 0:
            raise ValueError(f"Invalid event index {eidx} for CoDyAdapter.")

        t0 = self._tic()
        cf_example = self._explainer.explain(internal_event_id)
        elapsed = self._toc(t0)

        cf_events = np.asarray(cf_example.event_ids, dtype=int)
        raw_importances = cf_example.event_importances
        if raw_importances is None:
            raw_importances = []
        if len(raw_importances) == 0 or raw_importances[0] is None:
            cf_importances = np.ones(len(cf_events), dtype=float)
        else:
            if hasattr(cf_example, "get_absolute_importances"):
                cf_importances = np.asarray(cf_example.get_absolute_importances(), dtype=float)
            else:
                cf_importances = np.asarray(raw_importances, dtype=float)
        if len(cf_events) != len(cf_importances):
            cf_importances = np.ones(len(cf_events), dtype=float)

        cf_events_model = (cf_events + self._event_id_offset).astype(int)

        payload_candidate = None
        if context.subgraph and context.subgraph.payload:
            raw_candidate = context.subgraph.payload.get("candidate_eidx")
            if raw_candidate is not None:
                payload_candidate = [int(e) for e in raw_candidate]

        cody_candidate = self._resolve_candidate_eidx(internal_event_id)

        if cody_candidate:
            candidate = cody_candidate
        elif payload_candidate is not None:
            candidate = payload_candidate
        else:
            candidate = cf_events_model.tolist()

        if context.subgraph is not None:
            if context.subgraph.payload is None:
                context.subgraph.payload = {}
            if isinstance(context.subgraph.payload, dict):
                context.subgraph.payload["candidate_eidx"] = list(candidate)

        score_map = {int(e): float(s) for e, s in zip(cf_events_model, cf_importances)}
        importance_edges = [score_map.get(int(e), 0.0) for e in candidate]

        extras: Dict[str, Any] = {
            "event_idx": eidx,
            "candidate_eidx": candidate,
            "cf_event_ids": cf_events_model.tolist(),
            "cf_event_ids_raw": cf_events.tolist(),
            "cf_event_importances": cf_importances.tolist(),
            "original_prediction": cf_example.original_prediction,
            "counterfactual_prediction": cf_example.counterfactual_prediction,
            "achieves_counterfactual": cf_example.achieves_counterfactual_explanation,
            "elapsed_sec_cody": elapsed,
        }
        if payload_candidate is not None and cody_candidate is not None:
            if payload_candidate != list(candidate):
                extras["candidate_eidx_extractor"] = payload_candidate

        return ExplanationResult(
            run_id=context.run_id,
            explainer=self.alias,
            context_fp=context.fingerprint(),
            importance_edges=importance_edges,
            importance_nodes=None,
            importance_time=None,
            elapsed_sec=elapsed,
            extras=extras,
        )
