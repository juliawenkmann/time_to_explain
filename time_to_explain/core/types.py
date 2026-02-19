from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, runtime_checkable
import hashlib, json, time
from typing import Any, Dict, Optional, TypedDict
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

TargetKind = Literal["edge", "node", "graph"]

def _hash_dict(d: Dict[str, Any]) -> str:
    s = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(s).hexdigest()

@dataclass
class Subgraph:
    node_ids: List[int]
    edge_index: List[Tuple[int,int]]
    timestamps: Optional[List[float]] = None
    node_features: Optional[Any] = None
    edge_features: Optional[Any] = None
    payload: Optional[Any] = None  # plug in DGL/PyG/etc.

@dataclass
class ExplanationContext:
    run_id: str
    target_kind: TargetKind
    target: Dict[str, Any]  # e.g. {"u": int, "i": int, "ts": float, "edge_id": int}
    window: Optional[Tuple[float,float]] = None
    k_hop: int = 1
    num_neighbors: int = 50
    subgraph: Optional[Subgraph] = None
    label: Optional[Any] = None
    prediction: Optional[Any] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        base = dict(
            target_kind=self.target_kind, target=self.target, window=self.window,
            k_hop=self.k_hop, num_neighbors=self.num_neighbors,
            meta=self.meta,
        )
        return _hash_dict(base)

@dataclass
class ExplanationResult:
    run_id: str
    explainer: str
    context_fp: str
    importance_edges: Optional[List[float]] = None
    importance_nodes: Optional[List[float]] = None
    importance_time: Optional[List[float]] = None
    elapsed_sec: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)

@runtime_checkable
class ModelProtocol(Protocol):
    def predict_proba(self, subgraph: Subgraph, target: Dict[str, Any]) -> Any: ...
    def predict_proba_with_mask(self, subgraph: Subgraph, target: Dict[str, Any],
                                edge_mask: Optional[List[float]] = None,
                                node_mask: Optional[List[float]] = None) -> Any: ...

@runtime_checkable
class SubgraphExtractorProtocol(Protocol):
    name: str
    def extract(self, dataset: Any, anchor: Dict[str, Any], *, k_hop: int,
                num_neighbors: int, window: Optional[Tuple[float,float]] = None) -> Subgraph: ...

class BaseExplainer:
    def __init__(self, *, name: str, alias: Optional[str] = None) -> None:
        self.name = name
        self.alias = alias or name

    def prepare(self, *, model: ModelProtocol, dataset: Any) -> None:
        self._model = model
        self._dataset = dataset

    def explain(self, context: ExplanationContext) -> ExplanationResult:
        raise NotImplementedError

    def _tic(self) -> float: return time.perf_counter()
    def _toc(self, t0: float) -> float: return time.perf_counter() - t0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alias={self.alias!r}, name={self.name!r})"


# --- add this to time_to_explain/core/types.py ---
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class MetricResult:
    """
    Lightweight container for metric outputs.

    name:        metric name, e.g. "sparsity" or "fidelity"
    values:      flat mapping of metric components (e.g. {"sparsity_edges": 0.83})
    direction:   optional hint for dashboards ("higher-is-better" or "lower-is-better")
    run_id:      optional evaluation run id
    explainer:   optional explainer alias/name the metric was computed for
    context_fp:  optional ExplanationContext fingerprint
    extras:      free-form payload (confidence intervals, seeds, timing, etc.)
    """
    name: str
    values: Dict[str, float]
    direction: Optional[str] = None
    run_id: Optional[str] = None
    explainer: Optional[str] = None
    context_fp: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


class DatasetBundle(TypedDict):
    interactions: pd.DataFrame   # columns: u, i, ts, label (optional etype)
    node_features: Optional[np.ndarray]  # shape [num_nodes, d] or None
    edge_features: Optional[np.ndarray]  # per-interaction (or per-edge) shape
    metadata: Dict[str, Any]

class DatasetRecipe(ABC):
    RECIPE_NAME = "base"

    def __init__(self, **config):
        self.config = config

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return {}

    @abstractmethod
    def generate(self) -> DatasetBundle:
        ...