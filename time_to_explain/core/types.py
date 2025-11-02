from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, runtime_checkable
import hashlib, json, time

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
