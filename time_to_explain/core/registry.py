from __future__ import annotations
from typing import Any, TypeVar, Generic, Dict

T = TypeVar("T")

class Registry(Generic[T]):
    def __init__(self, name: str) -> None:
        self._name = name
        self._store: Dict[str, T] = {}

    def register(self, key: str, obj: T) -> None:
        if key in self._store:
            raise KeyError(f"{self._name} already has key {key}")
        self._store[key] = obj

    def get(self, key: str) -> T:
        return self._store[key]

    def keys(self): return list(self._store.keys())
    def items(self): return list(self._store.items())

EXPLAINERS: Registry[Any] = Registry("explainers")
METRICS: Registry[Any] = Registry("metrics")
EXTRACTORS: Registry[Any] = Registry("extractors")

def register_explainer(name: str):
    def deco(cls_or_fn):
        EXPLAINERS.register(name, cls_or_fn)
        return cls_or_fn
    return deco

def register_metric(name: str):
    def deco(fn):
        METRICS.register(name, fn)
        return fn
    return deco

def register_extractor(name: str):
    def deco(cls):
        EXTRACTORS.register(name, cls)
        return cls
    return deco
