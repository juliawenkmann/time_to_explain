from __future__ import annotations

"""Explainer catalog and lazy builder registry."""

import importlib
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs" / "explainer"

ExplainerBuilder = Tuple[type, type]
ExplainerBuilderSpec = Tuple[str, str, str]

_TEMPME_SPEC: ExplainerBuilderSpec = (
    "I_explainer_benchmark.src.explainers.adapters.tempme_adapter",
    "TempMEAdapterConfig",
    "TempMEAdapter",
)

_BUILDER_SPECS: Dict[str, ExplainerBuilderSpec] = {
    "tempme": _TEMPME_SPEC,
    "temgx": (
        "I_explainer_benchmark.src.explainers.adapters.temgx_adapter",
        "TemGXAdapterConfig",
        "TemGXAdapter",
    ),
    "pg": (
        "I_explainer_benchmark.src.explainers.adapters.pg_adapter",
        "PGAdapterConfig",
        "PGAdapter",
    ),
    "random": (
        "I_explainer_benchmark.src.explainers.adapters.random_adapter",
        "RandomAdapterConfig",
        "RandomAdapter",
    ),
    "khop": (
        "I_explainer_benchmark.src.explainers.adapters.khop_closer_adapter",
        "KHopCloserAdapterConfig",
        "KHopCloserAdapter",
    ),
}


def _resolve_builder_from_spec(spec: ExplainerBuilderSpec) -> ExplainerBuilder:
    module_name, cfg_name, explainer_name = spec
    module = importlib.import_module(module_name)
    return getattr(module, cfg_name), getattr(module, explainer_name)


class LazyExplainerBuilderMap:
    """Mapping-like adapter that lazily imports builder classes on access."""

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in _BUILDER_SPECS

    def __getitem__(self, key: str) -> ExplainerBuilder:
        if key not in _BUILDER_SPECS:
            raise KeyError(key)
        return _resolve_builder_from_spec(_BUILDER_SPECS[key])

    def get(self, key: str, default=None):
        if key not in _BUILDER_SPECS:
            return default
        return self[key]

    def keys(self) -> List[str]:
        return list(_BUILDER_SPECS.keys())

    def items(self) -> List[Tuple[str, ExplainerBuilder]]:
        return [(name, self[name]) for name in _BUILDER_SPECS]

    def values(self) -> List[ExplainerBuilder]:
        return [self[name] for name in _BUILDER_SPECS]

    def __iter__(self) -> Iterable[str]:
        return iter(_BUILDER_SPECS.keys())

    def __len__(self) -> int:
        return len(_BUILDER_SPECS)


EXPLAINER_BUILDERS = LazyExplainerBuilderMap()


def available_explainers() -> list[str]:
    """Return registered explainer builder names (sorted)."""
    return sorted(_BUILDER_SPECS.keys())


__all__ = [
    "CONFIGS_DIR",
    "EXPLAINER_BUILDERS",
    "ExplainerBuilder",
    "REPO_ROOT",
    "available_explainers",
]
