from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import pandas as pd

from time_to_explain.core.registry import EXTRACTORS, METRICS
from time_to_explain.data.io import load_processed_dataset
from time_to_explain.metrics import ensure_builtin_metrics_loaded

# Ensure metric registry is populated once.
ensure_builtin_metrics_loaded()


def resolve_callable(target: str) -> Callable[..., Any]:
    """
    Import a callable from a dotted path like ``package.module:function``.
    """
    if ":" in target:
        module_name, fn_name = target.split(":", 1)
    else:
        module_name, fn_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, fn_name)


def _kwargs(spec: Mapping[str, Any] | None) -> Dict[str, Any]:
    return dict(spec.get("kwargs") or {}) if spec else {}


# --------------------------------------------------------------------------- #
# Component builders
# --------------------------------------------------------------------------- #
def build_dataset(spec: Mapping[str, Any]) -> Any:
    if not spec:
        raise ValueError("Dataset config is missing.")
    if "builder" in spec:
        builder = spec["builder"]
        if builder == "processed":
            path = spec.get("path")
            if path is None:
                raise ValueError("Processed dataset builder requires 'path'.")
            return load_processed_dataset(path)
        raise ValueError(f"Unknown dataset builder: {builder}")
    if "callable" in spec:
        fn = resolve_callable(spec["callable"])
        return fn(**_kwargs(spec))
    raise ValueError("Dataset config must specify 'builder' or 'callable'.")


def build_model(spec: Mapping[str, Any], *, dataset: Any = None) -> Any:
    if not spec:
        raise ValueError("Model config is missing.")
    if "callable" in spec:
        fn = resolve_callable(spec["callable"])
        kwargs = _kwargs(spec)
        if spec.get("pass_dataset"):
            kwargs.setdefault("dataset", dataset)
        return fn(**kwargs)
    raise ValueError("Model config must specify 'callable'.")


def build_explainers(
    specs: Sequence[Mapping[str, Any]],
    *,
    model: Any = None,
    dataset: Any = None,
) -> List[Any]:
    explainers = []
    for spec in specs or []:
        if "callable" not in spec:
            raise ValueError("Each explainer config needs a 'callable'.")
        fn = resolve_callable(spec["callable"])
        kwargs = _kwargs(spec)
        if spec.get("pass_dataset"):
            kwargs.setdefault("dataset", dataset)
        if spec.get("pass_model"):
            kwargs.setdefault("model", model)
        explainers.append(fn(**kwargs))
    return explainers


def build_extractor(spec: Mapping[str, Any] | None, *, model: Any = None, dataset: Any = None) -> Any:
    if not spec:
        return None
    if "builder" in spec:
        builder = spec["builder"]
        if builder in EXTRACTORS.keys():
            cls = EXTRACTORS.get(builder)
            kwargs = _kwargs(spec)
            kwargs.setdefault("model", model)
            if dataset is not None:
                kwargs.setdefault("events", dataset.get("interactions") if isinstance(dataset, dict) else dataset)
            return cls(**kwargs)
        raise ValueError(f"Unknown extractor builder: {builder}")
    if "callable" in spec:
        fn = resolve_callable(spec["callable"])
        kwargs = _kwargs(spec)
        if spec.get("pass_dataset"):
            kwargs.setdefault("dataset", dataset)
        if spec.get("pass_model"):
            kwargs.setdefault("model", model)
        return fn(**kwargs)
    raise ValueError("Extractor config must specify 'builder' or 'callable'.")


def build_extractor_map(
    mapping: Mapping[str, Mapping[str, Any]] | None,
    *,
    model: Any = None,
    dataset: Any = None,
) -> Dict[str, Any]:
    if not mapping:
        return {}
    return {k: build_extractor(v, model=model, dataset=dataset) for k, v in mapping.items()}


def build_metrics(specs: Any) -> Any:
    """
    Validate metric specs and return a normalized spec for EvalConfig.
    """
    if specs is None:
        return []
    if isinstance(specs, str):
        if specs not in METRICS.keys():
            raise KeyError(f"Metric '{specs}' is not registered. Import its module or register it first.")
        return [specs]
    if isinstance(specs, dict):
        normalized: Dict[str, Dict[str, Any]] = {}
        for name, cfg in specs.items():
            if name not in METRICS.keys():
                raise KeyError(f"Metric '{name}' is not registered. Import its module or register it first.")
            if cfg is None:
                normalized[str(name)] = {}
            elif isinstance(cfg, dict):
                normalized[str(name)] = dict(cfg)
            else:
                raise TypeError("Metric config must be a mapping.")
        return normalized

    normalized = []
    for spec in specs:
        if isinstance(spec, str):
            name = spec
        elif isinstance(spec, dict):
            name = spec.get("builder") or spec.get("name")
            if not name:
                raise ValueError("Metric config must contain 'builder' or 'name'.")
        else:
            raise TypeError("Metric spec must be a string or mapping.")
        if name not in METRICS.keys():
            raise KeyError(f"Metric '{name}' is not registered. Import its module or register it first.")
        normalized.append(spec)
    return normalized


# --------------------------------------------------------------------------- #
# Anchors and utilities
# --------------------------------------------------------------------------- #
def load_anchors(source: Any) -> List[Dict[str, Any]]:
    if source is None:
        raise ValueError("Anchors are required. Provide 'anchors' or 'anchors_path'.")
    if isinstance(source, list):
        return list(source)

    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Anchors file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if path.suffix.lower() in {".json", ".js"}:
        return json.loads(path.read_text())
    if path.suffix.lower() in {".csv", ".tsv"}:
        df = pd.read_csv(path)
        return df.to_dict(orient="records")

    raise ValueError(f"Unsupported anchors format: {path.suffix}")
