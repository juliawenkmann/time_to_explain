from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from time_to_explain.core.runner import EvalConfig, EvaluationRunner

from .builders import (
    build_dataset,
    build_explainers,
    build_extractor,
    build_extractor_map,
    build_metrics,
    build_model,
    load_anchors,
)
from .config import load_yaml


def _build_eval_config(experiment_cfg: Dict[str, Any]) -> EvalConfig:
    return EvalConfig(
        out_dir=experiment_cfg.get("output_dir", "resources/results"),
        seed=int(experiment_cfg.get("seed", 42)),
        metrics=experiment_cfg.get("metrics", ["sparsity", "fidelity_minus", "fidelity_plus", "aufsc"]),
        save_jsonl=bool(experiment_cfg.get("save_jsonl", True)),
        save_csv=bool(experiment_cfg.get("save_csv", True)),
        compute_metrics=bool(experiment_cfg.get("compute_metrics", True)),
        resume=bool(experiment_cfg.get("resume", False)),
        overwrite_explainers=experiment_cfg.get("overwrite_explainers"),
    )


def _normalize_dataset_bundle(dataset: Any) -> Any:
    if not isinstance(dataset, dict):
        return dataset
    if "events" not in dataset and "interactions" in dataset:
        dataset["events"] = dataset["interactions"]
    if "dataset_name" not in dataset:
        metadata = dataset.get("metadata")
        if isinstance(metadata, dict):
            dataset_name = metadata.get("dataset_name") or metadata.get("dataset_alias")
            if dataset_name:
                dataset["dataset_name"] = dataset_name
    return dataset


def _normalize_explainer_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",")]
    else:
        try:
            items = [str(v).strip() for v in value]
        except TypeError:
            items = [str(value).strip()]
    items = [v for v in items if v]
    return items or None


def _explainer_key(explainer: Any) -> str:
    return str(getattr(explainer, "alias", None) or getattr(explainer, "name", None) or explainer.__class__.__name__)


def _filter_explainers(
    explainers: Sequence[Any],
    *,
    include: Any = None,
    exclude: Any = None,
) -> List[Any]:
    include_list = _normalize_explainer_list(include)
    exclude_list = _normalize_explainer_list(exclude)
    if not include_list and not exclude_list:
        return list(explainers)

    include_set = set(include_list or [])
    exclude_set = set(exclude_list or [])
    out: List[Any] = []
    for expl in explainers:
        key = _explainer_key(expl)
        if include_set and key not in include_set:
            continue
        if exclude_set and key in exclude_set:
            continue
        out.append(expl)
    if include_set and not out:
        raise ValueError(f"No explainers matched include filter: {sorted(include_set)}")
    return out


def run_eval(
    *,
    dataset: Any,
    model: Any,
    explainers: Sequence[Any],
    anchors: Sequence[Dict[str, Any]],
    extractor: Any = None,
    extractor_map: Optional[Dict[str, Any]] = None,
    experiment_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run EvaluationRunner given already-constructed objects (notebook-friendly).
    """
    dataset = _normalize_dataset_bundle(dataset)
    experiment_cfg = dict(experiment_cfg or {})
    eval_cfg = _build_eval_config(experiment_cfg)
    explainers = _filter_explainers(
        explainers,
        include=experiment_cfg.get("only_explainers") or experiment_cfg.get("include_explainers"),
        exclude=experiment_cfg.get("exclude_explainers"),
    )

    runner = EvaluationRunner(
        model=model,
        dataset=dataset,
        extractor=extractor,
        extractor_map=extractor_map,
        explainers=explainers,
        config=eval_cfg,
    )

    k_hop = experiment_cfg.get("k_hop")
    if k_hop is None:
        k_hop = getattr(model, "num_layers", 5) or 5
    num_neighbors = experiment_cfg.get("num_neighbors")
    if num_neighbors is None:
        num_neighbors = getattr(model, "num_neighbors", 20) or 20

    return runner.run(
        anchors,
        k_hop=k_hop,
        num_neighbors=num_neighbors,
        window=experiment_cfg.get("window"),
        run_id=experiment_cfg.get("run_id"),
    )


def run_from_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Build dataset/model/explainers/extractors from a YAML config and run eval.
    """
    cfg = load_yaml(config_path)
    exp_cfg = dict(cfg.get("experiment") or {})

    # Components
    dataset = _normalize_dataset_bundle(build_dataset(cfg["dataset"]))
    model = build_model(cfg["model"], dataset=dataset)
    explainers = build_explainers(cfg.get("explainers", []), model=model, dataset=dataset)
    extractor = build_extractor(cfg.get("extractor"), model=model, dataset=dataset)
    extractor_map = build_extractor_map(cfg.get("extractor_map"), model=model, dataset=dataset)

    # Metrics: validate and preserve config (dict or list of specs).
    if "metrics" in cfg:
        exp_cfg["metrics"] = build_metrics(cfg.get("metrics"))
    elif "metrics" in exp_cfg:
        exp_cfg["metrics"] = build_metrics(exp_cfg.get("metrics"))

    anchors_source = (
        exp_cfg.get("anchors")
        or exp_cfg.get("anchors_path")
        or cfg.get("anchors")
        or cfg.get("anchors_path")
    )
    anchors = load_anchors(anchors_source)

    return run_eval(
        dataset=dataset,
        model=model,
        explainers=explainers,
        extractor=extractor,
        extractor_map=extractor_map,
        anchors=anchors,
        experiment_cfg=exp_cfg,
    )


def sweep_from_glob(pattern: str) -> List[Dict[str, Any]]:
    """
    Run multiple configs matched by a glob pattern; returns list of results.
    """
    results = []
    for path in sorted(glob.glob(pattern)):
        results.append({"config": path, "result": run_from_config(path)})
    return results
