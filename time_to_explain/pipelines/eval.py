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
        out_dir=experiment_cfg.get("output_dir", "runs"),
        seed=int(experiment_cfg.get("seed", 42)),
        metrics=experiment_cfg.get("metrics", ["sparsity"]),
        save_jsonl=bool(experiment_cfg.get("save_jsonl", True)),
        save_csv=bool(experiment_cfg.get("save_csv", True)),
    )


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
    experiment_cfg = dict(experiment_cfg or {})
    eval_cfg = _build_eval_config(experiment_cfg)

    runner = EvaluationRunner(
        model=model,
        dataset=dataset,
        extractor=extractor,
        extractor_map=extractor_map,
        explainers=explainers,
        config=eval_cfg,
    )

    return runner.run(
        anchors,
        k_hop=experiment_cfg.get("k_hop", 2),
        num_neighbors=experiment_cfg.get("num_neighbors", 50),
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
    dataset = build_dataset(cfg["dataset"])
    model = build_model(cfg["model"], dataset=dataset)
    explainers = build_explainers(cfg.get("explainers", []), model=model, dataset=dataset)
    extractor = build_extractor(cfg.get("extractor"), model=model, dataset=dataset)
    extractor_map = build_extractor_map(cfg.get("extractor_map"), model=model, dataset=dataset)

    # Metrics: EvalConfig expects names only.
    metrics_spec: Iterable[Dict[str, Any]] = cfg.get("metrics", exp_cfg.get("metrics", []))
    metric_names = build_metrics(metrics_spec)
    if metric_names:
        exp_cfg["metrics"] = metric_names

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
