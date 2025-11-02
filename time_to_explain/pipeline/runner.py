from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

from time_to_explain.models.training import train_model_from_config
from time_to_explain.explainer.pgexplainer.train_pgexplainer import train_pgexplainer_from_config

from .config import ARTIFACTS_DIR, TEMP_ME_EXPERIMENTS, HTGNN_EXPERIMENTS, TempMEExperiment
from .htgnn import run_htgnn_experiment


LOG = logging.getLogger(__name__)


def _prepare_training_paths(base: Path) -> dict[str, str]:
    checkpoint_dir = base / "checkpoints"
    model_dir = base / "models"
    results_path = base / "train_results.npz"
    return {
        "output_dir": str(base),
        "checkpoint_dir": str(checkpoint_dir),
        "model_dir": str(model_dir),
        "results_path": str(results_path),
    }


def _prepare_explainer_paths(base: Path) -> dict[str, str]:
    checkpoint_dir = base / "checkpoints"
    model_dir = base / "models"
    results_path = base / "results.npz"
    return {
        "output_dir": str(base),
        "checkpoint_dir": str(checkpoint_dir),
        "model_dir": str(model_dir),
        "results_path": str(results_path),
    }


def run_tempme_experiment(exp: TempMEExperiment) -> None:
    dataset_path = Path(exp.dataset["path"])
    dataset_name = dataset_path.name
    model_name = exp.model["builder"]
    base_dir = ARTIFACTS_DIR / "tempme" / dataset_name / model_name

    train_cfg = {
        "dataset": exp.dataset,
        "model": exp.model,
        "training": {
            **_prepare_training_paths(base_dir),
            **exp.training,
        },
    }

    LOG.info("Training %s on %s", model_name, dataset_name)
    result = train_model_from_config(train_cfg)
    last_checkpoint = result.get("last_checkpoint")
    if last_checkpoint is None:
        model_dir = Path(train_cfg["training"]["model_dir"])
        candidate = sorted(model_dir.glob("*.pth"))
        if candidate:
            last_checkpoint = candidate[-1]
        else:
            raise RuntimeError("No checkpoint produced for %s on %s", model_name, dataset_name)

    for explainer in exp.explainers:
        expl_base = base_dir / "explainers" / explainer.name
        expl_cfg = {
            "dataset": exp.dataset,
            "model": {
                **exp.model,
                "checkpoint": str(last_checkpoint),
            },
            "training": {
                **_prepare_explainer_paths(expl_base),
                **explainer.training,
            },
        }
        LOG.info("Running %s for %s on %s", explainer.name, model_name, dataset_name)
        train_pgexplainer_from_config(expl_cfg)


def run_all_experiments(selected: Optional[Iterable[str]] = None) -> None:
    selected_set = set(selected) if selected else None

    for exp in TEMP_ME_EXPERIMENTS:
        dataset_name = Path(exp.dataset["path"]).name
        model_name = exp.model["builder"]
        experiment_name = f"{dataset_name}:{model_name}"
        if selected_set and experiment_name not in selected_set:
            continue
        run_tempme_experiment(exp)

    for job in HTGNN_EXPERIMENTS:
        experiment_name = f"htgnn:{job.name}"
        if selected_set and experiment_name not in selected_set:
            continue
        run_htgnn_experiment(job)
