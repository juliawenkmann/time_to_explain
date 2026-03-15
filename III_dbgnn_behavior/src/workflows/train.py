from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config import ExperimentConfig
from data.cache import save_dataset_bundle
from data.registry import get_dataset_loader
from train import load_or_train, save_checkpoint, train_model
from utils import get_device, make_run_name, set_seed


@dataclass(frozen=True)
class TrainingWorkflowConfig:
    dataset_name: str = "temporal_clusters"
    dataset_kwargs: Mapping[str, Any] = field(default_factory=dict)
    seed: int = 42
    device: str = "auto"
    num_test: float = 0.3
    epochs: int = 200
    lr: float = 0.005
    p_dropout: float = 0.4
    hidden_dims: tuple[int, ...] = (16, 32, 16)
    run_dir: str = "runs"
    run_name: str | None = None
    train_from_scratch: bool = False


@dataclass(frozen=True)
class SplitMetrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float

    def to_dict(self) -> dict[str, float]:
        return {
            "accuracy": float(self.accuracy),
            "precision_macro": float(self.precision_macro),
            "recall_macro": float(self.recall_macro),
            "f1_macro": float(self.f1_macro),
        }


@dataclass
class TrainingWorkflowResult:
    cfg: ExperimentConfig
    data: Any
    assets: Any
    adapter: Any
    device: torch.device
    dataset_dir: Path
    train_metrics: SplitMetrics
    test_metrics: SplitMetrics


def _resolve_run_name(cfg: TrainingWorkflowConfig) -> str:
    if cfg.run_name:
        return str(cfg.run_name)
    return make_run_name(cfg.dataset_name, cfg.dataset_kwargs, model_name="dbgnn")


@torch.no_grad()
def _predict_labels(model: torch.nn.Module, data) -> torch.Tensor:
    model.eval()
    logits = model(data)
    return logits.argmax(dim=1)


def _compute_split_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> SplitMetrics:
    labels = np.unique(y_true)
    return SplitMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_macro=float(precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        recall_macro=float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
    )


def evaluate_macro_metrics(model: torch.nn.Module, data) -> tuple[SplitMetrics, SplitMetrics]:
    y = data.y.detach().cpu().numpy()
    pred = _predict_labels(model, data).detach().cpu().numpy()

    train_mask = data.train_mask.detach().cpu().numpy().astype(bool)
    test_mask = data.test_mask.detach().cpu().numpy().astype(bool)
    labeled_mask = y >= 0

    train_idx = train_mask & labeled_mask
    test_idx = test_mask & labeled_mask

    train_metrics = _compute_split_metrics(y[train_idx], pred[train_idx])
    test_metrics = _compute_split_metrics(y[test_idx], pred[test_idx])
    return train_metrics, test_metrics


def _build_experiment_config(cfg: TrainingWorkflowConfig) -> ExperimentConfig:
    return ExperimentConfig(
        dataset_name=cfg.dataset_name,
        model_name="dbgnn",
        dataset_kwargs=dict(cfg.dataset_kwargs),
        seed=int(cfg.seed),
        device=str(cfg.device),
        epochs=int(cfg.epochs),
        lr=float(cfg.lr),
        p_dropout=float(cfg.p_dropout),
        hidden_dims=tuple(int(x) for x in cfg.hidden_dims),
        num_test=float(cfg.num_test),
        run_dir=str(cfg.run_dir),
        run_name=_resolve_run_name(cfg),
    )


def run_training_workflow(cfg: TrainingWorkflowConfig) -> TrainingWorkflowResult:
    exp_cfg = _build_experiment_config(cfg)
    set_seed(exp_cfg.seed)
    device = get_device(exp_cfg.device)

    loader = get_dataset_loader(exp_cfg.dataset_name)
    data, assets = loader(
        device=device,
        num_test=exp_cfg.num_test,
        seed=exp_cfg.seed,
        **dict(exp_cfg.dataset_kwargs),
    )

    if cfg.train_from_scratch:
        adapter, train_info = train_model(exp_cfg, data=data, assets=assets, device=device)
        save_checkpoint(exp_cfg, adapter=adapter, train_info=train_info)
    else:
        adapter = load_or_train(exp_cfg, data=data, assets=assets, device=device)

    dataset_dir = Path(exp_cfg.run_dir) / exp_cfg.run_name / "dataset"
    save_dataset_bundle(
        out_dir=dataset_dir,
        data=data,
        assets=assets,
        extra_meta={
            "dataset_name": exp_cfg.dataset_name,
            "dataset_kwargs": dict(exp_cfg.dataset_kwargs),
        },
    )

    train_metrics, test_metrics = evaluate_macro_metrics(adapter.model, data)
    return TrainingWorkflowResult(
        cfg=exp_cfg,
        data=data,
        assets=assets,
        adapter=adapter,
        device=device,
        dataset_dir=dataset_dir,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
    )
