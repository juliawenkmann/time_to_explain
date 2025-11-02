from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


ARTIFACTS_DIR = Path("artifacts")


@dataclass(frozen=True)
class ExplainerConfig:
    name: str
    training: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TempMEExperiment:
    dataset: Dict[str, object]
    model: Dict[str, object]
    training: Dict[str, object]
    explainers: List[ExplainerConfig]


TEMP_ME_EXPERIMENTS: List[TempMEExperiment] = [
    TempMEExperiment(
        dataset={
            "builder": "processed",
            "path": str(Path("resources/datasets/processed/wikipedia")),
            "directed": False,
            "bipartite": True,
        },
        model={
            "builder": builder_name,
            "device": "auto",
        },
        training={
            "epochs": 20,
            "learning_rate": 1e-4,
            "early_stop_patience": 5,
            "output_dir": str(ARTIFACTS_DIR / "tempme" / "models"),
        },
        explainers=[
            ExplainerConfig(
                name="pgexplainer",
                training={
                    "epochs": 50,
                    "learning_rate": 1e-4,
                    "batch_size": 32,
                    "output_dir": str(ARTIFACTS_DIR / "tempme" / "explainers" / builder_name),
                },
            )
        ],
    )
    for builder_name in ("graphmixer", "tgn", "tgat")
]


@dataclass(frozen=True)
class HTGNNExperiment:
    name: str
    dataset: str
    checkpoint: Path
    device: str = "auto"
    time_window: int = 5
    explainer: Dict[str, object] = field(default_factory=dict)


HTGNN_BASE = Path("submodules/explainer/htgexplainer")

HTGNN_EXPERIMENTS: List[HTGNNExperiment] = [
    HTGNNExperiment(
        name="mag",
        dataset="mag",
        checkpoint=HTGNN_BASE / "HTGNN" / "checkpoint_HTGNN_mag.pt",
        time_window=5,
        explainer={
            "epochs": 50,
            "learning_rate": 1e-3,
            "batch_size": 4,
            "warmup_epoch": 5,
            "es_epoch": 10,
        },
    ),
    HTGNNExperiment(
        name="ml",
        dataset="ml",
        checkpoint=HTGNN_BASE / "HTGNN" / "checkpoint_HTGNN_ml.pt",
        time_window=5,
        explainer={
            "epochs": 50,
            "learning_rate": 1e-3,
            "batch_size": 8,
            "warmup_epoch": 5,
            "es_epoch": 10,
        },
    ),
    HTGNNExperiment(
        name="covid",
        dataset="covid",
        checkpoint=HTGNN_BASE / "HTGNN" / "checkpoint_HTGNN_covid.pt",
        time_window=5,
        explainer={
            "epochs": 50,
            "learning_rate": 5e-4,
            "batch_size": 8,
            "warmup_epoch": 5,
            "es_epoch": 10,
        },
    ),
]
