from __future__ import annotations

from pathlib import Path
from typing import Any

from time_to_explain.core.registries import register_dataset
from time_to_explain.data.data import (
    ContinuousTimeDynamicGraphDataset,
    TrainTestDatasetParameters,
    create_dataset,
)


def _build_parameters(config: dict[str, Any]) -> TrainTestDatasetParameters:
    params_cfg = config.get("parameters")
    if params_cfg is None:
        return TrainTestDatasetParameters(0.2, 0.6, 0.8, 1000, 500, 500)
    if isinstance(params_cfg, TrainTestDatasetParameters):
        return params_cfg
    return TrainTestDatasetParameters(**params_cfg)


@register_dataset("processed")
def build_processed_dataset(config: dict[str, Any]) -> ContinuousTimeDynamicGraphDataset:
    path = config.get("path") or config.get("root") or config.get("dataset_dir")
    if path is None:
        raise ValueError("Processed dataset builder expects a 'path' (directory with *_data.csv files).")
    dataset_dir = Path(path)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    dataset = create_dataset(
        dataset_dir,
        directed=bool(config.get("directed", False)),
        bipartite=bool(config.get("bipartite", False)),
        parameters=_build_parameters(config),
    )
    dataset.metadata = {
        "path": str(dataset_dir),
        "name": config.get("name") or dataset.name,
        "directed": bool(config.get("directed", False)),
        "bipartite": bool(config.get("bipartite", False)),
    }
    return dataset
