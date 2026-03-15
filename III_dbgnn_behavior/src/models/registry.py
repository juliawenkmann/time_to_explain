from __future__ import annotations

from typing import Callable

import torch

from models.dbgnn import build_dbgnn_adapter


ModelBuilder = Callable[..., object]


MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "dbgnn": build_dbgnn_adapter,
}


def get_model_builder(name: str) -> ModelBuilder:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]
