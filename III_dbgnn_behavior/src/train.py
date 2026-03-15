from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import torch
from sklearn.metrics import balanced_accuracy_score

from config import ExperimentConfig
from models.registry import get_model_builder
from utils import ensure_dir


def evaluate_balanced_accuracy(model: torch.nn.Module, data) -> Tuple[float, float]:
    """Exact notebook evaluation function, returning (train_ba, test_ba)."""
    model.eval()
    with torch.no_grad():
        _, pred = model(data).max(dim=1)

    metrics_train = balanced_accuracy_score(data.y[data.train_mask].cpu(), pred[data.train_mask].cpu().numpy())
    metrics_test = balanced_accuracy_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu().numpy())
    return float(metrics_train), float(metrics_test)


def checkpoint_dir(cfg: ExperimentConfig) -> Path:
    return ensure_dir(Path(cfg.run_dir) / cfg.run_name)


def checkpoint_paths(cfg: ExperimentConfig) -> Dict[str, Path]:
    base = checkpoint_dir(cfg)
    return {
        "model": base / "model_state.pt",
        "meta": base / "meta.json",
        "losses": base / "losses.pt",
    }


def train_model(cfg: ExperimentConfig, *, data, assets, device: torch.device):
    """Train a model according to cfg and return a model adapter.

    Uses the same hyperparameters + training loop structure as the notebook.
    """
    model_builder = get_model_builder(cfg.model_name)
    adapter = model_builder(data=data, assets=assets, device=device, hidden_dims=cfg.hidden_dims, p_dropout=cfg.p_dropout)

    model = adapter.model

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_function = torch.nn.CrossEntropyLoss()

    losses = []

    for epoch in range(cfg.epochs):
        model.train()

        output = model(data)
        loss = loss_function(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(float(loss.detach().cpu().item()))

        if epoch % 10 == 0:
            train_ba, test_ba = evaluate_balanced_accuracy(model, data)
            print(
                f"Epoch: {epoch}, Loss: {loss.detach().item():.4f}, Train balanced accuracy: {train_ba:.4f}, Test balanced accuracy: {test_ba:.4f}"
            )

    return adapter, {"losses": losses}


def save_checkpoint(cfg: ExperimentConfig, *, adapter, train_info: Dict) -> None:
    paths = checkpoint_paths(cfg)

    torch.save(adapter.model.state_dict(), paths["model"])
    torch.save(train_info.get("losses", []), paths["losses"])

    with paths["meta"].open("w", encoding="utf-8") as f:
        json.dump({"cfg": asdict(cfg)}, f, indent=2)


def load_checkpoint(cfg: ExperimentConfig, *, data, assets, device: torch.device):
    """Load a previously saved model checkpoint (state_dict)."""
    paths = checkpoint_paths(cfg)
    if not paths["model"].exists():
        raise FileNotFoundError(f"Checkpoint not found: {paths['model']}")

    model_builder = get_model_builder(cfg.model_name)
    adapter = model_builder(data=data, assets=assets, device=device, hidden_dims=cfg.hidden_dims, p_dropout=cfg.p_dropout)
    state = torch.load(paths["model"], map_location=device)
    adapter.model.load_state_dict(state)
    return adapter


def load_or_train(cfg: ExperimentConfig, *, data, assets, device: torch.device):
    """Load checkpoint if available, otherwise train and save."""
    paths = checkpoint_paths(cfg)
    if paths["model"].exists():
        print(f"Loading checkpoint: {paths['model']}")
        return load_checkpoint(cfg, data=data, assets=assets, device=device)

    print("No checkpoint found. Training from scratch...")
    adapter, train_info = train_model(cfg, data=data, assets=assets, device=device)
    save_checkpoint(cfg, adapter=adapter, train_info=train_info)
    return adapter
