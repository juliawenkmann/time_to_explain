from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def get_device(device: str = "auto") -> torch.device:
    """Resolve a device string to a torch.device.

    Args:
        device: "auto", "cpu", or "cuda".
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    """Best-effort reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def choose_indices(mask: torch.Tensor, n: int, seed: int) -> list[int]:
    """Choose up to n indices where mask is True."""
    idx = mask.nonzero(as_tuple=False).view(-1).tolist()
    if not idx:
        return []
    rng = random.Random(seed)
    rng.shuffle(idx)
    return idx[: min(n, len(idx))]


def clone_data(data):
    """Clone a torch_geometric.Data (or compatible) object.

    Uses .clone() if available, otherwise falls back to deepcopy.
    """
    if hasattr(data, "clone"):
        return data.clone()

    # Fallback: deepcopy is expensive but works.
    from copy import deepcopy

    return deepcopy(data)


from typing import Any, Mapping


def make_run_name(dataset_name: str, dataset_kwargs: Mapping[str, Any] | None = None, *, model_name: str = "dbgnn") -> str:
    """Create a stable run name for checkpoints / logs.

    This is primarily used in notebooks to ensure that different dataset
    configurations (e.g., different netzschleuder networks within a record)
    don't overwrite each other.

    The convention is:
        <dataset_name>[_<record>][_ <network>]_<model_name>

    Notes:
        * If dataset_name is itself a netzschleuder record (via registry fallback),
          you typically only need to include the `network` if the record contains
          multiple networks.
        * If dataset_name == "netzschleuder", then `record` is pulled from
          dataset_kwargs.
    """
    kw = dict(dataset_kwargs or {})

    parts: list[str] = [str(dataset_name)]

    record = kw.get("record", None)
    if record is not None and str(record) and str(record) != str(dataset_name):
        parts.append(str(record))

    network = kw.get("network", None)
    if network is not None and str(network):
        parts.append(str(network))

    return "_".join(parts) + f"_{model_name}"
