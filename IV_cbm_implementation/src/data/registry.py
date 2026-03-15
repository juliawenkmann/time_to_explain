from __future__ import annotations

from typing import Callable, Tuple

import torch

from data.netzschleuder import (
    load_copenhagen,
    load_netzschleuder,
    load_sp_high_school,
    load_sp_hospital,
    load_sp_office,
)
from data.temporal_clusters import load_temporal_clusters

# A dataset loader returns (data, assets)
DatasetLoader = Callable[..., Tuple[object, object]]


DATASET_REGISTRY = {
    "temporal_clusters": load_temporal_clusters,
    "netzschleuder": load_netzschleuder,
    "copenhagen": load_copenhagen,
    "highschool": load_sp_high_school,
    "hospital": load_sp_hospital,
    "office": load_sp_office,
    "workplace": load_sp_office,
    "sp_high_school": load_sp_high_school,
    "sp_hospital": load_sp_hospital,
    "sp_office": load_sp_office,
}


def get_dataset_loader(name: str) -> DatasetLoader:
    """Return a dataset loader.

    If `name` is not explicitly registered, treat it as a netzschleuder record
    and return a loader equivalent to `load_netzschleuder(record=name, ...)`.
    """
    if name in DATASET_REGISTRY:
        return DATASET_REGISTRY[name]

    def _loader(*, device: torch.device, **kwargs):
        record = kwargs.pop("record", name)
        return load_netzschleuder(device=device, record=str(record), **kwargs)

    return _loader
