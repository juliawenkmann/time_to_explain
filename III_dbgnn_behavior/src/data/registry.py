from __future__ import annotations

from typing import Callable, Tuple

import torch

from data.covid import load_covid
from data.temporal_clusters import (
    load_temporal_clusters,
    load_temporal_clusters_connected,
    load_synthetic_tedges,
)
from data.netzschleuder import (
    load_netzschleuder,
    load_copenhagen,
    load_sp_colocation,
    load_sp_high_school,
    load_sp_high_school_new,
    load_sp_hospital,
    load_sp_primary_school,
    load_sp_office,
)

# A dataset loader returns (data, assets)
DatasetLoader = Callable[..., Tuple[object, object]]


DATASET_REGISTRY = {
    "temporal_clusters": load_temporal_clusters,
    "synthetic": load_synthetic_tedges,
    "synthetic_tedges": load_synthetic_tedges,
    # Synthetic variant with *connected* higher-order structure (hub motifs)
    "temporal_clusters_connected": load_temporal_clusters_connected,
    # Alias matching the user's notebook name
    "dbgnn_explainable_dataset_connected": load_temporal_clusters_connected,
    "covid": load_covid,

    # Generic netzschleuder entrypoint:
    #   dataset_name="netzschleuder"
    #   dataset_kwargs={"record": "...", "network": "...", "time_attr": "...", "target_attr": "..."}
    "netzschleuder": load_netzschleuder,

    # Convenience aliases (DBGNN paper / SocioPatterns / Copenhagen)
    "copenhagen": load_copenhagen,
    "sp_colocation": load_sp_colocation,
    "sp_high_school": load_sp_high_school,
    "sp_high_school_new": load_sp_high_school_new,
    "sp_hospital": load_sp_hospital,
    "sp_primary_school": load_sp_primary_school,
    "sp_office": load_sp_office,
}


def get_dataset_loader(name: str) -> DatasetLoader:
    """Return a dataset loader.

    Special case: if `name` is not in the registry, we treat it as a netzschleuder
    *record name* and return a loader equivalent to:

        load_netzschleuder(record=name, ...)

    This makes it easy to run on *any* netzschleuder dataset without adding a
    dedicated wrapper.
    """
    if name in DATASET_REGISTRY:
        return DATASET_REGISTRY[name]

    # Fallback: interpret as netzschleuder record name
    def _loader(*, device: torch.device, **kwargs):
        # Allow passing record explicitly in kwargs; but if both are given, kwargs wins.
        record = kwargs.pop("record", name)
        return load_netzschleuder(device=device, record=str(record), **kwargs)

    return _loader
