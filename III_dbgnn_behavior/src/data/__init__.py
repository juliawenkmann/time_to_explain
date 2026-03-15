from __future__ import annotations

# NOTE: This codebase supports optional dependencies (torch_geometric, pathpyG).
# We keep imports here best-effort so lightweight submodules (e.g. data.audit)
# work even when the optional deps are not installed.

__all__ = []

try:
    from data.temporal_clusters import (
        TemporalClustersAssets,
        load_temporal_clusters,
        load_temporal_clusters_connected,
        load_synthetic_tedges,
    )

    __all__ += [
        "TemporalClustersAssets",
        "load_temporal_clusters",
        "load_temporal_clusters_connected",
        "load_synthetic_tedges",
    ]
except Exception:
    # Optional deps missing (or not installed in this environment).
    pass
