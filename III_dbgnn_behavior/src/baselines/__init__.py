"""Small, dependency-light baselines.

These baselines are meant to answer one question early in the project:

    *Is the label signal already present in the representation we're feeding the model?*

They avoid torch_geometric/pathpyG so you can run them even before installing the
optional DBGNN dependencies.
"""

from .phase_c import (
    compute_debruijn_spectral_node_features,
    compute_static_degree_features,
    compute_static_spectral_features,
    infer_n_nodes,
    make_split,
    make_temporal_clusters_labels,
    run_phase_c_baselines,
)

__all__ = [
    "compute_debruijn_spectral_node_features",
    "compute_static_degree_features",
    "compute_static_spectral_features",
    "infer_n_nodes",
    "make_split",
    "make_temporal_clusters_labels",
    "run_phase_c_baselines",
]
