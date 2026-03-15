"""Probing / instrumentation utilities.

Phase D uses these to inspect intermediate representations of DBGNN.
"""

from .dbgnn_probe import DBGNNProbes, forward_with_probes, tsne_2d, silhouette_table

__all__ = [
    "DBGNNProbes",
    "forward_with_probes",
    "tsne_2d",
    "silhouette_table",
]
