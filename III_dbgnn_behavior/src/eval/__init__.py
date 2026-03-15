"""Evaluation entrypoints.

This package can be imported without the optional DBGNN dependencies.
When torch_geometric/pathpyG are missing, `run_benchmark` is simply not
importable from this namespace (but lightweight utilities like
`eval.metrics` remain usable).
"""

__all__ = []

# Lightweight helpers that do not depend on optional GNN packages.
from eval.flip_curve import compute_flip_curve

__all__ += [
    "compute_flip_curve",
]

try:
    from eval.runner import run_benchmark

    __all__ += [
        "run_benchmark",
    ]
except Exception:
    # Optional deps missing (or not installed in this environment).
    pass
