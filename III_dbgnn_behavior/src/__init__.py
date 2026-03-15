"""Minimal benchmarking harness for node-level GNN explainers.

This package is intentionally small and notebook-friendly:
- Notebooks should only import config + runner and plot results.
- All heavy lifting (data/model/train/explain/eval) lives in src/
"""

__all__ = [
    "config",
    "utils",
    "workflows",
]

__version__ = "0.1.0"
