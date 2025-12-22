"""
Lightweight, notebook-friendly pipelines for running evaluations and sweeps.

The functions here wrap `time_to_explain.core.runner.EvaluationRunner` and keep
the surface small so you can:
  * call them directly from notebooks, passing already-constructed objects, or
  * drive them from YAML via the CLI (`python -m time_to_explain.cli ...`).
"""

from .config import load_yaml
from .data import (
    DEFAULT_SYNTHETIC_RECIPES,
    format_uci_messages,
    prepare_real_data,
    prepare_synthetic_data,
)
from .eval import run_eval, run_from_config, sweep_from_glob

__all__ = [
    "load_yaml",
    "run_eval",
    "run_from_config",
    "sweep_from_glob",
    "prepare_real_data",
    "prepare_synthetic_data",
    "format_uci_messages",
    "DEFAULT_SYNTHETIC_RECIPES",
]
