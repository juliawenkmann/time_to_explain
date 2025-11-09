from __future__ import annotations

"""
Helpers to ensure built-in metrics register themselves with the global registry.
"""

from importlib import import_module
from typing import Iterable

_BUILTIN_METRIC_MODULES: tuple[str, ...] = (
    "time_to_explain.metrics.sparsity",
    "time_to_explain.metrics.fidelity",
    "time_to_explain.metrics.acc_auc",
    "time_to_explain.metrics.cohesiveness",
)


def ensure_builtin_metrics_loaded(extra_modules: Iterable[str] | None = None) -> None:
    """
    Import all built-in metric modules (and optional extras) so their
    @register_metric decorators run exactly once per process.
    """
    modules = list(_BUILTIN_METRIC_MODULES)
    if extra_modules:
        modules.extend(extra_modules)

    for module_name in modules:
        import_module(module_name)


# Load core metrics eagerly when the package is imported.
ensure_builtin_metrics_loaded()

__all__ = ["ensure_builtin_metrics_loaded"]
