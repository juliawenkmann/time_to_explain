"""Temporal Graph Explainability framework."""

from __future__ import annotations

import sys
from pathlib import Path
import warnings

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CODY_PATH = _REPO_ROOT / "submodules" / "explainer" / "CoDy"
if _CODY_PATH.is_dir() and str(_CODY_PATH) not in sys.path:
    sys.path.append(str(_CODY_PATH))

try:
    from time_to_explain.data import builders as _data_builders  # noqa: F401
except Exception as exc:  # pragma: no cover - import guard
    warnings.warn(f"Failed to register dataset builders: {exc!r}", RuntimeWarning)

try:
    from time_to_explain.explainer.greedy_and_cody import cody as _explainer_adapters  # noqa: F401
except Exception as exc:  # pragma: no cover - import guard
    warnings.warn(
        "Failed to register explainer adapters from CoDy. "
        "Ensure submodules/explainer/CoDy is available.",
        RuntimeWarning,
    )

from time_to_explain import metrics  # noqa: F401
from time_to_explain import models  # noqa: F401
from time_to_explain import samplers  # noqa: F401

__all__ = ["metrics", "models", "samplers"]
