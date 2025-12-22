from __future__ import annotations

import warnings

from time_to_explain.visualization import *  # noqa: F401,F403

warnings.warn(
    "time_to_explain.utils.visualization is deprecated; import from time_to_explain.visualization instead",
    DeprecationWarning,
    stacklevel=2,
)
