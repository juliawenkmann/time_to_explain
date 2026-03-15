from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
from IPython.display import display


def show_one_spot_metrics_summary(one_spot: Mapping[str, Any] | None) -> None:
    one_spot = dict(one_spot or {})
    print("One-spot metrics summary (official):")
    if isinstance(one_spot.get("combined"), pd.DataFrame) and not one_spot["combined"].empty:
        with pd.option_context("display.precision", 10):
            display(one_spot["combined"])
    else:
        if isinstance(one_spot.get("core"), pd.DataFrame) and not one_spot["core"].empty:
            with pd.option_context("display.precision", 10):
                display(one_spot["core"])
        if isinstance(one_spot.get("aux"), pd.DataFrame) and not one_spot["aux"].empty:
            with pd.option_context("display.precision", 10):
                display(one_spot["aux"])


__all__ = ["show_one_spot_metrics_summary"]
