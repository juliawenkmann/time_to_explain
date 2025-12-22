"""Reusable metric/visualization presets for notebooks and scripts."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List, Mapping, Sequence

from time_to_explain.metrics.eval_config import load_metrics_config
from time_to_explain.visualization import MetricCurveSpec

# ---------------------------------------------------------------------------
# Sparsity levels
# ---------------------------------------------------------------------------


_DEFAULT_METRICS_CFG = load_metrics_config()
DEFAULT_NOTEBOOK_SPARSITY_LEVELS: List[float] = list(
    _DEFAULT_METRICS_CFG.get("fidelity_drop", {}).get("sparsity_levels", [])
)
"""Default sparsity levels used throughout notebook 04."""


# ---------------------------------------------------------------------------
# Metric config factory
# ---------------------------------------------------------------------------


def _update_levels(
    cfg: Dict[str, Mapping[str, object]],
    levels: Sequence[float],
    keys: Iterable[str],
) -> None:
    for key in keys:
        if key in cfg:
            cfg[key]["sparsity_levels"] = list(levels)


def build_default_metric_config(
    *,
    fidelity_levels: Sequence[float] | None = None,
    prediction_levels: Sequence[float] | None = None,
) -> Dict[str, Mapping[str, object]]:
    """Return the default metric configuration dictionary used in Notebook 04."""
    cfg: Dict[str, Mapping[str, object]] = deepcopy(_DEFAULT_METRICS_CFG)

    if fidelity_levels is not None:
        _update_levels(
            cfg,
            fidelity_levels,
            [
                "fidelity_drop",
                "fidelity_keep",
                "temgx_fidelity_minus",
                "temgx_fidelity_plus",
                "fidelity_tempme",
                "cohesiveness",
                "prediction_profile",
                "singular_value",
            ],
        )

    if prediction_levels is not None:
        _update_levels(cfg, prediction_levels, ["prediction_profile"])

    return cfg


# ---------------------------------------------------------------------------
# Visualization presets
# ---------------------------------------------------------------------------


def get_notebook_curve_specs() -> List[MetricCurveSpec]:
    """MetricCurveSpec list used by Notebook 04 for the comparison plots."""

    return [
        MetricCurveSpec(
            prefix="fidelity_drop.@",
            title="Fidelity drop per anchor",
            color="tab:blue",
            axis_label_percent=(
                "Drop sparsity level (% of edges removed)",
                "Drop level (top-k edges removed)",
            ),
        ),
        MetricCurveSpec(
            prefix="fidelity_keep.@",
            title="Fidelity keep per anchor",
            color="tab:orange",
            axis_label_percent=(
                "Keep level (% of edges kept)",
                "Keep level (top-k edges kept)",
            ),
        ),
        MetricCurveSpec(
            prefix="fidelity_tempme.@",
            title="Fidelity TEMP-ME per anchor",
            color="tab:purple",
            ylabel="TEMP-ME fidelity",
            axis_label_percent=(
                "Explanation sparsity (% of edges kept)",
                "Explanation sparsity (kept edges)",
            ),
            y_min=None,
        ),
        MetricCurveSpec(
            prefix="cohesiveness.@",
            title="Cohesiveness per anchor",
            color="tab:red",
            ylabel="Cohesiveness",
            axis_label_percent=(
                "Explanation sparsity (% of edges kept)",
                "Explanation sparsity (kept edges)",
            ),
            y_min=0.0,
        ),
        MetricCurveSpec(
            prefix="temgx_fidelity_minus.@",
            title="TemGX fidelity- per anchor",
            color="tab:blue",
            ylabel="|Δ score|",
            axis_label_percent=(
                "Explanation sparsity (% of edges kept)",
                "Explanation sparsity (kept edges)",
            ),
        ),
        MetricCurveSpec(
            prefix="temgx_fidelity_plus.@",
            title="TemGX fidelity+ per anchor",
            color="tab:green",
            ylabel="1 - |Δ score|",
            axis_label_percent=(
                "Explanation sparsity (% of edges kept)",
                "Explanation sparsity (kept edges)",
            ),
            y_min=0.0,
        ),
        MetricCurveSpec(
            prefix="singular_value.@",
            title="Singular value per anchor",
            color="tab:gray",
            ylabel="Largest singular value",
            axis_label_percent=(
                "Explanation sparsity (% of edges kept)",
                "Explanation sparsity (kept edges)",
            ),
            y_min=0.0,
        ),
    ]


__all__ = [
    "DEFAULT_NOTEBOOK_SPARSITY_LEVELS",
    "build_default_metric_config",
    "get_notebook_curve_specs",
]
