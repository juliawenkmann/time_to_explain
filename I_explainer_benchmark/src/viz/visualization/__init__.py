from __future__ import annotations

from .metrics import (
    DEFAULT_METRIC_LABELS,
    compute_cody_style_report,
    filter_explainers,
    prepare_metrics_plotting,
    plot_explainer_metric_summary,
    plot_explainer_runtime,
    plot_prediction_match_rate,
    plot_selected_metrics,
)

__all__ = [
    # Metrics
    "filter_explainers",
    "prepare_metrics_plotting",
    "plot_explainer_metric_summary",
    "plot_explainer_runtime",
    "plot_prediction_match_rate",
    "plot_selected_metrics",
    "DEFAULT_METRIC_LABELS",
    "compute_cody_style_report",
]

try:
    from .utils import (
        COLORS,
        DIVERGING_COLORSCALE,
        PLOT_COLORWAY,
        PLOT_STYLE,
        SEQUENTIAL_COLORSCALE,
        _PLOTLY_TEMPLATE,
        _auto_show,
        _compute_bipartite_layout,
        _ensure_dataframe,
        _map_ratio_to_color,
        _maybe_save,
        _resolve_event_positions,
        apply_matplotlib_style,
        _require_matplotlib,
        _require_networkx,
        _require_plotly,
        _require_seaborn,
        go,
        make_subplots,
        choose_explain_indices,
    )

    __all__.extend(
        [
            "COLORS",
            "DIVERGING_COLORSCALE",
            "PLOT_COLORWAY",
            "PLOT_STYLE",
            "SEQUENTIAL_COLORSCALE",
            "_PLOTLY_TEMPLATE",
            "_auto_show",
            "_compute_bipartite_layout",
            "_ensure_dataframe",
            "_map_ratio_to_color",
            "_maybe_save",
            "_resolve_event_positions",
            "_require_plotly",
            "_require_networkx",
            "_require_matplotlib",
            "_require_seaborn",
            "apply_matplotlib_style",
            "choose_explain_indices",
            "go",
            "make_subplots",
        ]
    )
except Exception:
    pass

try:
    from .bipartite import (
        animate_bipartite_graph,
        animate_stick_figure,
        build_bipartite_graph,
        plot_bipartite_graph,
    )
    from .graphs import (
        plot_force_directed_graph,
        plot_ground_truth_subgraph,
        plot_nicolaus_motif,
        plot_triadic_closure_subgraph,
    )
    from .summary import (
        plot_dataset_quadrants,
        plot_explain_timeline,
        summarize_explain_instances,
    )

    __all__.extend(
        [
            "plot_dataset_quadrants",
            "plot_bipartite_graph",
            "plot_force_directed_graph",
            "plot_ground_truth_subgraph",
            "plot_triadic_closure_subgraph",
            "plot_nicolaus_motif",
            "animate_bipartite_graph",
            "animate_stick_figure",
            "plot_explain_timeline",
            "summarize_explain_instances",
            "build_bipartite_graph",
        ]
    )
except Exception:
    pass

try:
    from .exports import visualize_folder, visualize_to_files

    __all__.extend(["visualize_folder", "visualize_to_files"])
except Exception:
    pass

try:
    from .workflows import visualize_dataset

    __all__.append("visualize_dataset")
except Exception:
    pass
