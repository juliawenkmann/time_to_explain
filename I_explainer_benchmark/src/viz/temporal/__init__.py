from __future__ import annotations

from .common import ExplainerSelection, REQUIRED_EVENT_COLS, ROLE_COLORS
from .figure import (
    compute_overlap_stats,
    plot_temporal_explanation_figure,
    plot_temporal_explanation_separate_panels,
    save_figure,
)
from .loading import (
    assign_event_labels,
    build_clean_node_layout,
    extract_local_temporal_neighborhood,
    filter_events_at_or_before_target,
    focus_close_context_events,
    include_event_rows,
    load_ground_truth_event_indices,
    load_ground_truth_target_indices,
    load_temporal_graph,
    relabel_local_nodes,
)
from .selection import (
    discover_explainer_result_files,
    discover_latest_explainer_results,
    ensure_explainer_results_for_targets,
    list_available_targets_in_results,
    load_explainer_selection_from_results,
    resolve_explainer_selection,
    resolve_or_run_explainer_selection,
    run_explainer_for_target,
)

__all__ = [
    "ExplainerSelection",
    "REQUIRED_EVENT_COLS",
    "ROLE_COLORS",
    "assign_event_labels",
    "build_clean_node_layout",
    "compute_overlap_stats",
    "discover_explainer_result_files",
    "discover_latest_explainer_results",
    "ensure_explainer_results_for_targets",
    "extract_local_temporal_neighborhood",
    "filter_events_at_or_before_target",
    "focus_close_context_events",
    "include_event_rows",
    "list_available_targets_in_results",
    "load_explainer_selection_from_results",
    "load_ground_truth_event_indices",
    "load_ground_truth_target_indices",
    "load_temporal_graph",
    "plot_temporal_explanation_figure",
    "plot_temporal_explanation_separate_panels",
    "relabel_local_nodes",
    "resolve_explainer_selection",
    "resolve_or_run_explainer_selection",
    "run_explainer_for_target",
    "save_figure",
]
