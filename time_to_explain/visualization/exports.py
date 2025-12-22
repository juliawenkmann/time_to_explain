from __future__ import annotations

from itertools import count
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import warnings
from .plots import (
    animate_bipartite_graph,
    animate_stick_figure,
    plot_bipartite_graph,
    plot_dataset_quadrants,
    plot_force_directed_graph,
    plot_ground_truth_subgraph,
    plot_nicolaus_motif,
    plot_triadic_closure_subgraph,
    summarize_explain_instances,
)
from .utils import (
    _maybe_save,
    _require_plotly,
    go,
    infer_dataset_profile,
    load_dataset_bundle,
    select_ground_truth_event,
)


def visualize_folder(
    folder: str | Path | Dict[str, Any],
    *,
    root_dir: Optional[Path] = None,
    happen_interval: float = 0.5,
    explain_indices: Optional[Sequence[int]] = None,
    show: bool = True,
    structure_max_nodes: int = 60,
    structure_max_edges: int = 600,
) -> Dict[str, "go.Figure"]:
    """Quick interactive overview for a dataset name/path or bundle (auto-displays)."""
    _require_plotly()
    bundle = load_dataset_bundle(folder, root_dir=root_dir, verbose=False)
    df = bundle["interactions"]
    metadata = bundle.get("metadata") or {}
    dataset_ref = folder if isinstance(folder, (str, Path)) else None
    profile = infer_dataset_profile(metadata, dataset_name=dataset_ref)
    gt_event = select_ground_truth_event(df, metadata, explain_indices)

    figs: Dict[str, "go.Figure"] = {}
    figs["diagnostics"] = plot_dataset_quadrants(
        df,
        explain_indices=explain_indices,
        happen_interval=happen_interval,
        show=show,
    )
    try:
        figs["structure_graph"] = plot_force_directed_graph(
            bundle,
            metadata=metadata,
            max_nodes=structure_max_nodes,
            max_edges=structure_max_edges,
            show=show,
        )
    except (RuntimeError, ValueError) as exc:
        warnings.warn(f"Structure visualisation skipped: {exc}", RuntimeWarning)

    if gt_event is not None:
        try:
            figs["ground_truth"] = plot_ground_truth_subgraph(
                bundle,
                gt_event,
                metadata=metadata,
                show=show,
            )
        except (RuntimeError, ValueError) as exc:
            warnings.warn(f"Ground-truth visualisation skipped: {exc}", RuntimeWarning)

    if profile.is_nicolaus:
        try:
            figs["motif"] = plot_nicolaus_motif(bundle, show=show)
        except (RuntimeError, ValueError) as exc:
            warnings.warn(f"Nicolaus motif visualisation skipped: {exc}", RuntimeWarning)
    elif profile.is_triadic:
        try:
            figs["triadic_subset"] = plot_triadic_closure_subgraph(bundle, metadata=metadata, show=show)
        except (RuntimeError, ValueError) as exc:
            warnings.warn(f"Triadic-closure visualisation skipped: {exc}", RuntimeWarning)
    if profile.is_stick:
        try:
            figs["stick_figure_animation"] = animate_stick_figure(
                bundle,
                metadata=metadata,
                show=show,
            )
        except (RuntimeError, ValueError) as exc:
            warnings.warn(f"Stick-figure animation skipped: {exc}", RuntimeWarning)
    if profile.is_bipartite:
        try:
            figs["bipartite"] = plot_bipartite_graph(df, show=show)
        except (RuntimeError, ValueError) as exc:
            warnings.warn(f"Bipartite snapshot skipped: {exc}", RuntimeWarning)
        try:
            figs["animation"] = animate_bipartite_graph(df, show=show)
        except (RuntimeError, ValueError) as exc:
            warnings.warn(f"Bipartite animation skipped: {exc}", RuntimeWarning)
    return figs


def visualize_to_files(
    folder: str | Path | Dict[str, Any],
    out_dir: str | Path,
    *,
    root_dir: Optional[Path] = None,
    happen_interval: float = 0.5,
    include_graph: bool = True,
    graph_max_users: int = 40,
    graph_max_items: int = 40,
    graph_max_edges: int = 500,
    explain_indices: Optional[Sequence[int]] = None,
    explain_events_before: int = 5,
    explain_events_after: int = 5,
    explain_time_window: Optional[float] = None,
    export_format: str = "html",
    structure_max_nodes: int = 60,
    structure_max_edges: int = 600,
) -> Dict[str, List[str]]:
    """Persist a standard visualization bundle for a dataset name/path or bundle."""
    _require_plotly()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_dataset_bundle(folder, root_dir=root_dir, verbose=False)
    df = bundle["interactions"]
    metadata = bundle.get("metadata") or {}
    dataset_ref = folder if isinstance(folder, (str, Path)) else None
    profile = infer_dataset_profile(metadata, dataset_name=dataset_ref)
    gt_event = select_ground_truth_event(df, metadata, explain_indices)

    saved: Dict[str, List[str]] = {k: [] for k in ["diagnostics", "graph", "explain_summary"]}
    export_format = export_format.lstrip(".").lower()

    diag_path = _maybe_save(
        plot_dataset_quadrants(
            df,
            explain_indices=explain_indices,
            happen_interval=happen_interval,
            show=False,
        ),
        out_dir / f"01_dataset_diagnostics.{export_format}",
    )
    if diag_path:
        saved["diagnostics"].append(diag_path)

    file_counter = count(2)

    def _numbered_path(label: str, ext: str = "html") -> Path:
        idx = next(file_counter)
        return out_dir / f"{idx:02d}_{label}.{ext}"

    if include_graph:
        try:
            path = _maybe_save(
                plot_force_directed_graph(
                    bundle,
                    metadata=metadata,
                    max_nodes=structure_max_nodes,
                    max_edges=structure_max_edges,
                    show=False,
                ),
                _numbered_path("structure_graph", export_format),
            )
            if path:
                saved["graph"].append(path)
        except (RuntimeError, ValueError) as exc:
            warnings.warn(f"Structure visualisation skipped: {exc}", RuntimeWarning)

        if gt_event is not None:
            try:
                path = _maybe_save(
                    plot_ground_truth_subgraph(bundle, gt_event, metadata=metadata, show=False),
                    _numbered_path("ground_truth", export_format),
                )
                if path:
                    saved["graph"].append(path)
            except (RuntimeError, ValueError) as exc:
                warnings.warn(f"Ground-truth visualisation skipped: {exc}", RuntimeWarning)

        if profile.is_nicolaus:
            try:
                path = _maybe_save(
                    plot_nicolaus_motif(bundle, show=False),
                    _numbered_path("nicolaus_motif", export_format),
                )
                if path:
                    saved["graph"].append(path)
            except (RuntimeError, ValueError) as exc:
                warnings.warn(f"Nicolaus motif visualisation skipped: {exc}", RuntimeWarning)
        elif profile.is_triadic:
            try:
                path = _maybe_save(
                    plot_triadic_closure_subgraph(bundle, metadata=metadata, show=False),
                    _numbered_path("triadic_subset", export_format),
                )
                if path:
                    saved["graph"].append(path)
            except (RuntimeError, ValueError) as exc:
                warnings.warn(f"Triadic-closure visualisation skipped: {exc}", RuntimeWarning)
        if profile.is_stick:
            try:
                path = _maybe_save(
                    animate_stick_figure(bundle, metadata=metadata, show=False),
                    _numbered_path("stick_figure_animation", export_format),
                )
                if path:
                    saved["graph"].append(path)
            except (RuntimeError, ValueError) as exc:
                warnings.warn(f"Stick-figure animation skipped: {exc}", RuntimeWarning)
        if profile.is_bipartite:
            try:
                path = _maybe_save(
                    plot_bipartite_graph(
                        bundle,
                        max_users=graph_max_users,
                        max_items=graph_max_items,
                        max_edges=graph_max_edges,
                        show=False,
                    ),
                    _numbered_path("interaction_graph", export_format),
                )
                if path:
                    saved["graph"].append(path)
            except (RuntimeError, ValueError) as exc:
                warnings.warn(f"Graph visualisation skipped: {exc}", RuntimeWarning)

    if explain_indices:
        try:
            summary_df = summarize_explain_instances(
                bundle,
                explain_indices,
                events_before=explain_events_before,
                events_after=explain_events_after,
                time_window=explain_time_window,
            )
            if not summary_df.empty:
                idx = next(file_counter)
                csv_path = out_dir / f"{idx:02d}_explain_instances.csv"
                summary_df.to_csv(csv_path, index=False)
                table_fig = go.Figure(
                    data=[
                        go.Table(
                            header=dict(values=list(summary_df.columns), fill_color="#f5f5f5"),
                            cells=dict(values=[summary_df[c] for c in summary_df.columns]),
                        )
                    ]
                )
                html_path = _maybe_save(table_fig, out_dir / f"{idx:02d}_explain_instances.{export_format}")
                saved["explain_summary"].append(str(csv_path))
                if html_path:
                    saved["explain_summary"].append(html_path)
        except (RuntimeError, ValueError) as exc:
            warnings.warn(f"Explain visualisation skipped: {exc}", RuntimeWarning)

    return saved


__all__ = ["visualize_folder", "visualize_to_files"]
