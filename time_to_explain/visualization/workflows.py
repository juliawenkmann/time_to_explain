from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from time_to_explain.data.validate import basic_stats
from .exports import visualize_folder, visualize_to_files
from .datasets import animate_bipartite_graph, plot_bipartite_graph, plot_explain_timeline
from .utils import choose_explain_indices, infer_dataset_profile, load_dataset_bundle

def _normalize_explain_indices(
    explain_indices: Optional[Sequence[int]], num_events: int
) -> Sequence[int]:
    if explain_indices is None:
        return choose_explain_indices(num_events)
    return [int(idx) for idx in explain_indices]

def visualize_dataset(
    dataset_name: str | Path,
    *,
    root_dir: Optional[Path] = None,
    explain_indices: Optional[Sequence[int]] = None,
    show: bool = True,
    save_dir: Optional[Path] = None,
    export_format: str = "pdf",
    timeline_window: int = 15,
    max_base_points: int = 20_000,
    animate_bins: int = 25,
    animate_pruned: Optional[float] = 0.25,
    graph_max_users: int = 50,
    graph_max_items: int = 50,
) -> Dict[str, Any]:
    bundle = load_dataset_bundle(dataset_name, root_dir=root_dir, verbose=True)
    df = bundle["interactions"]
    metadata = bundle.get("metadata") or {}
    profile = infer_dataset_profile(metadata, dataset_name=dataset_name)

    explain_indices = _normalize_explain_indices(explain_indices, len(df))
    stats = basic_stats(df)
    print(f"Dataset: {profile.dataset_name}")
    print(json.dumps(stats, indent=2))
    if explain_indices:
        print(f"Explain indices: {list(explain_indices)}")

    figs: Dict[str, Any] = {}

    try:
        figs["overview"] = visualize_folder(
            bundle,
            explain_indices=explain_indices,
            show=show,
        )
    except Exception as exc:
        print(f"Overview visualization failed: {exc}")

    if profile.is_bipartite:
        try:
            figs["bipartite_snapshot"] = plot_bipartite_graph(
                bundle,
                max_users=graph_max_users,
                max_items=graph_max_items,
                show=show,
            )
        except Exception as exc:
            print(f"Bipartite snapshot skipped: {exc}")

        try:
            figs["bipartite_animation"] = animate_bipartite_graph(
                bundle,
                bins=animate_bins,
                cumulative=True,
                show=show,
            )
            if animate_pruned is not None:
                figs["bipartite_animation_pruned"] = animate_bipartite_graph(
                    bundle,
                    bins=animate_bins,
                    cumulative=True,
                    pruned=animate_pruned,
                    show=show,
                )
        except Exception as exc:
            print(f"Bipartite animation skipped: {exc}")

    if explain_indices:
        try:
            figs["explain_timeline"] = plot_explain_timeline(
                bundle,
                event_indices=explain_indices,
                window=timeline_window,
                max_base_points=max_base_points,
                show=show,
            )
        except Exception as exc:
            print(f"Explain timeline skipped: {exc}")

    if save_dir is not None:
        save_dir = Path(save_dir)
        if export_format.lower().lstrip(".") != "html":
            print("Static export requires kaleido for PDF/SVG/PNG output.")
        try:
            figs["saved"] = visualize_to_files(
                bundle,
                out_dir=save_dir,
                explain_indices=explain_indices,
                explain_time_window=timeline_window,
                export_format=export_format,
            )
            print(f"Saved visualization bundle to: {save_dir}")
        except Exception as exc:
            print(f"Saving visualization bundle failed: {exc}")

    return figs


__all__ = ["visualize_dataset"]
