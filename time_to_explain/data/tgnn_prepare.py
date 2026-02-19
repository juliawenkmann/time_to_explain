from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

# Ensure synthetic recipes are registered before we query the registry.
import time_to_explain.data.synthetic_recipes  # noqa: F401

from time_to_explain.core.registry import available_datasets
from time_to_explain.data.synthetic import prepare_dataset as prepare_synthetic_dataset
from time_to_explain.data.io import load_processed_dataset, resolve_repo_root
from time_to_explain.data.tgnn_paths import TGNNDatasetPaths, tgnn_dataset_paths
from time_to_explain.data.explain_index import generate_explain_index
from time_to_explain.data.tgnn_setup import setup_tgnn_data
from time_to_explain.data.validate import basic_stats
from time_to_explain.data.workflows import read_dataset_config

SYNTHETIC_ALIASES: Dict[str, str] = {
    "erdos_small": "erdos_temporal",
    "hawkes_small": "hawkes_exp",
    "sticky_figure": "stick_figure",
}

REAL_TGNN_DATASETS = {"wikipedia", "reddit", "simulate_v1", "simulate_v2", "multihost"}


def _resolve_repo_root(root_dir: Optional[Path] = None) -> Path:
    return Path(root_dir).resolve() if root_dir is not None else resolve_repo_root()


def _summarize_paths(paths: TGNNDatasetPaths) -> Dict[str, str]:
    return {
        "root_dir": str(paths.root_dir),
        "datasets_dir": str(paths.datasets_dir),
        "raw_dir": str(paths.raw_dir),
        "processed_dir": str(paths.processed_dir),
        "explain_dir": str(paths.explain_dir),
        "raw_csv": str(paths.raw_csv),
        "ml_csv": str(paths.ml_csv),
        "ml_edge": str(paths.ml_edge),
        "ml_node": str(paths.ml_node),
        "explain_idx": str(paths.explain_idx),
    }


def prepare_tgnn_dataset(
    dataset_name: str,
    *,
    root_dir: Optional[Path] = None,
    recipe: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    split: Optional[Tuple[float, float, float]] = None,
    from_cache: bool = True,
    overwrite: bool = False,
    seed: int = 0,
    verbose: bool = True,
    ensure_real: bool = True,
    force_download: bool = False,
    index_size: int = 500,
    do_index: bool = True,
) -> Tuple[TGNNDatasetPaths, Dict[str, Any]]:
    """
    Prepare a dataset in the resources layout (resources/datasets/{raw,processed,explain_index}).

    Returns:
        (paths, summary) where `paths` describes the resources/datasets layout.
    """
    repo_root = _resolve_repo_root(root_dir)
    paths = tgnn_dataset_paths(dataset_name, root_dir=repo_root)

    config_path, config_recipe, config_file = read_dataset_config(dataset_name, root_dir=repo_root)
    config: Dict[str, Any] = dict(config_file)
    if config_overrides:
        config.update(config_overrides)

    synthetic_names = set(available_datasets())
    chosen_recipe = recipe or config_recipe
    if chosen_recipe is None:
        if dataset_name in synthetic_names:
            chosen_recipe = dataset_name
        else:
            chosen_recipe = SYNTHETIC_ALIASES.get(dataset_name)

    if verbose:
        print(f"Dataset: {dataset_name}")
        if chosen_recipe:
            print(f"Recipe : {chosen_recipe}")
        if config_path:
            print(f"Config : {config_path}")
        print(f"Layout : {paths.processed_dir}")

    summary: Dict[str, Any] = {
        "dataset": dataset_name,
        "recipe": chosen_recipe,
        "config": config,
        "config_path": str(config_path) if config_path else None,
        "paths": _summarize_paths(paths),
        "stats": None,
        "explain_idx_path": None,
        "explain_idx_preview": None,
        "source": None,
    }

    if chosen_recipe:
        if chosen_recipe not in synthetic_names:
            raise ValueError(
                f"Unknown synthetic recipe '{chosen_recipe}'. Available: {sorted(synthetic_names)}"
            )

        cache_hit = paths.ml_csv.exists() and from_cache and not overwrite
        if verbose:
            print(f"Cache  : {'hit' if cache_hit else 'miss'}")

        if not cache_hit:
            prepare_synthetic_dataset(
                project_root=repo_root,
                dataset_name=dataset_name,
                recipe=chosen_recipe,
                config=config,
                split=split,
                overwrite=overwrite,
                export_tgn=True,
                processed_dir=paths.processed_dir,
                raw_dir=paths.raw_dir,
                seed=seed,
                verbose=verbose,
            )
            summary["source"] = "generated"
        else:
            summary["source"] = "cache"

        bipartite = bool(config.get("bipartite", True))
        if do_index and paths.ml_csv.exists():
            if overwrite or not paths.explain_idx.exists():
                generate_explain_index(
                    paths.ml_csv,
                    paths.explain_dir,
                    dataset_name,
                    size=index_size,
                    seed=seed,
                    bipartite=bipartite,
                )
    else:
        if dataset_name in REAL_TGNN_DATASETS and ensure_real:
            if verbose:
                print("TGNN setup: download + process + index")
            setup_tgnn_data(
                root=paths.root_dir,
                only=[dataset_name],
                force=force_download,
                do_process=True,
                do_index=do_index,
                seed=seed,
                index_size=index_size,
                data_dir=paths.raw_dir,
                proc_dir=paths.processed_dir,
                idx_dir=paths.explain_dir,
            )
            summary["source"] = "downloaded"
        else:
            summary["source"] = "processed"

        if do_index and paths.ml_csv.exists():
            if overwrite or not paths.explain_idx.exists():
                generate_explain_index(
                    paths.ml_csv,
                    paths.explain_dir,
                    dataset_name,
                    size=index_size,
                    seed=seed,
                )

    if paths.ml_csv.exists():
        bundle = load_processed_dataset(paths.ml_csv)
        stats = basic_stats(bundle["interactions"])
        summary["stats"] = stats
        if verbose:
            print(json.dumps(stats, indent=2))
    else:
        if verbose:
            print(f"Missing processed file: {paths.ml_csv}")

    if paths.explain_idx.exists():
        try:
            explain_df = pd.read_csv(paths.explain_idx)
            preview = explain_df.get("event_idx")
            if preview is not None:
                summary["explain_idx_path"] = str(paths.explain_idx)
                summary["explain_idx_preview"] = preview.head(5).astype(int).tolist()
        except Exception:
            pass

    return paths, summary


__all__ = ["prepare_tgnn_dataset", "TGNNDatasetPaths", "tgnn_dataset_paths"]
