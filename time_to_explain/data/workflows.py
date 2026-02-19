from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure synthetic recipes are registered before we query the registry.
import time_to_explain.data.synthetic_recipes  # noqa: F401

from time_to_explain.core.registry import available_datasets
from time_to_explain.core.types import DatasetBundle
from time_to_explain.data.synthetic import prepare_dataset as prepare_synthetic_dataset
from time_to_explain.data.io import load_processed_dataset, resolve_repo_root
from time_to_explain.data.tgnn_setup import setup_tgnn_data
from time_to_explain.data.validate import basic_stats

SYNTHETIC_ALIASES: Dict[str, str] = {
    "erdos_small": "erdos_temporal",
    "hawkes_small": "hawkes_exp",
    "sticky_figure": "stick_figure",
}

REAL_TGNN_DATASETS = {"wikipedia", "reddit", "simulate_v1", "simulate_v2"}


def _resolve_root(root_dir: Optional[Path] = None) -> Path:
    return Path(root_dir).resolve() if root_dir is not None else resolve_repo_root()


def dataset_config_dir(root_dir: Optional[Path] = None) -> Path:
    return _resolve_root(root_dir) / "configs" / "datasets"


def read_dataset_config(
    dataset_name: str, *, root_dir: Optional[Path] = None
) -> Tuple[Optional[Path], Optional[str], Dict[str, Any]]:
    config_path = dataset_config_dir(root_dir) / f"{dataset_name}.json"
    if not config_path.exists():
        return None, None, {}

    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Dataset config must be a JSON object: {config_path}")

    recipe = None
    config = data
    if isinstance(data.get("recipe"), str):
        recipe = data["recipe"]
        if isinstance(data.get("config"), dict):
            config = data["config"]
        elif isinstance(data.get("params"), dict):
            config = data["params"]
        else:
            config = {k: v for k, v in data.items() if k != "recipe"}

    return config_path, recipe, config


def list_processed_datasets(
    *, root_dir: Optional[Path] = None, include_flat: bool = True
) -> List[str]:
    root = _resolve_root(root_dir)
    processed_root = root / "resources" / "datasets" / "processed"
    names: set[str] = set()

    if processed_root.exists():
        for entry in processed_root.iterdir():
            if entry.is_dir():
                names.add(entry.name)

        if include_flat:
            for csv_path in processed_root.glob("ml_*.csv"):
                stem = csv_path.stem
                if stem.startswith("ml_"):
                    names.add(stem[3:])

    return sorted(names)


def is_synthetic_dataset(
    dataset_name: str,
    *,
    root_dir: Optional[Path] = None,
    synthetic_names: Optional[Iterable[str]] = None,
) -> bool:
    synthetic = set(synthetic_names or available_datasets())
    if dataset_name in synthetic:
        return True

    root = _resolve_root(root_dir)
    meta_path = root / "resources" / "datasets" / "processed" / dataset_name / f"ml_{dataset_name}.json"
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            if isinstance(meta, dict):
                recipe = meta.get("recipe") or (meta.get("metadata") or {}).get("recipe")
                return recipe is not None
        except Exception:
            return False
    return False


def ensure_ml_format(
    dataset_name: str, *, root_dir: Optional[Path] = None, verbose: bool = True
) -> Optional[Path]:
    """
    Ensure a TGAT-style processed CSV exists for `dataset_name`.

    Preferred layout (current):
        resources/datasets/processed/ml_<name>.csv

    Legacy fallback:
        resources/datasets/processed/<name>/ml_<name>.csv
    """
    root = _resolve_root(root_dir)
    processed_root = root / "resources" / "datasets" / "processed"

    # Current (flat) layout
    ml_csv = processed_root / f"ml_{dataset_name}.csv"
    if ml_csv.exists():
        return ml_csv

    # Legacy (folder) layout
    legacy = processed_root / dataset_name / f"ml_{dataset_name}.csv"
    if legacy.exists():
        if verbose:
            print(f"Found legacy processed CSV: {legacy}.")
        return legacy

    if verbose:
        print(f"Processed CSV not found for '{dataset_name}' in {processed_root}")
    return None


def _load_flat_processed_dataset(dataset_name: str, root: Path) -> Optional[DatasetBundle]:
    processed_root = root / "resources" / "datasets" / "processed"
    ml_csv = processed_root / f"ml_{dataset_name}.csv"
    if not ml_csv.exists():
        return None

    interactions = pd.read_csv(ml_csv)
    edge_path = processed_root / f"ml_{dataset_name}.npy"
    node_path = processed_root / f"ml_{dataset_name}_node.npy"
    meta_path = processed_root / f"ml_{dataset_name}.json"

    edge_features = np.load(edge_path) if edge_path.exists() else None
    node_features = np.load(node_path) if node_path.exists() else None
    metadata: Dict[str, Any] = {"dataset_name": dataset_name, "source": "processed"}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    return {
        "interactions": interactions,
        "edge_features": edge_features,
        "node_features": node_features,
        "metadata": metadata,
    }


def load_processed_dataset_safe(
    dataset_name: str, *, root_dir: Optional[Path] = None, verbose: bool = True
) -> DatasetBundle:
    root = _resolve_root(root_dir)
    resolved_name = SYNTHETIC_ALIASES.get(dataset_name, dataset_name)
    if resolved_name != dataset_name and verbose:
        print(f"Using dataset alias '{dataset_name}' -> '{resolved_name}'.")
    try:
        return load_processed_dataset(resolved_name, root_dir=root)
    except FileNotFoundError:
        bundle = _load_flat_processed_dataset(resolved_name, root)
        if bundle is not None:
            if verbose:
                print(f"Loaded flat processed files for '{resolved_name}'.")
            if resolved_name != dataset_name:
                metadata = bundle.get("metadata")
                if isinstance(metadata, dict):
                    metadata.setdefault("dataset_alias", dataset_name)
            return bundle

        ensured = ensure_ml_format(resolved_name, root_dir=root, verbose=verbose)
        if ensured is not None:
            return load_processed_dataset(resolved_name, root_dir=root)
        if resolved_name != dataset_name:
            raise FileNotFoundError(
                f"Processed dataset {dataset_name!r} (alias {resolved_name!r}) not found."
            )
        raise


def prepare_dataset_bundle(
    dataset_name: str,
    *,
    recipe: Optional[str] = None,
    root_dir: Optional[Path] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    split: Optional[Tuple[float, float, float]] = None,
    from_cache: bool = True,
    overwrite: bool = False,
    seed: int = 0,
    verbose: bool = True,
    ensure_real: bool = False,
    force_download: bool = False,
    index_size: int = 500,
) -> Tuple[DatasetBundle, Dict[str, Any]]:
    root = _resolve_root(root_dir)
    config_path, config_recipe, config_file = read_dataset_config(dataset_name, root_dir=root)
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

    processed_root = root / "resources" / "datasets" / "processed"
    processed_dir = processed_root / dataset_name
    flat_csv = processed_root / f"ml_{dataset_name}.csv"

    if chosen_recipe:
        if chosen_recipe not in synthetic_names:
            raise ValueError(
                f"Unknown synthetic recipe '{chosen_recipe}'. Available: {sorted(synthetic_names)}"
            )

        cache_hit = (processed_dir.exists() or flat_csv.exists()) and from_cache and not overwrite
        if verbose:
            print(f"Dataset: {dataset_name}")
            print(f"Recipe : {chosen_recipe}")
            if config_path:
                print(f"Config : {config_path}")
            print(f"Cache  : {'hit' if cache_hit else 'miss'}")

        if cache_hit:
            bundle = load_processed_dataset_safe(dataset_name, root_dir=root, verbose=verbose)
            stats = basic_stats(bundle["interactions"])
            if verbose:
                print(json.dumps(stats, indent=2))
            summary = {
                "dataset": dataset_name,
                "recipe": chosen_recipe,
                "config": config,
                "config_path": str(config_path) if config_path else None,
                "processed_dir": str(processed_dir) if processed_dir.exists() else str(processed_root),
                "stats": stats,
                "source": "cache",
            }
            return bundle, summary

        summary = prepare_synthetic_dataset(
            project_root=root,
            dataset_name=dataset_name,
            recipe=chosen_recipe,
            config=config,
            split=split,
            overwrite=overwrite,
            export_tgn=True,
            seed=seed,
            verbose=verbose,
        )
        summary["config_path"] = str(config_path) if config_path else None
        summary["source"] = "generated"

        bundle = load_processed_dataset_safe(dataset_name, root_dir=root, verbose=verbose)
        summary.setdefault("stats", basic_stats(bundle["interactions"]))
        return bundle, summary

    if dataset_name in REAL_TGNN_DATASETS and ensure_real:
        if verbose:
            print(f"Running TGNN setup for '{dataset_name}'.")
        setup_tgnn_data(
            root=root,
            only=[dataset_name],
            force=force_download,
            do_process=True,
            do_index=True,
            seed=seed,
            index_size=index_size,
        )

    bundle = load_processed_dataset_safe(dataset_name, root_dir=root, verbose=verbose)
    stats = basic_stats(bundle["interactions"])
    if verbose:
        print(json.dumps(stats, indent=2))
    summary = {
        "dataset": dataset_name,
        "processed_dir": str(processed_dir) if processed_dir.exists() else str(processed_root),
        "stats": stats,
        "source": "processed",
    }
    return bundle, summary


__all__ = [
    "dataset_config_dir",
    "read_dataset_config",
    "list_processed_datasets",
    "is_synthetic_dataset",
    "ensure_ml_format",
    "load_processed_dataset_safe",
    "prepare_dataset_bundle",
]
