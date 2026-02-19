from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Ensure synthetic recipes register themselves on import
import time_to_explain.data.synthetic_recipes  # noqa: F401

from time_to_explain.core.registry import get_dataset
from time_to_explain.data.io import export_tgn_csv, resolve_repo_root
from time_to_explain.data.validate import basic_stats, verify_interactions


def _merge_config(
    defaults: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    merged = dict(defaults)
    if cfg:
        merged.update(cfg)
    if overrides:
        merged.update(overrides)
    return merged


def _compute_split_info(df, split: Tuple[float, float, float] | None) -> Optional[Dict[str, Any]]:
    if not split:
        return None

    arr = np.array(split, dtype=float)
    if arr.ndim != 1 or arr.size != 3:
        raise ValueError("split must be a tuple of three floats (train, val, test)")

    total = float(arr.sum())
    if total <= 0:
        raise ValueError("split fractions must sum to a positive value")
    arr = arr / total

    n = len(df)
    if n == 0:
        counts = [0, 0, 0]
        boundaries = [0, 0]
    else:
        cumulative = np.floor(arr.cumsum() * n).astype(int)
        cumulative[-1] = n
        train_end, val_end, _ = cumulative
        counts = [train_end, max(val_end - train_end, 0), n - val_end]
        boundaries = [train_end, val_end]

    ts_sorted = df["ts"].to_numpy() if len(df) else np.array([], dtype=float)
    cutoffs: Dict[str, Optional[float]] = {
        "train_max_ts": float(ts_sorted[boundaries[0] - 1]) if boundaries[0] > 0 else None,
        "val_max_ts": float(ts_sorted[boundaries[1] - 1]) if boundaries[1] > 0 else None,
    }

    return {
        "fractions": arr.tolist(),
        "counts": {
            "train": int(counts[0]),
            "val": int(counts[1]),
            "test": int(counts[2]),
        },
        "cutoffs": cutoffs,
    }


def prepare_dataset(
    *,
    project_root: Optional[Path] = None,
    dataset_name: str,
    recipe: str,
    config_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
    split: Optional[Tuple[float, float, float]] = None,
    visualize: bool = False,
    visualization_dir: Optional[Path] = None,
    explain_indices: Optional[Sequence[int]] = None,
    overwrite: bool = False,
    export_tgn: bool = True,
    processed_dir: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    dry_run: bool = False,
    seed: Optional[int] = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    root_dir = Path(project_root).resolve() if project_root is not None else resolve_repo_root()

    recipe_cls = get_dataset(recipe)
    defaults = recipe_cls.default_config() if hasattr(recipe_cls, "default_config") else {}

    cfg_file: Dict[str, Any] = {}
    if config_path is not None:
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as f:
            cfg_file = json.load(f)

    cfg_overrides = dict(config or {})
    final_cfg = _merge_config(defaults, cfg_file, cfg_overrides)

    recipe_obj = recipe_cls(**final_cfg)
    bundle_raw = recipe_obj.generate()

    df = verify_interactions(bundle_raw["interactions"])
    stats = basic_stats(df)
    split_info = _compute_split_info(df, split)

    explain_indices_used: List[int] = list(explain_indices) if explain_indices is not None else []
    if visualize and not explain_indices_used:
        if "label" in df.columns:
            candidates = df[df["label"] > 0]
            if len(candidates) == 0:
                candidates = df
        else:
            candidates = df
        explain_indices_used = candidates.head(min(5, len(candidates))).index.astype(int).tolist()

    metadata = {
        **(bundle_raw.get("metadata") or {}),
        "recipe": recipe,
        "dataset_name": dataset_name,
        "config": final_cfg,
    }
    metadata["stats"] = stats
    if split_info is not None:
        metadata["split"] = split_info
    if explain_indices_used:
        metadata["explain_indices"] = explain_indices_used

    bundle_prepped = {
        "interactions": df,
        "node_features": bundle_raw.get("node_features"),
        "edge_features": bundle_raw.get("edge_features"),
        "metadata": metadata,
    }

    summary: Dict[str, Any] = {
        "dataset": dataset_name,
        "recipe": recipe,
        "config": final_cfg,
        "stats": stats,
        "split": split_info,
        "processed_dir": None,
        "raw_path": None,
        "visualizations": None,
        "visualization_dir": None,
        "explain_indices": explain_indices_used,
    }

    if verbose:
        print(f"Generated dataset '{dataset_name}' using recipe '{recipe}'.")
        print(json.dumps(stats, indent=2))

    if dry_run:
        return summary

    if visualize and not export_tgn:
        raise ValueError("Visualization requires export_tgn=True so processed files exist.")

    processed_out: Optional[Path] = None
    if export_tgn:
        processed_out = export_tgn_csv(
            bundle_prepped,
            dataset_name,
            root_dir=root_dir,
            processed_dir=processed_dir,
            raw_dir=raw_dir,
            overwrite=overwrite,
            seed=seed,
            metadata=metadata,
        )
        summary["processed_dir"] = str(processed_out)
        raw_base = Path(raw_dir) if raw_dir is not None else (root_dir / "resources" / "datasets" / "raw")
        raw_path = raw_base / f"{dataset_name}.csv"
        summary["raw_path"] = str(raw_path)

    if visualize:
        viz_out = Path(visualization_dir) if visualization_dir is not None else None
        if viz_out is None:
            base = processed_out if processed_out is not None else (root_dir / "resources" / "datasets" / "processed" / dataset_name)
            viz_out = Path(base) / "plots"
        try:
            from time_to_explain.visualization import visualize_to_files
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Visualization option requires matplotlib; install it to proceed."
            ) from exc

        viz_map = visualize_to_files(
            dataset_name,
            viz_out,
            explain_indices=explain_indices_used if explain_indices_used else None,
        )
        summary["visualizations"] = {
            section: [str(Path(p)) for p in paths] for section, paths in viz_map.items()
        }
        summary["visualization_dir"] = str(viz_out)

    if verbose and not dry_run:
        if summary["processed_dir"]:
            print(f"Processed data   : {summary['processed_dir']}")
        if summary["visualizations"]:
            print(f"Visualization dir: {summary['visualization_dir']}")

    return summary

__all__ = ["prepare_dataset"]
