"""I/O helpers for evaluation result files.

The evaluation pipeline in this repo produces both CSV and Parquet files.
Unfortunately, the CSV files often contain array-like columns serialized as
strings.

This module focuses purely on:
- discovering results files
- reading them reliably
- writing updated results back
- writing sidecar JSON metrics

Parsing of array-like columns is handled separately in :mod:`postprocess.parsing`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


def find_results_files(path: str | Path, recursive: bool = False) -> List[Path]:
    """Return a list of result files (.parquet/.csv) under *path*.

    - If *path* is a file, returns [path].
    - If *path* is a directory, returns all matching result files.

    We filter to files whose name starts with "results_" to avoid accidentally
    picking up unrelated CSVs (like logs or metrics).
    """
    p = Path(path)
    if p.is_file():
        return [p]

    if not p.exists():
        raise FileNotFoundError(f"Results path not found: {p}")

    exts = {".parquet", ".csv"}
    if recursive:
        candidates = [f for f in p.rglob("*") if f.is_file() and f.suffix in exts]
    else:
        candidates = [f for f in p.glob("*") if f.is_file() and f.suffix in exts]

    # Prefer parquet when both exist for same stem
    parquet_by_stem = {f.stem: f for f in candidates if f.suffix == ".parquet" and f.name.startswith("results_")}
    csv_by_stem = {f.stem: f for f in candidates if f.suffix == ".csv" and f.name.startswith("results_")}

    merged: Dict[str, Path] = {}
    merged.update(csv_by_stem)
    merged.update(parquet_by_stem)  # overwrite CSV with Parquet if both exist

    return [merged[k] for k in sorted(merged.keys())]


def _drop_pandas_index_column(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the common index column produced by DataFrame.to_csv(index=True)."""
    if len(df.columns) == 0:
        return df

    first = df.columns[0]
    if isinstance(first, str) and first.startswith("Unnamed"):
        return df.iloc[:, 1:]

    # also handle an empty column name
    if first in {"", None}:
        return df.iloc[:, 1:]

    return df


def load_results(filepath: str | Path) -> pd.DataFrame:
    """Load a results file (.csv or .parquet) into a DataFrame."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Results file not found: {p}")

    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
        return df

    if p.suffix == ".csv":
        df = pd.read_csv(p)
        return _drop_pandas_index_column(df)

    raise ValueError(f"Unsupported file extension for results: {p.suffix}")


def save_results(df: pd.DataFrame, filepath: str | Path, also_write_other_format: bool = True) -> None:
    """Save the dataframe back to *filepath* and optionally also to the other format."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.suffix == ".csv":
        df.to_csv(p, index=False)
        if also_write_other_format:
            _try_write_parquet(df, p.with_suffix(".parquet"))
        return

    if p.suffix == ".parquet":
        _try_write_parquet(df, p)
        if also_write_other_format:
            df.to_csv(p.with_suffix(".csv"), index=False)
        return

    raise ValueError(f"Unsupported file extension for results: {p.suffix}")


def _try_write_parquet(df: pd.DataFrame, parquet_path: Path) -> None:
    try:
        df.to_parquet(parquet_path)
    except ImportError:
        # pyarrow is optional in the repo
        return


def write_metrics_json(metrics: Dict, results_file: str | Path) -> Path:
    """Write a sidecar JSON file with the computed metrics."""
    p = Path(results_file)
    out = p.with_suffix("")  # remove suffix
    metrics_path = out.with_name(out.name + "_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics_path
