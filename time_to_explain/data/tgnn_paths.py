from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from time_to_explain.data.io import resolve_repo_root


@dataclass(frozen=True)
class TGNNDatasetPaths:
    dataset_name: str
    root_dir: Path
    datasets_dir: Path
    raw_dir: Path
    processed_dir: Path
    explain_dir: Path
    raw_csv: Path
    ml_csv: Path
    ml_edge: Path
    ml_node: Path
    explain_idx: Path

    @property
    def xgraph_dir(self) -> Path:
        # Backward-compatible alias for older summaries.
        return self.datasets_dir


def resolve_tgnn_root(root_dir: Optional[Path] = None) -> Path:
    base = Path(root_dir).resolve() if root_dir is not None else resolve_repo_root()
    return base


def tgnn_dataset_paths(dataset_name: str, *, root_dir: Optional[Path] = None) -> TGNNDatasetPaths:
    root = resolve_tgnn_root(root_dir)
    datasets_dir = root / "resources" / "datasets"
    raw_dir = datasets_dir / "raw"
    processed_dir = datasets_dir / "processed"
    explain_dir = datasets_dir / "explain_index"

    raw_csv = raw_dir / f"{dataset_name}.csv"
    ml_csv = processed_dir / f"ml_{dataset_name}.csv"
    ml_edge = processed_dir / f"ml_{dataset_name}.npy"
    ml_node = processed_dir / f"ml_{dataset_name}_node.npy"
    explain_idx = explain_dir / f"{dataset_name}.csv"

    return TGNNDatasetPaths(
        dataset_name=dataset_name,
        root_dir=root,
        datasets_dir=datasets_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        explain_dir=explain_dir,
        raw_csv=raw_csv,
        ml_csv=ml_csv,
        ml_edge=ml_edge,
        ml_node=ml_node,
        explain_idx=explain_idx,
    )


__all__ = ["TGNNDatasetPaths", "resolve_tgnn_root", "tgnn_dataset_paths"]
