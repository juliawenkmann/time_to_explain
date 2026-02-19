"""Path conventions used by the CoDy repository.

The original bash scripts build paths like:

- resources/datasets/processed/<dataset>
- resources/models/<dataset>/<model>-<dataset>.pth
- resources/models/<dataset>/pg_explainer/<model>_final.pth
- resources/results/<dataset>/<explainer>/...

Centralizing these in one module makes the pipeline code easier to follow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root (three levels above this file)."""
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class RepoPaths:
    root: Path

    @property
    def resources(self) -> Path:
        return self.root / "resources"

    @property
    def scripts(self) -> Path:
        return self.root / "scripts"

    def processed_dataset_dir(self, dataset_name: str) -> Path:
        return self.resources / "datasets" / "processed" / dataset_name

    def tgnn_model_path(self, model_type: str, dataset_name: str) -> Path:
        return self.resources / "models" / dataset_name / f"{model_type}-{dataset_name}.pth"

    def pgexplainer_model_path(self, model_type: str, dataset_name: str) -> Path:
        return self.resources / "models" / dataset_name / "pg_explainer" / f"{model_type}_final.pth"

    def eval_results_dir(self, dataset_name: str) -> Path:
        # matches common.bash RESULTS_DIR/<dataset>
        return self.resources / "results" / dataset_name

    def explainer_results_dir(self, dataset_name: str, explainer_name: str) -> Path:
        return self.eval_results_dir(dataset_name) / explainer_name

    def explained_ids_path(self, model_type: str, dataset_name: str) -> Path:
        return self.eval_results_dir(dataset_name) / f"{model_type}_evaluation_event_ids.npy"


def default_paths() -> RepoPaths:
    return RepoPaths(root=repo_root())
