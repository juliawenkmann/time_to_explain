from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for training + explainer evaluation.

    Notebooks should create one of these and then call:

        from eval.runner import run_benchmark
        df = run_benchmark(cfg)

    Defaults are chosen to match the original `dbgnn.ipynb`.
    """

    # Selection
    dataset_name: str = "temporal_clusters"
    model_name: str = "dbgnn"
    explainer_names: Sequence[str] = ("random_edges",)

    # Optional dataset-specific kwargs.
    #
    # This is the intended escape hatch for handling datasets with multiple
    # networks (e.g., netzschleuder records like `copenhagen` or
    # `sp_high_school`) or multiple possible label targets (e.g., predict
    # `gender` vs `class`).
    #
    # Example:
    #   dataset_name="sp_high_school"
    #   dataset_kwargs={"network": "proximity", "target_attr": "gender"}
    dataset_kwargs: Mapping[str, Any] = field(default_factory=dict)

    # Optional explainer-specific kwargs.
    # Example:
    #   explainer_kwargs={"mask_optim": {"steps": 200, "lr": 0.05}}
    explainer_kwargs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    # Reproducibility
    seed: int = 0

    # Device handling
    # - "auto": use cuda if available else cpu
    # - "cpu" or "cuda"
    device: str = "auto"

    # Training hyperparameters (match notebook)
    epochs: int = 50
    lr: float = 0.005
    p_dropout: float = 0.4
    hidden_dims: Sequence[int] = (16, 32, 8)

    # Random split (match notebook)
    num_test: float = 0.3

    # Benchmark
    n_nodes: int = 50
    topk_fracs: Sequence[float] = (0.01, 0.05, 0.1, 0.2)

    # Optional: counterfactual-style metric.
    # If >0, compute k_flip = minimal number of top-ranked edges that must be dropped
    # (within the candidate set) to flip the prediction, up to this cap.
    # Set to 0 to disable (recommended for very large candidate sets / many nodes).
    k_flip_max: int = 200

    # Diagnostics
    # When True, the benchmark will print lightweight sanity checks that help
    # detect common issues (e.g., graph perturbations not affecting predictions).
    sanity_checks: bool = True

    # Output
    run_dir: str = "runs"
    run_name: str = "temporal_clusters_dbgnn"
