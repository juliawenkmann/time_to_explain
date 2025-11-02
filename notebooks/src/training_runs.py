from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from time_to_explain.models.training import train_model_from_config


@dataclass(slots=True)
class ModelPaths:
    base: Path
    checkpoint_dir: Path
    model_export_dir: Path
    results_path: Path


@dataclass(slots=True)
class TrainingSpec:
    dataset_name: str
    model_key: str
    model_label: str
    dataset_config: Dict[str, object]
    model_config: Dict[str, object]
    training_config: Dict[str, object]
    paths: ModelPaths


@dataclass(slots=True)
class TrainingOutcome:
    dataset: str
    model: str
    status: str
    duration_sec: float | None = None
    best_checkpoint: str | None = None
    final_model: str | None = None
    results_path: str | None = None
    notes: str | None = None


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _clean_config(config: Mapping[str, object]) -> Dict[str, object]:
    return {k: v for k, v in config.items() if v is not None}


def build_training_specs(
    *,
    datasets: Sequence[str],
    model_types: Sequence[str],
    processed_root: Path,
    models_root: Path,
    dataset_defaults: Mapping[str, Mapping[str, object]] | None = None,
    model_builders: Mapping[str, str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    model_overrides: Mapping[str, Mapping[str, object]] | None = None,
    training_params: Mapping[str, object],
    shared_model_params: Mapping[str, object] | None = None,
) -> List[TrainingSpec]:
    """
    Assemble TrainingSpec entries for each dataset/model combination.
    """
    dataset_defaults = dataset_defaults or {}
    model_builders = model_builders or {key.lower(): key.lower() for key in model_types}
    model_labels = model_labels or {key.lower(): key for key in model_types}
    model_overrides = model_overrides or {}
    shared_model_params = shared_model_params or {}

    specs: List[TrainingSpec] = []

    for dataset_name in datasets:
        dataset_dir = processed_root / dataset_name
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        ds_defaults = dict(dataset_defaults.get(dataset_name, {}))
        dataset_model_overrides = ds_defaults.pop("model_overrides", {})
        ds_defaults.setdefault("directed", False)
        ds_defaults.setdefault("bipartite", False)

        base_dataset_cfg = {
            "builder": "processed",
            "path": str(dataset_dir),
            "directed": bool(ds_defaults["directed"]),
            "bipartite": bool(ds_defaults["bipartite"]),
        }

        for model_name in model_types:
            key = model_name.lower()
            if key not in model_builders:
                raise KeyError(f"No builder registered for model '{model_name}' (normalized '{key}').")

            label = model_labels.get(key, model_name)
            base_dir = models_root / dataset_name / key
            paths = ModelPaths(
                base=base_dir,
                checkpoint_dir=base_dir / "checkpoints",
                model_export_dir=base_dir / "models",
                results_path=base_dir / "train_history.npz",
            )

            model_cfg: Dict[str, object] = {
                "builder": model_builders[key],
                "model_type": key,
                **shared_model_params,
            }
            for overrides in (dataset_model_overrides, model_overrides.get(key, {})):
                if overrides:
                    model_cfg.update(overrides)

            spec = TrainingSpec(
                dataset_name=dataset_name,
                model_key=key,
                model_label=label,
                dataset_config=_clean_config(base_dataset_cfg),
                model_config=_clean_config(model_cfg),
                training_config={
                    **training_params,
                    "output_dir": str(paths.base),
                    "checkpoint_dir": str(paths.checkpoint_dir),
                    "model_dir": str(paths.model_export_dir),
                    "results_path": str(paths.results_path),
                },
                paths=paths,
            )
            specs.append(spec)
    return specs


def run_training_specs(
    specs: Iterable[TrainingSpec],
    *,
    force_retrain: bool = False,
    skip_if_model_exists: bool = True,
    final_model_name: str = "model_final.pth",
) -> List[TrainingOutcome]:
    """
    Execute training for each spec, returning a list of outcomes.
    """
    outcomes: List[TrainingOutcome] = []

    for spec in specs:
        paths = spec.paths
        for path in (paths.base, paths.checkpoint_dir, paths.model_export_dir, paths.results_path.parent):
            _ensure_directory(path)

        final_model_path = paths.model_export_dir / final_model_name

        if not force_retrain and skip_if_model_exists and final_model_path.exists():
            outcomes.append(
                TrainingOutcome(
                    dataset=spec.dataset_name,
                    model=spec.model_label,
                    status="skipped",
                    final_model=str(final_model_path),
                    results_path=str(paths.results_path),
                    notes="Existing final model detected",
                )
            )
            continue

        config_payload = {
            "dataset": dict(spec.dataset_config),
            "model": dict(spec.model_config),
            "training": dict(spec.training_config),
        }

        from time import perf_counter

        start = perf_counter()
        result = train_model_from_config(config_payload)
        elapsed = perf_counter() - start

        last_checkpoint = result.get("last_checkpoint")
        if last_checkpoint is None:
            candidates = sorted(paths.checkpoint_dir.glob("epoch-*.pth"))
            last_checkpoint = candidates[-1] if candidates else None

        outcomes.append(
            TrainingOutcome(
                dataset=spec.dataset_name,
                model=spec.model_label,
                status="trained",
                duration_sec=elapsed,
                best_checkpoint=str(last_checkpoint) if last_checkpoint else None,
                final_model=str(final_model_path),
                results_path=str(paths.results_path),
            )
        )

    return outcomes
