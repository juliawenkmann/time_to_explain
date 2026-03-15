from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..core.cli import find_repo_root
from ..core.device import resolve_device
from .notebook_config import load_explainer_notebook_config
from .notebook_helpers import prepend_sys_paths, resolve_explain_index_path


def _deep_update(dst: dict[str, Any], src: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, Mapping) and isinstance(dst.get(key), Mapping):
            nested = dict(dst.get(key, {}))
            dst[key] = _deep_update(nested, value)
        else:
            dst[key] = value
    return dst
def _resolve_bench_relative(bench_root: Path, raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (bench_root / path).resolve()


def resolve_runtime_config(config: Mapping[str, Any], dataset: str, model: str) -> dict[str, Any]:
    merged = dict(config.get("runtime") or {})
    dataset_key = str(dataset)
    model_key = str(model).strip().lower()

    by_dataset = config.get("runtime_by_dataset") or {}
    if dataset_key in by_dataset:
        _deep_update(merged, dict(by_dataset[dataset_key]))

    by_model = config.get("runtime_by_model") or {}
    if model_key in by_model:
        _deep_update(merged, dict(by_model[model_key]))

    by_combo = config.get("runtime_by_combo") or {}
    for combo_key in (
        f"{dataset_key}:{model_key}",
        f"{dataset_key.lower()}:{model_key}",
    ):
        if combo_key in by_combo:
            _deep_update(merged, dict(by_combo[combo_key]))

    return merged


def resolve_checkpoint(bench_root: Path, dataset: str, model: str) -> Path:
    models_root = bench_root / "resources" / "models"
    candidates = [
        models_root / dataset / model / f"{model}_{dataset}_best.pth",
        models_root / dataset / "checkpoints" / f"{model}_{dataset}_best.pth",
        models_root / "checkpoints" / f"{model}_{dataset}_best.pth",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find checkpoint. Checked:\n{checked}")


@dataclass(frozen=True)
class ExplainerNotebookEnv:
    project_root: Path
    bench_root: Path
    notebook_name: str
    config: dict[str, Any]
    paths: dict[str, Path]


@dataclass(frozen=True)
class ExplainerRunContext:
    env: ExplainerNotebookEnv
    dataset: str
    model: str
    settings: dict[str, Any]
    explain_index_path: Path
    checkpoint_path: Path
    device: Any


@dataclass(frozen=True)
class ExplainerNotebookBootstrap:
    project_root: Path
    bench_root: Path
    repo_root: Path
    notebook_name: str
    env: ExplainerNotebookEnv
    config: dict[str, Any]
    settings: dict[str, Any]
    paths: dict[str, Path]


def bootstrap_explainer_notebook(notebook_name: str, *, start: Path | None = None) -> ExplainerNotebookEnv:
    project_root = find_repo_root(start=start, marker="I_explainer_benchmark")
    bench_root = project_root / "I_explainer_benchmark"
    config = load_explainer_notebook_config(project_root, notebook_name)

    path_cfg = config.get("paths") or {}
    named_paths = {
        name: _resolve_bench_relative(bench_root, raw)
        for name, raw in dict(path_cfg.get("named") or {}).items()
    }
    sys_paths = [
        project_root,
        bench_root,
        *(_resolve_bench_relative(bench_root, raw) for raw in list(path_cfg.get("sys") or [])),
        *named_paths.values(),
    ]
    prepend_sys_paths(*(path for path in sys_paths if Path(path).exists()))

    return ExplainerNotebookEnv(
        project_root=project_root,
        bench_root=bench_root,
        notebook_name=str(notebook_name),
        config=config,
        paths=named_paths,
    )


def initialize_explainer_notebook(
    notebook_name: str,
    *,
    dataset: str,
    model: str,
    start: Path | None = None,
) -> ExplainerNotebookBootstrap:
    env = bootstrap_explainer_notebook(notebook_name, start=start)
    settings = resolve_runtime_config(env.config, dataset, model)
    return ExplainerNotebookBootstrap(
        project_root=env.project_root,
        bench_root=env.bench_root,
        repo_root=env.bench_root,
        notebook_name=env.notebook_name,
        env=env,
        config=env.config,
        settings=settings,
        paths=env.paths,
    )


def prepare_explainer_run(
    notebook_name: str,
    *,
    dataset: str,
    model: str,
    start: Path | None = None,
) -> ExplainerRunContext:
    boot = initialize_explainer_notebook(
        notebook_name,
        dataset=dataset,
        model=model,
        start=start,
    )
    explain_index_path = resolve_explain_index_path(boot.env.project_root, dataset)
    checkpoint_path = resolve_checkpoint(boot.env.bench_root, dataset, model)
    return ExplainerRunContext(
        env=boot.env,
        dataset=str(dataset),
        model=str(model).strip().lower(),
        settings=boot.settings,
        explain_index_path=explain_index_path,
        checkpoint_path=checkpoint_path,
        device=resolve_device(),
    )


__all__ = [
    "ExplainerNotebookBootstrap",
    "ExplainerNotebookEnv",
    "ExplainerRunContext",
    "bootstrap_explainer_notebook",
    "initialize_explainer_notebook",
    "prepare_explainer_run",
    "resolve_checkpoint",
    "resolve_device",
    "resolve_runtime_config",
]
