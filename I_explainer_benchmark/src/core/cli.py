from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence


def find_repo_root(start: Path | None = None, marker: str = "I_explainer_benchmark") -> Path:
    """Walk upwards until we locate the project root (contains `marker`)."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / marker).is_dir():
            return candidate
        if candidate.name == marker and candidate.is_dir():
            return candidate.parent
    raise RuntimeError(
        f"Could not locate the repository root from {here}. "
        "Set PROJECT_ROOT manually if your layout is unusual."
    )


def normalize_benchmark_root(base: Path, marker: str = "I_explainer_benchmark") -> Path:
    """Return the benchmark root for either a repo root or benchmark root input."""
    resolved = Path(base).expanduser().resolve()
    moved = resolved / marker
    if moved.is_dir():
        return moved
    if resolved.name == marker and resolved.is_dir():
        return resolved
    return resolved


def resolve_benchmark_root(
    root: os.PathLike[str] | str | None = None,
    *,
    marker: str = "I_explainer_benchmark",
    env_keys: Sequence[str] = ("PROJECT_ROOT", "REPO_ROOT", "ROOT", "TIME_TO_EXPLAIN_ROOT"),
) -> Path:
    """Resolve the benchmark root from an explicit path, env vars, or the shared repo finder."""
    if root is not None:
        return normalize_benchmark_root(Path(root), marker=marker)

    for key in env_keys:
        raw = os.environ.get(key)
        if raw and raw.strip():
            return normalize_benchmark_root(Path(raw), marker=marker)

    return normalize_benchmark_root(find_repo_root(marker=marker), marker=marker)


def slugify(value: str) -> str:
    sanitized = value.strip().lower().replace(" ", "-").replace("/", "-")
    return sanitized or "run"


def normalize_datasets(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    cleaned: list[str] = []
    for item in value:
        name = str(item).strip()
        if name:
            cleaned.append(name)
    if not cleaned:
        raise ValueError("datasets must contain at least one dataset name.")
    return cleaned


def resolve_path(path_like: str | None, *, root: Path) -> Path | None:
    if not path_like:
        return None
    path = Path(path_like)
    if path.is_absolute():
        return path

    direct = root / path
    moved_root = root / "I_explainer_benchmark"
    moved = moved_root / path

    top = path.parts[0] if path.parts else ""
    moved_top_levels = {
        "configs",
        "notebooks",
        "out",
        "paper",
        "resources",
        "scripts",
        "src",
        "submodules",
        "tests",
    }
    if top in moved_top_levels and not direct.exists() and moved.exists():
        return moved
    return direct


def format_arg_value(value, dataset: str):
    if isinstance(value, str):
        return value.format(dataset=dataset)
    return value


def args_dict_to_list(arg_dict: dict, dataset: str) -> list[str]:
    args: list[str] = []
    for key, value in arg_dict.items():
        flag = key if str(key).startswith("-") else f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                args.extend([flag, str(format_arg_value(item, dataset))])
            continue
        args.extend([flag, str(format_arg_value(value, dataset))])
    return args


__all__ = [
    "args_dict_to_list",
    "find_repo_root",
    "format_arg_value",
    "normalize_datasets",
    "normalize_benchmark_root",
    "resolve_path",
    "resolve_benchmark_root",
    "slugify",
]
