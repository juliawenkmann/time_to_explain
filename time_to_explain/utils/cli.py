from __future__ import annotations

from pathlib import Path
from typing import Iterable


def find_repo_root(start: Path | None = None, marker: str = "time_to_explain") -> Path:
    """Walk upwards until we locate the project root (contains `marker`)."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / marker).is_dir():
            return candidate
    raise RuntimeError(
        f"Could not locate the repository root from {here}. "
        "Set PROJECT_ROOT manually if your layout is unusual."
    )


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
    return path if path.is_absolute() else root / path


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
    "resolve_path",
    "slugify",
]
