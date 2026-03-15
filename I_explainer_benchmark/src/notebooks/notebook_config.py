from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _deep_update(dst: dict[str, Any], src: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, Mapping) and isinstance(dst.get(key), Mapping):
            nested = dict(dst.get(key, {}))
            dst[key] = _deep_update(nested, value)
        else:
            dst[key] = value
    return dst


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _bench_root_from_repo_root(repo_root: Path) -> Path:
    repo_root = repo_root.expanduser().resolve()
    bench_root = repo_root / "I_explainer_benchmark"
    if bench_root.is_dir():
        return bench_root
    if repo_root.name == "I_explainer_benchmark" and repo_root.is_dir():
        return repo_root
    raise FileNotFoundError(
        "Could not resolve I_explainer_benchmark root from "
        f"{repo_root}. Expected either <repo>/I_explainer_benchmark or that path itself."
    )


def load_explainer_notebook_config(repo_root: str | Path, notebook_filename: str) -> dict[str, Any]:
    """Load merged config for a notebook under notebooks/explainer_notebooks.

    Merge precedence (later wins):
    1) configs/global/notebooks.json
    2) configs/notebooks/explainer_notebooks/_common.json
    3) configs/notebooks/explainer_notebooks/<notebook_stem>.json
    """
    bench_root = _bench_root_from_repo_root(Path(repo_root))
    cfg_root = bench_root / "configs"
    notebook_stem = Path(str(notebook_filename)).stem

    merged: dict[str, Any] = {}
    for path in (
        cfg_root / "global" / "notebooks.json",
        cfg_root / "notebooks" / "explainer_notebooks" / "_common.json",
        cfg_root / "notebooks" / "explainer_notebooks" / f"{notebook_stem}.json",
    ):
        payload = _read_json(path)
        if payload:
            _deep_update(merged, payload)

    return merged


__all__ = ["load_explainer_notebook_config"]
