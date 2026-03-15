from __future__ import annotations

"""Load and normalize explainer configuration JSON."""

import json
from pathlib import Path
from typing import Any, Optional, Tuple

from ..core.constants import ASSET_ROOT, REPO_ROOT
from .catalog import CONFIGS_DIR


def _format_value(val: Any, *, model_type: str, dataset_name: str) -> Any:
    if isinstance(val, str):
        lowered = val.lower()
        if lowered in {"inf", "infinity"}:
            return float("inf")
        try:
            return val.format(model_type=model_type, dataset_name=dataset_name)
        except Exception:
            return val
    if isinstance(val, dict):
        return {k: _format_value(v, model_type=model_type, dataset_name=dataset_name) for k, v in val.items()}
    if isinstance(val, list):
        return [_format_value(v, model_type=model_type, dataset_name=dataset_name) for v in val]
    return val


def _resolve_path(value: str, *, repo_root: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    if not path.exists():
        moved_top_levels = {"resources", "submodules", "out", "paper", "notebooks"}
        parts = path.relative_to(repo_root).parts if path.is_relative_to(repo_root) else ()
        if parts and parts[0] in moved_top_levels:
            bench_path = repo_root / "I_explainer_benchmark" / Path(*parts)
            if bench_path.exists():
                path = bench_path
    return str(path)


def _explainer_config_dirs(configs_dir: Optional[Path]) -> tuple[Path, ...]:
    if configs_dir is not None:
        return (Path(configs_dir),)
    return tuple(
        dict.fromkeys(
            (
                ASSET_ROOT / "configs" / "explainer",
                CONFIGS_DIR,
                REPO_ROOT / "configs" / "explainer",
            )
        )
    )


def load_explainer_config(
    name: str,
    *,
    dataset_name: str,
    model_type: str,
    allow_missing: bool = False,
    configs_dir: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Tuple[Optional[str], Optional[dict], Optional[Path]]:
    root = repo_root or REPO_ROOT
    cfg_dirs = _explainer_config_dirs(configs_dir)
    candidates: list[Path] = []
    for cfg_dir in cfg_dirs:
        candidates.extend(
            [
                cfg_dir / f"{name.lower()}_{dataset_name}.json",
                cfg_dir / f"{name.lower()}.json",
            ]
        )
    config_path = next((p for p in candidates if p.exists()), None)
    if config_path is None:
        if allow_missing:
            return None, None, None
        raise FileNotFoundError(
            "Explainer config not found. Expected one of: "
            + ", ".join(str(p) for p in candidates)
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    adapter = config.get("adapter") or config.get("name") or name
    args = _format_value(dict(config.get("args") or {}), model_type=model_type, dataset_name=dataset_name)
    for key in (
        "results_dir",
        "mcts_saved_dir",
        "explainer_ckpt_dir",
        "explainer_ckpt",
        "checkpoint_path",
        "vendor_dir",
    ):
        if key in args and args[key]:
            args[key] = _resolve_path(str(args[key]), repo_root=root)
    nav_params = args.get("navigator_params")
    if isinstance(nav_params, dict) and nav_params.get("explainer_ckpt_dir"):
        nav_params["explainer_ckpt_dir"] = _resolve_path(str(nav_params["explainer_ckpt_dir"]), repo_root=root)
    return adapter, args, config_path


__all__ = ["load_explainer_config"]
