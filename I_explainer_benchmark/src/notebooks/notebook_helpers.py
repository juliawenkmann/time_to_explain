from __future__ import annotations

import os
import random
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any, Sequence


def prepend_sys_paths(*paths: Path | str) -> None:
    """Prepend paths to sys.path deterministically without duplicate entries."""
    for raw in map(str, paths):
        with suppress(ValueError):
            sys.path.remove(raw)
        sys.path.insert(0, raw)


def parse_bool(value: Any, default: bool) -> bool:
    """Parse common boolean-like notebook config values."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


def set_global_seed(seed: int) -> None:
    """Seed Python/NumPy/PyTorch RNGs and force deterministic CuDNN behavior."""
    import numpy as _np
    import torch as _torch

    value = int(seed)
    os.environ["PYTHONHASHSEED"] = str(value)
    random.seed(value)
    _np.random.seed(value)
    _torch.manual_seed(value)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(value)
    if hasattr(_torch.backends, "cudnn"):
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False


def require_existing_file(path: Path | str, *, what: str = "file") -> Path:
    """Resolve an existing file path or raise FileNotFoundError."""
    raw = Path(path).expanduser()
    try:
        return raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing {what}: {raw}") from exc


def resolve_explain_index_path(repo_root: Path | str, dataset_name: str) -> Path:
    """Resolve the explain-index path across the active benchmark layouts."""
    repo_root = Path(repo_root)
    dataset_file = f"{str(dataset_name)}.csv"
    candidates = [
        repo_root / "I_explainer_benchmark" / "resources" / "explainer" / "explain_index" / dataset_file,
        repo_root / "I_explainer_benchmark" / "resources" / "datasets" / "explain_index" / dataset_file,
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Missing explain index for dataset={str(dataset_name)!r}. Checked:\n{checked}"
    )


def load_explain_event_ids(explain_index_path: Path | str) -> list[int]:
    """Load ordered explain-index event ids from a canonical explain-index CSV."""
    from ..data.explain_index import load_explain_idx

    return [int(v) for v in load_explain_idx(str(explain_index_path), start=0)]


def forced_target_event_ids_for_combo(dataset_name: str, model_name: str) -> list[int]:
    """Return default forced target ids for specific notebook run combos."""
    dataset_key = str(dataset_name).strip().lower()
    model_key = str(model_name).strip().lower()
    if dataset_key == "simulate_v1" and model_key == "tgat":
        return [61]
    return []


_TGNNEXPLAINER_VENDOR_DATASET_ALIASES = {
    "uci": "wikipedia",
    "ucim": "wikipedia",
    "uci_messages": "wikipedia",
    "uci-messages": "wikipedia",
    "ucimessages": "wikipedia",
    "uci_motif": "wikipedia",
    "uci-motif": "wikipedia",
    "ucim_motif": "wikipedia",
    "ucim-motif": "wikipedia",
}


def resolve_tgnnexplainer_vendor_dataset(
    dataset_name: str,
    *,
    available: Sequence[str] | None = None,
) -> str:
    """Map benchmark dataset names to the closest vendored TGNNExplainer config key."""
    dataset_key = str(dataset_name).strip().lower()
    resolved = _TGNNEXPLAINER_VENDOR_DATASET_ALIASES.get(dataset_key, dataset_key)
    if available is None:
        return resolved

    available_keys = {str(value).strip().lower() for value in available}
    if resolved not in available_keys:
        raise KeyError(
            f"TGNNExplainer vendor dataset alias {resolved!r} is not available for {dataset_name!r}. "
            f"Available keys: {sorted(available_keys)}"
        )
    return resolved


def select_explain_event_ids(
    explain_index_path: Path | str,
    *,
    n_eval_events: int,
    start: int = 0,
    include_event_ids: Sequence[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Return (all_event_ids, selected_event_ids) with one uniform slicing rule."""
    all_event_ids = load_explain_event_ids(explain_index_path)
    if not all_event_ids:
        raise RuntimeError(f"No event indices loaded from explain index: {explain_index_path}")

    start_idx = max(0, int(start))
    budget = int(min(max(1, int(n_eval_events)), len(all_event_ids)))
    end_idx = start_idx + budget
    selected = all_event_ids[start_idx:end_idx]
    forced = [int(v) for v in (include_event_ids or [])]
    if forced:
        merged: list[int] = []
        seen: set[int] = set()
        for event_id in forced:
            if event_id in seen:
                continue
            merged.append(event_id)
            seen.add(event_id)
            if len(merged) >= budget:
                break
        for event_id in selected:
            if event_id in seen:
                continue
            merged.append(int(event_id))
            seen.add(int(event_id))
            if len(merged) >= budget:
                break
        selected = merged
    if not selected:
        raise RuntimeError(
            f"No target events selected from explain index {explain_index_path} "
            f"with start={start_idx}, n_eval_events={int(n_eval_events)}."
        )
    return all_event_ids, selected


def build_edge_anchors(event_ids: Sequence[int]) -> list[dict[str, int | str]]:
    """Build edge-target anchors from event ids."""
    return [{"target_kind": "edge", "event_idx": int(e)} for e in event_ids]


__all__ = [
    "prepend_sys_paths",
    "parse_bool",
    "set_global_seed",
    "require_existing_file",
    "resolve_explain_index_path",
    "load_explain_event_ids",
    "forced_target_event_ids_for_combo",
    "resolve_tgnnexplainer_vendor_dataset",
    "select_explain_event_ids",
    "build_edge_anchors",
]
