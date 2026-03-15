from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

from .cli import find_repo_root, resolve_benchmark_root

# ---- Derived paths you can rely on in notebooks ----
REPO_ROOT = find_repo_root(marker="I_explainer_benchmark")
BENCH_ROOT = resolve_benchmark_root(REPO_ROOT, marker="I_explainer_benchmark")
ASSET_ROOT = BENCH_ROOT if BENCH_ROOT.is_dir() else REPO_ROOT
SUBMODULES_ROOT = ASSET_ROOT / "submodules"
PKG_DIR = BENCH_ROOT / "src"
TGN_MODELS_ROOT = SUBMODULES_ROOT / "models"
TGN_SUBMODULE_ROOT = TGN_MODELS_ROOT / "tgn"

# Shared benchmark assets may live under I_explainer_benchmark/.
RESOURCES_DIR = ASSET_ROOT / "resources"
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", RESOURCES_DIR / "datasets" / "raw"))
PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR", RESOURCES_DIR / "datasets" / "processed"))
MODELS_ROOT = RESOURCES_DIR / "models"
RESULTS_ROOT = RESOURCES_DIR / "results"


def load_notebook_config(path: Path | None = None) -> dict:
    """
    Load a shared notebook config (seed/device) from configs/notebooks/global.json.
    """
    cfg_path = path or (REPO_ROOT / "configs" / "notebooks" / "global.json")
    if cfg_path.exists():
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    return {}


def ensure_repo_importable() -> None:
    """Make sure Python can import the active benchmark packages without editing global env."""
    ordered_paths = [
        str(ASSET_ROOT),
        str(TGN_MODELS_ROOT),
        str(TGN_SUBMODULE_ROOT),
        str(REPO_ROOT),
        str(PKG_DIR),
    ]
    for s in ordered_paths:
        while s in sys.path:
            sys.path.remove(s)
    for s in reversed(ordered_paths):
        sys.path.insert(0, s)
    if "utils" in sys.modules:
        del sys.modules["utils"]


def get_last_checkpoint(
    models_root: str | Path,
    dataset: str,
    model_type: str,
    *,
    checkpoints_subdir: str = "checkpoints",
    exts: Iterable[str] = ("pth",),         # also accepts ("pth","pt") if you use both
    strict: bool = False,                   # raise if missing dir or no matches
) -> Optional[Path]:
    """
    Find the checkpoint with the largest integer suffix:
        <models_root>/<dataset>/<checkpoints_subdir>/<model_type>-<dataset>-<N>.<ext>

    Examples:
      CoDy/resources/models/wikipedia/checkpoints/TGAT-wikipedia-19.pth

    Returns:
      Path to the highest-index checkpoint, or None if none found (unless strict=True).

    Raises:
      FileNotFoundError if strict=True and directory or matching file is not found.
    """
    ckpt_dir = Path(models_root) / dataset / checkpoints_subdir
    if not ckpt_dir.is_dir():
        if strict:
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
        return None

    # Build regex like: ^TGAT-wikipedia-(\d+)\.(?:pth|pt)$
    ext_pattern = "|".join(re.escape(e.lstrip(".")) for e in exts)
    pattern = re.compile(
        rf"^{re.escape(model_type)}-{re.escape(dataset)}-(\d+)\.(?:{ext_pattern})$"
    )

    best_path: Optional[Path] = None
    best_idx: int = -1

    for p in ckpt_dir.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx > best_idx:
            best_idx = idx
            best_path = p

    if best_path is None and strict:
        raise FileNotFoundError(
            f"No checkpoints like '{model_type}-{dataset}-<N>.({','.join(exts)})' in {ckpt_dir}"
        )
    return best_path
