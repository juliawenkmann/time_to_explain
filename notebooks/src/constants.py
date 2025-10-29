# notebooks/src/constants.py
from __future__ import annotations
from pathlib import Path
import os
import sys
from typing import Optional


def _find_repo_root(start: Path | None = None) -> Path:
    """Walk upwards until we find a directory that contains the package folder `time_to_explain`."""
    here = (start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        if (p / "time_to_explain").is_dir():
            return p
    # Fallback: assume CWD is repo root if we didn't find the package dir
    return here

# ---- Derived paths you can rely on in notebooks ----
REPO_ROOT = _find_repo_root()
PKG_DIR = REPO_ROOT / "time_to_explain"
TGN_SUBMODULE_ROOT = REPO_ROOT / "submodules" / "models" / "tgn" /"TTGN"

# Your repo keeps resources at the repo root (sibling to the package).
# Adjust these three lines if your layout differs.
RESOURCES_DIR = REPO_ROOT / "resources"
RAW_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR", RESOURCES_DIR / "datasets"/ "raw"))
PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR", RESOURCES_DIR / "datasets"/ "processed"))
MODELS_ROOT = RESOURCES_DIR / "models"
RESULTS_ROOT = RESOURCES_DIR / "results"

def ensure_repo_importable() -> None:
    """Make sure Python can import `time_to_explain` without editing global env."""
    s = str(REPO_ROOT)
    if s not in sys.path:
        sys.path.insert(0, s)


import re
from pathlib import Path
from typing import Iterable, Optional

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
