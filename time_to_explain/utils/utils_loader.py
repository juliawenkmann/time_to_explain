from typing import Union
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
# ── Set ROOT_DIR to the repository root (not the tgnnexplainer package dir) ──
from pathlib import Path
import os, sys, subprocess, importlib



def resolve_repo_root() -> Path:
    # 1) Respect an env var if you prefer to set it explicitly
    for key in ("PROJECT_ROOT", "REPO_ROOT", "TIME_TO_EXPLAIN_ROOT"):
        if key in os.environ:
            return Path(os.environ[key]).expanduser().resolve()
    # 2) Ask git
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return Path(out)
    except Exception:
        pass
    # 3) Walk upwards looking for common repo markers
    here = Path.cwd().resolve()
    markers = [".git", "pyproject.toml", "setup.cfg", "setup.py"]
    for p in (here, *here.parents):
        if any((p / m).exists() for m in markers):
            return p
    # 4) Fallback: current working directory
    return here

REPO_ROOT = resolve_repo_root()


def _add_notebooks_src_to_path():
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        candidate = p / "notebooks" / "src"
        if candidate.is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise FileNotFoundError("Could not find 'notebooks/src' from current working directory.")

def _find_checkpoint(models_root: Path, dataset_name: str, model_name: str) -> Path:
    model_name = model_name.lower()
    dataset_name = str(dataset_name)
    candidates = [
        models_root / dataset_name / model_name / f"{model_name}_{dataset_name}_best.pth",
        models_root / dataset_name / "checkpoints" / f"{model_name}_{dataset_name}_best.pth",
        models_root / "checkpoints" / f"{model_name}_{dataset_name}_best.pth",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    search_roots = [
        models_root / dataset_name / model_name,
        models_root / dataset_name,
        models_root / "checkpoints",
        models_root / "runs",
    ]
    for root in search_roots:
        matches = sorted(root.rglob(f"{model_name}*{dataset_name}*.pth"))
        for match in matches:
            if "best" in match.name:
                return match
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Checkpoint not found under {models_root} for {model_name}_{dataset_name}."
    )

def _resolve_checkpoint(models_root: Path, dataset_name: str, model_name: str, model_ref) -> Path:
    if model_ref:
        ref = str(model_ref).strip()
        if ref.lower() in {"tgn", "tgat"}:
            return _find_checkpoint(models_root, dataset_name, model_name)
        ref_path = Path(ref).expanduser()
        if not ref_path.is_absolute():
            ref_path = REPO_ROOT / ref_path
        if ref_path.is_file():
            return ref_path
        if ref_path.is_dir():
            matches = sorted(ref_path.rglob("*.pth"))
            if not matches:
                raise FileNotFoundError(f"No .pth checkpoints found under {ref_path}")
            for match in matches:
                if "best" in match.name:
                    return match
            return matches[0]
        raise FileNotFoundError(f"Checkpoint path not found: {ref_path}")
    return _find_checkpoint(models_root, dataset_name, model_name)