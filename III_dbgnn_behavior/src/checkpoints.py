from __future__ import annotations

"""Checkpoint helpers used by the notebooks.

The training notebook(s) in this repo save checkpoints under:

    runs/<run_name>/model_state.pt

and optionally a small JSON sidecar:

    runs/<run_name>/meta.json

Unfortunately, different training paths historically wrote slightly different
meta formats (e.g. the unified notebook vs. the COVID helper code). This
module provides a tolerant loader that can reconstruct a PathpyG DBGNN model
from a saved state_dict.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import json

import torch


def _find_checkpoint(run_dir: Path) -> Optional[Path]:
    """Locate a plausible checkpoint file inside a run directory."""

    candidates = [
        run_dir / "model_state.pt",
        run_dir / "model.pt",
        run_dir / "checkpoint.pt",
        run_dir / "best.pt",
        run_dir / "best_model.pt",
        run_dir / "state_dict.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _read_meta(run_dir: Path) -> Dict[str, Any]:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _meta_get(meta: Dict[str, Any], key: str, default):
    """Best-effort lookup that supports both {key: ...} and {cfg: {key: ...}}."""

    if key in meta:
        return meta.get(key, default)
    cfg = meta.get("cfg")
    if isinstance(cfg, dict) and key in cfg:
        return cfg.get(key, default)
    return default


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: torch.nn.Module
    checkpoint_path: Path
    meta: Dict[str, Any]


def load_dbgnn_checkpoint(
    *,
    run_dir: str | Path,
    run_name: str,
    data,
    device: torch.device,
    checkpoint_path: str | Path | None = None,
    hidden_dims: Tuple[int, ...] = (16, 32, 8),
    p_dropout: float = 0.4,
) -> LoadedCheckpoint:
    """Load a PathpyG DBGNN model checkpoint from ``runs/<run_name>``.

    This loader is intentionally notebook-friendly:
    - it infers input feature dimensions from ``data.x`` / ``data.x_h`` when available
    - it infers ``num_classes`` from ``data.y`` when available
    - it tolerates multiple ``meta.json`` formats

    Args:
        run_dir: base directory (e.g. "runs")
        run_name: experiment folder name (e.g. "temporal_clusters_dbgnn")
        data: PyG Data used to infer dimensions.
        device: torch device.
        checkpoint_path: explicit path to a state_dict file. If None, search in the run dir.
        hidden_dims: fallback hidden dims if meta.json is missing.
        p_dropout: fallback dropout if meta.json is missing.

    Returns:
        LoadedCheckpoint(model, checkpoint_path, meta)
    """

    # Import inside function so base package import does not require PathpyG.
    from pathpyG.nn.dbgnn import DBGNN  # type: ignore

    run_dir = Path(run_dir)
    run_paths = [run_dir / str(run_name)]
    # Common notebook layout: checkpoints saved under ./notebooks/runs/<run_name>
    if "notebooks" not in run_dir.parts:
        if run_dir.name == "runs":
            run_paths.append(run_dir.parent / "notebooks" / "runs" / str(run_name))
        else:
            run_paths.append(run_dir / "notebooks" / "runs" / str(run_name))
    # Legacy layout: checkpoints under .../legacy/runs/<run_name>
    legacy_paths = []
    for rp in list(run_paths):
        runs_dir = rp.parent
        if runs_dir.name == "runs":
            legacy_paths.append(runs_dir.parent / "legacy" / "runs" / str(run_name))
    run_paths.extend(legacy_paths)
    # De-duplicate while preserving order
    run_paths = list(dict.fromkeys(run_paths))

    ckpt = None
    chosen_run_path = None
    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path)
        chosen_run_path = ckpt.parent
    else:
        for rp in run_paths:
            ckpt = _find_checkpoint(rp)
            if ckpt is not None and ckpt.exists():
                chosen_run_path = rp
                break

    if ckpt is None or not ckpt.exists():
        tried = "\n".join(f"- {p}" for p in run_paths)
        raise FileNotFoundError(
            "Could not find a checkpoint.\n"
            f"Looked in:\n{tried}\n"
            "Looked for common names like model_state.pt / best.pt / state_dict.pt.\n"
            "Run the training notebook (01_train_unified.ipynb) or pass checkpoint_path explicitly."
        )

    meta = _read_meta(chosen_run_path) if chosen_run_path is not None else {}

    _hidden_dims = tuple(_meta_get(meta, "hidden_dims", hidden_dims))
    _p_dropout = float(_meta_get(meta, "p_dropout", p_dropout))

    # Infer model input dims from data.
    if hasattr(data, "x") and getattr(data, "x") is not None:
        fo_dim = int(getattr(data, "x").size(-1))
    else:
        raise ValueError("data.x is required to infer DBGNN input dimensions")

    if hasattr(data, "x_h") and getattr(data, "x_h") is not None:
        ho_dim = int(getattr(data, "x_h").size(-1))
    else:
        # Some older PathpyG versions used a different name; try best-effort.
        xh = getattr(data, "x_higher_order", None)
        if xh is None:
            raise ValueError("data.x_h (or data.x_higher_order) is required to infer DBGNN HO input dimensions")
        ho_dim = int(xh.size(-1))

    if hasattr(data, "y") and getattr(data, "y") is not None:
        y = getattr(data, "y")
        if isinstance(y, torch.Tensor) and y.numel() > 0:
            num_classes = int(torch.unique(y).numel())
        else:
            num_classes = int(_meta_get(meta, "num_classes", 2))
    else:
        num_classes = int(_meta_get(meta, "num_classes", 2))

    model = DBGNN(
        num_classes=num_classes,
        num_features=(fo_dim, ho_dim),
        hidden_dims=list(_hidden_dims),
        p_dropout=_p_dropout,
    ).to(device)

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return LoadedCheckpoint(model=model, checkpoint_path=ckpt, meta=meta)
