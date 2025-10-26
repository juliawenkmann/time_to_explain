from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

# -- Your variable-based wrapper factory (adjust if you named it differently)
from time_to_explain.models.wrapper import create_tgnn_wrapper
from time_to_explain.models.adapter.tgn import to_data_object  # helper used to build neighbor finder

# PGExplainer bits
from submodules.explainer.CoDy.cody.embedding import StaticEmbedding
from submodules.explainer.CoDy.cody.explainer.baseline.pgexplainer import TPGExplainer

# Neighbor finder for eval
from submodules.models.tgn.utils.utils import get_neighbor_finder


def _find_highest_checkpoint(
    models_root: str | Path,
    dataset: str,
    model_type: str,
    *,
    checkpoints_subdir: str = "checkpoints",
    exts: Iterable[str] = ("pth", "pt"),
) -> Optional[Path]:
    """
    Find the checkpoint with the largest integer suffix:
      <models_root>/<dataset>/<checkpoints_subdir>/<model_type>-<dataset>-<N>.<ext>
    Example:
      .../resources/models/wikipedia/checkpoints/TGAT-wikipedia-19.pth
    """
    ckpt_dir = Path(models_root) / dataset / checkpoints_subdir
    if not ckpt_dir.is_dir():
        return None

    ext_pattern = "|".join(re.escape(e.lstrip(".")) for e in exts)
    pat = re.compile(rf"^{re.escape(model_type)}-{re.escape(dataset)}-(\d+)\.(?:{ext_pattern})$")
    best, best_idx = None, -1
    for p in ckpt_dir.iterdir():
        if p.is_file():
            m = pat.match(p.name)
            if m:
                idx = int(m.group(1))
                if idx > best_idx:
                    best_idx, best = idx, p
    return best


def _resolve_base_model_checkpoint(
    *,
    models_root: str | Path,
    dataset: str,
    model_type: str,
    tgn_checkpoint: str | Path | None,
) -> Optional[str]:
    """
    Resolve the base (pretrained) TGN/TGAT file to explain.
    Priority:
      1) If tgn_checkpoint is a file -> use it.
      2) If tgn_checkpoint is a directory -> pick newest matching *.pth/*.pt in it by suffix.
      3) Else search <models_root>/<dataset>/checkpoints/<model_type>-<dataset>-<N>.pth with highest N.
      4) Else fall back to <models_root>/<dataset>/<model_type>-<dataset>.pth if present.
      5) Else return None (caller may choose to raise).
    """
    if tgn_checkpoint:
        p = Path(tgn_checkpoint)
        if p.is_file():
            return str(p)
        if p.is_dir():
            best = _find_highest_checkpoint(p.parent.parent, dataset, model_type)  # if user handed .../checkpoints
            if best:
                return str(best)
            # or newest *.pt/pth in that directory
            cands = sorted(list(p.glob("*.pt")) + list(p.glob("*.pth")), key=lambda x: x.stat().st_mtime)
            return str(cands[-1]) if cands else None

    # 3) Look for highest-index checkpoint in the standard layout
    best = _find_highest_checkpoint(models_root, dataset, model_type)
    if best:
        return str(best)

    # 4) Flat file without index
    flat = Path(models_root) / dataset / f"{model_type}-{dataset}.pth"
    if flat.exists():
        return str(flat)

    return None


def _ensure_full_neighbor_finder(wrapper) -> None:
    """Attach a full-graph neighbor finder to the wrapper if missing."""
    if getattr(wrapper, "full_ngh_finder", None) is None:
        data_obj = to_data_object(wrapper.dataset)
        wrapper.full_ngh_finder = get_neighbor_finder(data_obj, uniform=False)
    if getattr(wrapper, "ngh_finder", None) is None:
        wrapper.ngh_finder = wrapper.full_ngh_finder  # convenience


def train_pgexplainer(
    *,
    dataset_name: str,                 # e.g., "wikipedia"
    model_type: str,                   # "TGN" or "TGAT" (or "TTGN" if you kept that label)
    epochs: int = 100,
    learning_rate: float = 1e-4,
    batch_size: int = 16,
    directed: bool = False,
    bipartite: bool = False,
    device: str = "auto",              # "auto" | "cpu" | "cuda" | "mps"
    cuda: bool = True,                 # legacy flag still supported by your device selector
    candidates_size: int = 30,         # kept for compatibility (some wrappers read it)
    # Base model checkpoint to explain:
    tgn_checkpoint: str | Path | None = None,
    # Where to place the PGExplainer outputs:
    out_dir: str | Path | None = None,  # default: MODELS_ROOT/<dataset>/pg_explainer
    # Roots (normally from constants.py)
    models_root: str | Path = None,
    processed_root: str | Path = None,
):
    """
    Train PGExplainer in-notebook with variables only.

    Returns:
      explainer, wrapper, out_dir (Path)
    """
    dataset_dir = Path(processed_root) / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Processed dataset not found: {dataset_dir}")

    out_dir = Path(out_dir) if out_dir else Path(models_root) / dataset_name / "pg_explainer"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve base model checkpoint (file)
    resume_file = _resolve_base_model_checkpoint(
        models_root=models_root,
        dataset=dataset_name,
        model_type=model_type,
        tgn_checkpoint=tgn_checkpoint,
    )
    if resume_file is None:
        raise FileNotFoundError(
            "Could not resolve a base TGN/TGAT checkpoint to explain.\n"
            f"Searched under: {models_root}/{dataset_name}/checkpoints and "
            f"{models_root}/{dataset_name}/{model_type}-{dataset_name}.pth\n"
            "Pass `tgn_checkpoint` explicitly if needed."
        )

    # Build the wrapper (loads weights from resume_file)
    wrapper = create_tgnn_wrapper(
        model_type=model_type,
        dataset_dir=str(dataset_dir),
        directed=directed,
        bipartite=bipartite,
        device=device,
        cuda=cuda,
        update_memory_at_start=False,
        checkpoint_path=resume_file,   # <-- FILE to load
    )

    # Optional compatibility knob (some code reads this)
    try:
        setattr(wrapper, "explanation_candidates_size", candidates_size)
    except Exception:
        pass

    # Make sure eval neighbor finder exists
    _ensure_full_neighbor_finder(wrapper)

    # Build PGExplainer
    embedding = StaticEmbedding(wrapper.dataset, wrapper)
    explainer = TPGExplainer(wrapper, embedding, device=wrapper.device)

    # Train PGExplainer
    model_name = getattr(wrapper, "name", getattr(wrapper, "model_name", model_type))
    print(f"Training PGExplainer for base model '{model_name}' on dataset '{dataset_name}'")
    print("dataset dir   :", dataset_dir)
    print("output dir    :", out_dir)
    print("base checkpoint:", resume_file)
    print("device        :", wrapper.device)
    print("epochs        :", epochs, "| lr:", learning_rate, "| batch:", batch_size)

    explainer.train(
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        model_name=model_name,
        save_directory=str(out_dir),
    )

    return explainer, wrapper, out_dir
