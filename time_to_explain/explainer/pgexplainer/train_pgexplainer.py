from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping, Optional

from time_to_explain.core.factories import build_dataset, build_model
from time_to_explain.models.base import BaseTemporalGNN
from time_to_explain.models.models import models

# PGExplainer bits
from submodules.explainer.CoDy.cody.embedding import StaticEmbedding
from submodules.explainer.CoDy.cody.explainer.baseline.pgexplainer import TPGExplainer


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
    if hasattr(wrapper, "full_ngh_finder") and wrapper.full_ngh_finder is not None:
        return
    # TempMEWrapper already exposes full_ngh_finder; fallback to simple assignment
    if isinstance(wrapper, ):
        setattr(wrapper, "full_ngh_finder", wrapper.full_ngh_finder)
        return
    if hasattr(wrapper, "ngh_finder") and wrapper.ngh_finder is not None:
        wrapper.full_ngh_finder = wrapper.ngh_finder


def _train_impl(
    wrapper,
    *,
    dataset_name: str,
    model_name: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    candidates_size: int,
    out_dir: Path,
) -> tuple[TPGExplainer, object, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    if candidates_size:
        try:
            setattr(wrapper, "explanation_candidates_size", candidates_size)
        except Exception:
            pass

    _ensure_full_neighbor_finder(wrapper)

    embedding = StaticEmbedding(wrapper.dataset, wrapper)
    explainer = TPGExplainer(wrapper, embedding, device=wrapper.device)

    print(f"Training PGExplainer for base model '{model_name}' on dataset '{dataset_name}'")
    dataset_path = getattr(wrapper.dataset, "metadata", {}).get("path")
    if dataset_path:
        print("dataset dir   :", dataset_path)
    print("output dir    :", out_dir)
    base_checkpoint = getattr(wrapper, "checkpoint_path", None)
    if base_checkpoint is None:
        base_checkpoint = getattr(wrapper, "resume_file", None)
    if base_checkpoint:
        print("base checkpoint:", base_checkpoint)
    print("device        :", getattr(wrapper, "device", "cpu"))
    print("epochs        :", epochs, "| lr:", learning_rate, "| batch:", batch_size)

    explainer.train(
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        model_name=model_name,
        save_directory=str(out_dir),
    )
    return explainer, wrapper, out_dir


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
    dataset_bundle = build_dataset(
        {
            "builder": "processed",
            "path": str(dataset_dir),
            "directed": directed,
            "bipartite": bipartite,
        }
    )

    builder_name = {"TGN": "tgn", "TGAT": "tgat", "GRAPHMIXER": "graphmixer"}.get(model_type.upper(), model_type.lower())
    model_cfg = {
        "builder": builder_name,
        "device": device,
        "model_type": model_type.lower(),
        "checkpoint": str(resume_file),
    }
    wrapper = build_model(model_cfg, dataset_bundle.obj)

    model_name = getattr(wrapper, "name", getattr(wrapper, "model_name", model_type))
    setattr(wrapper, "checkpoint_path", resume_file)
    return _train_impl(
        wrapper,
        dataset_name=dataset_name,
        model_name=model_name,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        candidates_size=candidates_size,
        out_dir=out_dir,
    )


def train_pgexplainer_from_config(config: Mapping[str, object]) -> tuple[TPGExplainer, object, Path]:
    """
    Train PGExplainer using registry-backed dataset/model definitions.

    Expected schema:
      dataset: {...}   -> passed to build_dataset
      model: {...}     -> passed to build_model (must yield WrapperTemporalGNN)
      training:
        epochs: int
        learning_rate: float
        batch_size: int
        output_dir: str (optional)
        candidates_size: int (optional)
    """
    if "dataset" not in config or "model" not in config:
        raise KeyError("Config must contain 'dataset' and 'model' sections.")

    dataset_bundle = build_dataset(config["dataset"])
    model = build_model(config["model"], dataset_bundle.obj)
    if not isinstance(model, BaseTemporalGNN):
        raise TypeError("Model builder must return a BaseTemporalGNN-compatible wrapper for PGExplainer training.")

    wrapper = model.raw()
    required_attrs = ("initialize", "get_candidate_events", "predict", "reset_model")
    missing = [attr for attr in required_attrs if not hasattr(wrapper, attr)]
    if missing:
        raise TypeError(
            "Underlying model wrapper is missing required methods: "
            f"{', '.join(missing)}"
        )
    training_cfg = dict(config.get("training", {}))
    epochs = int(training_cfg.get("epochs", 100))
    learning_rate = float(training_cfg.get("learning_rate", 1e-4))
    batch_size = int(training_cfg.get("batch_size", 32))
    candidates_size = int(training_cfg.get("candidates_size", 30))

    output_dir = training_cfg.get("output_dir") or training_cfg.get("out_dir")
    if output_dir is None:
        model_alias = model.name.lower().replace(" ", "_")
        output_dir = Path("artifacts") / dataset_bundle.obj.name / model_alias / "pgexplainer"
    out_path = Path(output_dir)

    setattr(wrapper, "checkpoint_path", model.config.get("checkpoint_path"))
    dataset_meta = getattr(dataset_bundle.obj, "metadata", {})
    if dataset_meta:
        setattr(wrapper.dataset, "metadata", dataset_meta)

    return _train_impl(
        wrapper,
        dataset_name=dataset_bundle.obj.name,
        model_name=model.name,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        candidates_size=candidates_size,
        out_dir=out_path,
    )
