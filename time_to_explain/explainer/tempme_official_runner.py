from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple
import runpy
import shutil
import sys

import h5py
import numpy as np
import torch


_DEFAULT_RATIOS = [
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
    0.14,
    0.16,
    0.18,
    0.20,
    0.22,
    0.24,
    0.26,
    0.28,
    0.30,
]


@dataclass
class TempMEOfficialTrainingConfig:
    dataset_name: str
    base_type: str = "tgn"
    processed_dir: Path = Path("resources/datasets/processed")
    tempme_root: Optional[Path] = None
    base_ckpt_path: Optional[Path] = None
    output_ckpt: Optional[Path] = None
    prepare_if_missing: bool = True
    preprocess_batch_size: int = 1024
    preprocess_seed: Optional[int] = 2023
    preprocess_overwrite: bool = False
    preprocess_verbose: bool = True

    # Training hyperparams (defaults match temp_exp_main.py)
    bs: int = 500
    test_bs: int = 500
    n_degree: Optional[int] = None
    n_head: int = 4
    n_epoch: int = 150
    out_dim: int = 40
    hid_dim: int = 64
    temp: float = 0.07
    prior_p: float = 0.3
    lr: float = 1e-3
    drop_out: float = 0.1
    if_attn: bool = True
    if_bern: bool = True
    weight_decay: float = 0.0
    beta: float = 0.5
    lr_decay: float = 0.999
    verbose: int = 1
    test_threshold: bool = True
    save_model: bool = True
    ratios: list[float] = field(default_factory=lambda: list(_DEFAULT_RATIOS))

    device: Optional[torch.device] = None


def _resolve_device(device: Optional[torch.device | str]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None or str(device).lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")
    try:
        dev = torch.device(device)
    except Exception:
        return torch.device("cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if dev.type == "mps" and not torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("cpu")
    return dev


def _infer_n_degree(train_h5: Path) -> int:
    with h5py.File(str(train_h5), "r") as handle:
        if "subgraph_src_0" not in handle:
            raise KeyError("TempME train_cat.h5 missing 'subgraph_src_0'.")
        width = int(handle["subgraph_src_0"].shape[1])
    if width % 3 != 0:
        raise ValueError(f"Unexpected subgraph_src_0 width={width}; cannot infer n_degree.")
    return width // 3


def _link_or_copy(src: Path, dst: Path, *, overwrite: bool = False) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def _missing_processed_files(dataset: str, processed_dir: Path) -> list[Path]:
    required = [
        f"ml_{dataset}.csv",
        f"ml_{dataset}.npy",
        f"ml_{dataset}_node.npy",
        f"{dataset}_train_cat.h5",
        f"{dataset}_test_cat.h5",
        f"{dataset}_train_edge.npy",
        f"{dataset}_test_edge.npy",
    ]
    missing = []
    for name in required:
        src = processed_dir / name
        if not src.exists():
            missing.append(src)
    return missing


def _ensure_tempme_processed(
    dataset: str,
    *,
    processed_dir: Path,
    tempme_root: Path,
) -> None:
    processed_dir = Path(processed_dir)
    tempme_root = Path(tempme_root)
    tempme_processed = tempme_root / "processed"
    tempme_processed.mkdir(parents=True, exist_ok=True)

    required = [
        f"ml_{dataset}.csv",
        f"ml_{dataset}.npy",
        f"ml_{dataset}_node.npy",
        f"{dataset}_train_cat.h5",
        f"{dataset}_test_cat.h5",
        f"{dataset}_train_edge.npy",
        f"{dataset}_test_edge.npy",
    ]
    missing = []
    for name in required:
        src = processed_dir / name
        dst = tempme_processed / name
        if not src.exists():
            missing.append(src)
            continue
        _link_or_copy(src, dst)
    if missing:
        msg = "TempME processed files missing:\n" + "\n".join(f" - {p}" for p in missing)
        raise FileNotFoundError(msg)


def _load_tempme_module(tempme_root: Path) -> dict:
    tempme_root = Path(tempme_root).resolve()
    script = tempme_root / "temp_exp_main.py"
    if not script.exists():
        raise FileNotFoundError(f"TempME script not found: {script}")

    tempme_root_str = str(tempme_root)
    if tempme_root_str not in sys.path:
        sys.path.insert(0, tempme_root_str)
    argv_backup = list(sys.argv)
    try:
        sys.argv = [str(script)]
        mod_globals = runpy.run_path(str(script))
    finally:
        sys.argv = argv_backup
    return mod_globals


def train_tempme_official(
    cfg: TempMEOfficialTrainingConfig,
) -> Tuple[Optional[torch.nn.Module], Optional[Path]]:
    tempme_root = (
        Path(cfg.tempme_root)
        if cfg.tempme_root is not None
        else Path(__file__).resolve().parents[2] / "submodules" / "explainer" / "TempME"
    )
    device = _resolve_device(cfg.device)

    missing = _missing_processed_files(cfg.dataset_name, Path(cfg.processed_dir))
    if missing and cfg.prepare_if_missing:
        from time_to_explain.data.tempme_preprocess import (
            TempMEPreprocessConfig,
            prepare_tempme_dataset,
        )

        print("[TempME] Preprocessed files missing; generating from resources/datasets...")
        prep_cfg = TempMEPreprocessConfig(
            dataset_name=cfg.dataset_name,
            processed_dir=Path(cfg.processed_dir),
            output_dir=Path(cfg.processed_dir),
            n_degree=cfg.n_degree,
            batch_size=cfg.preprocess_batch_size,
            seed=cfg.preprocess_seed,
            overwrite=cfg.preprocess_overwrite,
            verbose=cfg.preprocess_verbose,
        )
        prepare_tempme_dataset(prep_cfg)

    _ensure_tempme_processed(
        cfg.dataset_name,
        processed_dir=cfg.processed_dir,
        tempme_root=tempme_root,
    )

    train_h5 = tempme_root / "processed" / f"{cfg.dataset_name}_train_cat.h5"
    test_h5 = tempme_root / "processed" / f"{cfg.dataset_name}_test_cat.h5"
    train_edge = tempme_root / "processed" / f"{cfg.dataset_name}_train_edge.npy"
    test_edge = tempme_root / "processed" / f"{cfg.dataset_name}_test_edge.npy"
    if cfg.n_degree is None:
        cfg.n_degree = _infer_n_degree(train_h5)

    base_ckpt = (
        Path(cfg.base_ckpt_path)
        if cfg.base_ckpt_path is not None
        else tempme_root / "params" / "tgnn" / f"{cfg.base_type}_{cfg.dataset_name}.pt"
    )
    if not base_ckpt.exists():
        raise FileNotFoundError(f"TempME base model not found: {base_ckpt}")

    tempme_mod = _load_tempme_module(tempme_root)

    args = SimpleNamespace(
        gpu=0,
        base_type=cfg.base_type,
        data=cfg.dataset_name,
        bs=int(cfg.bs),
        test_bs=int(cfg.test_bs),
        n_degree=int(cfg.n_degree),
        n_head=int(cfg.n_head),
        n_epoch=int(cfg.n_epoch),
        out_dim=int(cfg.out_dim),
        hid_dim=int(cfg.hid_dim),
        temp=float(cfg.temp),
        prior_p=float(cfg.prior_p),
        lr=float(cfg.lr),
        drop_out=float(cfg.drop_out),
        if_attn=bool(cfg.if_attn),
        if_bern=bool(cfg.if_bern),
        save_model=bool(cfg.save_model),
        test_threshold=bool(cfg.test_threshold),
        verbose=int(cfg.verbose),
        weight_decay=float(cfg.weight_decay),
        beta=float(cfg.beta),
        lr_decay=float(cfg.lr_decay),
        task_type="temporal explanation",
        ratios=list(cfg.ratios),
        device=device,
    )
    tempme_mod["args"] = args

    base_model = torch.load(base_ckpt, map_location=device).to(device)
    if cfg.base_type == "tgn" and hasattr(base_model, "forbidden_memory_update"):
        base_model.forbidden_memory_update = True

    with h5py.File(str(train_h5), "r") as train_handle, h5py.File(str(test_h5), "r") as test_handle:
        load_subgraph_margin = tempme_mod["load_subgraph_margin"]
        train_pack = load_subgraph_margin(args, train_handle)
        test_pack = load_subgraph_margin(args, test_handle)

    train_edge_arr = np.load(train_edge)
    test_edge_arr = np.load(test_edge)

    train_fn = tempme_mod["train"]
    train_fn(args, base_model, train_pack, test_pack, train_edge_arr, test_edge_arr)

    saved = tempme_root / "params" / "explainer" / cfg.base_type / f"{cfg.dataset_name}.pt"
    if not saved.exists():
        return None, None

    explainer = torch.load(saved, map_location=device)
    if cfg.output_ckpt:
        cfg.output_ckpt.parent.mkdir(parents=True, exist_ok=True)
        _link_or_copy(saved, cfg.output_ckpt, overwrite=True)
    return explainer, saved


__all__ = ["TempMEOfficialTrainingConfig", "train_tempme_official"]
