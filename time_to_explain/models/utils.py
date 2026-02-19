from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Mapping

from time_to_explain.utils.cli import slugify


def ensure_workdir(
    runs_root: Path,
    model_type: str,
    dataset: str,
    subdirs: Iterable[str] = ("", "saved_models", "saved_checkpoints", "results", "log"),
) -> Path:
    workdir = Path(runs_root) / f"{slugify(model_type)}_{slugify(dataset)}"
    for folder in subdirs:
        target = workdir if not folder else workdir / folder
        target.mkdir(parents=True, exist_ok=True)
    return workdir


def build_cmd(python_bin: str, script_path: Path | None, extra_args) -> list[str]:
    if script_path is None or not Path(script_path).exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")
    args = list(extra_args) if extra_args else []
    return [python_bin, str(script_path), *args]


def run_cmd(
    cmd: Iterable[str],
    *,
    env: Mapping[str, str] | None = None,
    workdir: Path | None = None,
    dry_run: bool = False,
) -> int:
    workdir = Path(workdir or Path.cwd())
    cmd_list = [str(c) for c in cmd]
    print("$ (cwd=", workdir, ") ", " ".join(shlex.quote(c) for c in cmd_list), sep="")
    if dry_run:
        print("[DRY_RUN] Skipping execution.")
        return 0
    proc = subprocess.run(cmd_list, env=env, cwd=str(workdir), check=False)
    if proc.returncode != 0:
        print(f"[ERROR] process exited with code {proc.returncode}")
    return proc.returncode


def prepare_env(
    *,
    project_root: Path,
    cuda_visible_devices: str | int | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
        print("Set CUDA_VISIBLE_DEVICES =", env["CUDA_VISIBLE_DEVICES"])
    current = env.get("PYTHONPATH")
    if current:
        env["PYTHONPATH"] = os.pathsep.join([str(project_root), current])
    else:
        env["PYTHONPATH"] = str(project_root)
    return env


def ensure_tempme_processed(
    dataset: str,
    *,
    processed_dir: Path,
    resources_datasets: Path,
) -> None:
    processed_dir = Path(processed_dir)
    resources_datasets = Path(resources_datasets)
    processed_dir.mkdir(parents=True, exist_ok=True)
    for suffix in [".csv", ".npy", "_node.npy"]:
        src = resources_datasets / f"ml_{dataset}{suffix}"
        dst = processed_dir / f"ml_{dataset}{suffix}"
        if not src.exists():
            raise FileNotFoundError(f"Expected dataset file missing: {src}")
        if dst.exists() or dst.is_symlink():
            continue
        dst.symlink_to(src)


def export_trained_models(
    model_type: str,
    dataset: str,
    workdir: Path,
    resources_models_dir: Path,
) -> list[Path]:
    """Copy saved `.pth` files into resources/models/<dataset>/<model>."""
    src_dir = Path(workdir) / "saved_models"
    if not src_dir.exists():
        print(f"[WARN] No saved_models directory at {src_dir}; nothing to export.")
        return []

    slug_model = slugify(model_type)
    dataset_slug = slugify(dataset)
    dest_dir = Path(resources_models_dir) / dataset_slug / slug_model
    dest_dir.mkdir(parents=True, exist_ok=True)

    exported: list[Path] = []
    for src_file in sorted(src_dir.glob("*.pth")):
        dest_path = dest_dir / src_file.name
        counter = 2
        while dest_path.exists():
            stem = src_file.stem
            dest_path = dest_dir / f"{stem}_{counter}{src_file.suffix}"
            counter += 1
        shutil.copy2(src_file, dest_path)
        exported.append(dest_path)

    if exported:
        print("Copied trained model(s) to:")
        for path in exported:
            print(" -", path)
    else:
        print(f"[WARN] No .pth files found under {src_dir}")
    return exported

