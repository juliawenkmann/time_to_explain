from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from tqdm.auto import tqdm


def runs_dir_for_context(project_root: str | Path, dataset_name: str, base_type: str) -> Path:
    out_dir = (
        Path(project_root).expanduser().resolve()
        / "I_explainer_benchmark"
        / "resources"
        / "explainer"
        / "tempme"
        / str(dataset_name)
        / str(base_type)
        / "runs"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def stage_cache_path(runs_dir: str | Path, stage: str) -> Path:
    return Path(runs_dir).expanduser().resolve() / f"latest_{stage}.json"


def latest_metrics_path(runs_dir: str | Path) -> Path | None:
    base = Path(runs_dir).expanduser().resolve()
    candidates = sorted(base.glob("metrics_*.csv"))
    if not candidates:
        return None
    non_empty = [p for p in candidates if p.stat().st_size > 0]
    return non_empty[-1] if non_empty else candidates[-1]


def _clean_metric_values(values: Mapping[str, Any]) -> dict[str, float]:
    return {str(k): float(v) for k, v in values.items() if pd.notna(v)}


def save_stage_cache(stage: str, metrics: Mapping[str, Any], logs: Sequence[str], runs_dir: str | Path) -> Path:
    path = stage_cache_path(runs_dir, stage)
    payload = {
        "stage": str(stage),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": dict(metrics),
        "logs": list(logs),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_stage_cache_metrics(stage: str, runs_dir: str | Path) -> dict[str, float]:
    cache_path = stage_cache_path(runs_dir, stage)
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        out = _clean_metric_values(payload.get("metrics") or {})
        if out:
            print(f"Loaded cached {stage} metrics from: {cache_path}")
        return out

    metrics_path = latest_metrics_path(runs_dir)
    if metrics_path is None or metrics_path.stat().st_size == 0:
        return {}

    df = pd.read_csv(metrics_path)
    if "stage" not in df.columns:
        return {}

    rows = df[df["stage"] == stage]
    if rows.empty:
        return {}

    row = rows.iloc[-1].to_dict()
    row.pop("stage", None)
    out = _clean_metric_values(row)
    if out:
        print(f"Loaded cached {stage} metrics from: {metrics_path}")
    return out


def _cli_args(args: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for key, value in args.items():
        out.extend([f"--{key}", str(value)])
    return out


def _cmd_arg_value(cmd: Sequence[str], key: str) -> str | None:
    flag = f"--{key}"
    for i, token in enumerate(cmd[:-1]):
        if token == flag:
            return str(cmd[i + 1])
    return None


def _expected_epochs_from_cmd(cmd: Sequence[str]) -> int | None:
    raw = _cmd_arg_value(cmd, "n_epoch")
    return None if raw is None else int(raw)


def _format_eta(seconds: float) -> str:
    total = max(int(seconds), 0)
    mins, sec = divmod(total, 60)
    hrs, mins = divmod(mins, 60)
    if hrs:
        return f"{hrs}h {mins:02d}m"
    if mins:
        return f"{mins}m {sec:02d}s"
    return f"{sec}s"


def run_and_capture(
    cmd: list[str],
    cwd: str | Path,
    *,
    project_root: str | Path,
    progress_title: str | None = None,
) -> tuple[int, list[str]]:
    env = os.environ.copy()
    root = str(Path(project_root).expanduser().resolve())
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{root}{os.pathsep}{pythonpath}" if pythonpath else root

    print("$", shlex.join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(cwd).expanduser().resolve()),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines: list[str] = []
    assert proc.stdout is not None

    expected_epochs = _expected_epochs_from_cmd(cmd)
    epoch_bar = None
    if progress_title and expected_epochs is not None and expected_epochs > 0:
        epoch_bar = tqdm(total=expected_epochs, desc=progress_title, unit="epoch", leave=True)

    epoch_re = re.compile(r"Training Epoch:\s*(\d+)\s*\|")
    t0 = time.perf_counter()
    last_completed_epoch = 0

    for raw in proc.stdout:
        print(raw, end="")
        line = raw.rstrip("\n")
        lines.append(line)

        if epoch_bar is None:
            continue

        m = epoch_re.search(line)
        if m is None:
            continue

        completed_epoch = int(m.group(1)) + 1
        if completed_epoch <= last_completed_epoch:
            continue

        step = completed_epoch - last_completed_epoch
        last_completed_epoch = completed_epoch
        epoch_bar.update(step)

        elapsed = time.perf_counter() - t0
        sec_per_epoch = elapsed / max(last_completed_epoch, 1)
        remaining_epochs = max(epoch_bar.total - last_completed_epoch, 0)
        eta_seconds = sec_per_epoch * remaining_epochs
        epoch_bar.set_postfix_str(f"eta~{_format_eta(eta_seconds)}", refresh=False)

    rc = proc.wait()

    if epoch_bar is not None:
        if rc == 0 and last_completed_epoch < epoch_bar.total:
            epoch_bar.update(epoch_bar.total - last_completed_epoch)
        epoch_bar.close()

    return rc, lines


def fallback_candidates(start: int, floor: int) -> list[int]:
    vals: list[int] = []
    cur = max(int(start), floor)
    while True:
        vals.append(cur)
        if cur <= floor:
            break
        nxt = max(cur // 2, floor)
        if nxt == cur:
            break
        cur = nxt
    return vals


def is_memory_error(lines: Sequence[str], rc: int | None = None) -> bool:
    if rc in {-9, 137}:
        return True
    text = "\n".join(str(x) for x in lines[-200:]).lower()
    tokens = [
        "out of memory",
        "cuda out of memory",
        "resource exhausted",
        "std::bad_alloc",
        "killed",
        "oom",
    ]
    return any(tok in text for tok in tokens)


def run_with_batch_fallback(
    script: str | Path,
    base_args: Mapping[str, Any],
    *,
    batch_keys: Sequence[str],
    candidates: Sequence[int],
    cwd: str | Path,
    python_bin: str | Path,
    project_root: str | Path,
    progress_title: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    script_path = Path(script).expanduser().resolve()
    for bs in candidates:
        args = dict(base_args)
        for key in batch_keys:
            args[str(key)] = int(bs)

        cmd = [str(python_bin), str(script_path), *_cli_args(args)]
        title = f"{progress_title} (bs={bs})" if progress_title else None
        rc, lines = run_and_capture(cmd, cwd, project_root=project_root, progress_title=title)
        if rc == 0:
            return args, lines
        if is_memory_error(lines, rc=rc):
            print(f"Retrying with smaller batch size after memory-related failure (bs={bs}, rc={rc}).")
            continue
        raise RuntimeError(f"{script_path.name} failed with exit code {rc}")

    raise RuntimeError(f"{script_path.name} failed for all batch-size candidates: {list(candidates)}")


def parse_learn_base_metrics(lines: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    p_acc = re.compile(r"train acc:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?),\s*test acc:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)")
    p_ap = re.compile(r"train ap:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?),\s*test ap:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)")
    p_auc = re.compile(r"train auc:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?),\s*test auc:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)")

    for line in lines:
        m = p_acc.search(line)
        if m:
            out["train_acc"] = float(m.group(1))
            out["test_acc"] = float(m.group(2))
        m = p_ap.search(line)
        if m:
            out["train_ap"] = float(m.group(1))
            out["test_ap"] = float(m.group(2))
        m = p_auc.search(line)
        if m:
            out["train_auc"] = float(m.group(1))
            out["test_auc"] = float(m.group(2))

    return out


def parse_testing_epoch_metrics(lines: Sequence[str]) -> dict[str, float]:
    testing_line = next((line for line in reversed(lines) if str(line).strip().startswith("Testing Epoch:")), None)
    if testing_line is None:
        return {}

    out: dict[str, float] = {}
    for part in str(testing_line).split("|"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        if not value:
            continue
        out[key] = float(value)

    if "ratio_acc" in out and "acc_auc" not in out:
        out["acc_auc"] = float(out["ratio_acc"])

    return out


def save_metrics_and_logs(
    metrics_rows: Sequence[Mapping[str, Any]],
    logs: Mapping[str, Sequence[str]],
    runs_dir: str | Path,
) -> tuple[Path, Path]:
    out_dir = Path(runs_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    metrics_df = pd.DataFrame(list(metrics_rows))
    if metrics_df.empty:
        metrics_df = pd.DataFrame(columns=["stage"])
    metrics_path = out_dir / f"metrics_{ts}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    logs_path = out_dir / f"logs_{ts}.json"
    logs_path.write_text(json.dumps(dict(logs), indent=2), encoding="utf-8")
    return metrics_path, logs_path


__all__ = [
    "runs_dir_for_context",
    "stage_cache_path",
    "latest_metrics_path",
    "save_stage_cache",
    "load_stage_cache_metrics",
    "run_and_capture",
    "fallback_candidates",
    "is_memory_error",
    "run_with_batch_fallback",
    "parse_learn_base_metrics",
    "parse_testing_epoch_metrics",
    "save_metrics_and_logs",
]

