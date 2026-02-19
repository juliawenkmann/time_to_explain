"""Helpers for running many subprocesses in parallel.

We intentionally run each evaluation / postprocess task as its own subprocess.
That keeps the GPU/torch state isolated and avoids issues with multiprocessing
+ CUDA (fork safety).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass(frozen=True)
class Task:
    cmd: List[str]
    label: str
    cwd: Optional[Path] = None
    env: Optional[Dict[str, str]] = None


def _safe_filename(label: str) -> str:
    keep = []
    for ch in label:
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)[:160]


def run_task(task: Task, log_dir: Optional[Path] = None) -> int:
    env = os.environ.copy()
    if task.env:
        env.update(task.env)

    stdout_target = None
    stderr_target = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        out_path = log_dir / f"{_safe_filename(task.label)}.log"
        stdout_target = open(out_path, "w", encoding="utf-8")
        stderr_target = subprocess.STDOUT

    try:
        proc = subprocess.run(
            task.cmd,
            cwd=str(task.cwd) if task.cwd else None,
            env=env,
            stdout=stdout_target,
            stderr=stderr_target,
            text=True,
            check=False,
        )
        return int(proc.returncode)
    finally:
        if stdout_target is not None:
            stdout_target.close()


def run_tasks_parallel(
    tasks: Sequence[Task],
    jobs: int = 1,
    log_dir: Optional[Path] = None,
) -> None:
    """Run tasks with up to *jobs* concurrent subprocesses.

    Raises RuntimeError if any task fails.
    """
    if not tasks:
        return

    jobs = max(1, int(jobs))

    failures: List[tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        future_to_task = {ex.submit(run_task, task, log_dir): task for task in tasks}
        for fut in as_completed(future_to_task):
            task = future_to_task[fut]
            code = fut.result()
            if code != 0:
                failures.append((task.label, code))

    if failures:
        msg = "\n".join([f"- {label}: exit code {code}" for label, code in failures])
        raise RuntimeError(f"Some tasks failed:\n{msg}")
