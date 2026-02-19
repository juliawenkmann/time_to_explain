"""Postprocess evaluation results to add fid-/fid+/AUFSC.

Historically this repository used `evaluate_fidelity_minus.py` only to add a
"prediction_explanation_events_only" column (the sufficiency score).

This updated version turns the script into a general postprocessor that...

- adds prediction_explanation_events_only (if missing)
- computes fidelity_plus and fidelity_minus
- computes AUFSC_plus and AUFSC_minus
- writes a *_metrics.json sidecar per results file

It can process a single file or a directory of results. When processing a
directory, it can run multiple files in parallel by spawning subprocesses.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from cody.data import TrainTestDatasetParameters

from common import (
    add_dataset_arguments,
    add_wrapper_model_arguments,
    create_dataset_from_args,
    create_tgnn_wrapper_from_args,
    parse_args,
)

from postprocess.io_utils import find_results_files
from postprocess.metrics import DecisionRule
from postprocess.process import process_results_file


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _parse_gpus(gpus: Optional[str]) -> List[str]:
    if not gpus:
        return []
    return [p.strip() for p in gpus.split(",") if p.strip()]


def _replace_or_add(argv: List[str], flag: str, value: Optional[str]) -> List[str]:
    """Replace `flag VALUE` in argv, or append it, or remove it if value is None."""
    out = list(argv)
    if flag in out:
        i = out.index(flag)
        # remove existing flag + value
        if i + 1 < len(out):
            out.pop(i + 1)
        out.pop(i)
    if value is not None:
        out.extend([flag, value])
    return out


def _remove_flag(argv: List[str], flag: str) -> List[str]:
    out = list(argv)
    while flag in out:
        out.remove(flag)
    return out


def _process_one_file(args: argparse.Namespace, results_file: Path) -> None:
    # Decide rule for labels
    threshold = args.decision_threshold
    if threshold is None:
        threshold = 0.5 if args.score_is_probability else 0.0
    rule = DecisionRule(score_is_probability=bool(args.score_is_probability), threshold=float(threshold))

    dataset = None
    tgn_wrapper = None
    if not args.no_infer_explanation_only:
        dataset = create_dataset_from_args(
            args,
            TrainTestDatasetParameters(0.2, 0.6, 0.8, 500, 500, 500),
        )
        tgn_wrapper = create_tgnn_wrapper_from_args(args, dataset)
        tgn_wrapper.set_evaluation_mode(True)

    process_results_file(
        results_file,
        dataset=dataset,
        tgnn=tgn_wrapper,
        decision_rule=rule,
        max_sparsity=float(args.max_sparsity),
        n_grid=int(args.n_grid),
        infer_explanation_only=not args.no_infer_explanation_only,
        show_progress=True,
    )


def _spawn_parallel(args: argparse.Namespace, files: List[Path]) -> None:
    """Spawn one subprocess per file (up to args.jobs concurrent)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    script_path = Path(__file__).resolve()
    gpus = _parse_gpus(args.gpus)
    base_argv = sys.argv[1:]

    def _run_child(idx: int, f: Path) -> int:
        env = os.environ.copy()
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = gpus[idx % len(gpus)]

        child_argv = list(base_argv)
        child_argv = _remove_flag(child_argv, "--all_files_in_dir")
        child_argv = _replace_or_add(child_argv, "--results", str(f))
        child_argv = _replace_or_add(child_argv, "--jobs", "1")

        cmd = [sys.executable, str(script_path)] + child_argv
        proc = subprocess.run(cmd, cwd=str(_script_dir()), env=env, check=False)
        return int(proc.returncode)

    jobs = max(1, int(args.jobs))
    failures = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futures = {ex.submit(_run_child, i, f): f for i, f in enumerate(files)}
        for fut in as_completed(futures):
            f = futures[fut]
            code = fut.result()
            if code != 0:
                failures.append((str(f), code))

    if failures:
        msg = "\n".join([f"- {f}: exit {code}" for f, code in failures])
        raise RuntimeError(f"Some postprocess jobs failed:\n{msg}")


def main() -> int:
    parser = argparse.ArgumentParser("Postprocess evaluation results (fid-/fid+/AUFSC)")
    add_dataset_arguments(parser)
    add_wrapper_model_arguments(parser)

    parser.add_argument(
        "--results",
        required=True,
        type=str,
        help="Path to a results file (.csv/.parquet) or to a directory containing results_*.csv/parquet",
    )
    parser.add_argument(
        "--all_files_in_dir",
        action="store_true",
        help="If set and --results is a directory, postprocess all results files in that directory.",
    )

    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs when processing multiple files")
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU ids (round-robin via CUDA_VISIBLE_DEVICES) for parallel jobs",
    )

    parser.add_argument("--score_is_probability", action="store_true")
    parser.add_argument(
        "--decision_threshold",
        type=float,
        default=None,
        help="Threshold for score->label. Default: 0.0 for logits, 0.5 for probabilities.",
    )

    parser.add_argument("--max_sparsity", type=float, default=1.0)
    parser.add_argument("--n_grid", type=int, default=101)

    parser.add_argument(
        "--no_infer_explanation_only",
        action="store_true",
        help="Skip model inference for prediction_explanation_events_only.",
    )

    args = parse_args(parser)

    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results path not found: {results_path}")

    # Discover files
    if results_path.is_dir():
        files = find_results_files(results_path, recursive=False)
        if not args.all_files_in_dir:
            if len(files) == 1:
                files = files[:1]
            else:
                raise ValueError(
                    f"{results_path} is a directory with {len(files)} results files. "
                    "Pass --all_files_in_dir to process all of them."
                )
    else:
        files = [results_path]

    if not files:
        raise FileNotFoundError(f"No results files found under: {results_path}")

    # Parallel mode only makes sense when we have multiple files.
    if len(files) > 1 and int(args.jobs) > 1:
        _spawn_parallel(args, files)
        return 0

    # Sequential: reuse the same dataset/model across all files
    # (unless inference is disabled).
    dataset = None
    tgn_wrapper = None

    threshold = args.decision_threshold
    if threshold is None:
        threshold = 0.5 if args.score_is_probability else 0.0
    rule = DecisionRule(score_is_probability=bool(args.score_is_probability), threshold=float(threshold))

    if not args.no_infer_explanation_only:
        dataset = create_dataset_from_args(
            args,
            TrainTestDatasetParameters(0.2, 0.6, 0.8, 500, 500, 500),
        )
        tgn_wrapper = create_tgnn_wrapper_from_args(args, dataset)
        tgn_wrapper.set_evaluation_mode(True)

    for f in files:
        process_results_file(
            f,
            dataset=dataset,
            tgnn=tgn_wrapper,
            decision_rule=rule,
            max_sparsity=float(args.max_sparsity),
            n_grid=int(args.n_grid),
            infer_explanation_only=not args.no_infer_explanation_only,
            show_progress=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
