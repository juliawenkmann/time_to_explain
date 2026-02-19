"""Run evaluation + postprocess as a single pipeline.

This script replaces the ad-hoc orchestration in evaluate.bash with a more
readable and parallelizable implementation.

It intentionally keeps the *user-facing* interface similar to the original
bash script:

Counterfactual explainers (cody/greedy):
    python run_pipeline.py MODEL DATASET EXPLAINER SAMPLER TIME_LIMIT [--bipartite]

Factual explainers (tgnnexplainer/pg_explainer):
    python run_pipeline.py MODEL DATASET EXPLAINER TIME_LIMIT [--bipartite]

Extra flags:
    --jobs_eval N    parallelize across samplers (only meaningful with SAMPLER=all)
    --jobs_post N    parallelize postprocessing across result files
    --gpus 0,1,...   assign tasks to GPUs round-robin via CUDA_VISIBLE_DEVICES

You can keep using bash evaluate.bash; in the updated version, it delegates to
this file.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import List, Optional

from pipeline.paths import default_paths
from pipeline.runner import Task, run_tasks_parallel
from pipeline.samplers import SAMPLERS


def _parse_gpus(gpus: Optional[str]) -> List[str]:
    if not gpus:
        return []
    parts = [p.strip() for p in gpus.split(",")]
    return [p for p in parts if p]


def _script_dir(paths: "RepoPaths") -> Path:
    return paths.scripts


def _python() -> str:
    return os.environ.get("PYTHON", "python")


def _run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    print("[pipeline]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")


def main() -> int:
    parser = argparse.ArgumentParser("Run CoDy evaluation + postprocess")
    parser.add_argument("model_type")
    parser.add_argument("dataset")
    parser.add_argument("explainer", choices=["cody", "greedy", "irand", "tgnnexplainer", "pg_explainer"])

    # We mirror evaluate.bash's positional interface by using two optional positionals.
    parser.add_argument("sampler_or_time", nargs="?", default=None)
    parser.add_argument("time_limit", nargs="?", default=None)

    parser.add_argument("--bipartite", action="store_true")
    parser.add_argument(
        "--cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA (default: on). Use --no-cuda to run on CPU.",
    )

    parser.add_argument("--jobs_eval", type=int, default=1, help="Parallel evaluation jobs (sampler=all only)")
    parser.add_argument("--jobs_post", type=int, default=1, help="Parallel postprocess jobs (per results file)")
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU ids to use for parallel tasks (sets CUDA_VISIBLE_DEVICES)",
    )

    parser.add_argument(
        "--parallel_samplers",
        action="store_true",
        help="If set and sampler=all, run each sampler in its own subprocess (can be faster on multi-GPU).",
    )

    # postprocess-related flags
    parser.add_argument("--score_is_probability", action="store_true")
    parser.add_argument("--decision_threshold", type=float, default=None)
    parser.add_argument("--max_sparsity", type=float, default=1.0)
    parser.add_argument("--n_grid", type=int, default=101)
    parser.add_argument("--no_postprocess", action="store_true")

    args = parser.parse_args()

    is_counterfactual = args.explainer in {"cody", "greedy", "irand"}

    if is_counterfactual:
        sampler = args.sampler_or_time or "all"
        time_limit = args.time_limit or "600"
    else:
        sampler = None
        time_limit = args.sampler_or_time or "600"

    time_limit = int(time_limit)

    paths = default_paths()
    root_path = str(paths.root)
    existing_py_path = os.environ.get("PYTHONPATH", "")
    if root_path not in existing_py_path.split(os.pathsep):
        os.environ["PYTHONPATH"] = (
            root_path + (os.pathsep + existing_py_path if existing_py_path else "")
        )
    script_dir = _script_dir(paths)

    processed_dataset = paths.processed_dataset_dir(args.dataset)
    tgnn_path = paths.tgnn_model_path(args.model_type, args.dataset)
    explained_ids_path = paths.explained_ids_path(args.model_type, args.dataset)

    results_dir = paths.explainer_results_dir(args.dataset, args.explainer)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ----- Evaluation -----
    if args.explainer in {"tgnnexplainer", "pg_explainer"}:
        # factual
        pg_path = paths.pgexplainer_model_path(args.model_type, args.dataset)
        results_file = results_dir / f"results_{args.model_type}_{args.dataset}_{args.explainer}.csv"

        cmd = [
            _python(),
            str(script_dir / "evaluate_factual_explainer.py"),
            "-d",
            str(processed_dataset),
            "--model",
            str(tgnn_path),
            "--type",
            args.model_type,
            "--candidates_size",
            "30",
            "--number_of_explained_events",
            "200",
            "--explained_ids",
            str(explained_ids_path),
            "--results",
            str(results_file),
            "--max_time",
            str(time_limit),
            "--explainer_model_path",
            str(pg_path),
        ]

        if args.explainer == "pg_explainer":
            cmd += ["--explainer", "pg_explainer"]
        else:
            cmd += ["--explainer", "t_gnnexplainer", "--rollout", "500", "--mcts_save_dir", str(results_dir) + "/"]

        if args.bipartite:
            cmd.append("--bipartite")
        if args.cuda:
            cmd.append("--cuda")

        _run(cmd, cwd=script_dir)

        if args.no_postprocess:
            return 0

        # 1) add counterfactual_prediction for factual explainers
        _run(
            [
                _python(),
                str(script_dir / "evaluate_factual_subgraphs.py"),
                "-d",
                str(processed_dataset),
                "--model",
                str(tgnn_path),
                "--type",
                args.model_type,
                "--results",
                str(results_file),
            ]
            + (["--bipartite"] if args.bipartite else [])
            + (["--cuda"] if args.cuda else []),
            cwd=script_dir,
        )

        # 2) compute prediction_explanation_events_only + fid +/- + AUFSC
        post_cmd = [
            _python(),
            str(script_dir / "evaluate_fidelity_minus.py"),
            "-d",
            str(processed_dataset),
            "--model",
            str(tgnn_path),
            "--type",
            args.model_type,
            "--results",
            str(results_file),
            "--jobs",
            str(args.jobs_post),
            "--max_sparsity",
            str(args.max_sparsity),
            "--n_grid",
            str(args.n_grid),
        ]
        if args.bipartite:
            post_cmd.append("--bipartite")
        if args.cuda:
            post_cmd.append("--cuda")
        if args.score_is_probability:
            post_cmd.append("--score_is_probability")
        if args.decision_threshold is not None:
            post_cmd += ["--decision_threshold", str(args.decision_threshold)]
        if args.gpus:
            post_cmd += ["--gpus", args.gpus]

        _run(post_cmd, cwd=script_dir)
        return 0

    # counterfactual explainers
    # If sampler==all, we can either use the built-in sequential evaluation (cached)
    # or fan out one subprocess per sampler.
    if sampler == "all" and args.parallel_samplers and args.jobs_eval > 1:
        gpus = _parse_gpus(args.gpus)
        tasks: List[Task] = []
        for idx, s in enumerate(SAMPLERS):
            env = {}
            if gpus:
                env["CUDA_VISIBLE_DEVICES"] = gpus[idx % len(gpus)]

            cmd = [
                _python(),
                str(script_dir / "evaluate_cf_explainer.py"),
                "-d",
                str(processed_dataset),
                "--model",
                str(tgnn_path),
                "--type",
                args.model_type,
                "--explainer",
                args.explainer,
                "--number_of_explained_events",
                "200",
                "--explained_ids",
                str(explained_ids_path),
                "--results",
                str(results_dir),
                "--dynamic",
                "--sample_size",
                "10",
                "--candidates_size",
                "64",
                "--sampler",
                s,
                "--max_time",
                str(time_limit),
                "--optimize",
            ]
            if args.explainer == "cody":
                cmd += ["--max_steps", "300"]

            if args.bipartite:
                cmd.append("--bipartite")
            if args.cuda:
                cmd.append("--cuda")

            tasks.append(Task(cmd=cmd, label=f"eval_{args.explainer}_{s}", cwd=script_dir, env=env))

        run_tasks_parallel(tasks, jobs=args.jobs_eval, log_dir=results_dir / "logs_eval")

    else:
        cmd = [
            _python(),
            str(script_dir / "evaluate_cf_explainer.py"),
            "-d",
            str(processed_dataset),
            "--model",
            str(tgnn_path),
            "--type",
            args.model_type,
            "--explainer",
            args.explainer,
            "--number_of_explained_events",
            "200",
            "--explained_ids",
            str(explained_ids_path),
            "--results",
            str(results_dir),
            "--dynamic",
            "--sample_size",
            "10",
            "--candidates_size",
            "64",
            "--sampler",
            sampler,
            "--max_time",
            str(time_limit),
            "--optimize",
        ]
        if args.explainer == "cody":
            cmd += ["--max_steps", "300"]

        if args.bipartite:
            cmd.append("--bipartite")
        if args.cuda:
            cmd.append("--cuda")

        _run(cmd, cwd=script_dir)

    if args.no_postprocess:
        return 0

    # ----- Postprocess -----
    post_cmd = [
        _python(),
        str(script_dir / "evaluate_fidelity_minus.py"),
        "-d",
        str(processed_dataset),
        "--model",
        str(tgnn_path),
        "--type",
        args.model_type,
        "--results",
        str(results_dir),
        "--all_files_in_dir",
        "--jobs",
        str(args.jobs_post),
        "--max_sparsity",
        str(args.max_sparsity),
        "--n_grid",
        str(args.n_grid),
    ]
    if args.bipartite:
        post_cmd.append("--bipartite")
    if args.cuda:
        post_cmd.append("--cuda")
    if args.score_is_probability:
        post_cmd.append("--score_is_probability")
    if args.decision_threshold is not None:
        post_cmd += ["--decision_threshold", str(args.decision_threshold)]
    if args.gpus:
        post_cmd += ["--gpus", args.gpus]

    _run(post_cmd, cwd=script_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
