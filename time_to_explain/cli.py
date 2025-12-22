from __future__ import annotations

import argparse
import json
from pathlib import Path

from time_to_explain.pipelines import (
    format_uci_messages,
    prepare_real_data,
    prepare_synthetic_data,
    run_from_config,
    sweep_from_glob,
)


def _print_result(result: dict) -> None:
    out_dir = result.get("out_dir") or result.get("result", {}).get("out_dir")
    jsonl = result.get("jsonl") or result.get("result", {}).get("jsonl")
    csv = result.get("csv") or result.get("result", {}).get("csv")
    if out_dir:
        print(f"output_dir: {out_dir}")
    if jsonl:
        print(f"jsonl     : {jsonl}")
    if csv:
        print(f"csv       : {csv}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ttx",
        description="Temporal Graph XAI pipeline runner",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # data prep
    data_p = sub.add_parser("data", help="Prepare datasets (download/process/generate).")
    data_sub = data_p.add_subparsers(dest="action", required=True)
    prep_p = data_sub.add_parser("prepare", help="Download/process real datasets and/or build synthetic ones.")
    prep_p.add_argument("--root", type=str, default=None, help="Repo root; defaults to auto-detection.")
    prep_p.add_argument("--only", type=str, help="Comma list for real datasets (reddit,wikipedia,simulate_v1,simulate_v2).")
    prep_p.add_argument("--force", action="store_true", help="Re-download even if file exists.")
    prep_p.add_argument("--no-process", dest="do_process", action="store_false", help="Skip processing real datasets.")
    prep_p.add_argument("--no-index", dest="do_index", action="store_false", help="Skip generating explain indices.")
    prep_p.add_argument("--seed", type=int, default=42, help="Seed for index sampling.")
    prep_p.add_argument("--index-size", type=int, default=500, help="How many explain indices to sample per dataset.")
    prep_p.add_argument("--synthetic", type=str, help="Comma list of synthetic recipes (e.g., erdos_small,hawkes_small).")
    prep_p.add_argument("--overwrite-synthetic", action="store_true", help="Overwrite existing synthetic datasets.")
    prep_p.add_argument("--visualize-synthetic", action="store_true", help="Generate plots for synthetic datasets.")
    prep_p.add_argument("--format-uci-input", type=str, help="Optional: path to raw UCI-Messages txt file.")
    prep_p.add_argument("--format-uci-output", type=str, help="Optional: path to write formatted UCI CSV.")

    # eval run
    eval_p = sub.add_parser("eval", help="Run evaluation(s) from YAML configs.")
    eval_sub = eval_p.add_subparsers(dest="action", required=True)

    run_p = eval_sub.add_parser("run", help="Run a single experiment config.")
    run_p.add_argument("--config", required=True, help="Path to experiment YAML.")

    sweep_p = eval_sub.add_parser("sweep", help="Run multiple configs via a glob.")
    sweep_p.add_argument("--glob", required=True, help="Glob pattern, e.g., 'configs/experiments/*.yaml'.")
    sweep_p.add_argument("--save-summary", help="Optional JSON path to write all run metadata.")

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # data pipeline
    if args.command == "data" and args.action == "prepare":
        only = [x.strip() for x in args.only.split(",")] if args.only else None
        synthetic = [x.strip() for x in args.synthetic.split(",")] if args.synthetic else []

        prepare_real_data(
            root=args.root,
            only=only,
            force=args.force,
            do_process=args.do_process,
            do_index=args.do_index,
            seed=args.seed,
            index_size=args.index_size,
        )

        if synthetic:
            res = prepare_synthetic_data(
                synthetic,
                root=args.root,
                overwrite=args.overwrite_synthetic,
                visualize=args.visualize_synthetic,
            )
            print(json.dumps(res, indent=2))

        if args.format_uci_input and args.format_uci_output:
            outp = format_uci_messages(args.format_uci_input, args.format_uci_output)
            print(f"Formatted UCI dataset saved to {outp}")
        return

    # eval run
    if args.command == "eval" and args.action == "run":
        result = run_from_config(args.config)
        _print_result(result)
        return

    if args.command == "eval" and args.action == "sweep":
        runs = sweep_from_glob(args.glob)
        for r in runs:
            cfg = r.get("config")
            print(f"\nconfig: {cfg}")
            _print_result(r["result"])
        if args.save_summary:
            summary_path = Path(args.save_summary).expanduser()
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
            print(f"\nSummary saved to {summary_path}")
        return

    raise SystemExit("Unknown command.")


if __name__ == "__main__":
    main()
