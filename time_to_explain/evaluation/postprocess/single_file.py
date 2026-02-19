"""Postprocess a single evaluation results file.

This script enriches a results file (CSV or Parquet) with:

- prediction_explanation_events_only
- explanation_size, sparsity (if missing)
- fidelity_plus, fidelity_minus
- AUFSC_plus, AUFSC_minus (constant columns; also written to *_metrics.json)

It is designed to be safe to run multiple times (it only recomputes missing
inference columns).

For parallelism, prefer running this script as a subprocess per file (see
:mod:`pipeline.runner`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cody.data import TrainTestDatasetParameters

from common import (
    add_dataset_arguments,
    add_wrapper_model_arguments,
    create_dataset_from_args,
    create_tgnn_wrapper_from_args,
    parse_args,
)

from postprocess.metrics import DecisionRule
from postprocess.process import process_results_file


def _safe_get_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"Missing required column '{col}' in results file")
    return df[col]


def main() -> int:
    parser = argparse.ArgumentParser("Postprocess a single CoDy evaluation results file")
    add_dataset_arguments(parser)
    add_wrapper_model_arguments(parser)

    parser.add_argument(
        "--results",
        required=True,
        type=str,
        help="Path to a single results file (.csv or .parquet).",
    )

    parser.add_argument(
        "--score_is_probability",
        action="store_true",
        help="Interpret stored scores as probabilities (0..1) instead of logits.",
    )
    parser.add_argument(
        "--decision_threshold",
        type=float,
        default=None,
        help="Threshold for converting scores -> labels. Defaults to 0.0 for logits and 0.5 for probabilities.",
    )

    parser.add_argument(
        "--max_sparsity",
        type=float,
        default=1.0,
        help="Upper sparsity limit for AUFSC integration.",
    )
    parser.add_argument(
        "--n_grid",
        type=int,
        default=101,
        help="Number of grid points for AUFSC integration.",
    )

    parser.add_argument(
        "--no_infer_explanation_only",
        action="store_true",
        help="Skip model inference for prediction_explanation_events_only.",
    )

    args = parse_args(parser)

    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    # Decide rule for labels
    if args.decision_threshold is None:
        threshold = 0.5 if args.score_is_probability else 0.0
    else:
        threshold = float(args.decision_threshold)
    rule = DecisionRule(score_is_probability=bool(args.score_is_probability), threshold=threshold)

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
        results_path,
        dataset=dataset,
        tgnn=tgn_wrapper,
        decision_rule=rule,
        max_sparsity=args.max_sparsity,
        n_grid=args.n_grid,
        infer_explanation_only=not args.no_infer_explanation_only,
        show_progress=True,
        also_write_other_format=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
