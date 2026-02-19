"""High-level postprocessing routine.

This module provides a single function (`process_results_file`) that loads a
results file, performs optional model inference for missing columns, computes
fid+/fid-/AUFSC, saves the enriched results, and writes a metrics JSON.

It is used by `evaluate_fidelity_minus.py` and can also be called from other
scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from postprocess.inference import (
    add_explanation_size_and_sparsity,
    compute_prediction_explanation_events_only,
    infer_explanation_column,
    normalize_explanation_ids,
)
from postprocess.io_utils import load_results, save_results, write_metrics_json
from postprocess.metrics import DecisionRule, compute_aufsc, fidelity_minus, fidelity_plus, summarize_binary


def process_results_file(
    results_file: str | Path,
    *,
    dataset: Any | None,
    tgnn: Any | None,
    decision_rule: DecisionRule,
    max_sparsity: float = 1.0,
    n_grid: int = 101,
    infer_explanation_only: bool = True,
    show_progress: bool = True,
    also_write_other_format: bool = True,
) -> Dict:
    """Postprocess a single file and return the computed metrics.

    Parameters
    ----------
    dataset, tgnn:
        Only required when `infer_explanation_only=True` and the file does not
        already contain `prediction_explanation_events_only`.
    """
    p = Path(results_file)
    df = load_results(p)

    # Normalize and cache explanation ids
    expl_col = infer_explanation_column(df)
    expl_ids = normalize_explanation_ids(df, expl_col)

    # Ensure explanation_size and sparsity exist
    df = add_explanation_size_and_sparsity(df, expl_ids)

    # Model inference (if requested)
    if infer_explanation_only:
        if dataset is None or tgnn is None:
            # We still might not need inference if the column is already filled.
            if "prediction_explanation_events_only" not in df.columns or df["prediction_explanation_events_only"].isna().any():
                raise ValueError(
                    "dataset/tgnn must be provided when infer_explanation_only=True and "
                    "prediction_explanation_events_only is missing/NaN"
                )
        else:
            df = compute_prediction_explanation_events_only(
                results_df=df,
                dataset=dataset,
                tgnn=tgnn,
                explanation_ids=expl_ids,
                show_progress=show_progress,
            )

    # Compute fidelity columns
    if "fidelity_plus" not in df.columns:
        df["fidelity_plus"] = np.nan
    if "fidelity_minus" not in df.columns:
        df["fidelity_minus"] = np.nan

    if "counterfactual_prediction" in df.columns and "original_prediction" in df.columns:
        df["fidelity_plus"] = [
            fidelity_plus(o, c, decision_rule)
            for o, c in zip(df["original_prediction"].astype(float), df["counterfactual_prediction"].astype(float))
        ]

    if "prediction_explanation_events_only" in df.columns and "original_prediction" in df.columns:
        df["fidelity_minus"] = [
            fidelity_minus(o, e, decision_rule)
            for o, e in zip(
                df["original_prediction"].astype(float),
                df["prediction_explanation_events_only"].astype(float),
            )
        ]

    # Compute AUFSC
    sparsity = df.get("sparsity")
    aufsc_plus = 0.0
    aufsc_minus = 0.0

    if sparsity is not None and "fidelity_plus" in df.columns:
        mask = (~pd.isna(df["fidelity_plus"])) & (~pd.isna(sparsity))
        aufsc_plus = compute_aufsc(
            sparsity=np.asarray(sparsity[mask], dtype=float),
            fidelity=np.asarray(df.loc[mask, "fidelity_plus"], dtype=float),
            max_sparsity=max_sparsity,
            n_grid=n_grid,
        )

    if sparsity is not None and "fidelity_minus" in df.columns:
        mask = (~pd.isna(df["fidelity_minus"])) & (~pd.isna(sparsity))
        aufsc_minus = compute_aufsc(
            sparsity=np.asarray(sparsity[mask], dtype=float),
            fidelity=np.asarray(df.loc[mask, "fidelity_minus"], dtype=float),
            max_sparsity=max_sparsity,
            n_grid=n_grid,
        )

    df["AUFSC_plus"] = float(aufsc_plus)
    df["AUFSC_minus"] = float(aufsc_minus)

    # Save outputs
    save_results(df, p, also_write_other_format=also_write_other_format)

    metrics = {
        "file": str(p),
        "decision_rule": {
            "score_is_probability": bool(decision_rule.score_is_probability),
            "threshold": float(decision_rule.threshold),
        },
        "AUFSC_plus": float(aufsc_plus),
        "AUFSC_minus": float(aufsc_minus),
        "fidelity_plus": summarize_binary(df["fidelity_plus"].dropna().astype(float).values),
        "fidelity_minus": summarize_binary(df["fidelity_minus"].dropna().astype(float).values),
        "rows": int(len(df)),
    }
    write_metrics_json(metrics, p)

    return metrics
