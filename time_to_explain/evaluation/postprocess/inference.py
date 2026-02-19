"""Model-inference helpers for postprocessing.

This module is responsible for computing the *missing* prediction columns that
require running the TGNN:

- prediction_explanation_events_only: p(f(X), ε)

where X is the explanation events.

It intentionally does not compute fid+/fid- or AUFSC (those are pure dataframe
metrics handled in :mod:`postprocess.metrics`).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from postprocess.parsing import parse_int_array


def infer_explanation_column(results_df: pd.DataFrame) -> str:
    """Return the column containing explanation event ids."""
    if "explanation_event_ids" in results_df.columns:
        return "explanation_event_ids"
    if "cf_example_event_ids" in results_df.columns:
        return "cf_example_event_ids"
    raise KeyError(
        "Could not find an explanation event-id column. Expected one of: "
        "{'explanation_event_ids', 'cf_example_event_ids'}"
    )


def normalize_explanation_ids(results_df: pd.DataFrame, column: str) -> List[np.ndarray]:
    """Parse and return the explanation id arrays for each row.

    For CSV inputs, this also makes downstream processing deterministic.
    """
    parsed: List[np.ndarray] = [parse_int_array(v) for v in results_df[column].values]
    return parsed


def add_explanation_size_and_sparsity(
    results_df: pd.DataFrame,
    explanation_ids: Sequence[np.ndarray],
    candidate_size_col: str = "candidate_size",
) -> pd.DataFrame:
    """Add explanation_size and sparsity columns if missing."""
    if "explanation_size" not in results_df.columns:
        results_df["explanation_size"] = [int(arr.size) for arr in explanation_ids]

    if "sparsity" not in results_df.columns:
        if candidate_size_col not in results_df.columns:
            # leave sparsity missing; caller can decide what to do.
            return results_df
        # avoid divide-by-zero
        denom = results_df[candidate_size_col].replace(0, np.nan)
        results_df["sparsity"] = results_df["explanation_size"] / denom

    return results_df


def compute_prediction_explanation_events_only(
    results_df: pd.DataFrame,
    dataset: Any,
    tgnn: Any,
    explanation_ids: Sequence[np.ndarray],
    explained_event_id_col: str = "explained_event_id",
    output_col: str = "prediction_explanation_events_only",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compute prediction_explanation_events_only and store it in *output_col*.

    This is the expensive step because it runs TGNN inference per explained event.

    Parameters
    ----------
    dataset:
        A dataset object created by `create_dataset_from_args`. It must provide
        `edge_ids` (a 1D array of edge/event ids).
    tgnn:
        A TGNNWrapper created by `create_tgnn_wrapper_from_args`.

    Notes
    -----
    We keep the call signature compatible with the original repo script.
    """
    if explained_event_id_col not in results_df.columns:
        raise KeyError(f"Missing column: {explained_event_id_col}")

    # Only compute for missing values; keep existing computations.
    needs = None
    if output_col not in results_df.columns:
        results_df[output_col] = np.nan
        needs = np.ones(len(results_df), dtype=bool)
    else:
        needs = results_df[output_col].isna().to_numpy()

    if not needs.any():
        return results_df

    # Progress bar is optional (repo provides cody.utils.ProgressBar)
    progress_bar = None
    if show_progress:
        try:
            from cody.utils import ProgressBar  # type: ignore

            progress_bar = ProgressBar(int(needs.sum()), prefix="Inferring explanation-only scores")
        except Exception:
            progress_bar = None

    all_edge_ids = np.asarray(dataset.edge_ids, dtype=int)

    out_scores: List[float] = results_df[output_col].astype(float, errors="ignore").tolist()

    for i, (row, exp_ids) in enumerate(zip(results_df.itertuples(index=False), explanation_ids)):
        if not needs[i]:
            continue

        # Using getattr because row is a namedtuple from itertuples.
        explained_event_id = int(getattr(row, explained_event_id_col))

        # Ensure we keep the explained event itself.
        keep_ids = np.concatenate([exp_ids.astype(int, copy=False), np.asarray([explained_event_id], dtype=int)])

        # Drop everything that is NOT in keep_ids
        edges_to_drop = all_edge_ids[~np.isin(all_edge_ids, keep_ids)]

        # Important for stateful models
        try:
            tgnn.reset_model()
        except Exception:
            pass

        try:
            pred, _ = tgnn.compute_edge_probabilities_for_subgraph(
                explained_event_id,
                edges_to_drop,
                True,
                exp_ids,
            )
        except TypeError:
            # Fallback to keyword-only signature used elsewhere in the repo.
            pred, _ = tgnn.compute_edge_probabilities_for_subgraph(
                explained_event_id,
                edges_to_drop=np.asarray(edges_to_drop, dtype=int),
                result_as_logit=True,
            )

        # torch tensor -> float
        try:
            score = float(pred.detach().cpu().item())
        except Exception:
            score = float(pred)

        out_scores[i] = score

        if progress_bar is not None:
            progress_bar.next()

    if progress_bar is not None:
        progress_bar.close()

    results_df[output_col] = out_scores
    return results_df
