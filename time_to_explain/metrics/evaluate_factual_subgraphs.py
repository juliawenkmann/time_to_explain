import argparse
import json
import ast
import os
from pathlib import Path
from typing import Optional, Union, Sequence, Any

import numpy as np
import pandas as pd

from time_to_explain.data.data import TrainTestDatasetParameters
from time_to_explain.setup.utils import ProgressBar
from time_to_explain.utils.common import (
    add_dataset_arguments,
    add_wrapper_model_arguments,
    create_dataset_from_args,
    create_tgnn_wrapper_from_args,
    parse_args,
)

__all__ = ["evaluate_explainer_results"]


def _parse_results_cell(cell: Any) -> Sequence[dict]:
    """
    Normalize the 'results' cell into a list[dict] with keys like
    'prediction' and 'event_ids_in_explanation'.

    Accepts:
      - list[dict]
      - numpy arrays of dicts
      - pandas Series of dicts
      - JSON string or stringified Python literal representing a list[dict]
    """
    # Already a list of dicts?
    if isinstance(cell, list):
        return cell

    # Numpy array or pandas Series of dicts -> to list
    if hasattr(cell, "tolist"):
        try:
            as_list = cell.tolist()
            if isinstance(as_list, list):
                return as_list
        except Exception:
            pass

    # String -> try JSON first, then Python literal safely
    if isinstance(cell, str):
        # Try straightforward JSON
        try:
            parsed = json.loads(cell)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

        # Some logs store single quotes; try a replacer
        try:
            normalized = cell.replace("'", '"')
            parsed = json.loads(normalized)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

        # Last resort: safe literal_eval
        try:
            parsed = ast.literal_eval(cell)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    raise ValueError("Could not parse a valid list of results from cell.")


def _read_results_file(
    path: Union[str, Path],
    drop_first_csv_col: bool = True,
) -> tuple[pd.DataFrame, Path, bool]:
    """
    Read results file as CSV or Parquet and compute an output stem.
    Returns: (df, base_stem, read_from_csv)
    - base_stem is the input path without suffix (e.g., '/x/y/foo' for '/x/y/foo.csv')
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Failed to locate the file containing the results: {p}")

    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
        is_csv = False
    else:
        # Default to CSV if it's not '.parquet'
        df = pd.read_csv(p)
        if drop_first_csv_col and df.shape[1] >= 2:
            # Preserve original behavior: drop first column unconditionally if requested
            df = df.iloc[:, 1:]
        is_csv = True

    base_stem = p.with_suffix("")  # e.g., '/path/to/file'
    return df, base_stem, is_csv


def evaluate_explainer_results(
    results_path: Union[str, Path],
    tgn_wrapper: Any,
    *,
    show_progress: bool = True,
    drop_first_csv_col: bool = True,
    save_outputs: bool = True,
    save_csv: bool = True,
    save_parquet: bool = True,
) -> pd.DataFrame:
    """
    Compute counterfactual metrics from a results file and a ready-to-use TGNN wrapper.

    Parameters
    ----------
    results_path : str | Path
        Path to the file containing the results of the factual explainer evaluation (.csv or .parquet).
        The file is expected to have at least:
          - a 'results' column (list/dict/JSON with 'prediction' and 'event_ids_in_explanation')
          - an 'explained_event_id' column (id used for compute_edge_probabilities_for_subgraph)
          - optionally a 'candidate_size' column (used to compute 'sparsity')
    tgn_wrapper : Any
        A TGNN wrapper instance created for the target dataset/model. Must implement:
          - set_evaluation_mode(bool)
          - compute_edge_probabilities_for_subgraph(event_id, edges_to_drop: np.ndarray, result_as_logit: bool)
          - reset_model()
    show_progress : bool, default True
        Show a progress bar while iterating rows.
    drop_first_csv_col : bool, default True
        When reading a CSV, drop the first column (replicates the original script behavior).
    save_outputs : bool, default True
        If True, save CSV/Parquet next to the input file (same stem).
    save_csv : bool, default True
        If True and save_outputs is True, write a '.csv'.
    save_parquet : bool, default True
        If True and save_outputs is True, attempt to write a '.parquet' (requires pyarrow or fastparquet).

    Returns
    -------
    pd.DataFrame
        The augmented results DataFrame with columns:
          - 'explanation_event_ids'
          - 'explanation_size'
          - 'sparsity' (if 'candidate_size' exists)
          - 'counterfactual_prediction'
    """
    results_df, base_stem, _ = _read_results_file(results_path, drop_first_csv_col=drop_first_csv_col)

    # Ensure evaluation mode
    if hasattr(tgn_wrapper, "set_evaluation_mode"):
        tgn_wrapper.set_evaluation_mode(True)

    explanation_events: list[list[int]] = []
    explanation_sizes: list[int] = []
    scores: list[float] = []

    total = len(results_df)
    progress_bar = ProgressBar(total, "Adding counterfactual information") if show_progress else None

    for _, row in results_df.iterrows():
        if progress_bar:
            progress_bar.next()

        # Parse the 'results' cell to a list[dict]
        parsed_results = _parse_results_cell(row["results"])

        # Pick the best result by highest prediction
        best_result = max(parsed_results, key=lambda r: r.get("prediction", float("-inf")))
        ev_ids = best_result.get("event_ids_in_explanation", []) or []
        explanation_events.append(ev_ids)
        explanation_sizes.append(len(ev_ids))

        # Compute counterfactual prediction
        edges_to_drop = np.array(ev_ids)
        event_id = row["explained_event_id"]
        score, _ = tgn_wrapper.compute_edge_probabilities_for_subgraph(
            event_id,
            edges_to_drop=edges_to_drop,
            result_as_logit=True,
        )

        # Robust extraction of scalar
        try:
            value = float(score.detach().cpu().item())
        except Exception:
            value = float(score)
        scores.append(value)

        # reset between iterations (matches original script)
        if hasattr(tgn_wrapper, "reset_model"):
            tgn_wrapper.reset_model()

    if progress_bar:
        progress_bar.close()

    # Augment DataFrame
    results_df["explanation_event_ids"] = explanation_events
    results_df["explanation_size"] = explanation_sizes

    if "candidate_size" in results_df.columns:
        # Avoid division by zero (NaN if zero)
        denom = results_df["candidate_size"].replace(0, np.nan)
        results_df["sparsity"] = results_df["explanation_size"] / denom
    else:
        # Keep compatibility with consumers expecting the column
        results_df["sparsity"] = np.nan

    results_df["counterfactual_prediction"] = scores

    # Save (optional)
    if save_outputs:
        if save_csv:
            csv_path = base_stem.with_suffix(".csv")
            results_df.to_csv(csv_path, index=False)
            print(f"Results saved to {csv_path}")

        if save_parquet:
            parquet_path = base_stem.with_suffix(".parquet")
            try:
                results_df.to_parquet(parquet_path, index=False)
                print(f"Results saved to {parquet_path}")
            except ModuleNotFoundError:
                print(
                    "Could not save to parquet format. Install 'pyarrow' or 'fastparquet' if you want to export to parquet."
                )

    return results_df


def main():
    """
    CLI entry point:
    1) Parse arguments
    2) Build dataset + wrapper
    3) Call the callable function above
    """
    parser = argparse.ArgumentParser("Explainer Evaluation")
    add_dataset_arguments(parser)
    add_wrapper_model_arguments(parser)
    parser.add_argument(
        "--results",
        required=True,
        type=str,
        help="Path to the file containing the results of the factual explainer evaluation",
    )

    args = parse_args(parser)

    # Create dataset & wrapper just like the original script
    dataset = create_dataset_from_args(
        args,
        TrainTestDatasetParameters(0.2, 0.6, 0.8, 500, 500, 500),
    )
    tgn_wrapper = create_tgnn_wrapper_from_args(args, dataset)

    # Call the function using variables (no parsing inside the function)
    evaluate_explainer_results(
        results_path=args.results,
        tgn_wrapper=tgn_wrapper,
        show_progress=True,
        drop_first_csv_col=True,  # preserves the original behavior on CSV input
        save_outputs=True,
        save_csv=True,
        save_parquet=True,
    )


if __name__ == "__main__":
    main()
