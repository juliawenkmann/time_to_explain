import os
import ast
from typing import Optional, Union

import numpy as np
import pandas as pd


def evaluate_explainer_results(
    results: Union[str, pd.DataFrame],
    dataset,
    tgn_wrapper,
    *,
    output_path_root: Optional[str] = None,
    save_every: int = 5,
    save_csv: bool = True,
    save_parquet: bool = True,
    result_as_logit: bool = True,
    drop_potential_index_col: bool = True,
) -> pd.DataFrame:
    """
    Evaluate factual/counterfactual and explanation-only predictions at k = 6, 12, 18.

    Parameters
    ----------
    results : str | pd.DataFrame
        Path to a CSV/Parquet file (must include columns: 'explained_event_id', 'event_ids'),
        or a pre-loaded DataFrame with those columns. If a CSV has an auto-saved index column
        (e.g., 'Unnamed: 0'), it can be dropped automatically.
    dataset : object
        Must expose `edge_ids` as a 1D numpy array (or array-like) of all edge IDs.
    tgn_wrapper : object
        Must expose:
          - set_evaluation_mode(bool)          (optional but used if present)
          - reset_model()                       (required)
          - compute_edge_probabilities_for_subgraph(
                explained_event_id: int,
                edges_to_drop=None,
                result_as_logit=False,
                edges_to_keep=None
            ) -> (tensor_like, any)
    output_path_root : str | None
        Output file *root* (no extension). If None and `results` is a path, this is derived
        from that path (extension stripped). If None and `results` is a DataFrame, defaults
        to './explainer_eval_results'.
    save_every : int
        Save intermediate checkpoints every N processed rows (<=0 disables checkpoints).
    save_csv : bool
        Whether to write a CSV to '{output_path_root}.csv'.
    save_parquet : bool
        Whether to write a Parquet file to '{output_path_root}.parquet' (requires pyarrow).
    result_as_logit : bool
        Passed to `compute_edge_probabilities_for_subgraph` for counterfactual scores.
        (Explanation-only calls also use True, matching original script.)
    drop_potential_index_col : bool
        If True and the first column looks like an index column, drop it.

    Returns
    -------
    pd.DataFrame
        The input results with the following columns appended:
          - 'counterfactual_prediction_at_6'
          - 'counterfactual_prediction_at_12'
          - 'counterfactual_prediction_at_18'
          - 'prediction_explanation_events_only_at_6'
          - 'prediction_explanation_events_only_at_12'
          - 'prediction_explanation_events_only_at_18'
    """
    # ---- Helpers ----
    def _pad(seq, n):
        return list(seq) + [None] * (n - len(seq))

    def _parse_event_ids(val) -> list[int]:
        """Robustly parse event_ids that may be list-like or a stringified list."""
        if isinstance(val, (list, tuple, np.ndarray)):
            return [int(x) for x in val]
        if pd.isna(val):
            return []
        if isinstance(val, str):
            # Try literal_eval first (handles "[1, 2, 3]")
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, (list, tuple, np.ndarray)):
                    return [int(x) for x in parsed]
            except Exception:
                # Fallback: split on commas/semicolons after stripping brackets
                parts = (
                    val.strip().strip("[](){}")
                       .replace(";", ",")
                       .split(",")
                )
                parts = [p.strip() for p in parts if p.strip() != ""]
                return [int(p) for p in parts]
        # Last resort
        raise ValueError(f"Cannot parse event_ids from value: {val!r}")

    def _save_checkpoint(df_like: pd.DataFrame):
        if output_path_root is None:
            return
        if save_csv:
            df_like.to_csv(f"{output_path_root}.csv", index=False)
        if save_parquet:
            try:
                df_like.to_parquet(f"{output_path_root}.parquet")
            except ModuleNotFoundError:
                # Keep behavior consistent with original script
                print("Could not save to Parquet format. Install pyarrow if you want to export to Parquet.")

    # ---- Load inputs & prepare outputs ----
    if isinstance(results, str):
        if not os.path.exists(results):
            raise FileNotFoundError("Failed to locate the file containing the results")

        ext = os.path.splitext(results)[1].lower()
        if ext == ".parquet":
            results_df = pd.read_parquet(results)
        elif ext == ".csv":
            results_df = pd.read_csv(results)
        else:
            raise ValueError(f"Unsupported results file extension: {ext}")

        if output_path_root is None:
            output_path_root = os.path.splitext(results)[0]

    elif isinstance(results, pd.DataFrame):
        results_df = results.copy()
        if output_path_root is None:
            output_path_root = os.path.abspath("./explainer_eval_results")
    else:
        raise TypeError("`results` must be a file path (str) or a pandas DataFrame")

    # Drop a typical auto-saved index column, if present
    if drop_potential_index_col and results_df.shape[1] > 0:
        first_col = results_df.columns[0]
        if str(first_col).lower().startswith("unnamed") or str(first_col).lower() in ("index",):
            results_df = results_df.drop(columns=[first_col])

    # Sanity checks
    required_cols = {"explained_event_id", "event_ids"}
    missing = required_cols - set(results_df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # Ensure wrapper is in eval mode if supported
    if hasattr(tgn_wrapper, "set_evaluation_mode"):
        tgn_wrapper.set_evaluation_mode(True)

    # Edge universe as ndarray[int]
    edge_ids = np.asarray(getattr(dataset, "edge_ids"), dtype=int)

    # Output collectors
    scores_at_6, scores_at_12, scores_at_18 = [], [], []
    expl_only_6, expl_only_12, expl_only_18 = [], [], []

    # Optional nice progress bar if available
    progress = None
    try:
        from time_to_explain.setup.utils import ProgressBar  # available in your project
        progress = ProgressBar(len(results_df), "Adding information to results")
    except Exception:
        progress = None

    # ---- Main loop ----
    total_rows = len(results_df)
    for i, (_, row) in enumerate(results_df.iterrows(), start=1):
        explained_event_id = int(row["explained_event_id"])
        event_ids_list = _parse_event_ids(row["event_ids"])

        exp_at_6 = np.asarray(event_ids_list[:6], dtype=int)
        exp_at_12 = np.asarray(event_ids_list[:12], dtype=int)
        exp_at_18 = np.asarray(event_ids_list[:18], dtype=int)

        # --- Counterfactual scores (drop explanation edges) ---
        s6, _ = tgn_wrapper.compute_edge_probabilities_for_subgraph(
            explained_event_id,
            edges_to_drop=exp_at_6,
            result_as_logit=result_as_logit,
        )
        tgn_wrapper.reset_model()
        scores_at_6.append(float(s6.detach().cpu().item()))

        s12, _ = tgn_wrapper.compute_edge_probabilities_for_subgraph(
            explained_event_id,
            edges_to_drop=exp_at_12,
            result_as_logit=result_as_logit,
        )
        tgn_wrapper.reset_model()
        scores_at_12.append(float(s12.detach().cpu().item()))

        s18, _ = tgn_wrapper.compute_edge_probabilities_for_subgraph(
            explained_event_id,
            edges_to_drop=exp_at_18,
            result_as_logit=result_as_logit,
        )
        tgn_wrapper.reset_model()
        scores_at_18.append(float(s18.detach().cpu().item()))

        # --- Explanation-only predictions (keep only explanation edges + the explained edge) ---
        # Build drop set: everything except exp_at_k and explained_event_id
        keep_6 = np.concatenate([exp_at_6, np.array([explained_event_id], dtype=int)])
        drop_6 = edge_ids[~np.isin(edge_ids, keep_6)]
        p6, _ = tgn_wrapper.compute_edge_probabilities_for_subgraph(
            explained_event_id, drop_6, True, exp_at_6
        )
        tgn_wrapper.reset_model()
        expl_only_6.append(float(p6.detach().cpu().item()))

        keep_12 = np.concatenate([exp_at_12, np.array([explained_event_id], dtype=int)])
        drop_12 = edge_ids[~np.isin(edge_ids, keep_12)]
        p12, _ = tgn_wrapper.compute_edge_probabilities_for_subgraph(
            explained_event_id, drop_12, True, exp_at_12
        )
        tgn_wrapper.reset_model()
        expl_only_12.append(float(p12.detach().cpu().item()))

        keep_18 = np.concatenate([exp_at_18, np.array([explained_event_id], dtype=int)])
        drop_18 = edge_ids[~np.isin(edge_ids, keep_18)]
        p18, _ = tgn_wrapper.compute_edge_probabilities_for_subgraph(
            explained_event_id, drop_18, True, exp_at_18
        )
        tgn_wrapper.reset_model()
        expl_only_18.append(float(p18.detach().cpu().item()))

        if progress is not None:
            progress.next()

        # --- Intermediate checkpointing ---
        if save_every > 0 and (i % save_every == 0 or i == total_rows):
            tmp = results_df.copy()
            n = len(results_df)
            tmp["counterfactual_prediction_at_6"] = _pad(scores_at_6, n)
            tmp["counterfactual_prediction_at_12"] = _pad(scores_at_12, n)
            tmp["counterfactual_prediction_at_18"] = _pad(scores_at_18, n)
            tmp["prediction_explanation_events_only_at_6"] = _pad(expl_only_6, n)
            tmp["prediction_explanation_events_only_at_12"] = _pad(expl_only_12, n)
            tmp["prediction_explanation_events_only_at_18"] = _pad(expl_only_18, n)
            _save_checkpoint(tmp)

    if progress is not None:
        progress.close()

    # ---- Finalize outputs ----
    results_df["counterfactual_prediction_at_6"] = scores_at_6
    results_df["counterfactual_prediction_at_12"] = scores_at_12
    results_df["counterfactual_prediction_at_18"] = scores_at_18
    results_df["prediction_explanation_events_only_at_6"] = expl_only_6
    results_df["prediction_explanation_events_only_at_12"] = expl_only_12
    results_df["prediction_explanation_events_only_at_18"] = expl_only_18

    # Persist final
    _save_checkpoint(results_df)

    return results_df
