from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from time_to_explain.data.validate import verify_dataframe_unify


def load_explain_idx(
    explain_idx_filepath: Path | str,
    *,
    start: int = 0,
    end: Optional[int] = None,
    column: str = "event_idx",
    verbose: bool = True,
) -> list[int]:
    """
    Load 1-based event indices from an explain-index CSV.
    """
    df = pd.read_csv(explain_idx_filepath)
    if column not in df.columns:
        raise KeyError(f"Missing '{column}' column in {explain_idx_filepath}")
    event_idxs = df[column].to_list()
    if end is not None:
        event_idxs = event_idxs[start:end]
    else:
        event_idxs = event_idxs[start:]
    if verbose:
        print(f"{len(event_idxs)} events to explain")
    return [int(e) for e in event_idxs]


def generate_explain_index(
    ml_csv: Path | str,
    out_dir: Path | str,
    dataset_name: str,
    *,
    size: int = 500,
    seed: int = 42,
    explain_idx_name: Optional[str] = None,
    verbose: bool = True,
    bipartite: Optional[bool] = None,
) -> Path:
    """
    Create a CSV listing event indices to be explained.

    For simulate_v* we sample positives by label==1.
    For wikipedia/reddit we sample uniformly from the 70%-99% tail of event indices.
    """
    rng = np.random.default_rng(seed)
    ml_csv = Path(ml_csv)
    out_dir = Path(out_dir)
    df = pd.read_csv(ml_csv)
    if bipartite is None:
        try:
            u_max = int(df["u"].max())
            i_min = int(df["i"].min())
            bipartite = i_min >= (u_max + 1)
        except Exception:
            bipartite = True

    verify_dataframe_unify(df, bipartite=bool(bipartite))

    candidates: np.ndarray
    if dataset_name in {"simulate_v1", "simulate_v2"}:
        positive_mask = df["label"] == 1
        candidates = df.loc[positive_mask, "e_idx"].to_numpy()
        if len(candidates) == 0:
            candidates = df["e_idx"].to_numpy()
    elif dataset_name in {"wikipedia", "reddit"}:
        e_num = len(df)
        low = int(e_num * 0.70)
        high = max(int(e_num * 0.99), low + 1)
        candidates = df["e_idx"].to_numpy()[low:high]
        if len(candidates) == 0:
            candidates = df["e_idx"].to_numpy()
    else:
        if "label" in df.columns:
            positives = df.loc[df["label"] > 0, "e_idx"].to_numpy()
            candidates = positives if len(positives) else df["e_idx"].to_numpy()
        else:
            candidates = df["e_idx"].to_numpy()

    if len(candidates) == 0:
        raise ValueError(f"No candidate events found for dataset '{dataset_name}'.")
    if size > len(candidates):
        if verbose:
            print(
                f"[{dataset_name}] Requested {size} explain indices, "
                f"but only {len(candidates)} candidates available. Using all candidates."
            )
        size = len(candidates)
    explain_idxs = rng.choice(candidates, size=size, replace=False)

    explain_idxs = sorted(int(x) for x in explain_idxs)
    explain_df = pd.DataFrame({"event_idx": explain_idxs})

    if explain_idx_name is None:
        out_file = out_dir / f"{dataset_name}.csv"
    else:
        out_file = out_dir / f"{explain_idx_name}.csv"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    explain_df.to_csv(out_file, index=False)
    if verbose:
        print(f"Explain index saved to {out_file}")
    return out_file


__all__ = ["load_explain_idx", "generate_explain_index"]
