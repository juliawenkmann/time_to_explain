from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def summarize_benchmark(df: pd.DataFrame, *, frac_ref: Optional[float] = None) -> pd.DataFrame:
    """Return a compact, checkable per-explainer summary.

    This project is easiest to interpret with a small set of metrics:

    - KeepAgree@k: keeping the top-k edges preserves the prediction
    - DropFlip@k: dropping the top-k edges flips the prediction
    - GTPrec@k: agreement with a chosen ground-truth label (if available)
    - GT_AP: ranking agreement with GT (per node; independent of k)
    - Time/node

    Args:
        df: benchmark table returned by run_benchmark (one row per (explainer,node,frac)).
        frac_ref: which fraction to report for KeepAgree/DropFlip/GTPrec/GTRec.
                  If None, uses 0.05 if present, else the median available frac.

    Returns:
        DataFrame with one row per explainer.
    """
    if df.empty:
        return pd.DataFrame()

    # Decide which frac to display in the compact summary.
    fracs = np.sort(df["frac"].astype(float).unique())
    if frac_ref is None:
        if np.any(np.isclose(fracs, 0.05)):
            frac_ref = 0.05
        else:
            frac_ref = float(fracs[len(fracs) // 2])

    # Per-node metrics (gt_ap, base_rate, time) are repeated across fracs in df.
    per_node = (
        df.sort_values(["explainer", "node", "frac"], kind="mergesort")
          .groupby(["explainer", "node"], sort=False)
          .first()
          .reset_index()
    )

    base_aggs = {
        "GT_AP_mean": ("gt_ap", "mean"),
        "GT_AP_std": ("gt_ap", "std"),
        "GT_BaseRate_mean": ("gt_base_rate", "mean"),
        "Time_s_mean": ("explain_time_s", "mean"),
    }

    # Counterfactual-style metrics (if present)
    if "flip_success" in per_node.columns:
        base_aggs.update({"FlipSuccess_mean": ("flip_success", "mean")})
    if "k_flip" in per_node.columns:
        base_aggs.update({
            "KFlip_mean": ("k_flip", "mean"),
            "KFlip_median": ("k_flip", "median"),
        })
    if "frac_flip" in per_node.columns:
        base_aggs.update({"FracFlip_mean": ("frac_flip", "mean")})

    base = per_node.groupby("explainer", sort=False).agg(**base_aggs).reset_index()

    # k-dependent metrics at the chosen frac
    at_frac = df[np.isclose(df["frac"].astype(float), float(frac_ref))].copy()
    if at_frac.empty:
        # Fallback: use the smallest frac
        at_frac = df[df["frac"].astype(float) == float(fracs[0])].copy()
        frac_ref = float(fracs[0])

    kstats = at_frac.groupby("explainer", sort=False).agg(
        frac=("frac", "first"),
        KeepAgree_mean=("keep_agree", "mean"),
        DropFlip_mean=("drop_flip", "mean"),
        GTPrec_mean=("gt_precision_at_k", "mean"),
        GTRec_mean=("gt_recall_at_k", "mean"),
    ).reset_index()

    out = base.merge(kstats, on="explainer", how="left")
    sort_cols = [c for c in ["GT_AP_mean", "DropFlip_mean", "FlipSuccess_mean"] if c in out.columns]
    if sort_cols:
        # Prefer interpretable ordering: GT agreement first, then flips.
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols), kind="mergesort")
    return out
