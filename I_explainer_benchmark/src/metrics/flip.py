from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_flip_success_summary(
    rows: pd.DataFrame,
    *,
    budget: float = 0.9,
    area_threshold: float = 0.4,
    point_threshold: float | None = None,
    anchor_col: str = "anchor_idx",
    sparsity_col: str = "sparsity",
    prediction_full_col: str = "prediction_full",
    prediction_drop_col: str = "prediction_drop",
) -> Dict[str, float]:
    """Compute legacy flip metrics used in `time_to_explain_official-1`.

    Legacy definition:
    - For each anchor, derive keep-score from TGNN `fid_inv` and full score:
      keep = full + fid_inv (full >= 0), else keep = full - fid_inv.
    - A flip happens when sign(keep) differs from sign(full).
    - `flip_success_rate` is the anchor-level mean of any-flip indicators.
    - `first_flip_sparsity` is mean first flip sparsity; if no anchor flips: 1.0.

    The threshold/budget arguments are accepted for API compatibility and ignored
    by this legacy formulation.
    """
    _ = float(budget)
    _ = float(area_threshold)
    _ = float(point_threshold if point_threshold is not None else area_threshold)

    out: Dict[str, float] = {
        "flip_success_rate": float("nan"),
        "first_flip_sparsity": float("nan"),
        "first_flip_score": float("nan"),
        "flip_progress_auc": float("nan"),
        "n_valid_anchors": 0.0,
    }

    required_cols = {anchor_col, sparsity_col, prediction_full_col, "fid_inv"}
    if rows is None or rows.empty or not required_cols.issubset(set(rows.columns)):
        return out

    success_vals: list[float] = []
    first_vals: list[float] = []

    for _, g_anchor in rows.groupby(anchor_col, as_index=False):
        g_anchor = g_anchor.sort_values(sparsity_col)

        full_arr = pd.to_numeric(g_anchor[prediction_full_col], errors="coerce").to_numpy(dtype=float)
        full_arr = full_arr[np.isfinite(full_arr)]
        if full_arr.size == 0:
            continue
        full_score = float(full_arr[0])

        spars_arr = pd.to_numeric(g_anchor[sparsity_col], errors="coerce").to_numpy(dtype=float)
        fid_arr = pd.to_numeric(g_anchor["fid_inv"], errors="coerce").to_numpy(dtype=float)
        # TGNN fid_inv_tg sign convention from legacy notebook:
        # full>=0 -> fid = keep-full ; full<0 -> fid = full-keep
        keep_arr = full_score + fid_arr if full_score >= 0.0 else full_score - fid_arr

        valid_mask = np.isfinite(spars_arr) & np.isfinite(keep_arr)
        if not np.any(valid_mask):
            continue

        s_valid = spars_arr[valid_mask]
        k_valid = keep_arr[valid_mask]
        order = np.argsort(s_valid)
        s_valid = s_valid[order]
        k_valid = k_valid[order]

        base_label = bool(full_score >= 0.0)
        flip_mask = (k_valid >= 0.0) != base_label
        success = bool(np.any(flip_mask))
        success_vals.append(1.0 if success else 0.0)
        if success:
            first_vals.append(float(s_valid[np.where(flip_mask)[0][0]]))

    if not success_vals:
        return out

    out["flip_success_rate"] = float(np.mean(np.asarray(success_vals, dtype=float)))
    out["first_flip_sparsity"] = float(np.mean(np.asarray(first_vals, dtype=float))) if first_vals else 1.0
    out["first_flip_score"] = float("nan")
    out["flip_progress_auc"] = float("nan")
    out["n_valid_anchors"] = float(len(success_vals))
    return out


__all__ = ["compute_flip_success_summary"]
