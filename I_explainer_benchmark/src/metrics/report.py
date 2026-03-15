from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd

from ..notebooks.notebook_runtime_common import (
    as_dataframe as _as_dataframe,
    read_csv_if_exists as _read_csv_if_exists,
    resolve_path,
)
from .cody import _apply_harmonic_charr


def _normalize_run_dirs(run_dirs: Iterable[Any] | None) -> list[Path]:
    out: list[Path] = []
    if run_dirs is None:
        return out
    for item in run_dirs:
        if item is None:
            continue
        try:
            p = resolve_path(item)
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        if p.exists() and p.is_dir():
            out.append(p)
    return out
def resolve_one_spot_metric_inputs(
    *,
    summary_long: Any = None,
    cody_summary: Any = None,
    run_dirs: Iterable[Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve summary-long + CoDy summary from memory or exported run folders."""
    long_df = _as_dataframe(summary_long)
    cody_df = _as_dataframe(cody_summary)
    dirs = _normalize_run_dirs(run_dirs)

    if long_df.empty:
        for d in dirs:
            candidate = _read_csv_if_exists(d / "metric_summary_long.csv")
            if not candidate.empty:
                long_df = candidate
                break

    if cody_df.empty:
        for d in dirs:
            candidates = [d / "cody_style_summary.csv"]
            candidates.extend(sorted(d.glob("*_cody_metrics_summary.csv")))
            candidates.extend(sorted(d.glob("*cody*summary*.csv")))
            for cpath in candidates:
                candidate = _read_csv_if_exists(cpath)
                if candidate.empty:
                    continue
                if any(
                    col in candidate.columns
                    for col in ("cody_AUFSC_plus", "cody_AUFSC_minus", "cody_CHARR")
                ):
                    cody_df = candidate
                    break
            if not cody_df.empty:
                break

    return long_df, cody_df


def build_one_spot_metrics_tables(
    *,
    summary_long: Any = None,
    cody_summary: Any = None,
    dataset_name: str | None = None,
    model_name: str | None = None,
    variant: str = "official",
) -> Dict[str, pd.DataFrame]:
    """Build one-spot notebook metric tables from summary-long + CoDy summary.

    Metric definitions are not recomputed here; this function only merges already
    computed outputs into clean notebook tables.
    """
    long_df = _as_dataframe(summary_long)
    if not long_df.empty:
        if "variant" in long_df.columns:
            long_df = long_df[long_df["variant"].astype(str) == str(variant)]
        if dataset_name and "dataset" in long_df.columns:
            long_df = long_df[long_df["dataset"].astype(str) == str(dataset_name)]
        if model_name and "model" in long_df.columns:
            long_df = long_df[long_df["model"].astype(str) == str(model_name)]

    if long_df.empty:
        wide = pd.DataFrame(columns=["explainer"])
    else:
        index_cols = [c for c in ["dataset", "model", "explainer", "variant", "n_events"] if c in long_df.columns]
        wide = (
            long_df.pivot_table(
                index=index_cols,
                columns="metric",
                values="value",
                aggfunc="mean",
            )
            .reset_index()
            .sort_values(["explainer"] if "explainer" in index_cols else index_cols)
            .reset_index(drop=True)
        )

    cody_df = _as_dataframe(cody_summary)
    if not cody_df.empty:
        if dataset_name and "dataset" in cody_df.columns:
            cody_df = cody_df[cody_df["dataset"].astype(str) == str(dataset_name)]
        if model_name and "model" in cody_df.columns:
            cody_df = cody_df[cody_df["model"].astype(str) == str(model_name)]

        cody_keep = [
            c
            for c in [
                "dataset",
                "model",
                "explainer",
                "n_events",
                "cody_rows",
                "kept_ratio",
                "sparsity_ratio",
                "elapsed_sec",
                "cody_AUFSC_plus",
                "cody_AUFSC_minus",
                "cody_CHARR",
            ]
            if c in cody_df.columns
        ]
        cody_small = cody_df[cody_keep].copy()
        cody_small = (
            cody_small.groupby(
                [c for c in ["dataset", "model", "explainer"] if c in cody_small.columns],
                as_index=False,
            ).mean(numeric_only=True)
        )
        cody_small = _apply_harmonic_charr(cody_small, weight_plus=0.5, weight_minus=0.5)

        if wide.empty:
            wide = cody_small.copy()
            if "variant" not in wide.columns:
                wide["variant"] = str(variant)
        else:
            join_keys = [c for c in ["dataset", "model", "explainer"] if c in wide.columns and c in cody_small.columns]
            if join_keys:
                wide = wide.merge(cody_small, on=join_keys, how="left", suffixes=("", "_cody"))
                if "n_events_cody" in wide.columns and "n_events" in wide.columns:
                    wide["n_events"] = wide["n_events"].fillna(wide["n_events_cody"])
                    wide = wide.drop(columns=["n_events_cody"])

    base_cols = [c for c in ["dataset", "model", "explainer", "variant", "n_events"] if c in wide.columns]
    combined_metric_cols = [
        "best_fid",
        "best_fid_sparsity",
        "aufsc",
        "best_minus_aufsc",
        "best_fid_raw",
        "best_fid_raw_sparsity",
        "flip_success_rate",
        "first_flip_sparsity",
        "first_flip_score",
        "tempme_acc_auc.ratio_acc",
        "tempme_acc_auc.ratio_prob",
        "tempme_acc_auc.ratio_logit",
        "tempme_acc_auc.ratio_aps",
        "tempme_acc_auc.ratio_auc",
        "temgx_aufsc",
        "cody_AUFSC_plus",
        "cody_AUFSC_minus",
        "cody_CHARR",
        "elapsed_sec",
    ]
    core_metric_cols = [
        "best_fid",
        "best_fid_sparsity",
        "aufsc",
        "best_minus_aufsc",
        "best_fid_raw",
        "best_fid_raw_sparsity",
        "flip_success_rate",
        "first_flip_sparsity",
        "first_flip_score",
        "elapsed_sec",
    ]
    aux_metric_cols = [
        "tempme_acc_auc.ratio_acc",
        "tempme_acc_auc.ratio_prob",
        "tempme_acc_auc.ratio_logit",
        "tempme_acc_auc.ratio_aps",
        "tempme_acc_auc.ratio_auc",
        "temgx_aufsc",
        "cody_AUFSC_plus",
        "cody_AUFSC_minus",
        "cody_CHARR",
    ]

    combined_cols = base_cols + [c for c in combined_metric_cols if c in wide.columns]
    core_cols = base_cols + [c for c in core_metric_cols if c in wide.columns]
    aux_cols = base_cols + [c for c in aux_metric_cols if c in wide.columns]

    combined = wide[combined_cols].copy() if combined_cols else pd.DataFrame()
    core = wide[core_cols].copy() if core_cols else pd.DataFrame()
    aux = wide[aux_cols].copy() if aux_cols else pd.DataFrame()

    if "explainer" in combined.columns:
        combined = combined.sort_values("explainer").reset_index(drop=True)
    if "explainer" in core.columns:
        core = core.sort_values("explainer").reset_index(drop=True)
    if "explainer" in aux.columns:
        aux = aux.sort_values("explainer").reset_index(drop=True)

    return {"wide": wide, "combined": combined, "core": core, "aux": aux}


__all__ = ["resolve_one_spot_metric_inputs", "build_one_spot_metrics_tables"]
