from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..viz.visualization.metrics import compute_cody_style_report


def _apply_harmonic_charr(
    summary: pd.DataFrame,
    *,
    weight_plus: float = 0.5,
    weight_minus: float = 0.5,
) -> pd.DataFrame:
    if summary is None or summary.empty:
        return summary
    if not {"cody_AUFSC_plus", "cody_AUFSC_minus"}.issubset(set(summary.columns)):
        return summary

    out = summary.copy()
    plus = pd.to_numeric(out["cody_AUFSC_plus"], errors="coerce").to_numpy(dtype=float)
    minus = pd.to_numeric(out["cody_AUFSC_minus"], errors="coerce").to_numpy(dtype=float)
    w_plus = float(weight_plus)
    w_minus = float(weight_minus)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = (w_plus / plus) + (w_minus / minus)
    valid = (
        np.isfinite(plus)
        & np.isfinite(minus)
        & (plus != 0.0)
        & (minus != 0.0)
        & np.isfinite(denom)
        & (denom != 0.0)
    )
    charr = np.full_like(plus, np.nan, dtype=float)
    charr[valid] = (w_plus + w_minus) / denom[valid]
    out["cody_CHARR"] = charr
    return out


def compute_cody_paper_report(
    *,
    results_jsonl: str | Path,
    model: Any = None,
    dataset: Any = None,
    events: pd.DataFrame | None = None,
    score_is_logit: bool = True,
    decision_threshold: float = 0.0,
    max_sparsity: float = 1.0,
    n_grid: int = 101,
    allow_model_inference: bool = True,
    char_weight_plus: float = 0.5,
    char_weight_minus: float = 0.5,
) -> Dict[str, pd.DataFrame]:
    """Compute CoDy metrics and enforce harmonic CHARR from AUFSC+/AUFSC-."""
    resolved_dataset = dataset
    if resolved_dataset is None and events is not None:
        resolved_dataset = {"events": events}

    report = compute_cody_style_report(
        results_jsonl=results_jsonl,
        model=model,
        dataset=resolved_dataset,
        score_is_logit=score_is_logit,
        decision_threshold=decision_threshold,
        max_sparsity=max_sparsity,
        n_grid=n_grid,
        allow_model_inference=allow_model_inference,
        fidelity_mode="score",
        charr_mode="score",
    )
    summary = report.get("summary")
    if isinstance(summary, pd.DataFrame):
        report["summary"] = _apply_harmonic_charr(
            summary,
            weight_plus=float(char_weight_plus),
            weight_minus=float(char_weight_minus),
        )
    return report


def save_cody_report_csv(
    *,
    report: Dict[str, pd.DataFrame],
    out_dir: str | Path,
    base_name: str,
    sort_summary_by_explainer: bool = True,
) -> Dict[str, Any]:
    """Persist CoDy summary/detail CSV files with canonical naming."""
    out_dir_path = Path(out_dir).expanduser().resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    summary = report.get("summary")
    detail = report.get("detail")

    summary_csv = out_dir_path / f"{base_name}_cody_metrics_summary.csv"
    detail_csv = out_dir_path / f"{base_name}_cody_metrics_detail.csv"

    if isinstance(summary, pd.DataFrame) and not summary.empty:
        if sort_summary_by_explainer and "explainer" in summary.columns:
            summary = summary.sort_values("explainer").reset_index(drop=True)
        summary.to_csv(summary_csv, index=False)

    if isinstance(detail, pd.DataFrame) and not detail.empty:
        detail.to_csv(detail_csv, index=False)

    return {
        "summary": summary,
        "detail": detail,
        "summary_csv": summary_csv,
        "detail_csv": detail_csv,
    }


def compute_and_save_cody_paper_metrics(
    *,
    results_jsonl: str | Path,
    model: Any = None,
    dataset: Any = None,
    events: pd.DataFrame | None = None,
    out_dir: str | Path,
    base_name: str,
    score_is_logit: bool = True,
    decision_threshold: float = 0.0,
    max_sparsity: float = 1.0,
    n_grid: int = 101,
    allow_model_inference: bool = True,
    char_weight_plus: float = 0.5,
    char_weight_minus: float = 0.5,
) -> Dict[str, Any]:
    """Compute CoDy paper metrics and save CSV artifacts."""
    report = compute_cody_paper_report(
        results_jsonl=results_jsonl,
        model=model,
        dataset=dataset,
        events=events,
        score_is_logit=score_is_logit,
        decision_threshold=decision_threshold,
        max_sparsity=max_sparsity,
        n_grid=n_grid,
        allow_model_inference=allow_model_inference,
        char_weight_plus=char_weight_plus,
        char_weight_minus=char_weight_minus,
    )
    saved = save_cody_report_csv(
        report=report,
        out_dir=out_dir,
        base_name=base_name,
        sort_summary_by_explainer=True,
    )
    saved["report"] = report
    return saved


def build_cody_aufsc_curve_table(
    *,
    detail: pd.DataFrame,
    fidelity_col: str = "cody_fidelity_plus_change",
    sparsity_col: str = "cody_sparsity",
    group_col: str = "explainer",
    max_sparsity: float = 1.0,
    n_grid: int = 101,
) -> pd.DataFrame:
    """Reconstruct the exact AUFSC curve used in CoDy aggregation.

    This mirrors the curve logic in `compute_cody_style_report`:
    for each sparsity threshold on a uniform grid, average fidelity values of
    rows with sparsity <= threshold, then integrate that curve for AUFSC.
    """
    if detail is None or detail.empty:
        return pd.DataFrame(
            columns=[group_col, "sparsity", "fidelity", "n_prefix", "aufsc"]
        )
    required_cols = {str(fidelity_col), str(sparsity_col)}
    if not required_cols.issubset(set(detail.columns)):
        return pd.DataFrame(
            columns=[group_col, "sparsity", "fidelity", "n_prefix", "aufsc"]
        )
    if group_col not in detail.columns:
        work = detail.copy()
        work[group_col] = "explainer"
    else:
        work = detail.copy()

    sp = float(max(float(max_sparsity), 1e-12))
    grid = np.linspace(0.0, sp, int(max(2, n_grid)))
    rows: list[dict[str, Any]] = []

    for group_value, g in work.groupby(group_col, dropna=False):
        s = pd.to_numeric(g[sparsity_col], errors="coerce").to_numpy(dtype=float)
        v = pd.to_numeric(g[fidelity_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(s) & np.isfinite(v)
        s = s[valid]
        v = v[valid]
        if s.size == 0:
            continue

        s = np.clip(s, 0.0, sp)
        order = np.argsort(s, kind="mergesort")
        s_sorted = s[order]
        v_sorted = v[order]
        v_prefix = np.cumsum(v_sorted)

        y = np.zeros_like(grid, dtype=float)
        n_prefix = np.zeros_like(grid, dtype=int)
        seen = 0
        for idx, thr in enumerate(grid):
            while seen < s_sorted.size and s_sorted[seen] <= float(thr):
                seen += 1
            n_prefix[idx] = int(seen)
            y[idx] = 0.0 if seen == 0 else float(v_prefix[seen - 1]) / float(seen)

        aufsc = float(np.trapz(y, grid) / sp)
        for s_thr, f_val, n_val in zip(grid.tolist(), y.tolist(), n_prefix.tolist()):
            rows.append(
                {
                    group_col: str(group_value),
                    "sparsity": float(s_thr),
                    "fidelity": float(f_val),
                    "n_prefix": int(n_val),
                    "aufsc": float(aufsc),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[group_col, "sparsity", "fidelity", "n_prefix", "aufsc"]
        )
    out = pd.DataFrame(rows)
    out = out.sort_values([group_col, "sparsity"]).reset_index(drop=True)
    return out


def save_cody_fidelity_curve_plot(
    *,
    curve_df: pd.DataFrame,
    out_path: str | Path,
    title: str = "CoDy Fidelity+ Curve (used for AUFSC+)",
    group_col: str = "explainer",
    dpi: int = 180,
) -> Path | None:
    """Save a fidelity-vs-sparsity plot for a CoDy AUFSC curve table."""
    if curve_df is None or curve_df.empty:
        return None

    import matplotlib.pyplot as plt

    path = Path(out_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for group_value, g in curve_df.groupby(group_col, dropna=False):
        tab = g.sort_values("sparsity")
        x = pd.to_numeric(tab["sparsity"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(tab["fidelity"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            continue
        x = x[finite]
        y = y[finite]
        aufsc_val = float(pd.to_numeric(tab.get("aufsc"), errors="coerce").dropna().iloc[0]) if "aufsc" in tab.columns and pd.to_numeric(tab.get("aufsc"), errors="coerce").notna().any() else float("nan")
        label = f"{group_value} (AUFSC+={aufsc_val:.4f})" if np.isfinite(aufsc_val) else str(group_value)
        ax.plot(x, y, linewidth=2.0, label=label)

    ax.set_xlabel("Sparsity")
    ax.set_ylabel("Fidelity+")
    ax.set_title(str(title))
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    if path.suffix.lower() != ".png":
        fig.savefig(path.with_suffix(".png"), dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return path


__all__ = [
    "_apply_harmonic_charr",
    "compute_cody_paper_report",
    "save_cody_report_csv",
    "compute_and_save_cody_paper_metrics",
    "build_cody_aufsc_curve_table",
    "save_cody_fidelity_curve_plot",
]
