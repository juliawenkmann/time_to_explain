from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import pandas as pd

from ..core.runner import EvalConfig, EvaluationRunner


TEMPME_PAPER_LEVELS = [
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
    0.14,
    0.16,
    0.18,
    0.20,
    0.22,
    0.24,
    0.26,
    0.28,
    0.30,
]


def _infer_base_type(model: Any, *, base_type: str | None) -> str:
    if isinstance(base_type, str) and base_type.strip():
        return base_type.strip().lower()

    names: list[str] = []
    backbone = getattr(model, "backbone", None)
    for raw in (
        getattr(backbone, "base_type", None),
        getattr(backbone, "__class__", type("", (), {})).__name__ if backbone is not None else None,
        getattr(model, "__class__", type("", (), {})).__name__ if model is not None else None,
    ):
        if isinstance(raw, str) and raw:
            names.append(raw.lower())

    joined = " ".join(names)
    if "graphmixer" in joined:
        return "graphmixer"
    if "tgat" in joined or "tgan" in joined:
        return "tgat"
    return "tgn"


def _infer_n_degree(model: Any, *, n_degree: int | None, fallback: int = 20) -> int:
    if n_degree is not None:
        try:
            return max(1, int(n_degree))
        except Exception:
            pass
    for raw in (
        getattr(model, "num_neighbors", None),
        getattr(getattr(model, "backbone", None), "num_neighbors", None),
    ):
        try:
            if raw is not None:
                return max(1, int(raw))
        except Exception:
            continue
    return max(1, int(fallback))


def build_tempme_paper_metric_config(
    *,
    model: Any,
    base_type: str | None = None,
    n_degree: int | None = None,
    levels: Sequence[float] | None = None,
    n_negative_samples: int = 1,
    label_threshold: float = 0.5,
    result_as_logit: bool = True,
    use_official_num_edge: bool = True,
    cap_num_edge_to_candidates: bool = False,
    scale_to_percent: bool = False,
    metric_overrides: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    metric_cfg: Dict[str, Any] = {
        "levels": [float(x) for x in (levels or TEMPME_PAPER_LEVELS)],
        "result_as_logit": bool(result_as_logit),
        "label_threshold": float(label_threshold),
        "base_type": _infer_base_type(model, base_type=base_type),
        "n_degree": _infer_n_degree(model, n_degree=n_degree),
        "use_official_num_edge": bool(use_official_num_edge),
        "cap_num_edge_to_candidates": bool(cap_num_edge_to_candidates),
        "n_negative_samples": max(1, int(n_negative_samples)),
        "scale_to_percent": bool(scale_to_percent),
        # TempME threshold_test ranks by raw explainer score only.
        "ranking": {"prefer_selected": False, "tie_break": "candidate_order"},
    }
    if metric_overrides:
        metric_cfg.update(dict(metric_overrides))
    return metric_cfg


def summarize_tempme_metrics(
    metrics_df: pd.DataFrame,
    *,
    dataset_name: str,
    model_name: str,
    variant: str = "official",
    metric_cols: Sequence[str] | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    if metric_cols is None:
        metric_cols = [
            "tempme_acc_auc.ratio_acc",
            "tempme_acc_auc.ratio_prob",
            "tempme_acc_auc.ratio_logit",
            "tempme_acc_auc.ratio_aps",
            "tempme_acc_auc.ratio_auc",
        ]

    rows: list[dict[str, Any]] = []
    for explainer_name, g in metrics_df.groupby("explainer"):
        row: dict[str, Any] = {
            "dataset": str(dataset_name),
            "model": str(model_name),
            "explainer": str(explainer_name),
            "n_events": int(g["anchor_idx"].nunique()) if "anchor_idx" in g.columns else int(len(g)),
            "variant": str(variant),
        }
        if extra_fields:
            row.update(dict(extra_fields))
        for col in metric_cols:
            if col in g.columns:
                row[col] = float(pd.to_numeric(g[col], errors="coerce").mean())
            else:
                row[col] = float("nan")
        rows.append(row)

    if not rows:
        base: dict[str, Any] = {
            "dataset": str(dataset_name),
            "model": str(model_name),
            "explainer": "",
            "n_events": 0,
            "variant": str(variant),
        }
        if extra_fields:
            base.update(dict(extra_fields))
        for col in metric_cols:
            base[col] = float("nan")
        return pd.DataFrame([base]).iloc[0:0]

    return pd.DataFrame(rows).sort_values(["explainer", "variant"]).reset_index(drop=True)


def compute_and_save_tempme_paper_metrics(
    *,
    results_jsonl: str | Path,
    model: Any,
    events: pd.DataFrame,
    extractor: Any,
    out_dir: str | Path,
    base_name: str,
    dataset_name: str,
    model_name: str,
    seed: int = 42,
    base_type: str | None = None,
    n_degree: int | None = None,
    levels: Sequence[float] | None = None,
    n_negative_samples: int = 1,
    label_threshold: float = 0.5,
    result_as_logit: bool = True,
    use_official_num_edge: bool = True,
    cap_num_edge_to_candidates: bool = False,
    scale_to_percent: bool = False,
    metric_overrides: Mapping[str, Any] | None = None,
    variant: str = "official",
    metric_cols: Sequence[str] | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    metrics_dir = Path(out_dir).expanduser().resolve() / f"{base_name}_tempme_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    tempme_metric_cfg = build_tempme_paper_metric_config(
        model=model,
        base_type=base_type,
        n_degree=n_degree,
        levels=levels,
        n_negative_samples=n_negative_samples,
        label_threshold=label_threshold,
        result_as_logit=result_as_logit,
        use_official_num_edge=use_official_num_edge,
        cap_num_edge_to_candidates=cap_num_edge_to_candidates,
        scale_to_percent=scale_to_percent,
        metric_overrides=metric_overrides,
    )

    cfg = EvalConfig(
        out_dir=str(metrics_dir),
        metrics={"tempme_acc_auc": tempme_metric_cfg},
        seed=int(seed),
        resume=False,
        show_progress=False,
    )

    runner = EvaluationRunner(
        model=model,
        dataset={"events": events},
        extractor=extractor,
        explainers=[],
        config=cfg,
    )

    out = runner.compute_metrics_from_results(
        results_path=str(results_jsonl),
        out_dir=str(metrics_dir),
        resume=False,
    )
    metrics_csv = Path(out["csv"]).expanduser().resolve()
    metrics_df = pd.read_csv(metrics_csv)

    summary_df = summarize_tempme_metrics(
        metrics_df=metrics_df,
        dataset_name=dataset_name,
        model_name=model_name,
        variant=variant,
        metric_cols=metric_cols,
        extra_fields=extra_fields,
    )
    summary_csv = metrics_dir / f"{base_name}_tempme_acc_auc_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    return {
        "metrics_dir": metrics_dir,
        "metrics_csv": metrics_csv,
        "summary_csv": summary_csv,
        "metrics": metrics_df,
        "summary": summary_df,
        "metric_config": tempme_metric_cfg,
    }


__all__ = [
    "TEMPME_PAPER_LEVELS",
    "build_tempme_paper_metric_config",
    "summarize_tempme_metrics",
    "compute_and_save_tempme_paper_metrics",
]
