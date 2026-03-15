from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..core.runner import EvalConfig, EvaluationRunner
from ..explainers.extractors.khop import KHopCandidatesExtractor
from ..metrics.cody import (
    build_cody_aufsc_curve_table,
    compute_and_save_cody_paper_metrics,
)
from ..models.adapters.tg_model_adapter import TemporalGNNModelAdapter
from ..metrics.flip import compute_flip_success_summary
from ..metrics.report import build_one_spot_metrics_tables, resolve_one_spot_metric_inputs
from .notebook_runtime_common import (
    coerce_float as _as_float,
    resolve_path as _path,
    sorted_sparsity_cols as _sorted_sparsity_cols,
    upsert_csv as _upsert_csv,
)
from ..metrics.tempme import compute_and_save_tempme_paper_metrics

DEFAULT_TGNN_LEVELS: tuple[float, ...] = tuple(i / 20.0 for i in range(21))
DEFAULT_RANKING_CFG: dict[str, Any] = {
    "prefer_selected": True,
    "tie_break": "edge_id",
    "uninformative_fallback": "none",
    "natural_support": "nonzero",
}
SUMMARY_LONG_ID_COLS: tuple[str, ...] = (
    "run_id",
    "source_notebook",
    "created_at",
    "dataset",
    "model",
    "explainer",
    "variant",
    "n_events",
)
CURVE_LONG_EXPORT_COLS: tuple[str, ...] = (
    "run_id",
    "source_notebook",
    "created_at",
    "dataset",
    "model",
    "explainer",
    "variant",
    "target_idx",
    "sparsity",
    "fid_inv",
    "fid_inv_best",
)


def _compute_metrics_from_results(
    *,
    results_jsonl: Path,
    model: Any,
    events: pd.DataFrame,
    extractor: KHopCandidatesExtractor,
    out_dir: Path,
    metrics: Mapping[str, Any],
    seed: int,
    show_progress: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    cfg = EvalConfig(
        out_dir=str(out_dir),
        metrics=dict(metrics),
        seed=int(seed),
        resume=False,
        show_progress=bool(show_progress),
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
        out_dir=str(out_dir),
        resume=False,
    )
    return out, pd.read_csv(out["csv"])


def _with_run_metadata(
    df: pd.DataFrame,
    *,
    run_id: str,
    dataset_name: str,
    model_name: str,
    source_notebook: str,
    created_at: str,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["run_id"] = str(run_id)
    out["dataset"] = str(dataset_name)
    out["model"] = str(model_name)
    out["source_notebook"] = str(source_notebook)
    out["created_at"] = str(created_at)
    return out


def _melt_metric_summary(df: pd.DataFrame, metric_cols: Sequence[str]) -> pd.DataFrame:
    metric_cols = [str(col) for col in metric_cols if str(col) in df.columns]
    if df.empty or not metric_cols:
        return pd.DataFrame()
    return df.melt(
        id_vars=[col for col in SUMMARY_LONG_ID_COLS if col in df.columns],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )


def _prepare_metric_summary_export(
    df: pd.DataFrame,
    *,
    metric_cols: Sequence[str],
    run_id: str,
    dataset_name: str,
    model_name: str,
    source_notebook: str,
    created_at: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    export_df = _with_run_metadata(
        df,
        run_id=run_id,
        dataset_name=dataset_name,
        model_name=model_name,
        source_notebook=source_notebook,
        created_at=created_at,
    )
    return export_df, _melt_metric_summary(export_df, metric_cols)


def _concat_aligned_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame.copy() for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    if len(non_empty) == 1:
        return non_empty[0].reset_index(drop=True)
    all_cols = sorted({col for frame in non_empty for col in frame.columns})
    for frame in non_empty:
        for col in all_cols:
            if col not in frame.columns:
                frame[col] = pd.NA
    return pd.concat([frame[all_cols] for frame in non_empty], ignore_index=True)


def _save_csv_if_not_empty(df: pd.DataFrame, path: Path) -> None:
    if isinstance(df, pd.DataFrame) and not df.empty:
        df.to_csv(path, index=False)


def _upsert_csv_if_not_empty(path: Path, df: pd.DataFrame, key_cols: Sequence[str]) -> None:
    if isinstance(df, pd.DataFrame) and not df.empty:
        _upsert_csv(path, df, list(key_cols))


def _build_runtime_metric_long(
    *,
    results_jsonl: Path,
    run_id: str,
    dataset_name: str,
    model_name: str,
    source_notebook: str,
    created_at: str,
    variant: str,
) -> pd.DataFrame:
    runtime_rows: list[dict[str, Any]] = []
    with results_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            try:
                rec = json.loads(text)
            except json.JSONDecodeError:
                continue
            result = rec.get("result") if isinstance(rec, dict) else None
            if not isinstance(result, dict):
                continue
            elapsed = result.get("elapsed_sec")
            explainer = result.get("explainer")
            if elapsed is None or explainer is None:
                continue
            try:
                runtime_rows.append({"explainer": str(explainer), "elapsed_sec": float(elapsed)})
            except (TypeError, ValueError, OverflowError):
                continue

    if not runtime_rows:
        return pd.DataFrame()

    runtime_long = (
        pd.DataFrame(runtime_rows)
        .groupby("explainer", as_index=False)
        .agg(value=("elapsed_sec", "mean"), n_events=("elapsed_sec", "count"))
    )
    runtime_long["metric"] = "elapsed_sec"
    runtime_long["variant"] = str(variant)
    return _with_run_metadata(
        runtime_long,
        run_id=run_id,
        dataset_name=dataset_name,
        model_name=model_name,
        source_notebook=source_notebook,
        created_at=created_at,
    )


def _ensure_metric_model(model_obj: Any, events: pd.DataFrame) -> Any:
    """Ensure model object provides predict_proba API required by fidelity metrics."""
    if hasattr(model_obj, "predict_proba") and callable(getattr(model_obj, "predict_proba", None)):
        return model_obj
    if hasattr(model_obj, "get_prob") and callable(getattr(model_obj, "get_prob", None)):
        device = getattr(model_obj, "device", None)
        return TemporalGNNModelAdapter(model_obj, events, device=device)
    raise AttributeError(
        "Metrics pipeline requires a model with predict_proba(...) "
        "or a TGNN backbone exposing get_prob(...)."
    )


def materialize_notebook_results(
    *,
    results_jsonl: str | Path | None,
    out_obj: Mapping[str, Any] | None,
    out_dir: str | Path,
    base_name: str | None,
    dataset_name: str,
    model_name: str,
    explainer_name: str,
    result_records: Sequence[Mapping[str, Any]] | None = None,
    results_df: pd.DataFrame | None = None,
) -> tuple[Path, Path, str]:
    """Resolve/write notebook result artifacts used by shared metric pipeline."""
    out_dir_path = _path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    base = str(base_name or "").strip()
    if not base:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{str(dataset_name)}_{str(model_name)}_official_{str(explainer_name)}_{ts}"

    resolved_jsonl: Path | None = None
    if results_jsonl is not None:
        resolved_jsonl = _path(results_jsonl)
    elif isinstance(out_obj, Mapping) and out_obj.get("jsonl") is not None:
        resolved_jsonl = _path(str(out_obj.get("jsonl")))

    if resolved_jsonl is None:
        resolved_jsonl = out_dir_path / f"{base}.jsonl"
        if not result_records:
            raise RuntimeError(
                "Could not resolve results JSONL path. "
                "Provide out_jsonl/out['jsonl'] or result_records."
            )
        with resolved_jsonl.open("w", encoding="utf-8") as fh:
            for rec in result_records:
                rec_dict = dict(rec)
                rec_dict.setdefault("run_id", base)
                fh.write(json.dumps(rec_dict) + "\n")
    else:
        if not base_name:
            base = resolved_jsonl.stem

    if isinstance(results_df, pd.DataFrame) and not results_df.empty:
        csv_path = out_dir_path / f"{base}.csv"
        if not csv_path.exists():
            results_df.to_csv(csv_path, index=False)

    return resolved_jsonl, out_dir_path, base


def run_notebook_metrics_from_namespace(
    namespace: Mapping[str, Any],
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = dict(cfg or {})
    bench_root = Path(
        namespace.get(
            "BENCHMARK_ROOT",
            (namespace["PROJECT_ROOT"] / "I_explainer_benchmark")
            if "PROJECT_ROOT" in namespace
            else (namespace["REPO_ROOT"] / "I_explainer_benchmark"),
        )
    )
    model_name = str(
        namespace["MODEL_TYPE"]
        if "MODEL_TYPE" in namespace
        else namespace.get("MODEL_NAME", namespace.get("BASE_TYPE", ""))
    )
    dataset_name = str(namespace["DATASET_NAME"])
    explainer_alias = str(
        namespace.get("EXPLAINER_ALIAS")
        or namespace.get("winning_alias")
        or namespace.get("EXPLAINER_NAME")
        or cfg.get("explainer_alias")
        or cfg.get("out_dir_name", "explainer")
    ).strip().lower()
    if explainer_alias.startswith("official_"):
        explainer_alias = explainer_alias[len("official_") :]
    if explainer_alias == "metric_optimized_cf_upper":
        explainer_alias = "cf_metric_opt_upper"
    if not explainer_alias:
        explainer_alias = "explainer"

    records = None
    for records_name in (
        "result_records",
        "cody_result_records",
        "tgnn_result_records",
        "temgx_result_records",
        "tempme_result_records",
    ):
        candidate = namespace.get(records_name)
        if isinstance(candidate, list) and len(candidate) > 0:
            records = candidate
            break

    if isinstance(records, list) and len(records) > 0:
        first = records[0]
        rec_explainer = None
        if isinstance(first, dict):
            result_obj = first.get("result")
            if isinstance(result_obj, dict):
                rec_explainer = result_obj.get("explainer")
            if rec_explainer is None:
                rec_explainer = first.get("explainer")
        if rec_explainer:
            explainer_alias = str(rec_explainer).strip().lower()

    results_df = None
    for results_df_name in (
        "results_df",
        "cody_results_df",
        "temgx_results_df",
        "tgnn_results_df",
        "random_results_df",
    ):
        candidate = namespace.get(results_df_name)
        if candidate is not None:
            results_df = candidate
            break

    summary_root = bench_root / "resources" / "results" / "summary_ready"
    default_out_dir = bench_root / "resources" / "results" / f"official_{explainer_alias}"

    results_jsonl, run_out_dir, base_name = materialize_notebook_results(
        results_jsonl=namespace.get("out_jsonl"),
        out_obj=namespace.get("out") if isinstance(namespace.get("out"), dict) else None,
        out_dir=namespace.get(
            "out_dir",
            namespace.get("official_tempme_dir", namespace.get("METRICS_OUT_DIR", default_out_dir)),
        ),
        base_name=namespace.get("base_name"),
        dataset_name=dataset_name,
        model_name=model_name,
        explainer_name=explainer_alias,
        result_records=records,
        results_df=results_df,
    )

    metrics_out_dir = Path(namespace.get("METRICS_OUT_DIR", run_out_dir))
    model_for_metrics = namespace.get(
        "model_for_tgnn_metrics",
        namespace.get("model", namespace.get("backbone")),
    )
    if model_for_metrics is None:
        raise RuntimeError("Could not resolve model object for shared metrics pipeline.")
    if "events" not in namespace:
        raise RuntimeError("`events` must be defined before running shared metrics pipeline.")

    use_navigator = bool(namespace.get("USE_NAVIGATOR", False))
    navigator_type = str(namespace.get("NAVIGATOR_TYPE", "none")).strip().lower() or "none"

    pipeline_out = run_explainer_metric_pipeline(
        results_jsonl=results_jsonl,
        model=model_for_metrics,
        events=namespace["events"],
        out_dir=metrics_out_dir,
        base_name=base_name,
        dataset_name=dataset_name,
        model_name=model_name,
        seed=int(namespace.get("SEED", 42)),
        candidates_size=int(namespace.get("CANDIDATES_SIZE", 64)),
        source_notebook=str(namespace.get("SOURCE_NOTEBOOK", "")),
        target_event_idxs=list(namespace.get("target_event_idxs", [])),
        run_dir=namespace.get("run_dir"),
        summary_ready_root=summary_root,
        summary_ready_group=(str(namespace.get("BASE_TYPE")) if "BASE_TYPE" in namespace else None),
        model_for_tgnn_metrics=namespace.get("model_for_tgnn_metrics", None),
        tempme_base_type=model_name,
        tempme_out_dir=metrics_out_dir,
        tempme_extra_fields={"navigator": (navigator_type if use_navigator else "none")},
    )

    pipeline_out["out_jsonl"] = results_jsonl
    pipeline_out["out_dir"] = run_out_dir
    pipeline_out["base_name"] = base_name
    pipeline_out["run_dir_export"] = Path(pipeline_out.get("run_dir_export", run_out_dir))
    pipeline_out["export_root"] = Path(pipeline_out.get("export_root", summary_root))
    pipeline_out["CURRENT_RUN_ID"] = base_name
    return pipeline_out


def _build_eval_rows(
    curve_df: pd.DataFrame,
    *,
    s_levels: np.ndarray,
    fid_cols: Sequence[str] | None = None,
    fid_best_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    raw_cols = list(fid_cols or [])
    best_cols = list(fid_best_cols or [])
    for _, row in curve_df.iterrows():
        if best_cols:
            fid_inv_best = np.asarray([float(row[c]) for c in best_cols], dtype=float)
            if raw_cols:
                fid_inv = np.asarray([float(row[c]) for c in raw_cols], dtype=float)
            else:
                # For MCTS-exported curves we only have the cumulative-best series.
                fid_inv = fid_inv_best.copy()
        else:
            fid_inv = np.asarray([float(row[c]) for c in raw_cols], dtype=float)
            fid_inv_best = np.maximum.accumulate(fid_inv)
        pred_full = float("nan")
        for pred_col in (
            "prediction_profile.prediction_full",
            "best_fid.prediction_full",
            "tgnn_aufsc.prediction_full",
            "prediction_full",
        ):
            if pred_col in row.index and pd.notna(row[pred_col]):
                pred_full = float(row[pred_col])
                break

        drop_cols = sorted(
            [c for c in row.index if str(c).startswith("prediction_profile.prediction_drop.@s=")],
            key=lambda c: float(str(c).split("@s=")[-1]),
        )
        drop_map = {
            float(str(c).split("@s=")[-1]): float(row[c]) if pd.notna(row[c]) else float("nan")
            for c in drop_cols
        }

        for s, f_raw, f_best in zip(s_levels.tolist(), fid_inv.tolist(), fid_inv_best.tolist()):
            drop_val = drop_map.get(float(s), float("nan"))
            if not np.isfinite(drop_val) and drop_map:
                nearest_s = min(drop_map.keys(), key=lambda ss: abs(float(ss) - float(s)))
                drop_val = float(drop_map.get(nearest_s, float("nan")))

            rows.append(
                {
                    "explainer": str(row["explainer"]),
                    "anchor_idx": int(row["anchor_idx"]),
                    "sparsity": float(s),
                    "fid_inv": float(f_raw),
                    "fid_inv_best": float(f_best),
                    "prediction_full": float(pred_full),
                    "prediction_keep": float(pred_full + f_raw) if np.isfinite(pred_full) else float("nan"),
                    "prediction_drop": float(drop_val),
                }
            )
    return pd.DataFrame(rows)


def _aggregate_eval(
    eval_df: pd.DataFrame,
    *,
    dataset_name: str,
    model_name: str,
    variant: str,
    flip_budget: float,
    flip_area_threshold: float,
    flip_point_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    curve_tables: list[pd.DataFrame] = []

    for explainer, g in eval_df.groupby("explainer", as_index=False):
        tab = g.groupby("sparsity", as_index=True).mean(numeric_only=True).sort_index()
        x = tab.index.to_numpy(dtype=float)
        y_raw = tab["fid_inv"].to_numpy(dtype=float)
        y_best = tab["fid_inv_best"].to_numpy(dtype=float)

        best_fid = float(np.max(y_best))
        best_fid_sparsity = float(x[int(np.argmax(y_best))])
        aufsc = float(np.trapz(y_best, x))
        direct_aufsc = pd.to_numeric(g.get("tgnn_aufsc.value"), errors="coerce") if "tgnn_aufsc.value" in g.columns else None
        if (x.size < 2 or not np.any(np.diff(x) > 0)) and direct_aufsc is not None and direct_aufsc.notna().any():
            aufsc = float(direct_aufsc.mean())

        flip_stats = compute_flip_success_summary(
            g,
            budget=flip_budget,
            area_threshold=flip_area_threshold,
            point_threshold=flip_point_threshold,
            anchor_col="anchor_idx",
            sparsity_col="sparsity",
            prediction_full_col="prediction_full",
            prediction_drop_col="prediction_drop",
        )

        summary_rows.append(
            {
                "dataset": str(dataset_name),
                "model": str(model_name),
                "explainer": str(explainer),
                "n_events": int(g["anchor_idx"].nunique()),
                "variant": str(variant),
                "best_fid": best_fid,
                "best_fid_sparsity": best_fid_sparsity,
                "aufsc": aufsc,
                "best_minus_aufsc": float(best_fid - aufsc),
                "fid_best_flat_curve": bool(np.allclose(y_best, y_best[0], rtol=1e-12, atol=1e-12)),
                "best_fid_raw": float(np.max(y_raw)),
                "best_fid_raw_sparsity": float(x[int(np.argmax(y_raw))]),
                "flip_success_rate": float(flip_stats["flip_success_rate"]),
                "first_flip_sparsity": float(flip_stats["first_flip_sparsity"]),
            }
        )

        tab_out = tab.reset_index().copy()
        tab_out.insert(0, "explainer", str(explainer))
        tab_out.insert(1, "variant", str(variant))
        curve_tables.append(tab_out)

    summary = pd.DataFrame(summary_rows).sort_values(["explainer", "variant"]).reset_index(drop=True)
    curve = pd.concat(curve_tables, axis=0, ignore_index=True) if curve_tables else pd.DataFrame()
    return summary, curve


def run_explainer_metric_pipeline(
    *,
    results_jsonl: str | Path,
    model: Any,
    events: pd.DataFrame,
    out_dir: str | Path,
    base_name: str,
    dataset_name: str,
    model_name: str,
    seed: int = 42,
    candidates_size: int = 64,
    tgnn_levels: Sequence[float] | None = None,
    ranking_cfg: Mapping[str, Any] | None = None,
    flip_progress_budget: float = 0.9,
    flip_progress_area_threshold: float = 0.4,
    flip_progress_point_threshold: float | None = None,
    source_notebook: str = "",
    target_event_idxs: Sequence[int] | None = None,
    run_dir: str | Path | None = None,
    summary_ready_root: str | Path | None = None,
    summary_ready_group: str | None = None,
    model_for_tgnn_metrics: Any | None = None,
    run_temgx_metric: bool = True,
    run_tempme_metric: bool = True,
    tempme_base_type: str | None = None,
    tempme_out_dir: str | Path | None = None,
    tempme_extra_fields: Mapping[str, Any] | None = None,
    variant: str = "official",
) -> dict[str, Any]:
    """Compute shared notebook metrics + exports and return one-spot tables."""
    out_jsonl = _path(results_jsonl)
    out_dir_path = _path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    if run_dir is None:
        run_dir_path = out_jsonl.parent
    else:
        run_dir_path = _path(run_dir)

    model_for_metrics = _ensure_metric_model(model_for_tgnn_metrics or model, events)
    ranking_src = ranking_cfg if ranking_cfg is not None else DEFAULT_RANKING_CFG
    ranking = dict(ranking_src)
    levels_src = tgnn_levels if tgnn_levels is not None else DEFAULT_TGNN_LEVELS
    tgnn_levels = [float(x) for x in list(levels_src)]

    # 1) CoDy paper metrics.
    cody_saved = compute_and_save_cody_paper_metrics(
        results_jsonl=out_jsonl,
        model=model_for_metrics,
        events=events,
        out_dir=out_dir_path,
        base_name=str(base_name),
        score_is_logit=True,
        decision_threshold=0.0,
        max_sparsity=1.0,
        n_grid=101,
        allow_model_inference=True,
    )
    cody_summary = cody_saved["summary"]
    cody_detail = cody_saved["detail"]

    # 2) TGNN curve metrics (official + zero_at_s0) + flip.
    tgnn_metrics_dir = out_dir_path / f"{base_name}_tgnn_metrics"
    tgnn_metrics_dir.mkdir(parents=True, exist_ok=True)

    extractor_for_metrics = KHopCandidatesExtractor(
        model=model_for_metrics,
        events=events,
        candidates_size=max(1, int(candidates_size)),
        num_hops=int(getattr(model_for_metrics, "num_layers", 2) or 2),
    )

    tgnn_out, tgnn_metrics_df = _compute_metrics_from_results(
        results_jsonl=out_jsonl,
        model=model_for_metrics,
        events=events,
        extractor=extractor_for_metrics,
        out_dir=tgnn_metrics_dir,
        metrics={
            "tgnn_aufsc": {
                "sparsity_levels": tgnn_levels,
                "result_as_logit": True,
                "ensure_min_one": True,
                "clamp_non_negative": False,
                "include_series": True,
                "order_strategy": "strict",
                "ranking": dict(ranking),
            },
            "prediction_profile": {
                "mode": "drop",
                "levels": tgnn_levels,
                "label_threshold": 0.5,
                "order_strategy": "strict",
                "ranking": dict(ranking),
            },
        },
        seed=int(seed),
    )

    zero_s0_dir = tgnn_metrics_dir / "zero_s0"
    zero_s0_dir.mkdir(parents=True, exist_ok=True)
    zero_out, zero_df = _compute_metrics_from_results(
        results_jsonl=out_jsonl,
        model=model_for_metrics,
        events=events,
        extractor=extractor_for_metrics,
        out_dir=zero_s0_dir,
        metrics={
            "tgnn_aufsc": {
                "sparsity_levels": [0.0],
                "result_as_logit": True,
                "ensure_min_one": False,
                "clamp_non_negative": False,
                "include_series": True,
                "order_strategy": "strict",
                "ranking": dict(ranking),
            },
        },
        seed=int(seed),
    )

    fid_raw_cols = _sorted_sparsity_cols(tgnn_metrics_df, "tgnn_aufsc.fid_inv.@s=")
    fid_best_cols = _sorted_sparsity_cols(tgnn_metrics_df, "tgnn_aufsc.best.@s=")
    fid_curve_cols = _sorted_sparsity_cols(tgnn_metrics_df, "tgnn_aufsc.@s=")

    curve_kind = "raw"
    selected_curve_cols: list[str] = []
    if fid_best_cols:
        curve_kind = "best"
        selected_curve_cols = fid_best_cols
    elif fid_raw_cols:
        curve_kind = "raw"
        selected_curve_cols = fid_raw_cols
    elif fid_curve_cols:
        # MCTS-exported TGNN curve: cumulative-best series is emitted directly as @s=*.
        curve_kind = "curve"
        selected_curve_cols = fid_curve_cols
    else:
        fallback_col = "tgnn_aufsc.fid_inv.@s=0.0"
        tgnn_metrics_df[fallback_col] = pd.to_numeric(tgnn_metrics_df.get("tgnn_aufsc.value", np.nan), errors="coerce")
        zero_df[fallback_col] = pd.to_numeric(zero_df.get("tgnn_aufsc.value", np.nan), errors="coerce")
        fid_raw_cols = [fallback_col]
        selected_curve_cols = fid_raw_cols
        curve_kind = "raw"

    if curve_kind == "best":
        fid0_cols = _sorted_sparsity_cols(zero_df, "tgnn_aufsc.best.@s=")
    elif curve_kind == "curve":
        fid0_cols = _sorted_sparsity_cols(zero_df, "tgnn_aufsc.@s=")
    else:
        fid0_cols = _sorted_sparsity_cols(zero_df, "tgnn_aufsc.fid_inv.@s=")
    if not fid0_cols:
        fid0_cols = [selected_curve_cols[0]]
        if fid0_cols[0] not in zero_df.columns:
            zero_df[fid0_cols[0]] = np.nan
    fid0_col = min(fid0_cols, key=lambda c: abs(float(str(c).split("@s=")[-1])))

    s_levels = np.asarray([float(c.split("@s=")[-1]) for c in selected_curve_cols], dtype=float)
    if curve_kind == "raw":
        eval_df_official = _build_eval_rows(
            tgnn_metrics_df,
            s_levels=s_levels,
            fid_cols=fid_raw_cols,
        )
    else:
        eval_df_official = _build_eval_rows(
            tgnn_metrics_df,
            s_levels=s_levels,
            fid_best_cols=selected_curve_cols,
        )
    flip_kwargs = {
        "flip_budget": _as_float(flip_progress_budget, 0.9),
        "flip_area_threshold": _as_float(flip_progress_area_threshold, 0.4),
        "flip_point_threshold": _as_float(
            flip_progress_point_threshold if flip_progress_point_threshold is not None else flip_progress_area_threshold,
            0.4,
        ),
    }
    summary_official, curve_official = _aggregate_eval(
        eval_df_official,
        dataset_name=dataset_name,
        model_name=model_name,
        variant="official",
        **flip_kwargs,
    )

    zero_map = {(str(r["explainer"]), int(r["anchor_idx"])): float(r[fid0_col]) for _, r in zero_df.iterrows()}
    eval_df_zero = eval_df_official.copy()
    for idx in eval_df_zero.index[eval_df_zero["sparsity"] == 0.0].tolist():
        key = (str(eval_df_zero.at[idx, "explainer"]), int(eval_df_zero.at[idx, "anchor_idx"]))
        if key in zero_map:
            eval_df_zero.at[idx, "fid_inv"] = float(zero_map[key])
    eval_df_zero = eval_df_zero.sort_values(["explainer", "anchor_idx", "sparsity"]).reset_index(drop=True)
    eval_df_zero["fid_inv_best"] = (
        eval_df_zero.groupby(["explainer", "anchor_idx"], as_index=False)["fid_inv"].cummax()
    )

    summary_zero, curve_zero = _aggregate_eval(
        eval_df_zero,
        dataset_name=dataset_name,
        model_name=model_name,
        variant="zero_at_s0",
        **flip_kwargs,
    )

    summary_both = pd.concat([summary_official, summary_zero], axis=0, ignore_index=True)
    curve_both = pd.concat([curve_official, curve_zero], axis=0, ignore_index=True)

    compare_df = summary_official.merge(
        summary_zero,
        on=["dataset", "model", "explainer", "n_events"],
        suffixes=("_official", "_zero_at_s0"),
    )
    compare_df["delta_best_fid"] = compare_df["best_fid_zero_at_s0"] - compare_df["best_fid_official"]
    compare_df["delta_aufsc"] = compare_df["aufsc_zero_at_s0"] - compare_df["aufsc_official"]

    summary_csv = tgnn_metrics_dir / f"{base_name}_tgnn_aufsc_bestfid_summary_both.csv"
    official_curve_csv = tgnn_metrics_dir / f"{base_name}_tgnn_official_fid_curve.csv"
    zero_curve_csv = tgnn_metrics_dir / f"{base_name}_tgnn_zero_at_s0_fid_curve.csv"
    compare_csv = tgnn_metrics_dir / f"{base_name}_tgnn_official_vs_zero_at_s0.csv"

    _save_csv_if_not_empty(summary_both, summary_csv)
    _save_csv_if_not_empty(curve_official, official_curve_csv)
    _save_csv_if_not_empty(curve_zero, zero_curve_csv)
    _save_csv_if_not_empty(compare_df, compare_csv)

    # 3) TempME metric.
    tempme_summary = pd.DataFrame()
    tempme_saved: dict[str, Any] | None = None
    if run_tempme_metric:
        tempme_saved = compute_and_save_tempme_paper_metrics(
            results_jsonl=out_jsonl,
            model=model_for_metrics,
            events=events,
            extractor=extractor_for_metrics,
            out_dir=_path(tempme_out_dir) if tempme_out_dir is not None else out_dir_path,
            base_name=str(base_name),
            dataset_name=str(dataset_name),
            model_name=str(model_name),
            seed=int(seed),
            base_type=str(tempme_base_type or model_name),
            extra_fields=dict(tempme_extra_fields or {}),
        )
        tempme_summary = tempme_saved["summary"]

    # 4) TemGX AUFSC metric.
    temgx_summary = pd.DataFrame()
    temgx_saved: dict[str, Any] | None = None
    if run_temgx_metric:
        temgx_metrics_dir = out_dir_path / f"{base_name}_temgx_metrics"
        temgx_metrics_dir.mkdir(parents=True, exist_ok=True)

        temgx_out, temgx_metrics_df = _compute_metrics_from_results(
            results_jsonl=out_jsonl,
            model=model_for_metrics,
            events=events,
            extractor=extractor_for_metrics,
            out_dir=temgx_metrics_dir,
            metrics={
                "temgx_aufsc": {
                    "sparsity_levels": list(DEFAULT_TGNN_LEVELS),
                    "mode": "minus",
                    "result_as_logit": False,
                    "order_strategy": "strict",
                    "ranking": dict(ranking),
                },
            },
            seed=int(seed),
            show_progress=False,
        )

        temgx_rows: list[dict[str, Any]] = []
        for explainer_name, g in temgx_metrics_df.groupby("explainer"):
            temgx_col = "temgx_aufsc.value" if "temgx_aufsc.value" in g.columns else (
                "temgx_aufsc.aufsc" if "temgx_aufsc.aufsc" in g.columns else None
            )
            temgx_rows.append(
                {
                    "dataset": str(dataset_name),
                    "model": str(model_name),
                    "explainer": str(explainer_name),
                    "n_events": int(g["anchor_idx"].nunique()) if "anchor_idx" in g.columns else int(len(g)),
                    "variant": "official",
                    "temgx_aufsc": float(pd.to_numeric(g[temgx_col], errors="coerce").mean()) if temgx_col else np.nan,
                }
            )
        temgx_summary = pd.DataFrame(temgx_rows).sort_values(["explainer", "variant"]).reset_index(drop=True)
        temgx_summary_csv = temgx_metrics_dir / f"{base_name}_temgx_aufsc_summary.csv"
        _save_csv_if_not_empty(temgx_summary, temgx_summary_csv)
        temgx_saved = {"metrics_csv": temgx_out["csv"], "summary_csv": temgx_summary_csv, "summary": temgx_summary}

    # 5) Shared summary-ready export.
    if summary_ready_root is None:
        summary_ready_root = out_dir_path.parent / "summary_ready"
    export_root = _path(summary_ready_root)
    export_root.mkdir(parents=True, exist_ok=True)
    export_model_root = export_root / str(summary_ready_group) if summary_ready_group else export_root
    export_model_root.mkdir(parents=True, exist_ok=True)

    run_dir_export = export_model_root / str(base_name)
    run_dir_export.mkdir(parents=True, exist_ok=True)

    created_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    export_meta = {
        "run_id": str(base_name),
        "dataset_name": str(dataset_name),
        "model_name": str(model_name),
        "source_notebook": str(source_notebook),
        "created_at": created_at,
    }

    curve_parts = [
        frame.assign(variant=variant_name)
        for frame, variant_name in ((eval_df_official, "official"), (eval_df_zero, "zero_at_s0"))
        if isinstance(frame, pd.DataFrame) and not frame.empty
    ]
    if not curve_parts:
        raise RuntimeError("No curve data produced for export.")

    curve_long = pd.concat(curve_parts, axis=0, ignore_index=True)
    if "explainer" not in curve_long.columns:
        curve_long["explainer"] = "unknown"
    if "anchor_idx" in curve_long.columns:
        curve_long["target_idx"] = pd.to_numeric(curve_long["anchor_idx"], errors="coerce")
    elif "event_idx" in curve_long.columns:
        curve_long["target_idx"] = pd.to_numeric(curve_long["event_idx"], errors="coerce")
    else:
        curve_long["target_idx"] = pd.NA

    curve_long = _with_run_metadata(curve_long, **export_meta)
    curve_long = curve_long[
        [c for c in CURVE_LONG_EXPORT_COLS if c in curve_long.columns]
    ].sort_values(
        [c for c in ["explainer", "variant", "target_idx", "sparsity"] if c in curve_long.columns]
    ).reset_index(drop=True)

    curve_mean = (
        curve_long.groupby(
            ["run_id", "source_notebook", "created_at", "dataset", "model", "explainer", "variant", "sparsity"],
            as_index=False,
        )[[c for c in ["fid_inv", "fid_inv_best"] if c in curve_long.columns]].mean(numeric_only=True)
    )

    metric_cols_pref = [
        "best_fid",
        "best_fid_sparsity",
        "aufsc",
        "best_minus_aufsc",
        "best_fid_raw",
        "best_fid_raw_sparsity",
        "best_fid_raw_lt1",
        "best_fid_raw_lt1_sparsity",
        "flip_success_rate",
        "first_flip_sparsity",
        "elapsed_sec",
    ]
    summary_wide, summary_long = _prepare_metric_summary_export(
        summary_both,
        metric_cols=metric_cols_pref,
        **export_meta,
    )
    runtime_long = _build_runtime_metric_long(
        results_jsonl=out_jsonl,
        variant=str(variant),
        **export_meta,
    )

    # Append TempME summary metrics.
    tempme_summary_export, tempme_long = _prepare_metric_summary_export(
        tempme_summary,
        metric_cols=[
            "tempme_acc_auc.ratio_acc",
            "tempme_acc_auc.ratio_prob",
            "tempme_acc_auc.ratio_logit",
            "tempme_acc_auc.ratio_aps",
            "tempme_acc_auc.ratio_auc",
        ],
        **export_meta,
    )
    _save_csv_if_not_empty(tempme_summary_export, run_dir_export / "tempme_metric_summary.csv")

    # Append TemGX summary metric.
    temgx_summary_export, temgx_long = _prepare_metric_summary_export(
        temgx_summary,
        metric_cols=["temgx_aufsc"],
        **export_meta,
    )
    _save_csv_if_not_empty(temgx_summary_export, run_dir_export / "temgx_metric_summary.csv")
    summary_long = _concat_aligned_frames([summary_long, runtime_long, tempme_long, temgx_long])

    # Save run-local files.
    curve_long_path = run_dir_export / "fidelity_curve_long.csv"
    curve_mean_path = run_dir_export / "fidelity_curve_mean.csv"
    summary_wide_path = run_dir_export / "metric_summary_wide.csv"
    summary_long_path = run_dir_export / "metric_summary_long.csv"

    curve_long.to_csv(curve_long_path, index=False)
    curve_mean.to_csv(curve_mean_path, index=False)
    _save_csv_if_not_empty(summary_wide, summary_wide_path)
    _save_csv_if_not_empty(summary_long, summary_long_path)

    cody_summary_export = _with_run_metadata(cody_summary, **export_meta)
    _save_csv_if_not_empty(cody_summary_export, run_dir_export / "cody_style_summary.csv")

    cody_detail_export = _with_run_metadata(cody_detail, **export_meta)
    _save_csv_if_not_empty(cody_detail_export, run_dir_export / "cody_style_detail.csv")

    cody_plus_curve_export = pd.DataFrame()
    cody_plus_curve_csv_path: Path | None = None
    if not cody_detail_export.empty:
        cody_plus_curve_export = build_cody_aufsc_curve_table(
            detail=cody_detail_export,
            fidelity_col="cody_fidelity_plus_change",
            sparsity_col="cody_sparsity",
            group_col="explainer",
            max_sparsity=1.0,
            n_grid=101,
        )
        if not cody_plus_curve_export.empty:
            cody_plus_curve_csv_path = run_dir_export / "cody_fidelity_plus_curve.csv"
            _save_csv_if_not_empty(cody_plus_curve_export, cody_plus_curve_csv_path)

    # Global upserts.
    _upsert_csv(
        export_root / "all_fidelity_curve_long.csv",
        curve_long,
        ["run_id", "source_notebook", "explainer", "variant", "target_idx", "sparsity"],
    )
    _upsert_csv(
        export_root / "all_fidelity_curve_mean.csv",
        curve_mean,
        ["run_id", "source_notebook", "explainer", "variant", "sparsity"],
    )
    _upsert_csv_if_not_empty(
        export_root / "all_metric_summary_long.csv",
        summary_long,
        ["run_id", "source_notebook", "explainer", "variant", "metric"],
    )
    _upsert_csv_if_not_empty(
        export_root / "all_cody_style_summary.csv",
        cody_summary_export,
        ["run_id", "source_notebook", "explainer"],
    )
    if not cody_plus_curve_export.empty:
        cody_plus_curve_global = _with_run_metadata(cody_plus_curve_export, **export_meta)
        _upsert_csv(
            export_root / "all_cody_fidelity_plus_curve.csv",
            cody_plus_curve_global,
            ["run_id", "source_notebook", "explainer", "sparsity"],
        )

    manifest_row = pd.DataFrame(
        [
            {
                "run_id": str(base_name),
                "source_notebook": str(source_notebook),
                "dataset": str(dataset_name),
                "model": str(model_name),
                "explainer_group": f"official_{str(model_name).lower()}",
                "n_targets": int(len(target_event_idxs)) if target_event_idxs is not None else pd.NA,
                "run_dir": str(run_dir_export),
                "created_at": created_at,
            }
        ]
    )
    _upsert_csv(export_root / "manifest.csv", manifest_row, ["run_id", "source_notebook"])

    # 6) One-spot tables.
    summary_long_resolved, cody_summary_resolved = resolve_one_spot_metric_inputs(
        summary_long=summary_long,
        cody_summary=cody_summary_export,
        run_dirs=[run_dir_export, run_dir_path],
    )
    one_spot = build_one_spot_metrics_tables(
        summary_long=summary_long_resolved,
        cody_summary=cody_summary_resolved,
        dataset_name=str(dataset_name),
        model_name=str(model_name),
        variant=str(variant),
    )

    return {
        "cody_saved": cody_saved,
        "tgnn_metrics_csv": tgnn_out["csv"],
        "tgnn_zero_csv": zero_out["csv"],
        "summary_both": summary_both,
        "curve_both": curve_both,
        "compare_df": compare_df,
        "eval_df_official": eval_df_official,
        "eval_df_zero": eval_df_zero,
        "tempme_saved": tempme_saved,
        "temgx_saved": temgx_saved,
        "summary_long": summary_long,
        "cody_summary": cody_summary_export,
        "cody_plus_curve": cody_plus_curve_export,
        "run_dir_export": run_dir_export,
        "export_root": export_root,
        "one_spot": one_spot,
        "paths": {
            "curve_long": curve_long_path,
            "curve_mean": curve_mean_path,
            "summary_wide": summary_wide_path,
            "summary_long": summary_long_path,
            "summary_both_csv": summary_csv,
            "official_curve_csv": official_curve_csv,
            "zero_curve_csv": zero_curve_csv,
            "compare_csv": compare_csv,
            "cody_plus_curve_csv": cody_plus_curve_csv_path,
        },
    }


__all__ = [
    "materialize_notebook_results",
    "run_explainer_metric_pipeline",
    "run_notebook_metrics_from_namespace",
]
