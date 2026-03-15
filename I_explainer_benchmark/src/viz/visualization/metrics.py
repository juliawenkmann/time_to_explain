from __future__ import annotations

import colorsys
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import (
    COLORS,
    PLOT_COLORWAY,
    _PLOTLY_TEMPLATE,
    _auto_show,
    _maybe_save,
    _rgba,
    _slugify,
    apply_matplotlib_style,
    go,
)

DEFAULT_METRIC_LABELS: Dict[str, str] = {
    "fidelity_minus.value": "Fidelity- (drop explanation edges)",
    "fidelity_plus.value": "Fidelity+ (explanation only)",
    "fidelity_minus.at_10pct": "Fidelity- @10% sparsity",
    "fidelity_minus.at_20pct": "Fidelity- @20% sparsity",
    "fidelity_plus.at_20pct": "Fidelity+ @20% sparsity",
    "tgnn_fid_inv.at_10pct": "TGNN fid_inv @10% sparsity (signed)",
    "tgnn_fid_inv.at_20pct": "TGNN fid_inv @20% sparsity (signed)",
    "best_fid.value": "Best FID (TGNNExplainer style)",
    "fidelity_best.best": "Best Fidelity (drop sweep)",
    "sparsity.ratio": "Sparsity (|E_expl|/|E_candidates|)",
    "aufsc.value": "AUFSC",
    "tgnn_aufsc.value": "AUFSC (TGNNExplainer definition)",
    "temgx_aufsc.value": "AUFSC (TemGX definition)",
    "tempme_acc_auc.ratio_acc": "TempME Ratio ACC (paper)",
    "tempme_acc_auc": "TempME ACC-AUC",
    "acc_auc.auc": "ACC-AUC (prediction match)",
    "seed_stability.value": "Seed Stability (top-k Jaccard)",
    "perturbation_robustness.value": "Perturbation Robustness",
    "monotonicity.spearman_rho": "Monotonicity (importance-impact Spearman)",
    "elapsed_sec": "Runtime (s)",
}

DEFAULT_METRIC_DIRECTIONS: Dict[str, str] = {
    "fidelity_minus.value": "higher",
    "fidelity_plus.value": "higher",
    "fidelity_minus.at_10pct": "higher",
    "fidelity_minus.at_20pct": "higher",
    "fidelity_plus.at_20pct": "higher",
    "tgnn_fid_inv.at_10pct": "higher",
    "tgnn_fid_inv.at_20pct": "higher",
    "best_fid.value": "higher",
    "fidelity_best.best": "higher",
    "aufsc.value": "higher",
    "tgnn_aufsc.value": "higher",
    "temgx_aufsc.value": "higher",
    "tempme_acc_auc.ratio_acc": "higher",
    "tempme_acc_auc": "higher",
    "acc_auc.auc": "higher",
    "seed_stability.value": "higher",
    "perturbation_robustness.value": "higher",
    "monotonicity.spearman_rho": "higher",
    "cody_AUFSC_plus": "higher",
    "cody_AUFSC_minus": "higher",
    "cody_CHARR": "higher",
    "elapsed_sec": "lower",
}

DEFAULT_FIDELITY_CURVE_LABELS: Dict[str, str] = {
    "fidelity_minus": "Fidelity-",
    "fidelity_plus": "Fidelity+",
    "fidelity_drop": "Fidelity- (drop)",
    "fidelity_keep": "Fidelity+ (keep)",
    "fidelity_tempme": "Fidelity (TempME)",
    "temgx_fidelity_minus": "TemGX Fidelity-",
    "temgx_fidelity_plus": "TemGX Fidelity+",
    "temgx_fidelity_minus_logit": "TemGX Fidelity- (logit)",
    "temgx_fidelity_plus_logit": "TemGX Fidelity+ (logit)",
}


def _resolve_metric_spec(
    entry: Union[str, Tuple[str, str], List[str]],
    *,
    metric_labels: Dict[str, str],
) -> Tuple[str, str]:
    if isinstance(entry, (tuple, list)) and len(entry) == 2:
        label, metric_id = entry
        return str(label), str(metric_id)
    metric_id = str(entry)
    if metric_id.endswith(".lift_vs_random"):
        base = metric_id[: -len(".lift_vs_random")]
        base_label = metric_labels.get(base, base.replace("_", " ").title())
        return f"{base_label} (lift vs random)", metric_id
    label = metric_labels.get(metric_id, metric_id.replace("_", " ").title())
    return label, metric_id


def _resolve_metric_direction(
    metric_id: str,
    *,
    metric_directions: Mapping[str, str] | None = None,
) -> str | None:
    if metric_directions and metric_id in metric_directions:
        return str(metric_directions[metric_id]).strip().lower()
    if metric_id in DEFAULT_METRIC_DIRECTIONS:
        return str(DEFAULT_METRIC_DIRECTIONS[metric_id]).strip().lower()

    metric_id_l = metric_id.lower()
    if metric_id_l.endswith(".lift_vs_random"):
        return "higher"
    if any(tok in metric_id_l for tok in ("acc", "auc", "fidelity", "aufsc", "charr", "precision", "recall", "f1")):
        return "higher"
    if any(tok in metric_id_l for tok in ("elapsed", "runtime", "latency", "time", "loss", "error")):
        return "lower"
    return None


def filter_explainers(
    metrics_df: pd.DataFrame,
    *,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
) -> pd.DataFrame:
    if "explainer" not in metrics_df.columns:
        raise KeyError("metrics_df must contain an 'explainer' column.")
    out = metrics_df.copy()
    if include:
        include_set = {str(x) for x in include}
        out = out[out["explainer"].astype(str).isin(include_set)]
    if exclude:
        exclude_set = {str(x) for x in exclude}
        out = out[~out["explainer"].astype(str).isin(exclude_set)]
    return out


def prepare_metrics_plotting(
    metrics_df: pd.DataFrame,
    *,
    explainer_order: Sequence[str] | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    if "explainer" not in metrics_df.columns:
        raise KeyError("metrics_df must contain an 'explainer' column.")
    order = list(explainer_order or sorted(metrics_df["explainer"].astype(str).unique()))
    out = metrics_df.copy()
    out["explainer"] = pd.Categorical(out["explainer"].astype(str), categories=order, ordered=True)
    out = out.sort_values("explainer")
    palette = list(PLOT_COLORWAY)
    palette = (palette * (len(order) // len(palette) + 1))[: len(order)]
    return out, palette


def _aggregate_metrics(
    metrics_df: pd.DataFrame,
    *,
    group_col: str,
    metric_columns: Sequence[str],
    agg: str,
) -> pd.DataFrame:
    return metrics_df.groupby(group_col, dropna=False)[list(metric_columns)].agg(agg).reset_index()


def _plot_bar_chart(
    ax: Any,
    *,
    labels: Sequence[str],
    values: Sequence[float],
    colors: Sequence[str],
    title: str,
    ylabel: str,
) -> None:
    ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("explainer")
    ax.tick_params(axis="x", rotation=45)


def plot_explainer_metric_summary(
    metrics_df: pd.DataFrame,
    *,
    metric_columns: Sequence[Tuple[str, str]],
    agg: str = "mean",
    palette: Sequence[str] | None = None,
    group_col: str = "explainer",
    show: bool = True,
) -> pd.DataFrame:
    apply_matplotlib_style()
    metric_ids = [col for _, col in metric_columns]
    summary = _aggregate_metrics(metrics_df, group_col=group_col, metric_columns=metric_ids, agg=agg)

    colors = list(palette or PLOT_COLORWAY)
    if len(colors) < len(summary):
        colors = (colors * (len(summary) // len(colors) + 1))[: len(summary)]

    fig, axes = plt.subplots(1, max(1, len(metric_columns)), figsize=(4 * max(1, len(metric_columns)), 4))
    axes_list = axes if isinstance(axes, (list, np.ndarray)) else [axes]

    for ax, (label, col) in zip(axes_list, metric_columns):
        values = summary[col].to_numpy(dtype=float)
        labels = summary[group_col].astype(str).tolist()
        _plot_bar_chart(
            ax,
            labels=labels,
            values=values,
            colors=colors,
            title=label,
            ylabel=col,
        )

    fig.tight_layout()
    if show:
        plt.show()

    return summary


def plot_explainer_runtime(
    metrics_df: pd.DataFrame,
    *,
    palette: Sequence[str] | None = None,
    group_col: str = "explainer",
    agg: str = "mean",
) -> pd.DataFrame:
    return plot_explainer_metric_summary(
        metrics_df,
        metric_columns=[("Runtime (s)", "elapsed_sec")],
        agg=agg,
        palette=palette,
        group_col=group_col,
    )


def plot_prediction_match_rate(
    metrics_df: pd.DataFrame,
    *,
    palette: Sequence[str] | None = None,
    group_col: str = "explainer",
    agg: str = "mean",
    prefix: str = "prediction_profile.match_",
) -> pd.DataFrame:
    match_cols = [c for c in metrics_df.columns if c.startswith(prefix)]
    if not match_cols:
        raise KeyError("No prediction_profile.match_ columns found in metrics_df.")
    temp = metrics_df.copy()
    temp["match_rate"] = temp[match_cols].mean(axis=1)
    return plot_explainer_metric_summary(
        temp,
        metric_columns=[("Prediction match rate", "match_rate")],
        agg=agg,
        palette=palette,
        group_col=group_col,
    )


def plot_selected_metrics(
    metrics_df: pd.DataFrame,
    metrics: Sequence[Union[str, Tuple[str, str]]],
    *,
    palette: Sequence[str] | None = None,
    group_col: str = "explainer",
    metric_labels: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    labels = dict(DEFAULT_METRIC_LABELS)
    if metric_labels:
        labels.update(metric_labels)

    metric_columns: List[Tuple[str, str]] = []
    missing: List[str] = []
    for entry in metrics:
        label, metric_id = _resolve_metric_spec(entry, metric_labels=labels)
        if metric_id in metrics_df.columns:
            metric_columns.append((label, metric_id))
        else:
            missing.append(metric_id)

    results: Dict[str, Any] = {}
    if metric_columns:
        summary = plot_explainer_metric_summary(
            metrics_df,
            metric_columns=metric_columns,
            palette=palette,
            group_col=group_col,
        )
        results["summary"] = summary

    if missing:
        print("Skipped missing metrics:", ", ".join(missing))

    return results


def _resolve_plot_order(
    metrics_df: pd.DataFrame,
    *,
    group_col: str,
    explainer_order: Sequence[str] | None = None,
) -> List[str]:
    if explainer_order is not None:
        return [str(x) for x in explainer_order]
    series = metrics_df[group_col]
    if isinstance(series.dtype, pd.CategoricalDtype):
        return [str(x) for x in series.cat.categories.tolist()]
    return [str(x) for x in pd.unique(series.astype(str))]


def _as_scalar_score(value: Any) -> float:
    if hasattr(value, "detach"):
        try:
            value = value.detach()
        except Exception:
            pass
    if hasattr(value, "cpu"):
        try:
            value = value.cpu()
        except Exception:
            pass
    if hasattr(value, "numpy"):
        try:
            value = value.numpy()
        except Exception:
            pass
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return float("nan")
    if arr.size == 1:
        return float(arr[0])
    return float(np.max(arr))


def _is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _label_from_score(score: float, *, score_is_logit: bool, threshold: float) -> int:
    if not np.isfinite(score):
        return 0
    if score_is_logit:
        return int(float(score) >= float(threshold))
    return int(float(score) >= float(threshold))


def _first_finite_value(*values: Any) -> float | None:
    for value in values:
        if _is_finite_number(value):
            return float(value)
    return None

def _compute_cody_aufsc(
    sparsity: np.ndarray,
    fidelity: np.ndarray,
    *,
    max_sparsity: float = 1.0,
    n_grid: int = 101,
) -> float:
    if sparsity.size == 0:
        return 0.0
    s = np.asarray(sparsity, dtype=float)
    f = np.asarray(fidelity, dtype=float)
    max_s = float(max_sparsity)
    s = np.clip(s, 0.0, max_s)
    grid = np.linspace(0.0, max_s, int(n_grid))
    y = np.zeros_like(grid)
    order = np.argsort(s)
    s_sorted = s[order]
    f_sorted = f[order]
    prefix_sum = np.cumsum(f_sorted)
    idx = 0
    for i, t in enumerate(grid):
        while idx < s_sorted.size and s_sorted[idx] <= t:
            idx += 1
        y[i] = 0.0 if idx == 0 else float(prefix_sum[idx - 1]) / float(idx)
    area = float(np.trapz(y, grid))
    if max_s > 0:
        area /= max_s
    return area


def _as_int_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        out: List[int] = []
        for x in value:
            try:
                out.append(int(x))
            except Exception:
                continue
        return out
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = raw.strip("[]")
            if not parsed:
                return []
            out = []
            for part in parsed.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    out.append(int(part))
                except Exception:
                    continue
            return out
        return _as_int_list(parsed)
    try:
        return [int(value)]
    except Exception:
        return []


def _resolve_events_frame(*, dataset: Any, model: Any) -> Any | None:
    if isinstance(dataset, Mapping) and dataset.get("events") is not None:
        return dataset.get("events")
    if dataset is not None and getattr(dataset, "events", None) is not None:
        return getattr(dataset, "events")
    if getattr(model, "events", None) is not None:
        return getattr(model, "events")
    return None


def _resolve_all_edge_ids(*, dataset: Any, events_df: Any | None) -> np.ndarray:
    if isinstance(dataset, Mapping):
        if dataset.get("edge_ids") is not None:
            return np.asarray(dataset.get("edge_ids"), dtype=int).reshape(-1)
        if dataset.get("edge_idxs") is not None:
            return np.asarray(dataset.get("edge_idxs"), dtype=int).reshape(-1)
    if dataset is not None:
        if getattr(dataset, "edge_ids", None) is not None:
            return np.asarray(getattr(dataset, "edge_ids"), dtype=int).reshape(-1)
        if getattr(dataset, "edge_idxs", None) is not None:
            return np.asarray(getattr(dataset, "edge_idxs"), dtype=int).reshape(-1)

    if events_df is None:
        return np.empty((0,), dtype=int)

    columns = getattr(events_df, "columns", [])
    for key in ("idx", "e_idx", "event_idx", "event_id", "id"):
        if key not in columns:
            continue
        vals = pd.to_numeric(events_df[key], errors="coerce")
        vals = vals.dropna().astype(int).to_numpy()
        if vals.size:
            return np.asarray(vals, dtype=int)

    n_rows = int(len(events_df))
    if n_rows <= 0:
        return np.empty((0,), dtype=int)
    return np.arange(1, n_rows + 1, dtype=int)


def _prepare_event_lookup(events_df: Any | None) -> Dict[str, Any]:
    if events_df is None or len(events_df) == 0:
        return {}

    def _col(name: str, fallback_idx: int, cast: Any) -> np.ndarray:
        if name in events_df.columns:
            return events_df[name].to_numpy(dtype=cast)
        return events_df.iloc[:, fallback_idx].to_numpy(dtype=cast)

    try:
        src = _col("u", 0, int).reshape(-1)
        dst = _col("i", 1, int).reshape(-1)
        ts = _col("ts", 2, float).reshape(-1)
    except Exception:
        return {}

    idx_to_pos: Dict[int, int] = {}
    for key in ("idx", "e_idx", "event_idx", "event_id", "id"):
        if key not in events_df.columns:
            continue
        vals = pd.to_numeric(events_df[key], errors="coerce").dropna().astype(int).to_numpy()
        if vals.size != src.size:
            continue
        idx_to_pos = {int(e): int(i) for i, e in enumerate(vals.tolist())}
        if idx_to_pos:
            break

    return {"src": src, "dst": dst, "ts": ts, "idx_to_pos": idx_to_pos}


def _resolve_event_triplet(
    *,
    event_idx: int,
    extras: Mapping[str, Any],
    lookup: Mapping[str, Any],
) -> Tuple[int, int, float] | None:
    if all(k in extras for k in ("u", "i", "ts")):
        try:
            return int(extras["u"]), int(extras["i"]), float(extras["ts"])
        except Exception:
            pass

    src = np.asarray(lookup.get("src", []), dtype=int).reshape(-1)
    dst = np.asarray(lookup.get("dst", []), dtype=int).reshape(-1)
    ts = np.asarray(lookup.get("ts", []), dtype=float).reshape(-1)
    if src.size == 0 or dst.size == 0 or ts.size == 0:
        return None

    idx_to_pos = lookup.get("idx_to_pos", {}) or {}
    pos: int | None = None
    if int(event_idx) in idx_to_pos:
        pos = int(idx_to_pos[int(event_idx)])
    elif 1 <= int(event_idx) <= int(src.size):
        pos = int(event_idx) - 1
    elif 0 <= int(event_idx) < int(src.size):
        pos = int(event_idx)

    if pos is None or pos < 0 or pos >= int(src.size):
        return None
    return int(src[pos]), int(dst[pos]), float(ts[pos])


def _reset_model_state(model: Any) -> None:
    for obj in (model, getattr(model, "backbone", None)):
        if obj is None:
            continue
        for method_name in ("reset_model", "reset_state", "reset_memory"):
            method = getattr(obj, method_name, None)
            if not callable(method):
                continue
            try:
                method()
                return
            except Exception:
                continue


def _predict_event_with_preserve(
    *,
    model: Any,
    src: int,
    dst: int,
    ts: float,
    preserve_eidx: Sequence[int] | None,
    score_is_logit: bool,
) -> float:
    _reset_model_state(model)

    src_arr = np.asarray([int(src)], dtype=np.int64)
    dst_arr = np.asarray([int(dst)], dtype=np.int64)
    ts_arr = np.asarray([float(ts)], dtype=np.float32)

    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "get_prob"):
        kwargs: Dict[str, Any] = {"logit": bool(score_is_logit)}
        if preserve_eidx is not None:
            kwargs["edge_idx_preserve_list"] = [int(e) for e in preserve_eidx]
        out = backbone.get_prob(src_arr, dst_arr, ts_arr, **kwargs)
        return _as_scalar_score(out)

    from ...core.types import Subgraph

    payload: Dict[str, Any] = {"u": int(src), "i": int(dst), "ts": float(ts)}
    target = {"u": int(src), "i": int(dst), "ts": float(ts)}
    if preserve_eidx is None:
        subgraph = Subgraph(node_ids=[], edge_index=[], payload=payload)
        return _as_scalar_score(model.predict_proba(subgraph, target))

    keep = [int(e) for e in preserve_eidx]
    payload["candidate_eidx"] = keep
    subgraph = Subgraph(node_ids=[], edge_index=[], payload=payload)
    edge_mask = [1.0] * len(keep)
    return _as_scalar_score(model.predict_proba_with_mask(subgraph, target, edge_mask=edge_mask))


def _prediction_metric_values(metric_details: Mapping[str, Any]) -> Tuple[float | None, float | None, float | None]:
    prediction_full: float | None = None
    prediction_drop: float | None = None
    prediction_keep: float | None = None
    for metric_name in ("fidelity_plus", "fidelity_minus"):
        detail = metric_details.get(metric_name) or []
        if not detail or not isinstance(detail[0], Mapping):
            continue
        values = detail[0].get("values", {})
        if not isinstance(values, Mapping):
            continue
        prediction_full = _first_finite_value(prediction_full, values.get("prediction_full"))
        prediction_drop = _first_finite_value(prediction_drop, values.get("prediction_drop.@s=1"))
        prediction_keep = _first_finite_value(prediction_keep, values.get("prediction_keep.@s=1"))
    return prediction_full, prediction_drop, prediction_keep

def _infer_cody_score(
    *,
    model: Any,
    subgraph: Any,
    target_min: Mapping[str, Any],
    triplet: Tuple[int, int, float] | None,
    score_is_logit: bool,
    preserve_eidx: Sequence[int] | None = None,
    edge_mask: Sequence[float] | None = None,
) -> float:
    if triplet is not None:
        src, dst, ts = triplet
        return _predict_event_with_preserve(
            model=model,
            src=src,
            dst=dst,
            ts=ts,
            preserve_eidx=preserve_eidx,
            score_is_logit=bool(score_is_logit),
        )
    if edge_mask is None:
        return _as_scalar_score(model.predict_proba(subgraph, target_min))
    return _as_scalar_score(model.predict_proba_with_mask(subgraph, target_min, edge_mask=list(edge_mask)))


def _resolve_plot_export(
    *,
    save_dir: str | Path | None,
    export_ext: str,
) -> Tuple[Path | None, str]:
    save_root = Path(save_dir) if save_dir is not None else None
    if save_root is not None:
        save_root.mkdir(parents=True, exist_ok=True)
    return save_root, export_ext.lstrip(".").lower() or "pdf"

def _export_plotly_figure(
    *,
    fig: Any,
    save_root: Path | None,
    ext: str,
    stem: str,
    metric_id: str,
    saved_paths: Dict[str, str],
    warning_prefix: str,
) -> None:
    if save_root is None:
        return
    save_path = save_root / f"{stem}.{ext}"
    try:
        written = _maybe_save(fig, save_path)
        saved_paths[metric_id] = written
    except Exception as exc:
        warnings.warn(
            f"Failed to export {warning_prefix} '{metric_id}' to '{save_path}': {exc}",
            RuntimeWarning,
        )


def _resolve_selected_eidx(
    *,
    candidate_eidx: Sequence[int],
    importance_edges: Sequence[float] | None,
    extras: Mapping[str, Any],
) -> List[int]:
    candidate = [int(e) for e in candidate_eidx]
    for key in ("selected_eidx", "cf_event_ids", "coalition_eidx", "explanation_event_ids"):
        raw = extras.get(key)
        if raw is None:
            continue
        selected_raw = _as_int_list(raw)
        selected_set = {int(e) for e in selected_raw}
        selected = [int(e) for e in candidate if int(e) in selected_set]
        return selected

    imp = np.asarray(list(importance_edges or []), dtype=float).reshape(-1)
    n = min(len(candidate), int(imp.size))
    if n <= 0:
        return []
    selected = [int(candidate[i]) for i in range(n) if float(imp[i]) > 0.0]
    if selected:
        return selected
    best_i = int(np.nanargmax(imp[:n]))
    return [int(candidate[best_i])] if n > 0 else []


def _cody_delta_ratio(original_score: float, perturbed_score: float) -> float:
    orig = float(original_score)
    pert = float(perturbed_score)
    denom = max(abs(orig), 1e-12)
    if orig * pert < 0:
        delta = abs(orig) + abs(pert)
    else:
        delta = abs(orig) - abs(pert)
    return max(0.0, float(delta) / float(denom))


def compute_cody_style_report(
    *,
    results_jsonl: str | Path,
    model: Any,
    dataset: Any | None = None,
    score_is_logit: bool = True,
    decision_threshold: float = 0.0,
    max_sparsity: float = 1.0,
    n_grid: int = 101,
    allow_model_inference: bool = False,
    fidelity_mode: str = "binary",
    charr_mode: str = "binary",
) -> Dict[str, pd.DataFrame]:
    from ...core.types import Subgraph

    path = Path(results_jsonl)
    if not path.exists():
        raise FileNotFoundError(f"Results JSONL not found: {path}")

    events_df = _resolve_events_frame(dataset=dataset, model=model)
    event_lookup = _prepare_event_lookup(events_df)
    all_edge_ids = _resolve_all_edge_ids(dataset=dataset, events_df=events_df)
    if all_edge_ids.size:
        all_edge_ids = np.unique(np.asarray(all_edge_ids, dtype=int).reshape(-1))

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            result = rec.get("result") or {}
            context = rec.get("context") or {}
            target = context.get("target") or {}
            extras = result.get("extras") or {}
            metric_details = rec.get("metric_details") or {}

            explainer = str(result.get("explainer", "unknown"))
            context_fp = rec.get("context_fp")
            event_idx = (
                target.get("event_idx")
                or target.get("index")
                or target.get("idx")
                or extras.get("event_idx")
            )
            if event_idx is None:
                continue
            event_idx = int(event_idx)

            raw_candidate = extras.get("candidate_eidx")
            if raw_candidate is None:
                raw_candidate = list(range(len(result.get("importance_edges") or [])))
            try:
                candidate_eidx = _as_int_list(raw_candidate)
            except Exception:
                continue
            if not candidate_eidx:
                continue

            selected_eidx = _resolve_selected_eidx(
                candidate_eidx=candidate_eidx,
                importance_edges=result.get("importance_edges"),
                extras=extras,
            )
            selected_set = {int(e) for e in selected_eidx}
            sparsity_ratio = float(len(selected_eidx)) / float(max(1, len(candidate_eidx)))

            subgraph = Subgraph(node_ids=[], edge_index=[], payload={"event_idx": int(event_idx), "candidate_eidx": list(candidate_eidx)})
            target_min = {"event_idx": int(event_idx)}

            prediction_full_metric, prediction_drop_metric, prediction_keep_metric = _prediction_metric_values(metric_details)

            triplet = _resolve_event_triplet(event_idx=event_idx, extras=extras, lookup=event_lookup)

            original_score = _first_finite_value(extras.get("original_prediction"), prediction_full_metric)
            if original_score is None and bool(allow_model_inference):
                original_score = _infer_cody_score(
                    model=model,
                    subgraph=subgraph,
                    target_min=target_min,
                    triplet=triplet,
                    preserve_eidx=None,
                    score_is_logit=bool(score_is_logit),
                )
            if original_score is None:
                continue

            counterfactual_score = _first_finite_value(
                extras.get("counterfactual_prediction"),
                prediction_drop_metric,
            )
            if counterfactual_score is None and bool(allow_model_inference):
                preserve_list = None
                edge_mask = None
                if triplet is not None and all_edge_ids.size:
                    preserve_cf = (
                        np.setdiff1d(
                            all_edge_ids,
                            np.asarray(sorted(selected_set), dtype=int),
                            assume_unique=True,
                        )
                        if selected_set
                        else all_edge_ids
                    )
                    preserve_list = preserve_cf.tolist()
                else:
                    edge_mask = [0.0 if int(e) in selected_set else 1.0 for e in candidate_eidx]
                counterfactual_score = _infer_cody_score(
                    model=model,
                    subgraph=subgraph,
                    target_min=target_min,
                    triplet=triplet if preserve_list is not None else None,
                    preserve_eidx=preserve_list,
                    edge_mask=edge_mask,
                    score_is_logit=bool(score_is_logit),
                )
            if counterfactual_score is None:
                continue

            explanation_only_score = _first_finite_value(
                extras.get("prediction_explanation_events_only"),
                prediction_keep_metric,
            )
            if explanation_only_score is None and bool(allow_model_inference):
                keep_ids = None
                edge_mask = None
                if triplet is not None and all_edge_ids.size:
                    keep_ids = np.asarray(sorted(selected_set | {int(event_idx)}), dtype=int).tolist()
                else:
                    edge_mask = [1.0 if int(e) in selected_set else 0.0 for e in candidate_eidx]
                explanation_only_score = _infer_cody_score(
                    model=model,
                    subgraph=subgraph,
                    target_min=target_min,
                    triplet=triplet if keep_ids is not None else None,
                    preserve_eidx=keep_ids,
                    edge_mask=edge_mask,
                    score_is_logit=bool(score_is_logit),
                )
            if explanation_only_score is None:
                continue

            mode = str(fidelity_mode).strip().lower()
            if mode == "score":
                fidelity_plus_change = float(_cody_delta_ratio(original_score, counterfactual_score))
                fidelity_minus_same = float(
                    max(0.0, 1.0 - _cody_delta_ratio(original_score, explanation_only_score))
                )
            else:
                y_orig = _label_from_score(
                    original_score,
                    score_is_logit=bool(score_is_logit),
                    threshold=float(decision_threshold),
                )
                y_cf = _label_from_score(
                    counterfactual_score,
                    score_is_logit=bool(score_is_logit),
                    threshold=float(decision_threshold),
                )
                y_exp = _label_from_score(
                    explanation_only_score,
                    score_is_logit=bool(score_is_logit),
                    threshold=float(decision_threshold),
                )
                fidelity_plus_change = float(int(y_orig != y_cf))
                fidelity_minus_same = float(int(y_orig == y_exp))

            ch_mode = str(charr_mode).strip().lower()
            has_flag = ("achieves_counterfactual" in extras) or ("achieves_counterfactual_explanation" in extras)
            if ch_mode == "auto":
                ch_mode = "binary" if has_flag else ("score" if mode == "score" else "binary")

            if ch_mode == "score":
                charr_row = float(fidelity_plus_change)
            elif "achieves_counterfactual" in extras:
                charr_row = float(bool(extras.get("achieves_counterfactual")))
            elif "achieves_counterfactual_explanation" in extras:
                charr_row = float(bool(extras.get("achieves_counterfactual_explanation")))
            else:
                charr_row = float(fidelity_plus_change)

            rows.append(
                {
                    "context_fp": context_fp,
                    "explainer": explainer,
                    "cody_fidelity_plus_change": fidelity_plus_change,
                    "cody_fidelity_minus_same": fidelity_minus_same,
                    "cody_sparsity": sparsity_ratio,
                    "cody_charr_row": charr_row,
                }
            )

    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "explainer",
                "cody_AUFSC_plus",
                "cody_AUFSC_minus",
                "cody_CHARR",
                "cody_rows",
            ]
        )
        return {"detail": detail_df, "summary": summary_df}

    summary_rows: List[Dict[str, Any]] = []
    for explainer, group in detail_df.groupby("explainer", dropna=False):
        s = np.asarray(group["cody_sparsity"], dtype=float)
        fp = np.asarray(group["cody_fidelity_plus_change"], dtype=float)
        fm = np.asarray(group["cody_fidelity_minus_same"], dtype=float)
        ch = np.asarray(group["cody_charr_row"], dtype=float)
        summary_rows.append(
            {
                "explainer": str(explainer),
                "cody_AUFSC_plus": float(
                    _compute_cody_aufsc(s, fp, max_sparsity=float(max_sparsity), n_grid=int(n_grid))
                ),
                "cody_AUFSC_minus": float(
                    _compute_cody_aufsc(s, fm, max_sparsity=float(max_sparsity), n_grid=int(n_grid))
                ),
                "cody_CHARR": float(np.nanmean(ch)) if ch.size else float("nan"),
                "cody_rows": int(len(group)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    return {"detail": detail_df, "summary": summary_df}

def _generate_distinct_colors(
    count: int,
    *,
    start_index: int = 0,
    saturation: float = 0.70,
    value: float = 0.86,
) -> List[str]:
    if count <= 0:
        return []
    golden = 0.618033988749895
    out: List[str] = []
    for i in range(count):
        idx = start_index + i
        hue = (0.11 + golden * idx) % 1.0
        sat = saturation - 0.08 * float(idx % 3 == 1) + 0.05 * float(idx % 3 == 2)
        val = value - 0.06 * float(idx % 2 == 1)
        sat = float(np.clip(sat, 0.52, 0.88))
        val = float(np.clip(val, 0.64, 0.93))
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        out.append(f"#{int(round(r * 255)):02x}{int(round(g * 255)):02x}{int(round(b * 255)):02x}")
    return out


def _resolve_palette_map(order: Sequence[str], palette: Sequence[str] | None) -> Dict[str, str]:
    requested = [str(c) for c in list(palette or PLOT_COLORWAY) if c]
    if not requested:
        requested = list(PLOT_COLORWAY)

    unique: List[str] = []
    seen: set[str] = set()
    for color in requested:
        key = color.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(color)
        if len(unique) >= len(order):
            break

    if len(unique) < len(order):
        needed = len(order) - len(unique)
        extras = _generate_distinct_colors(needed, start_index=len(unique))
        for color in extras:
            key = color.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(color)
            if len(unique) >= len(order):
                break

    if len(unique) < len(order):
        unique.extend(_generate_distinct_colors(len(order) - len(unique), start_index=len(unique)))

    return {name: unique[idx] for idx, name in enumerate(order)}

def _collect_fidelity_sparsity_cols(metrics_df: pd.DataFrame) -> Dict[str, List[Tuple[float, str]]]:
    out: Dict[str, List[Tuple[float, str]]] = {}
    for col in metrics_df.columns:
        if ".@s=" not in col:
            continue
        metric_name, s_key = col.split(".", 1)
        if not s_key.startswith("@s="):
            continue
        try:
            s_val = float(s_key.split("=", 1)[1])
        except Exception:
            continue
        out.setdefault(metric_name, []).append((s_val, col))
    return {metric: sorted(cols, key=lambda item: item[0]) for metric, cols in out.items()}


def plot_metrics_one_per_figure(
    metrics_df: pd.DataFrame,
    metrics: Sequence[Union[str, Tuple[str, str]]],
    *,
    palette: Sequence[str] | None = None,
    group_col: str = "explainer",
    metric_labels: Mapping[str, str] | None = None,
    explainer_order: Sequence[str] | None = None,
    save_dir: str | Path | None = None,
    export_ext: str = "pdf",
    show: bool = True,
    width: int = 980,
    height: int = 560,
    sort_by_mean: bool = True,
    target_overlays: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    method_targets: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    metric_directions: Mapping[str, str] | None = None,
    show_direction_captions: bool = True,
) -> Dict[str, Any]:
    if group_col not in metrics_df.columns:
        raise KeyError(f"metrics_df must contain '{group_col}' column.")

    labels = dict(DEFAULT_METRIC_LABELS)
    if metric_labels:
        labels.update(metric_labels)

    order = _resolve_plot_order(metrics_df, group_col=group_col, explainer_order=explainer_order)
    color_map = _resolve_palette_map(order, palette)

    figures: Dict[str, Any] = {}
    summaries: Dict[str, pd.DataFrame] = {}
    saved_paths: Dict[str, str] = {}
    missing: List[str] = []

    save_root, ext = _resolve_plot_export(save_dir=save_dir, export_ext=export_ext)
    overlays = dict(target_overlays or {})
    method_target_specs = dict(method_targets or {})

    for entry in metrics:
        label, metric_id = _resolve_metric_spec(entry, metric_labels=labels)
        if metric_id not in metrics_df.columns:
            missing.append(metric_id)
            continue

        data = metrics_df[[group_col, metric_id]].copy()
        data[group_col] = data[group_col].astype(str)
        data[metric_id] = pd.to_numeric(data[metric_id], errors="coerce")
        data = data.dropna(subset=[metric_id])
        if data.empty:
            missing.append(metric_id)
            continue

        valid_explainers = set(data[group_col].tolist())
        display_order = [name for name in order if name in valid_explainers]
        if not display_order:
            display_order = sorted(valid_explainers)

        summary = (
            data.groupby(group_col, dropna=False, observed=False)[metric_id]
            .agg(["mean", "median", "std", "count"])
            .reindex(display_order)
            .reset_index()
            .rename(columns={group_col: "explainer"})
        )
        if sort_by_mean and not summary.empty:
            summary = summary.sort_values("mean", ascending=False).reset_index(drop=True)
            display_order = summary["explainer"].astype(str).tolist()
        summaries[metric_id] = summary

        fig = go.Figure()
        rng = np.random.default_rng(7)
        y_pos = {name: idx for idx, name in enumerate(display_order)}

        for explainer_name in display_order:
            values = data.loc[data[group_col] == explainer_name, metric_id].to_numpy(dtype=float)
            color = color_map.get(explainer_name, PLOT_COLORWAY[0])
            y_center = float(y_pos[explainer_name])
            jitter = rng.normal(0.0, 0.06, size=len(values))

            fig.add_trace(
                go.Scatter(
                    x=values,
                    y=(y_center + jitter),
                    mode="markers",
                    name=explainer_name,
                    showlegend=False,
                    marker=dict(size=5, color=color, opacity=0.35),
                    hovertemplate=(
                        f"{group_col}: {explainer_name}<br>"
                        f"{metric_id}: %{{x:.5g}}<extra></extra>"
                    ),
                )
            )

            n = int(np.isfinite(values).sum())
            mean_value = summary.loc[summary["explainer"] == explainer_name, "mean"].iloc[0]
            std_value = summary.loc[summary["explainer"] == explainer_name, "std"].iloc[0]
            ci95 = float(1.96 * std_value / np.sqrt(n)) if n > 1 and pd.notna(std_value) else 0.0
            if pd.notna(mean_value):
                fig.add_trace(
                    go.Scatter(
                        x=[float(mean_value)],
                        y=[y_center],
                        mode="markers",
                        marker=dict(
                            symbol="diamond-wide",
                            size=13,
                            color=color,
                            line=dict(width=1.6, color=COLORS["background"]),
                        ),
                        error_x=dict(
                            type="data",
                            array=[ci95],
                            visible=True,
                            thickness=1.3,
                            width=0,
                            color=_rgba(color, 0.9),
                        ),
                        showlegend=False,
                        hovertemplate=(
                            f"{group_col}: {explainer_name}<br>"
                            f"mean({metric_id}): {float(mean_value):.5g}<br>"
                            f"95% CI: ±{ci95:.4g}<extra></extra>"
                        ),
                    )
                )

        for spec in overlays.get(metric_id, []):
            if not isinstance(spec, Mapping):
                continue
            color = str(spec.get("color", "#E07A5F"))
            text = str(spec.get("label", "target"))
            dash = str(spec.get("dash", "dash"))
            if "value" in spec and _is_finite_number(spec.get("value")):
                fig.add_vline(
                    x=float(spec["value"]),
                    line_dash=dash,
                    line_width=float(spec.get("width", 2.0)),
                    line_color=color,
                    opacity=float(spec.get("opacity", 0.95)),
                    annotation_text=text,
                    annotation_position="top",
                    annotation_font=dict(color=color, size=11),
                )
            elif _is_finite_number(spec.get("min")) and _is_finite_number(spec.get("max")):
                lo = float(spec["min"])
                hi = float(spec["max"])
                if hi < lo:
                    lo, hi = hi, lo
                fig.add_vrect(
                    x0=lo,
                    x1=hi,
                    fillcolor=color,
                    opacity=float(spec.get("opacity", 0.12)),
                    line_width=0,
                    annotation_text=text,
                    annotation_position="top left",
                    annotation_font=dict(color=color, size=11),
                )

        method_specs = method_target_specs.get(metric_id, [])
        seen_goal_labels: set[str] = set()
        for spec in method_specs:
            if not isinstance(spec, Mapping):
                continue
            method = str(spec.get("explainer") or spec.get("method") or "").strip()
            if not method or method not in y_pos:
                continue
            if not _is_finite_number(spec.get("value")):
                continue

            value = float(spec["value"])
            color = str(spec.get("color", color_map.get(method, "#C44536")))
            symbol = str(spec.get("symbol", "x"))
            size = float(spec.get("size", 12))
            base_label = str(spec.get("label", f"{method} goal"))
            source = str(spec.get("source", "")).strip()
            legend_label = f"{base_label} ({source})" if source and source not in base_label else base_label
            show_legend = legend_label not in seen_goal_labels
            seen_goal_labels.add(legend_label)

            fig.add_trace(
                go.Scatter(
                    x=[value],
                    y=[float(y_pos[method])],
                    mode="markers",
                    name=legend_label,
                    showlegend=show_legend,
                    marker=dict(
                        symbol=symbol,
                        size=size,
                        color=color,
                        opacity=0.95,
                        line=dict(width=1.2, color=_rgba(COLORS["background"], 0.95)),
                    ),
                    hovertemplate=(
                        f"{group_col}: {method}<br>"
                        f"goal({metric_id}): %{{x:.5g}}<br>"
                        + (f"{source}<extra>{base_label}</extra>" if source else f"<extra>{base_label}</extra>")
                    ),
                )
            )

        x_vals = data[metric_id].to_numpy(dtype=float)
        finite_x = x_vals[np.isfinite(x_vals)]
        if finite_x.size:
            x_span = float(np.nanmax(finite_x) - np.nanmin(finite_x))
            pad = 0.10 * x_span if x_span > 0 else 0.1
            x_min = float(np.nanmin(finite_x) - pad)
            x_max = float(np.nanmax(finite_x) + pad)
        else:
            x_min, x_max = -1.0, 1.0

        if overlays.get(metric_id):
            for spec in overlays.get(metric_id, []):
                if "value" in spec and _is_finite_number(spec.get("value")):
                    v = float(spec["value"])
                    x_min = min(x_min, v)
                    x_max = max(x_max, v)
                if _is_finite_number(spec.get("min")) and _is_finite_number(spec.get("max")):
                    x_min = min(x_min, float(spec["min"]))
                    x_max = max(x_max, float(spec["max"]))
            x_pad = 0.08 * (x_max - x_min if x_max > x_min else 1.0)
            x_min -= x_pad
            x_max += x_pad

        if method_specs:
            for spec in method_specs:
                if not _is_finite_number(spec.get("value")):
                    continue
                v = float(spec["value"])
                x_min = min(x_min, v)
                x_max = max(x_max, v)
            x_pad = 0.08 * (x_max - x_min if x_max > x_min else 1.0)
            x_min -= x_pad
            x_max += x_pad

        if x_min < 0 < x_max:
            fig.add_vline(x=0.0, line_dash="dot", line_width=1.2, line_color=_rgba(COLORS["text"], 0.40))

        direction = _resolve_metric_direction(metric_id, metric_directions=metric_directions)
        if show_direction_captions and direction == "higher":
            direction_text = "Higher is better"
            direction_color = "#2A9D8F"
            fig.add_annotation(
                x=1.0,
                y=1.10,
                xref="paper",
                yref="paper",
                text=direction_text,
                showarrow=False,
                xanchor="right",
                yanchor="bottom",
                font=dict(size=11, color=direction_color),
                bgcolor=_rgba("#FFFFFF", 0.85),
                bordercolor=_rgba(direction_color, 0.25),
                borderwidth=1,
                borderpad=4,
            )

        fig.update_layout(
            template=_PLOTLY_TEMPLATE,
            width=width,
            height=height,
            title=dict(text=f"{label}", x=0.01, xanchor="left"),
            margin=dict(l=180, r=45, t=85, b=70),
            xaxis=dict(
                title=label,
                range=[x_min, x_max],
                showgrid=True,
                gridcolor=_rgba(COLORS["text"], 0.12),
                zeroline=False,
            ),
            yaxis=dict(
                title="Explainer",
                tickmode="array",
                tickvals=[y_pos[name] for name in display_order],
                ticktext=display_order,
                range=[-0.8, max(0.0, len(display_order) - 0.2)],
                showgrid=False,
            ),
            plot_bgcolor="white",
        )
        fig.update_xaxes(showline=True, linewidth=1.0, linecolor=_rgba(COLORS["text"], 0.22))
        fig.update_yaxes(showline=True, linewidth=1.0, linecolor=_rgba(COLORS["text"], 0.22))

        figures[metric_id] = fig
        _auto_show(fig, show=show)

        _export_plotly_figure(
            fig=fig,
            save_root=save_root,
            ext=ext,
            stem=_slugify(metric_id),
            metric_id=metric_id,
            saved_paths=saved_paths,
            warning_prefix="metric",
        )

    if missing:
        print("Skipped missing metrics:", ", ".join(missing))

    return {
        "figures": figures,
        "summaries": summaries,
        "saved_paths": saved_paths,
        "missing_metrics": missing,
    }

def plot_fidelity_sparsity_curves(
    metrics_df: pd.DataFrame,
    *,
    palette: Sequence[str] | None = None,
    group_col: str = "explainer",
    metric_labels: Mapping[str, str] | None = None,
    explainer_order: Sequence[str] | None = None,
    agg: str = "mean",
    save_dir: str | Path | None = None,
    export_ext: str = "pdf",
    show: bool = True,
    width: int = 960,
    height: int = 560,
) -> Dict[str, Any]:
    if group_col not in metrics_df.columns:
        raise KeyError(f"metrics_df must contain '{group_col}' column.")

    labels = dict(DEFAULT_FIDELITY_CURVE_LABELS)
    if metric_labels:
        labels.update(metric_labels)

    fidelity_cols = _collect_fidelity_sparsity_cols(metrics_df)
    if not fidelity_cols:
        return {
            "figures": {},
            "saved_paths": {},
            "metric_columns": {},
            "message": "No metric.@s=<level> columns found.",
        }

    order = _resolve_plot_order(metrics_df, group_col=group_col, explainer_order=explainer_order)
    color_map = _resolve_palette_map(order, palette)

    save_root, ext = _resolve_plot_export(save_dir=save_dir, export_ext=export_ext)

    figures: Dict[str, Any] = {}
    saved_paths: Dict[str, str] = {}

    for metric_name, cols in fidelity_cols.items():
        xs = [float(s) for s, _ in cols]
        col_ids = [col for _, col in cols]

        agg_df = (
            metrics_df[[group_col, *col_ids]]
            .copy()
            .assign(**{group_col: metrics_df[group_col].astype(str)})
            .groupby(group_col, dropna=False, observed=False)[col_ids]
            .agg(agg)
        )
        valid_explainers = set(agg_df.index.astype(str).tolist())
        display_order = [name for name in order if name in valid_explainers]
        if not display_order:
            display_order = sorted(valid_explainers)

        fig = go.Figure()
        for explainer_name in display_order:
            row = agg_df.loc[explainer_name]
            ys = [float(row[col]) if pd.notna(row[col]) else float("nan") for col in col_ids]
            color = color_map.get(explainer_name, PLOT_COLORWAY[0])
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name=str(explainer_name),
                    line=dict(width=3.0, color=color, shape="spline", smoothing=0.45),
                    marker=dict(size=8.5, color=color, line=dict(width=1.0, color=COLORS["background"])),
                    hovertemplate=(
                        f"{group_col}: {explainer_name}<br>"
                        "sparsity: %{x:.3f}<br>"
                        "value: %{y:.5g}<extra></extra>"
                    ),
                )
            )

        if xs:
            x_min = max(0.0, min(xs))
            x_max = max(xs)
            fig.update_xaxes(range=[x_min, x_max], dtick=0.05 if x_max - x_min >= 0.1 else None)

        y_vals = agg_df.to_numpy(dtype=float)
        if np.isfinite(y_vals).any():
            y_min = float(np.nanmin(y_vals))
            y_max = float(np.nanmax(y_vals))
            if y_min < 0 < y_max:
                fig.add_hline(y=0.0, line_dash="dot", line_width=1.2, line_color=_rgba(COLORS["text"], 0.35))

        label = labels.get(metric_name, metric_name.replace("_", " ").title())
        fig.update_layout(
            template=_PLOTLY_TEMPLATE,
            width=width,
            height=height,
            title=dict(text=f"{label} Over Sparsity", x=0.01, xanchor="left"),
            margin=dict(l=68, r=28, t=74, b=74),
            xaxis=dict(title="Sparsity Level"),
            yaxis=dict(title=label),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1.0,
                bgcolor=_rgba(COLORS["background"], 0.85),
            ),
        )
        fig.update_xaxes(showline=True, linewidth=1.0, linecolor=_rgba(COLORS["text"], 0.22))
        fig.update_yaxes(showline=True, linewidth=1.0, linecolor=_rgba(COLORS["text"], 0.22))

        figures[metric_name] = fig
        _auto_show(fig, show=show)

        _export_plotly_figure(
            fig=fig,
            save_root=save_root,
            ext=ext,
            stem=f"{_slugify(metric_name)}_over_sparsity",
            metric_id=metric_name,
            saved_paths=saved_paths,
            warning_prefix="fidelity curve",
        )

    return {
        "figures": figures,
        "saved_paths": saved_paths,
        "metric_columns": fidelity_cols,
    }


__all__ = [
    "DEFAULT_METRIC_LABELS",
    "DEFAULT_FIDELITY_CURVE_LABELS",
    "filter_explainers",
    "prepare_metrics_plotting",
    "plot_explainer_metric_summary",
    "plot_explainer_runtime",
    "plot_prediction_match_rate",
    "plot_selected_metrics",
    "plot_metrics_one_per_figure",
    "plot_fidelity_sparsity_curves",
    "compute_cody_style_report",
]
