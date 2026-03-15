from __future__ import annotations

"""Compact fidelity-style metrics.

This module keeps the same metric names as the old implementation, but with a
small and explicit code path focused on readability.

Registered names:
- fidelity_drop (alias: fidelity_minus)
- fidelity_keep
- fidelity_keep_graph
- fidelity_best
- best_fid
- fidelity_tempme
- graph_sparsity
- acc_auc
- tempme_acc_auc
- prediction_profile
- monotonicity
- seed_stability
- perturbation_robustness
- temgx_fidelity_minus
- temgx_fidelity_plus
- temgx_fidelity_minus_logit
- temgx_fidelity_plus_logit
- temgx_sparsity
- singular_value
"""

from collections.abc import Mapping, Sequence
import hashlib
import importlib.util
import sys
from typing import Any, Callable

import numpy as np

from ..core.constants import ASSET_ROOT
from ..core.registry import METRICS, register_metric
from ..core.types import ExplanationContext, ExplanationResult, MetricResult, Subgraph
from .base import BaseMetric, MetricDirection
from .builtin import aufsc, fidelity_minus, fidelity_plus, sparsity
from .selection import (
    extract_explicit_selected,
    importance_for_candidates,
    rank_candidate_positions,
)


def _payload(context: ExplanationContext) -> dict[str, Any]:
    payload = context.subgraph.payload if context.subgraph else None
    return payload if isinstance(payload, dict) else {}


def _to_array_1d(values: Any, *, dtype: Any = float) -> np.ndarray:
    if values is None:
        return np.empty((0,), dtype=dtype)
    # Torch tensors may require grad; convert safely without triggering
    # "Can't call numpy() on Tensor that requires grad" errors.
    if hasattr(values, "detach") and callable(getattr(values, "detach")):
        try:
            values = values.detach().cpu().numpy()
        except Exception:
            pass
    arr = np.asarray(values, dtype=dtype)
    return arr.reshape(-1)


def _score(prediction: Any) -> float:
    arr = _to_array_1d(prediction, dtype=float)
    if arr.size == 0:
        return float("nan")
    if arr.size == 1:
        return float(arr[0])
    return float(np.max(arr))


def _label(prediction: Any, *, threshold: float = 0.5) -> int:
    arr = _to_array_1d(prediction, dtype=float)
    if arr.size <= 1:
        return int(float(arr[0]) >= float(threshold)) if arr.size == 1 else 0
    return int(np.argmax(arr))


def _to_probability(score: float, *, score_is_logit: bool) -> float:
    if not np.isfinite(score):
        return float("nan")
    if score_is_logit:
        clipped = float(np.clip(score, -60.0, 60.0))
        return float(1.0 / (1.0 + np.exp(-clipped)))
    return float(score)


def _label_from_score(score: float, *, score_is_logit: bool, threshold: float) -> int:
    prob = _to_probability(score, score_is_logit=score_is_logit)
    return int(float(prob) > float(threshold))


def _candidate_eidx(context: ExplanationContext, result: ExplanationResult) -> list[int]:
    payload = _payload(context)
    if "candidate_eidx" in payload:
        return [int(eidx) for eidx in payload["candidate_eidx"]]
    if "candidate_eidx" in result.extras:
        return [int(eidx) for eidx in result.extras["candidate_eidx"]]
    n = len(result.importance_edges or [])
    return list(range(n))


def _importance(result: ExplanationResult, n: int) -> np.ndarray:
    return importance_for_candidates(result.importance_edges, n)


def _ranked_order(
    metric: BaseMetric,
    context: ExplanationContext,
    result: ExplanationResult,
    candidate: Sequence[int],
    imp: np.ndarray,
) -> list[int]:
    order, _ = rank_candidate_positions(
        model=metric.model if hasattr(metric, "model") else None,
        context=context,
        result=result,
        candidate_eidx=candidate,
        importance=imp,
        config=metric.config,
    )
    return order


def _curve_order(
    metric: BaseMetric,
    context: ExplanationContext,
    result: ExplanationResult,
    candidate: Sequence[int],
    imp: np.ndarray,
    *,
    default: str = "strict",
) -> list[int]:
    cfg = metric.config if isinstance(metric.config, Mapping) else {}
    ranking_cfg = cfg.get("ranking", {}) if isinstance(cfg, Mapping) else {}

    strategy = cfg.get("order_strategy")
    if strategy is None and isinstance(ranking_cfg, Mapping):
        strategy = ranking_cfg.get("strategy")
    if strategy is None:
        strategy = default
    strategy_s = str(strategy).strip().lower()

    if strategy_s in {"strict", "importance", "explainer", "scores"}:
        return _ranked_order_strict_importance(metric, result, candidate, imp)
    if strategy_s in {"model", "impact"}:
        return _ranked_order(metric, context, result, candidate, imp)
    if strategy_s == "auto":
        fallback = str(ranking_cfg.get("uninformative_fallback", "")).strip().lower()
        if fallback in {"", "none", "off", "false"}:
            return _ranked_order_strict_importance(metric, result, candidate, imp)
        return _ranked_order(metric, context, result, candidate, imp)

    # Unknown strategy -> keep robust paper-style ranking.
    return _ranked_order_strict_importance(metric, result, candidate, imp)


def _ranked_order_strict_importance(
    metric: BaseMetric,
    result: ExplanationResult,
    candidate: Sequence[int],
    imp: np.ndarray,
) -> list[int]:
    """Explainer-only ranking (no model-impact fallback), closer to paper setups."""
    n = len(candidate)
    if n <= 0:
        return []

    scores = np.asarray(imp, dtype=float).reshape(-1)
    if scores.size >= n:
        scores = scores[:n]
    else:
        padded = np.zeros((n,), dtype=float)
        padded[: scores.size] = scores
        scores = padded
    scores = np.nan_to_num(scores, nan=-np.inf)

    ranking_cfg = metric.config.get("ranking", {}) if isinstance(metric.config, Mapping) else {}
    prefer_selected = bool(ranking_cfg.get("prefer_selected", True))
    tie_break = str(ranking_cfg.get("tie_break", "candidate_order"))

    _, selected_set = extract_explicit_selected(result, candidate)
    if prefer_selected and selected_set:
        selected_bonus = np.asarray([1 if int(e) in selected_set else 0 for e in candidate], dtype=np.int64)
    else:
        selected_bonus = np.zeros((n,), dtype=np.int64)

    if tie_break == "edge_id":
        tie = np.asarray([int(e) for e in candidate], dtype=np.int64)
    elif tie_break == "reverse_candidate_order":
        tie = -np.arange(n, dtype=np.int64)
    else:
        tie = np.arange(n, dtype=np.int64)

    order = np.lexsort((tie, -scores, -selected_bonus))
    return [int(i) for i in order.tolist()]


def _ranked_order_tempme_official(candidate: Sequence[int], imp: np.ndarray) -> list[int]:
    """TempME threshold_test ordering: sort by explainer score only, stable ties."""
    n = len(candidate)
    if n <= 0:
        return []

    scores = np.asarray(imp, dtype=float).reshape(-1)
    if scores.size >= n:
        scores = scores[:n]
    else:
        padded = np.full((n,), -np.inf, dtype=float)
        if scores.size > 0:
            padded[: scores.size] = scores
        scores = padded
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)

    order = np.argsort(-scores, kind="mergesort")
    return [int(i) for i in order.tolist()]


def _resolve_levels(config: Mapping[str, Any], default: Sequence[float]) -> list[float]:
    levels = config.get("sparsity_levels", config.get("levels"))
    if levels is None:
        return [float(x) for x in default]
    if isinstance(levels, (float, int)):
        return [float(levels)]
    parsed = [float(x) for x in levels]
    return parsed if parsed else [float(x) for x in default]


def _resolve_k_max(config: Mapping[str, Any], n_candidates: int) -> int:
    if config.get("k_max") is None:
        return int(n_candidates)
    return int(config["k_max"])


def _infer_tempme_base_type(metric: BaseMetric) -> str:
    config = metric.config if isinstance(metric.config, Mapping) else {}
    for key in ("base_type", "model_type", "backbone_type"):
        raw = config.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip().lower()

    model = getattr(metric, "model", None)
    names: list[str] = []
    backbone = getattr(model, "backbone", None) if model is not None else None
    for candidate in (
        getattr(backbone, "base_type", None),
        getattr(backbone, "__class__", type("", (), {})).__name__ if backbone is not None else None,
        getattr(model, "__class__", type("", (), {})).__name__ if model is not None else None,
    ):
        if isinstance(candidate, str) and candidate:
            names.append(candidate.lower())

    joined = " ".join(names)
    if "graphmixer" in joined:
        return "graphmixer"
    if "tgat" in joined or "tgan" in joined:
        return "tgat"
    return "tgn"


def _resolve_tempme_num_edge(metric: BaseMetric, *, n_candidates: int) -> int:
    if metric.config.get("k_max") is not None:
        return _resolve_k_max(metric.config, n_candidates)
    if not bool(metric.config.get("use_official_num_edge", True)):
        return int(n_candidates)

    n_degree = metric.config.get("n_degree")
    if n_degree is None:
        n_degree = getattr(metric.model, "num_neighbors", None)
    if n_degree is None:
        backbone = getattr(metric.model, "backbone", None)
        n_degree = getattr(backbone, "num_neighbors", None)
    try:
        n_degree_i = max(1, int(n_degree))
    except Exception:
        n_degree_i = max(1, int(n_candidates))

    base_type = _infer_tempme_base_type(metric)
    if base_type in {"tgn", "tgat"}:
        num_edge = n_degree_i + n_degree_i * n_degree_i
    elif base_type == "graphmixer":
        num_edge = n_degree_i
    else:
        num_edge = int(n_candidates)
    return max(1, int(num_edge))


def _resolve_ks(config: Mapping[str, Any], n_candidates: int) -> list[int]:
    if config.get("k") is not None:
        raw = config["k"]
    elif config.get("topk") is not None:
        raw = config["topk"]
    elif config.get("sparsity") is not None:
        raw = int(round(float(config["sparsity"]) * float(n_candidates)))
    else:
        raw = [n_candidates]

    if isinstance(raw, int):
        items = [raw]
    else:
        items = [int(v) for v in raw]

    clamped = [max(0, min(int(k), int(n_candidates))) for k in items]
    unique = sorted(set(clamped))
    return unique if unique else [0]


def _k_from_level(
    level: float,
    *,
    n_candidates: int,
    k_max: int,
    ensure_min_one: bool,
) -> int:
    k = int(round(float(level) * float(k_max)))
    if ensure_min_one and level > 0.0 and k == 0 and n_candidates > 0:
        k = 1
    return max(0, min(k, int(n_candidates)))


def _edge_mask(candidate_eidx: Sequence[int], selected_eidx: Sequence[int], *, mode: str) -> list[float]:
    selected = {int(eidx) for eidx in selected_eidx}
    if mode == "keep":
        return [1.0 if int(eidx) in selected else 0.0 for eidx in candidate_eidx]
    return [0.0 if int(eidx) in selected else 1.0 for eidx in candidate_eidx]


def _int_list(values: Any) -> list[int]:
    if values is None:
        return []
    if isinstance(values, (list, tuple, np.ndarray)):
        out: list[int] = []
        for v in values:
            try:
                out.append(int(v))
            except Exception:
                continue
        return out
    try:
        return [int(values)]
    except Exception:
        return []


def _base_eidx_from_context(
    context: ExplanationContext,
    result: ExplanationResult,
    candidate_eidx: Sequence[int],
) -> list[int]:
    payload = _payload(context)
    candidate_set = {int(e) for e in candidate_eidx}

    if "base_eidx" in payload:
        base = _int_list(payload.get("base_eidx"))
    elif "support_eidx" in payload:
        support = _int_list(payload.get("support_eidx"))
        base = [int(e) for e in support if int(e) not in candidate_set]
    elif isinstance(result.extras, Mapping) and "base_eidx" in result.extras:
        base = _int_list(result.extras.get("base_eidx"))
    elif isinstance(result.extras, Mapping) and "support_eidx" in result.extras:
        support = _int_list(result.extras.get("support_eidx"))
        base = [int(e) for e in support if int(e) not in candidate_set]
    else:
        base = []

    # Preserve original ordering while removing duplicates.
    seen: set[int] = set()
    ordered: list[int] = []
    for e in base:
        if e in seen:
            continue
        seen.add(int(e))
        ordered.append(int(e))
    return ordered


def _preserve_from_selected(
    context: ExplanationContext,
    result: ExplanationResult,
    candidate_eidx: Sequence[int],
    selected_eidx: Sequence[int],
    *,
    mode: str = "keep",
) -> list[int]:
    selected = {int(eidx) for eidx in selected_eidx}
    base = _base_eidx_from_context(context, result, candidate_eidx)
    if mode == "drop":
        dynamic = [int(eidx) for eidx in candidate_eidx if int(eidx) not in selected]
    else:
        dynamic = [int(eidx) for eidx in candidate_eidx if int(eidx) in selected]
    preserve = [*base, *dynamic]
    seen: set[int] = set()
    out: list[int] = []
    for e in preserve:
        if int(e) in seen:
            continue
        seen.add(int(e))
        out.append(int(e))
    return out


def _predict_masked(
    metric: BaseMetric,
    context: ExplanationContext,
    candidate_eidx: Sequence[int],
    selected_eidx: Sequence[int],
    *,
    mode: str,
) -> float:
    edge_mask = _edge_mask(candidate_eidx, selected_eidx, mode=mode)
    prediction = metric.model.predict_proba_with_mask(
        context.subgraph,
        context.target,
        edge_mask=edge_mask,
    )
    return _score(prediction)


def _resolve_event_triplet(model: Any, context: ExplanationContext) -> tuple[int, int, float] | None:
    try:
        if hasattr(model, "_resolve_event"):
            event = model._resolve_event(context.target, subgraph=context.subgraph)  # type: ignore[attr-defined]
            return int(event.src), int(event.dst), float(event.ts)
    except Exception:
        pass

    payload = _payload(context)
    if {"u", "i", "ts"}.issubset(payload.keys()):
        try:
            return int(payload["u"]), int(payload["i"]), float(payload["ts"])
        except Exception:
            pass

    target = context.target if isinstance(context.target, dict) else {}
    if {"u", "i", "ts"}.issubset(target.keys()):
        try:
            return int(target["u"]), int(target["i"]), float(target["ts"])
        except Exception:
            pass

    events = getattr(model, "events", None)
    eidx = (
        target.get("event_idx")
        or target.get("index")
        or target.get("idx")
        or payload.get("event_idx")
        or payload.get("index")
        or payload.get("idx")
    )
    if events is None or eidx is None:
        return None

    try:
        row = events.iloc[int(eidx) - 1]
        src = int(row["u"] if "u" in row else row.iloc[0])
        dst = int(row["i"] if "i" in row else row.iloc[1])
        ts = float(row["ts"] if "ts" in row else row.iloc[2])
        return src, dst, ts
    except Exception:
        return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text and text.lstrip("-").isdigit():
            return int(text)
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        simple = text.replace("-", "", 1).replace(".", "", 1)
        if text and simple.isdigit():
            return float(text)
    return None


def _resolve_event_idx(model: Any, context: ExplanationContext) -> int | None:
    del model

    payload = _payload(context)
    target = context.target if isinstance(context.target, dict) else {}
    for source in (target, payload):
        for key in ("event_idx", "index", "idx"):
            if key not in source:
                continue
            value = _coerce_int(source.get(key))
            if value is not None:
                return value
    return None


def _full_prior_event_ids(model: Any, context: ExplanationContext) -> list[int]:
    events = getattr(model, "events", None)
    event_idx = _resolve_event_idx(model, context)
    if events is None or event_idx is None or int(event_idx) <= 0:
        return []

    if hasattr(events, "columns") and "e_idx" in events.columns:
        raw_ids = list(events["e_idx"].tolist())
    elif hasattr(events, "columns") and "idx" in events.columns:
        raw_ids = list(events["idx"].tolist())
    else:
        raw_ids = list(range(1, len(events) + 1))

    event_ids = [value for value in (_coerce_int(v) for v in raw_ids) if value is not None]
    full_prior = sorted({int(v) for v in event_ids if int(v) < int(event_idx)})
    return [int(v) for v in full_prior]


def _build_full_prior_subgraph(model: Any, context: ExplanationContext) -> Subgraph | None:
    event_idx = _resolve_event_idx(model, context)
    triplet = _resolve_event_triplet(model, context)
    if event_idx is None or triplet is None:
        return None

    src, dst, ts = triplet
    payload = {
        "event_idx": int(event_idx),
        "u": int(src),
        "i": int(dst),
        "ts": float(ts),
        "candidate_eidx": _full_prior_event_ids(model, context),
        # Empty base means the mask controls the entire prior graph.
        "base_eidx": [],
    }
    return Subgraph(node_ids=[], edge_index=[], payload=payload)


def _support_eps_from_config(config: Mapping[str, Any] | None) -> float:
    if isinstance(config, Mapping):
        nested = config.get("ranking")
        if isinstance(nested, Mapping) and "support_eps" in nested:
            value = _coerce_float(nested.get("support_eps"))
            if value is not None:
                return value
        if "support_eps" in config:
            value = _coerce_float(config.get("support_eps"))
            if value is not None:
                return value
    return 1e-12


def _ordered_explanation_support(
    metric: BaseMetric,
    context: ExplanationContext,
    result: ExplanationResult,
    candidate: Sequence[int],
    imp: np.ndarray,
) -> list[int]:
    if not candidate:
        return []

    order = _ranked_order(metric, context, result, candidate, imp)
    _, selected_set = extract_explicit_selected(result, candidate)
    if selected_set:
        support = [int(candidate[i]) for i in order if int(candidate[i]) in selected_set]
    else:
        support_eps = _support_eps_from_config(metric.config)
        support = [int(candidate[i]) for i in order if abs(float(imp[i])) > support_eps]

    seen: set[int] = set()
    ordered: list[int] = []
    for e in support:
        if int(e) in seen:
            continue
        seen.add(int(e))
        ordered.append(int(e))
    return ordered


def _sample_negative_dst(model: Any, context: ExplanationContext, *, positive_dst: int) -> int | None:
    payload = _payload(context)
    target = context.target if isinstance(context.target, dict) else {}
    for key in ("negative_dst", "neg_dst", "dst_neg", "negative_node"):
        for source in (target, payload):
            if key not in source:
                continue
            try:
                dst = int(source[key])
            except Exception:
                continue
            if dst != int(positive_dst):
                return dst

    events = getattr(model, "events", None)
    if events is None:
        return None

    try:
        if "i" in events.columns:
            pool = np.asarray(events["i"].to_numpy(dtype=int), dtype=int)
        else:
            pool = np.asarray(events.iloc[:, 1].to_numpy(dtype=int), dtype=int)
    except Exception:
        return None

    pool = np.unique(pool)
    pool = pool[pool != int(positive_dst)]
    if pool.size == 0:
        return None

    salt = f"{context.fingerprint()}|{int(positive_dst)}"
    seed = int(hashlib.sha1(salt.encode("utf-8")).hexdigest()[:8], 16)
    return int(pool[seed % int(pool.size)])


def _sample_negative_dsts(
    model: Any,
    context: ExplanationContext,
    *,
    positive_dst: int,
    n_samples: int,
) -> list[int]:
    """Deterministically sample multiple negative destinations for one anchor."""
    n = max(0, int(n_samples))
    if n <= 0:
        return []

    out: list[int] = []
    seen: set[int] = set()

    first = _sample_negative_dst(model, context, positive_dst=positive_dst)
    if first is not None and int(first) != int(positive_dst):
        out.append(int(first))
        seen.add(int(first))
        if len(out) >= n:
            return out

    events = getattr(model, "events", None)
    if events is None:
        return out
    try:
        if "i" in events.columns:
            pool = np.asarray(events["i"].to_numpy(dtype=int), dtype=int)
        else:
            pool = np.asarray(events.iloc[:, 1].to_numpy(dtype=int), dtype=int)
    except Exception:
        return out

    pool = np.unique(pool)
    pool = pool[pool != int(positive_dst)]
    if pool.size == 0:
        return out

    fp = context.fingerprint()
    for idx in range(max(int(pool.size), n * 2)):
        salt = f"{fp}|{int(positive_dst)}|neg#{idx}"
        seed = int(hashlib.sha1(salt.encode("utf-8")).hexdigest()[:8], 16)
        dst = int(pool[seed % int(pool.size)])
        if dst in seen:
            continue
        out.append(dst)
        seen.add(dst)
        if len(out) >= n:
            break
    return out


def _predict_raw_pair_score(
    model: Any,
    *,
    src: int,
    dst: int,
    ts: float,
    preserve_eidx: Sequence[int] | None,
    result_as_logit: bool,
) -> float | None:
    backbone = getattr(model, "backbone", None)
    if backbone is None or not hasattr(backbone, "get_prob"):
        return None

    src_arr = np.asarray([int(src)], dtype=np.int64)
    dst_arr = np.asarray([int(dst)], dtype=np.int64)
    ts_arr = np.asarray([float(ts)], dtype=np.float32)

    kwargs: dict[str, Any] = {"logit": bool(result_as_logit)}
    if preserve_eidx is not None:
        kwargs["edge_idx_preserve_list"] = [int(e) for e in preserve_eidx]

    try:
        import torch
        with torch.no_grad():
            pred = backbone.get_prob(src_arr, dst_arr, ts_arr, **kwargs)
    except Exception:
        return None

    try:
        return float(_score(pred))
    except Exception:
        return None


def _series_keys(*, levels: list[float] | None, ks: list[int] | None) -> list[str]:
    if levels is not None:
        return [f"@s={float(level):g}" for level in levels]
    assert ks is not None
    return [f"@{int(k)}" for k in ks]


_TEMGX_METRICS_MODULE: Any | None = None


def _load_temgx_metrics_module() -> Any | None:
    global _TEMGX_METRICS_MODULE
    if _TEMGX_METRICS_MODULE is not None:
        return _TEMGX_METRICS_MODULE

    module_path = (
        ASSET_ROOT
        / "submodules"
        / "explainer"
        / "TemGX"
        / "link"
        / "temgxlib"
        / "explain"
        / "metrics.py"
    )
    if not module_path.exists():
        return None

    module_dir = module_path.parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

    spec = importlib.util.spec_from_file_location("temgx_official_metrics", module_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _TEMGX_METRICS_MODULE = mod
    return mod


def _temgx_fidelity_minus(z_full: float, z_removed: float) -> float:
    mod = _load_temgx_metrics_module()
    if mod is not None and hasattr(mod, "fidelity_minus"):
        return float(mod.fidelity_minus(float(z_full), float(z_removed)))
    return float(fidelity_minus(float(z_full), float(z_removed)))


def _temgx_fidelity_plus(z_full: float, z_expl: float) -> float:
    mod = _load_temgx_metrics_module()
    if mod is not None and hasattr(mod, "fidelity_plus"):
        return float(mod.fidelity_plus(float(z_full), float(z_expl)))
    return float(fidelity_plus(float(z_full), float(z_expl)))


def _temgx_aufsc(points: Sequence[tuple[float, float]]) -> float:
    """TemGX-style AUFSC: trapezoidal area over sorted sparsity-fidelity points."""
    pts = [
        (float(s), float(v))
        for s, v in points
        if np.isfinite(float(s)) and np.isfinite(float(v))
    ]
    if len(pts) < 2:
        return 0.0
    mod = _load_temgx_metrics_module()
    if mod is not None and hasattr(mod, "aufsc"):
        return float(mod.aufsc(list(pts)))
    pts = sorted(pts, key=lambda x: x[0])
    area = 0.0
    for (s0, f0), (s1, f1) in zip(pts[:-1], pts[1:]):
        area += 0.5 * (float(f0) + float(f1)) * (float(s1) - float(s0))
    return float(area)


def _fidelity_inv_tg(ori_score: float, important_score: float) -> float:
    """TGNNExplainer fidelity_inv_tg (logit-oriented, sign-aware)."""
    if float(ori_score) >= 0.0:
        return float(important_score) - float(ori_score)
    return float(ori_score) - float(important_score)


def _extract_tgnn_mcts_curve(
    result: ExplanationResult,
    *,
    levels: Sequence[float],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Reconstruct TGNNExplainer fid_inv_best curve from saved MCTS nodes.

    This mirrors the official evaluator behavior used by TGNNExplainer for
    MCTS-style explainers: sort node rewards by sparsity, take cumulative best,
    then sample at fixed sparsity thresholds.
    """
    extras = result.extras if isinstance(result.extras, Mapping) else {}
    sparsity_raw = extras.get("mcts_tree_nodes_sparsity")
    reward_raw = extras.get("mcts_tree_nodes_reward")
    if sparsity_raw is None or reward_raw is None:
        return None

    try:
        sparsity = np.asarray(sparsity_raw, dtype=float).reshape(-1)
        reward = np.asarray(reward_raw, dtype=float).reshape(-1)
    except Exception:
        return None
    if sparsity.size == 0 or reward.size == 0 or sparsity.size != reward.size:
        return None

    finite = np.isfinite(sparsity) & np.isfinite(reward)
    if not np.any(finite):
        return None
    sparsity = sparsity[finite]
    reward = reward[finite]

    sort_idx = np.argsort(sparsity)
    sparsity = sparsity[sort_idx]
    reward = reward[sort_idx]
    reward_best = np.maximum.accumulate(reward)

    x = np.asarray([float(v) for v in levels], dtype=float)
    y: list[float] = []
    for threshold in x:
        idx = np.where(sparsity <= float(threshold))[0]
        # Fallback to first available node if no node is below threshold.
        pos = int(idx.max()) if idx.size > 0 else 0
        y.append(float(reward_best[pos]))

    return x, np.asarray(y, dtype=float)


def _fidelity_sweep(
    metric: BaseMetric,
    context: ExplanationContext,
    result: ExplanationResult,
    *,
    mask_mode: str,
    value_fn: Callable[[float, float], float],
    prediction_prefix: str,
    ensure_min_one_for_levels: bool,
) -> tuple[dict[str, float], list[tuple[float, float]]]:
    candidate = _candidate_eidx(context, result)
    n_candidates = len(candidate)
    imp = _importance(result, n_candidates)
    order = _ranked_order(metric, context, result, candidate, imp)

    pred_full = metric.model.predict_proba(context.subgraph, context.target)
    z_full = _score(pred_full)

    values: dict[str, float] = {"prediction_full": float(z_full)}
    points: list[tuple[float, float]] = []

    has_levels = metric.config.get("sparsity_levels") is not None or metric.config.get("levels") is not None
    if has_levels:
        levels = _resolve_levels(metric.config, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        k_max = _resolve_k_max(metric.config, n_candidates)
        for level in levels:
            k = _k_from_level(
                float(level),
                n_candidates=n_candidates,
                k_max=k_max,
                ensure_min_one=ensure_min_one_for_levels,
            )
            selected = [candidate[i] for i in order[:k]]
            z_masked = _predict_masked(
                metric,
                context,
                candidate,
                selected,
                mode=mask_mode,
            )
            key = f"@s={float(level):g}"
            values[f"prediction_{prediction_prefix}.{key}"] = float(z_masked)
            values[key] = float(value_fn(float(z_full), float(z_masked)))
            points.append((float(level), float(values[key])))
        return values, points

    ks = _resolve_ks(metric.config, n_candidates)
    for k in ks:
        selected = [candidate[i] for i in order[:k]]
        z_masked = _predict_masked(
            metric,
            context,
            candidate,
            selected,
            mode=mask_mode,
        )
        key = f"@{int(k)}"
        values[f"prediction_{prediction_prefix}.{key}"] = float(z_masked)
        values[key] = float(value_fn(float(z_full), float(z_masked)))
        sparsity_x = float(k) / float(max(1, n_candidates))
        points.append((sparsity_x, float(values[key])))

    return values, points


def _edge_index_arrays(edge_index_like: Any) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(edge_index_like)
    if arr.ndim != 2:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=int)
    if arr.shape[0] == 2:
        src = arr[0]
        dst = arr[1]
    elif arr.shape[1] == 2:
        src = arr[:, 0]
        dst = arr[:, 1]
    else:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=int)
    return src.astype(int).reshape(-1), dst.astype(int).reshape(-1)


def _candidate_endpoints(
    context: ExplanationContext,
    result: ExplanationResult,
    candidate_eidx: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    payload = _payload(context)
    n_candidates = len(candidate_eidx)

    if "candidate_edge_index" in payload:
        src, dst = _edge_index_arrays(payload["candidate_edge_index"])
        n = min(src.size, dst.size, n_candidates)
        return src[:n], dst[:n]

    if "edge_index" in payload:
        src_all, dst_all = _edge_index_arrays(payload["edge_index"])
        if src_all.size > 0 and dst_all.size > 0 and n_candidates > 0:
            idx = np.asarray(candidate_eidx, dtype=int)
            if idx.max(initial=-1) < src_all.size:
                return src_all[idx], dst_all[idx]

    if context.subgraph and context.subgraph.edge_index:
        edge_list = np.asarray(context.subgraph.edge_index, dtype=int)
        if edge_list.ndim == 2 and edge_list.shape[1] == 2 and n_candidates > 0:
            idx = np.asarray(candidate_eidx, dtype=int)
            if idx.max(initial=-1) < edge_list.shape[0]:
                chosen = edge_list[idx]
                return chosen[:, 0], chosen[:, 1]

    return np.empty((0,), dtype=int), np.empty((0,), dtype=int)


def _rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.zeros_like(values, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)
    return ranks


def _deterministic_seed(metric_name: str, context: ExplanationContext, result: ExplanationResult) -> int:
    token = f"{metric_name}|{context.fingerprint()}|{result.explainer}"
    return int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16)


def _resolve_metric_topk(
    metric: BaseMetric,
    *,
    n_candidates: int,
    default_sparsity: float = 0.2,
    ensure_min_one: bool = True,
) -> int:
    if n_candidates <= 0:
        return 0

    if any(metric.config.get(key) is not None for key in ("k", "topk", "sparsity")):
        ks = _resolve_ks(metric.config, n_candidates)
        k = int(ks[-1]) if ks else 0
    else:
        level = float(metric.config.get("sparsity_level", default_sparsity))
        k = _k_from_level(
            level,
            n_candidates=n_candidates,
            k_max=_resolve_k_max(metric.config, n_candidates),
            ensure_min_one=ensure_min_one,
        )

    if ensure_min_one and n_candidates > 0 and k <= 0:
        k = 1
    return max(0, min(int(k), int(n_candidates)))


def _resolve_graph_metric_topk(
    metric: BaseMetric,
    *,
    n_graph: int,
    n_available: int,
    default_sparsity: float = 0.2,
    ensure_min_one: bool = True,
) -> int:
    if n_available <= 0:
        return 0

    if any(metric.config.get(key) is not None for key in ("k", "topk", "sparsity")):
        ks = _resolve_ks(metric.config, max(0, int(n_graph)))
        k = int(ks[-1]) if ks else 0
    else:
        level = float(metric.config.get("sparsity_level", default_sparsity))
        k = _k_from_level(
            level,
            n_candidates=n_available,
            k_max=max(0, int(n_graph)),
            ensure_min_one=ensure_min_one,
        )

    if ensure_min_one and n_available > 0 and k <= 0:
        k = 1
    return max(0, min(int(k), int(n_available)))


__all__ = [
    name
    for name in globals()
    if name not in {"__builtins__", "__annotations__", "__all__"}
]
