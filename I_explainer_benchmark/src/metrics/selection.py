from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from typing import Any

import numpy as np

from ..core.types import ExplanationContext, ExplanationResult, ModelProtocol


_EDGE_IMPACT_CACHE: dict[tuple[Any, ...], np.ndarray] = {}
_CACHE_MAX_ITEMS = 4096


def _prune_cache(cache: dict[tuple[Any, ...], Any], *, max_items: int = _CACHE_MAX_ITEMS) -> None:
    overflow = len(cache) - int(max_items)
    if overflow <= 0:
        return
    for key in list(cache.keys())[:overflow]:
        cache.pop(key, None)


def _to_scalar_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except Exception:
        pass
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return float("nan")
            return float(value.reshape(-1)[0].detach().cpu().item())
    except Exception:
        pass
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(arr[0])


def importance_for_candidates(importance_edges: Sequence[float] | None, n_candidates: int) -> np.ndarray:
    if n_candidates <= 0:
        return np.empty((0,), dtype=float)
    if importance_edges is None:
        raw: Sequence[float] = []
    else:
        raw = importance_edges
    imp = np.asarray(list(raw), dtype=float).reshape(-1)
    if imp.size >= n_candidates:
        return imp[:n_candidates]
    out = np.zeros((n_candidates,), dtype=float)
    out[: imp.size] = imp
    return out


def extract_explicit_selected(
    result: ExplanationResult,
    candidate_eidx: Sequence[int],
) -> tuple[list[int], set[int]]:
    candidate = [int(e) for e in candidate_eidx]
    keys = (
        "selected_eidx",
        "cf_event_ids",
        "coalition_eidx",
        "explanation_event_ids",
        "omitted_edge_idxs",
    )
    for key in keys:
        raw = result.extras.get(key)
        if raw is None:
            continue
        try:
            raw_list = [int(e) for e in raw]
        except Exception:
            continue
        raw_set = {int(e) for e in raw_list}
        selected = [int(e) for e in candidate if int(e) in raw_set]
        return selected, set(selected)
    return [], set()


def _ranking_cfg(config: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg: Mapping[str, Any] = {}
    if isinstance(config, Mapping):
        nested = config.get("ranking")
        if isinstance(nested, Mapping):
            cfg = nested

    def _get(key: str, default: Any) -> Any:
        if key in cfg:
            return cfg.get(key)
        if isinstance(config, Mapping) and key in config:
            return config.get(key)
        return default

    return {
        "prefer_selected": bool(_get("prefer_selected", True)),
        "tie_break": str(_get("tie_break", "candidate_order")),
        "support_eps": float(_get("support_eps", 1e-12)),
        "uninformative_fallback": str(_get("uninformative_fallback", "none")),
        "fallback_max_candidates": int(_get("fallback_max_candidates", 96)),
        "natural_support": str(_get("natural_support", "nonzero")),
        "natural_top_fraction": float(_get("natural_top_fraction", 0.1)),
    }


def _is_uninformative(scores: np.ndarray, *, eps: float) -> bool:
    if scores.size <= 1:
        return True
    finite = scores[np.isfinite(scores)]
    if finite.size <= 1:
        return True
    span = float(np.max(finite) - np.min(finite))
    return span <= float(eps)


def _hash_ints(values: Sequence[int]) -> str:
    arr = np.asarray([int(v) for v in values], dtype=np.int64).reshape(-1)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _edge_impacts(
    *,
    model: ModelProtocol,
    context: ExplanationContext,
    result: ExplanationResult,
    candidate_eidx: Sequence[int],
    mode: str,
) -> np.ndarray | None:
    if context.subgraph is None:
        return None
    candidate = [int(e) for e in candidate_eidx]
    n = len(candidate)
    if n <= 0:
        return np.empty((0,), dtype=float)

    cache_key = (
        id(model),
        str(result.context_fp),
        str(result.explainer),
        _hash_ints(candidate),
        str(mode),
    )
    cached = _EDGE_IMPACT_CACHE.get(cache_key)
    if cached is not None and cached.size == n:
        return cached.copy()

    z_full = _to_scalar_float(model.predict_proba(context.subgraph, context.target))
    if not np.isfinite(z_full):
        return None

    impacts = np.full((n,), np.nan, dtype=float)
    ones = np.ones((n,), dtype=float)
    zeros = np.zeros((n,), dtype=float)
    for i in range(n):
        if mode == "single_edge_keep":
            mask = zeros.copy()
            mask[i] = 1.0
        else:
            mask = ones.copy()
            mask[i] = 0.0
        try:
            z_masked = _to_scalar_float(
                model.predict_proba_with_mask(
                    context.subgraph,
                    context.target,
                    edge_mask=mask.tolist(),
                )
            )
        except Exception:
            continue
        if np.isfinite(z_masked):
            impacts[i] = float(abs(float(z_full) - float(z_masked)))

    _EDGE_IMPACT_CACHE[cache_key] = impacts.copy()
    _prune_cache(_EDGE_IMPACT_CACHE)
    return impacts


def _rank_by_scores(
    *,
    candidate_eidx: Sequence[int],
    scores: np.ndarray,
    selected_set: set[int],
    prefer_selected: bool,
    tie_break: str,
) -> list[int]:
    n = len(candidate_eidx)
    if n <= 0:
        return []

    score_arr = np.asarray(scores, dtype=float).reshape(-1)
    if score_arr.size >= n:
        score_arr = score_arr[:n]
    else:
        padded = np.zeros((n,), dtype=float)
        padded[: score_arr.size] = score_arr
        score_arr = padded
    score_arr = np.nan_to_num(score_arr, nan=-np.inf)

    if tie_break == "edge_id":
        tie = np.asarray([int(e) for e in candidate_eidx], dtype=np.int64)
    elif tie_break == "reverse_candidate_order":
        tie = -np.arange(n, dtype=np.int64)
    else:
        tie = np.arange(n, dtype=np.int64)

    if prefer_selected and selected_set:
        selected_bonus = np.asarray(
            [1 if int(e) in selected_set else 0 for e in candidate_eidx],
            dtype=np.int64,
        )
    else:
        selected_bonus = np.zeros((n,), dtype=np.int64)

    order = np.lexsort((tie, -score_arr, -selected_bonus))
    return [int(i) for i in order.tolist()]


def rank_candidate_positions(
    *,
    model: ModelProtocol | None,
    context: ExplanationContext | None,
    result: ExplanationResult,
    candidate_eidx: Sequence[int],
    importance: Sequence[float] | np.ndarray | None,
    config: Mapping[str, Any] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    candidate = [int(e) for e in candidate_eidx]
    n = len(candidate)
    if n <= 0:
        return [], {"reason": "empty_candidate"}

    cfg = _ranking_cfg(config)
    support_eps = float(cfg["support_eps"])
    tie_break = str(cfg["tie_break"])
    prefer_selected = bool(cfg["prefer_selected"])
    fallback_mode = str(cfg["uninformative_fallback"]).strip().lower()
    fallback_max_candidates = max(0, int(cfg["fallback_max_candidates"]))

    selected_list, selected_set = extract_explicit_selected(result, candidate)
    imp = importance_for_candidates(importance, n)
    scores = imp.copy()
    source = "importance"
    used_fallback = False

    if _is_uninformative(scores, eps=support_eps):
        if (
            fallback_mode in {"single_edge_drop", "single_edge_keep"}
            and model is not None
            and context is not None
            and n <= fallback_max_candidates
        ):
            impacts = _edge_impacts(
                model=model,
                context=context,
                result=result,
                candidate_eidx=candidate,
                mode=fallback_mode,
            )
            if impacts is not None and impacts.size == n and not _is_uninformative(impacts, eps=support_eps):
                scores = impacts
                source = fallback_mode
                used_fallback = True
        elif selected_set:
            source = "explicit_selected"

    order = _rank_by_scores(
        candidate_eidx=candidate,
        scores=scores,
        selected_set=selected_set,
        prefer_selected=prefer_selected,
        tie_break=tie_break,
    )

    return order, {
        "source": source,
        "used_fallback": bool(used_fallback),
        "prefer_selected": bool(prefer_selected),
        "tie_break": tie_break,
        "selected_count": int(len(selected_list)),
        "n_candidates": int(n),
    }


def select_candidate_eidx(
    *,
    result: ExplanationResult,
    candidate_eidx: Sequence[int],
    order: Sequence[int],
    importance: Sequence[float] | np.ndarray | None,
    k: int | None,
    config: Mapping[str, Any] | None = None,
) -> list[int]:
    candidate = [int(e) for e in candidate_eidx]
    n = len(candidate)
    if n <= 0:
        return []

    cfg = _ranking_cfg(config)
    support_eps = float(cfg["support_eps"])
    natural_support = str(cfg.get("natural_support", "nonzero")).strip().lower()
    natural_top_fraction = float(cfg.get("natural_top_fraction", 0.1))

    ordered = [int(i) for i in order if 0 <= int(i) < n]
    if not ordered:
        return []

    selected_list, selected_set = extract_explicit_selected(result, candidate)
    if k is None:
        if selected_set:
            return [int(candidate[i]) for i in ordered if int(candidate[i]) in selected_set]
        imp = importance_for_candidates(importance, n)

        if natural_support == "elbow":
            ranked_abs = np.asarray([abs(float(imp[i])) for i in ordered], dtype=float)
            if ranked_abs.size == 1:
                return [int(candidate[ordered[0]])]
            if ranked_abs.size > 1:
                gaps = ranked_abs[:-1] - ranked_abs[1:]
                if np.isfinite(gaps).any() and float(np.max(gaps)) > support_eps:
                    k_elbow = int(np.argmax(gaps)) + 1
                    k_elbow = max(1, min(k_elbow, n))
                    return [int(candidate[i]) for i in ordered[:k_elbow]]

        if natural_support == "top_fraction":
            frac = min(1.0, max(0.0, float(natural_top_fraction)))
            k_frac = int(round(frac * float(n)))
            if frac > 0.0 and k_frac == 0 and n > 0:
                k_frac = 1
            k_frac = max(0, min(k_frac, n))
            if k_frac > 0:
                return [int(candidate[i]) for i in ordered[:k_frac]]

        support = [int(i) for i in ordered if abs(float(imp[i])) > support_eps]
        if support:
            return [int(candidate[i]) for i in support]
        return [int(candidate[ordered[0]])]

    limit = max(0, min(int(k), n))
    return [int(candidate[i]) for i in ordered[:limit]]


__all__ = [
    "importance_for_candidates",
    "extract_explicit_selected",
    "rank_candidate_positions",
    "select_candidate_eidx",
]
