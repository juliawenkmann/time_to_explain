from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REQUIRED_EVENT_COLS = ("u", "i", "ts", "label", "idx")

ROLE_COLORS: Dict[str, str] = {
    "context": "#9AA0A6",  # gray
    "ground_truth": "#F28E2B",  # orange
    "explainer": "#2CA25F",  # green
    "overlap": "#9C755F",  # brown
    "target": "#2F6DF3",  # blue
}


@dataclass(frozen=True)
class ExplainerSelection:
    """Selected event indices for one explainer/target pair."""

    explainer_name: str
    target_idx: int
    selected_event_idxs: List[int]
    candidate_event_idxs: List[int]
    source: str
    results_path: Path
    run_id: Optional[str]


def _validate_required_columns(
    df: pd.DataFrame,
    required: Sequence[str] = REQUIRED_EVENT_COLS,
    context: str = "events dataframe",
) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{context} is missing required columns: {missing}. "
            f"Expected at least {list(required)}."
        )


def _stable_sorted_events(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["ts", "idx"], kind="mergesort").reset_index(drop=True)


def _as_int_list(value: object) -> List[int]:
    if not isinstance(value, (list, tuple)):
        return []
    out: List[int] = []
    for item in value:
        try:
            out.append(int(item))
        except Exception:
            continue
    return out


def _deduplicate_preserve_order(values: Iterable[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _normalize_event_time(ts: np.ndarray) -> np.ndarray:
    ts = np.asarray(ts, dtype=float)
    if len(ts) == 0:
        return ts
    t_min, t_max = float(np.min(ts)), float(np.max(ts))
    if math.isclose(t_min, t_max):
        base = np.linspace(0.0, 1.0, len(ts), endpoint=True)
    else:
        base = (ts - t_min) / (t_max - t_min)
    return base + np.linspace(0.0, 1e-3, len(ts), endpoint=True)


def _role_style(role: str) -> Dict[str, object]:
    if role == "target":
        return {"color": ROLE_COLORS["target"], "lw": 3.0, "linestyle": "solid", "alpha": 0.99}
    if role == "overlap":
        return {"color": ROLE_COLORS["overlap"], "lw": 3.0, "linestyle": "solid", "alpha": 0.98}
    if role == "ground_truth":
        return {"color": ROLE_COLORS["ground_truth"], "lw": 2.7, "linestyle": "solid", "alpha": 0.96}
    if role == "explainer":
        return {"color": ROLE_COLORS["explainer"], "lw": 2.4, "linestyle": "solid", "alpha": 0.95}
    return {
        "color": ROLE_COLORS["context"],
        "lw": 1.3,
        "linestyle": (0, (4, 3)),
        "alpha": 0.92,
    }


def _format_node_mapping_table(node_mapping: Dict[int, str]) -> pd.DataFrame:
    rows = [
        {"local_label": local, "original_node_id": node}
        for node, local in sorted(node_mapping.items(), key=lambda kv: int(kv[1][1:]))
    ]
    return pd.DataFrame(rows)


def _avoid_text_overlap(
    candidates: Sequence[Tuple[float, float]],
    min_dx: float = 0.03,
    min_dy: float = 0.12,
    max_tries: int = 25,
) -> List[Tuple[float, float]]:
    placed: List[Tuple[float, float]] = []
    out: List[Tuple[float, float]] = []
    for x, y in candidates:
        y_new = y
        for _ in range(max_tries):
            collides = any(abs(x - px) < min_dx and abs(y_new - py) < min_dy for px, py in placed)
            if not collides:
                break
            y_new += min_dy
        placed.append((x, y_new))
        out.append((x, y_new))
    return out


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _extract_record_explainer(record: Dict[str, object]) -> str:
    result = record.get("result")
    if isinstance(result, dict):
        value = result.get("explainer")
        if isinstance(value, str) and value.strip():
            return _normalize_name(value)
    value = record.get("explainer")
    if isinstance(value, str) and value.strip():
        return _normalize_name(value)
    return ""


def _extract_record_target_idx(record: Dict[str, object]) -> Optional[int]:
    context = record.get("context")
    if isinstance(context, dict):
        target = context.get("target")
        if isinstance(target, dict):
            for key in ("event_idx", "e_idx", "idx"):
                if key in target:
                    try:
                        return int(target[key])  # type: ignore[arg-type]
                    except Exception:
                        pass

    result = record.get("result")
    if isinstance(result, dict):
        extras = result.get("extras")
        if isinstance(extras, dict):
            for key in ("event_idx", "e_idx", "idx"):
                if key in extras:
                    try:
                        return int(extras[key])  # type: ignore[arg-type]
                    except Exception:
                        pass
    return None


def _derive_selected_event_idxs(
    record: Dict[str, object],
    top_k_fallback: int = 8,
) -> Tuple[List[int], List[int], str]:
    result = record.get("result")
    if not isinstance(result, dict):
        return [], [], "none"

    extras = result.get("extras")
    extras_dict = extras if isinstance(extras, dict) else {}
    candidate_eidx = _as_int_list(extras_dict.get("candidate_eidx"))

    for key in ("selected_eidx", "cf_event_ids", "coalition_eidx"):
        explicit = _as_int_list(extras_dict.get(key))
        if explicit:
            return _deduplicate_preserve_order(explicit), _deduplicate_preserve_order(candidate_eidx), key

    importance = result.get("importance_edges")
    if isinstance(importance, (list, tuple)) and candidate_eidx:
        imp = np.asarray(list(importance), dtype=float)
        n = min(len(candidate_eidx), len(imp))
        if n > 0:
            order = np.argsort(-np.abs(imp[:n]))
            k = min(max(1, int(top_k_fallback)), n)
            selected = [int(candidate_eidx[i]) for i in order[:k]]
            return _deduplicate_preserve_order(selected), _deduplicate_preserve_order(candidate_eidx), "top_abs_importance"

    return [], _deduplicate_preserve_order(candidate_eidx), "none"

__all__ = [
    "ExplainerSelection",
    "REQUIRED_EVENT_COLS",
    "ROLE_COLORS",
]
