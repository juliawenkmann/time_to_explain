from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


def as_dataframe(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.DataFrame()


def resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def upsert_csv(path: Path, df: pd.DataFrame, subset_cols: list[str]) -> None:
    if df is None or df.empty:
        return
    merged = df.copy()
    if path.exists():
        prev = pd.read_csv(path)
        merged = pd.concat([prev, merged], ignore_index=True)
    subset = [c for c in subset_cols if c in merged.columns]
    if subset:
        merged = merged.drop_duplicates(subset=subset, keep="last")
    merged.to_csv(path, index=False)


def sorted_sparsity_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if str(c).startswith(prefix)]
    return sorted(cols, key=lambda c: float(str(c).split("@s=")[-1]))


def jsonl_has_records(path: str | Path) -> bool:
    resolved = resolve_path(path)
    if not resolved.exists() or resolved.stat().st_size <= 0:
        return False
    with resolved.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                return True
    return False


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    resolved = resolve_path(path)
    with resolved.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_csv_if_exists(path: str | Path) -> pd.DataFrame:
    resolved = resolve_path(path)
    if not resolved.exists() or not resolved.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(resolved)
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def jsonl_contains_targets(path: str | Path, target_ids: Sequence[int]) -> bool:
    needed = {int(v) for v in target_ids}
    if not needed:
        return True
    found: set[int] = set()
    try:
        records = load_jsonl_records(path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    for rec in records:
        ctx = rec.get("context") or {}
        tgt = ctx.get("target") or {}
        event_idx = tgt.get("event_idx")
        if event_idx is None:
            event_idx = rec.get("anchor_idx", rec.get("target_idx"))
        if event_idx is None:
            continue
        found.add(int(event_idx))
        if needed.issubset(found):
            return True
    return needed.issubset(found)


def event_triplet_from_events(events_df: pd.DataFrame, event_idx: int) -> tuple[int, int, float]:
    if 1 <= int(event_idx) <= int(len(events_df)):
        row = events_df.iloc[int(event_idx) - 1]
    else:
        row = events_df.iloc[int(event_idx)]

    src = int(row["u"] if "u" in events_df.columns else row.iloc[0])
    dst = int(row["i"] if "i" in events_df.columns else row.iloc[1])
    ts = float(row["ts"] if "ts" in events_df.columns else row.iloc[2])
    return src, dst, ts


__all__ = [
    "as_dataframe",
    "coerce_float",
    "coerce_int",
    "event_triplet_from_events",
    "jsonl_contains_targets",
    "jsonl_has_records",
    "load_jsonl_records",
    "read_csv_if_exists",
    "resolve_path",
    "sorted_sparsity_cols",
    "upsert_csv",
]
