from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .common import (
    REQUIRED_EVENT_COLS,
    _as_int_list,
    _deduplicate_preserve_order,
    _format_node_mapping_table,
    _stable_sorted_events,
    _validate_required_columns,
)

def load_ground_truth_event_indices(
    ground_truth_path: Path | str,
    target_idx: int,
) -> List[int]:
    """
    Load motif event indices for a target from a ground-truth JSON file.

    Expected common schema:
    - {"examples": [{"target_event_idx": ..., "injected_edge_indices": [...]}, ...]}
    """
    path = Path(ground_truth_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Ground-truth file has unsupported schema: {path}")

    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError(f"Ground-truth JSON has no `examples` list: {path}")

    matches: List[int] = []
    for example in examples:
        if not isinstance(example, dict):
            continue
        ex_target = example.get("target_event_idx")
        try:
            ex_target_int = int(ex_target)  # type: ignore[arg-type]
        except Exception:
            continue
        if ex_target_int != int(target_idx):
            continue
        matches = _as_int_list(example.get("injected_edge_indices"))
        break

    return _deduplicate_preserve_order(matches)


def load_ground_truth_target_indices(ground_truth_path: Path | str) -> List[int]:
    """
    Load all target event indices from a ground-truth JSON file.
    """
    path = Path(ground_truth_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Ground-truth file has unsupported schema: {path}")

    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError(f"Ground-truth JSON has no `examples` list: {path}")

    targets: List[int] = []
    for example in examples:
        if not isinstance(example, dict):
            continue
        raw = example.get("target_event_idx")
        try:
            targets.append(int(raw))  # type: ignore[arg-type]
        except Exception:
            continue
    return _deduplicate_preserve_order(targets)


def load_temporal_graph(csv_path: Path | str) -> pd.DataFrame:
    """
    Load a temporal graph CSV in `ml_<dataset>.csv` format.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)
    _validate_required_columns(df, REQUIRED_EVENT_COLS, context=str(path))

    out = df.copy()
    for col in ("u", "i", "idx"):
        out[col] = pd.to_numeric(out[col], errors="raise").astype(int)
    out["ts"] = pd.to_numeric(out["ts"], errors="raise").astype(float)
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(int)

    if "is_motif" in out.columns:
        out["is_motif"] = out["is_motif"].astype(bool)
    if "is_target" in out.columns:
        out["is_target"] = out["is_target"].astype(bool)
    if "event_name" in out.columns:
        out["event_name"] = out["event_name"].astype(str)

    out = _stable_sorted_events(out)

    if not out["idx"].is_unique:
        warnings.warn("Column `idx` is not unique. Unique event indices are expected.", stacklevel=2)
    expected_idx = np.arange(int(out["idx"].min()), int(out["idx"].min()) + len(out))
    if not np.array_equal(out["idx"].to_numpy(), expected_idx):
        warnings.warn(
            "`idx` is not contiguous in sorted order. This is allowed, but contiguous indexing is recommended.",
            stacklevel=2,
        )
    return out


def extract_local_temporal_neighborhood(
    df: pd.DataFrame,
    target_idx: int,
    num_hops: int = 2,
    max_past_events: int = 30,
    time_window: Optional[float] = None,
    only_before_target: bool = True,
) -> pd.DataFrame:
    """
    Extract historical local events around a target event.
    """
    _validate_required_columns(df, REQUIRED_EVENT_COLS)
    target_rows = df.loc[df["idx"] == int(target_idx)]
    if target_rows.empty:
        raise KeyError(f"target_idx={target_idx} not found in dataframe column `idx`.")

    target_row = target_rows.iloc[0]
    target_ts = float(target_row["ts"])
    target_u = int(target_row["u"])
    target_i = int(target_row["i"])

    if only_before_target:
        before_mask = (df["ts"] < target_ts) | ((df["ts"] == target_ts) & (df["idx"] < int(target_idx)))
        candidate_df = df.loc[before_mask].copy()
    else:
        candidate_df = df.loc[df["idx"] != int(target_idx)].copy()

    if time_window is not None:
        lower_bound = target_ts - float(time_window)
        candidate_df = candidate_df.loc[candidate_df["ts"] >= lower_bound].copy()

    candidate_df = _stable_sorted_events(candidate_df)

    node_set = {target_u, target_i}
    frontier = {target_u, target_i}
    hops = max(1, int(num_hops))
    for _ in range(hops):
        if not frontier:
            break
        hop_mask = candidate_df["u"].isin(frontier) | candidate_df["i"].isin(frontier)
        hop_events = candidate_df.loc[hop_mask]
        if hop_events.empty:
            break
        touched_nodes = set(hop_events["u"].astype(int)).union(set(hop_events["i"].astype(int)))
        new_nodes = touched_nodes - node_set
        node_set |= touched_nodes
        frontier = new_nodes

    local_df = candidate_df.loc[candidate_df["u"].isin(node_set) & candidate_df["i"].isin(node_set)].copy()
    local_df = _stable_sorted_events(local_df)
    if max_past_events is not None and max_past_events > 0 and len(local_df) > int(max_past_events):
        local_df = local_df.tail(int(max_past_events)).copy()

    if len(local_df) < 3:
        warnings.warn(
            "Very small local neighborhood extracted; using the most recent historical events as fallback.",
            stacklevel=2,
        )
        fallback_n = int(max_past_events) if max_past_events and max_past_events > 0 else 30
        local_df = candidate_df.tail(fallback_n).copy()

    out = pd.concat([local_df, target_rows.head(1)], ignore_index=True)
    out = out.drop_duplicates(subset="idx", keep="last")
    return _stable_sorted_events(out)


def filter_events_at_or_before_target(
    events_df: pd.DataFrame,
    target_idx: int,
) -> pd.DataFrame:
    """
    Keep only events that are causally at or before `target_idx`.

    Rule:
    - keep events with `ts < target_ts`
    - keep events with `ts == target_ts` and `idx <= target_idx`
    """
    _validate_required_columns(events_df, REQUIRED_EVENT_COLS)
    out = _stable_sorted_events(events_df.copy())
    out["idx"] = out["idx"].astype(int)

    target_rows = out.loc[out["idx"] == int(target_idx)]
    if target_rows.empty:
        raise KeyError(f"target_idx={target_idx} not found in selected events.")

    target_row = target_rows.iloc[0]
    target_ts = float(target_row["ts"])
    target_idx_int = int(target_idx)

    keep_mask = (out["ts"].astype(float) < target_ts) | (
        (out["ts"].astype(float) == target_ts) & (out["idx"].astype(int) <= target_idx_int)
    )
    filtered = out.loc[keep_mask].copy()
    filtered = filtered.drop_duplicates(subset="idx", keep="last")
    return _stable_sorted_events(filtered)


def include_event_rows(
    events_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    event_idxs: Sequence[int],
    *,
    target_idx: Optional[int] = None,
    only_before_target: bool = False,
) -> pd.DataFrame:
    """
    Ensure specific event indices are present in the plotted event subset.

    Parameters
    ----------
    target_idx:
        Required when `only_before_target=True`.
    only_before_target:
        If `True`, drop any included rows that occur after the target event.
    """
    if not event_idxs:
        return _stable_sorted_events(events_df.copy())

    _validate_required_columns(events_df, REQUIRED_EVENT_COLS)
    _validate_required_columns(reference_df, REQUIRED_EVENT_COLS, context="reference dataframe")

    wanted = set(int(v) for v in event_idxs)
    add_rows = reference_df.loc[reference_df["idx"].isin(list(wanted))].copy()
    merged = pd.concat([events_df.copy(), add_rows], ignore_index=True)
    merged = merged.drop_duplicates(subset="idx", keep="last")
    merged = _stable_sorted_events(merged)

    if only_before_target:
        if target_idx is None:
            raise ValueError("`target_idx` must be provided when `only_before_target=True`.")
        merged = filter_events_at_or_before_target(merged, int(target_idx))
    return merged


def focus_close_context_events(
    events_df: pd.DataFrame,
    target_idx: int,
    *,
    motif_event_idxs: Optional[Sequence[int]] = None,
    explainer_event_idxs: Optional[Sequence[int]] = None,
    max_context_events: int = 12,
    time_radius: Optional[float] = None,
) -> pd.DataFrame:
    """
    Keep target/highlighted events and only the closest normal context events.

    Closeness priority for context events:
    1) shares node with highlighted/target event nodes
    2) smaller absolute time distance to target
    3) recency tie-breaker
    """
    _validate_required_columns(events_df, REQUIRED_EVENT_COLS)
    out = _stable_sorted_events(events_df.copy())
    out["idx"] = out["idx"].astype(int)

    target_rows = out.loc[out["idx"] == int(target_idx)]
    if target_rows.empty:
        raise KeyError(f"target_idx={target_idx} not found in selected events.")
    target_ts = float(target_rows.iloc[0]["ts"])

    keep_idxs: set[int] = {int(target_idx)}
    for seq in (motif_event_idxs, explainer_event_idxs):
        if not seq:
            continue
        for raw in seq:
            try:
                idx = int(raw)
            except Exception:
                continue
            if idx in set(out["idx"].tolist()):
                keep_idxs.add(idx)

    key_rows = out.loc[out["idx"].isin(list(keep_idxs))]
    key_nodes = set(key_rows["u"].astype(int)).union(set(key_rows["i"].astype(int)))

    if time_radius is not None:
        lower = target_ts - float(time_radius)
        upper = target_ts + float(time_radius)
        out = out.loc[(out["ts"] >= lower) & (out["ts"] <= upper)].copy()
        out = _stable_sorted_events(out)

    key_rows = out.loc[out["idx"].isin(list(keep_idxs))]
    context_rows = out.loc[~out["idx"].isin(list(keep_idxs))].copy()
    if context_rows.empty or max_context_events <= 0:
        return _stable_sorted_events(key_rows)

    context_rows["_shares_key_node"] = (
        context_rows["u"].astype(int).isin(list(key_nodes))
        | context_rows["i"].astype(int).isin(list(key_nodes))
    ).astype(int)
    context_rows["_time_abs_dist"] = (context_rows["ts"].astype(float) - target_ts).abs()

    context_rows = context_rows.sort_values(
        by=["_shares_key_node", "_time_abs_dist", "ts", "idx"],
        ascending=[False, True, False, False],
        kind="mergesort",
    )
    context_rows = context_rows.head(int(max_context_events)).copy()
    context_rows = context_rows.drop(columns=["_shares_key_node", "_time_abs_dist"], errors="ignore")

    merged = pd.concat([key_rows, context_rows], ignore_index=True)
    merged = merged.drop_duplicates(subset="idx", keep="last")
    return _stable_sorted_events(merged)


def relabel_local_nodes(events_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, str], pd.DataFrame]:
    """
    Relabel node ids to compact labels S1, S2, ...
    """
    nodes = sorted(set(events_df["u"].astype(int)).union(set(events_df["i"].astype(int))))
    mapping = {node_id: f"S{k}" for k, node_id in enumerate(nodes, start=1)}

    out = events_df.copy()
    out["u_local"] = out["u"].astype(int).map(mapping)
    out["i_local"] = out["i"].astype(int).map(mapping)
    return out, mapping, _format_node_mapping_table(mapping)


def assign_event_labels(
    events_df: pd.DataFrame,
    motif_event_idxs: Optional[Sequence[int]],
    target_idx: int,
    explainer_event_idxs: Optional[Sequence[int]] = None,
) -> Tuple[pd.DataFrame, List[int], List[int], int]:
    """
    Assign event roles and labels.

    Role precedence:
    target > overlap(gt+explainer) > ground_truth > explainer > context
    """
    _validate_required_columns(events_df, REQUIRED_EVENT_COLS)
    out = _stable_sorted_events(events_df.copy())
    out["idx"] = out["idx"].astype(int)
    idx_set = set(out["idx"].tolist())

    resolved_target_idx = int(target_idx)
    if resolved_target_idx not in idx_set:
        if "is_target" in out.columns:
            candidate_targets = out.loc[out["is_target"].astype(bool), "idx"].astype(int).tolist()
            if candidate_targets:
                resolved_target_idx = int(candidate_targets[0])
        if resolved_target_idx not in idx_set:
            raise KeyError(
                f"target_idx={target_idx} not found in selected events and no valid `is_target` row exists."
            )

    motif_set: set[int] = set()
    if motif_event_idxs:
        invalid = []
        for raw in motif_event_idxs:
            try:
                idx = int(raw)
            except Exception:
                invalid.append(raw)
                continue
            if idx in idx_set:
                motif_set.add(idx)
            else:
                invalid.append(raw)
        if invalid:
            warnings.warn(f"Ignored motif indices not present in selected events: {invalid}", stacklevel=2)
    if "is_motif" in out.columns:
        motif_set.update(out.loc[out["is_motif"].astype(bool), "idx"].astype(int).tolist())

    explainer_set: set[int] = set()
    if explainer_event_idxs:
        invalid = []
        for raw in explainer_event_idxs:
            try:
                idx = int(raw)
            except Exception:
                invalid.append(raw)
                continue
            if idx in idx_set:
                explainer_set.add(idx)
            else:
                invalid.append(raw)
        if invalid:
            warnings.warn(f"Ignored explainer indices not present in selected events: {invalid}", stacklevel=2)

    motif_set.discard(resolved_target_idx)
    explainer_set.discard(resolved_target_idx)

    overlap_set = motif_set.intersection(explainer_set)
    gt_only_set = motif_set - overlap_set
    expl_only_set = explainer_set - overlap_set

    out["event_role"] = "context"
    out.loc[out["idx"].isin(list(gt_only_set)), "event_role"] = "ground_truth"
    out.loc[out["idx"].isin(list(expl_only_set)), "event_role"] = "explainer"
    out.loc[out["idx"].isin(list(overlap_set)), "event_role"] = "overlap"
    out.loc[out["idx"] == resolved_target_idx, "event_role"] = "target"

    out["event_label"] = ""

    gt_rows = out.loc[out["idx"].isin(list(motif_set))].sort_values(["ts", "idx"], kind="mergesort")
    gt_label_map: Dict[int, str] = {}
    for k, row in enumerate(gt_rows.itertuples(index=False), start=1):
        gt_label_map[int(row.idx)] = f"e{k}"

    expl_rows = out.loc[out["idx"].isin(list(explainer_set))].sort_values(["ts", "idx"], kind="mergesort")
    expl_label_map: Dict[int, str] = {}
    for k, row in enumerate(expl_rows.itertuples(index=False), start=1):
        expl_label_map[int(row.idx)] = f"x{k}"

    for row in out.itertuples():
        idx = int(row.idx)
        role = str(row.event_role)
        if role == "target":
            out.loc[out["idx"] == idx, "event_label"] = "target"
        elif role == "overlap":
            label = gt_label_map.get(idx, expl_label_map.get(idx, ""))
            out.loc[out["idx"] == idx, "event_label"] = f"{label}*" if label else "*"
        elif role == "ground_truth":
            out.loc[out["idx"] == idx, "event_label"] = gt_label_map.get(idx, "")
        elif role == "explainer":
            out.loc[out["idx"] == idx, "event_label"] = expl_label_map.get(idx, "")

    return _stable_sorted_events(out), sorted(motif_set), sorted(explainer_set), resolved_target_idx


def build_clean_node_layout(
    events_df: pd.DataFrame,
    target_idx: int,
    motif_event_idxs: Optional[Sequence[int]] = None,
) -> Dict[int, Tuple[float, float]]:
    """
    Deterministic manual node layout for clean, thesis-style figures.
    """
    _ = motif_event_idxs
    target_rows = events_df.loc[events_df["idx"].astype(int) == int(target_idx)]
    if target_rows.empty:
        raise KeyError(f"target_idx={target_idx} missing from events_df in layout builder.")

    target_row = target_rows.iloc[0]
    src, dst = int(target_row["u"]), int(target_row["i"])
    nodes = sorted(set(events_df["u"].astype(int)).union(set(events_df["i"].astype(int))))

    pos: Dict[int, Tuple[float, float]] = {}
    if src == dst:
        pos[src] = (0.0, 0.0)
    else:
        pos[src] = (-1.45, 0.0)
        pos[dst] = (1.45, 0.0)

    motif_rows = events_df.loc[events_df.get("event_role", "context").isin(["ground_truth", "overlap"])]
    motif_nodes = sorted(set(motif_rows["u"].astype(int)).union(set(motif_rows["i"].astype(int))) - {src, dst})

    expl_rows = events_df.loc[events_df.get("event_role", "context").isin(["explainer", "overlap"])]
    expl_nodes = sorted(set(expl_rows["u"].astype(int)).union(set(expl_rows["i"].astype(int))) - {src, dst})

    featured_nodes = sorted(set(motif_nodes).union(set(expl_nodes)))
    context_nodes = [n for n in nodes if n not in {src, dst} and n not in featured_nodes]

    def _place_arc(node_ids: List[int], radius: float, start_deg: float, end_deg: float, y_shift: float = 0.0) -> None:
        if not node_ids:
            return
        if len(node_ids) == 1:
            angles = np.array([(start_deg + end_deg) / 2.0])
        else:
            angles = np.linspace(start_deg, end_deg, len(node_ids))
        for node_id, deg in zip(node_ids, angles):
            rad = np.deg2rad(deg)
            pos[node_id] = (radius * np.cos(rad), radius * np.sin(rad) + y_shift)

    _place_arc(featured_nodes, radius=2.0, start_deg=30, end_deg=150, y_shift=0.36)
    if len(context_nodes) <= 8:
        _place_arc(context_nodes, radius=2.35, start_deg=220, end_deg=320, y_shift=-0.16)
    else:
        split = int(np.ceil(len(context_nodes) / 2))
        _place_arc(context_nodes[:split], radius=2.3, start_deg=205, end_deg=285, y_shift=-0.10)
        _place_arc(context_nodes[split:], radius=2.8, start_deg=250, end_deg=335, y_shift=-0.34)

    leftovers = [n for n in nodes if n not in pos]
    if leftovers:
        angles = np.linspace(0, 2 * np.pi, len(leftovers), endpoint=False)
        for node_id, ang in zip(leftovers, angles):
            pos[node_id] = (2.6 * np.cos(ang), 2.6 * np.sin(ang))
    return pos

__all__ = [
    "assign_event_labels",
    "build_clean_node_layout",
    "extract_local_temporal_neighborhood",
    "filter_events_at_or_before_target",
    "focus_close_context_events",
    "include_event_rows",
    "load_ground_truth_event_indices",
    "load_ground_truth_target_indices",
    "load_temporal_graph",
    "relabel_local_nodes",
]
