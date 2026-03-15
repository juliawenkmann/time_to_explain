from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _as_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        out: list[int] = []
        for v in values:
            try:
                out.append(int(v))
            except Exception:
                pass
        return out
    return []


def _as_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        out: list[float] = []
        for v in values:
            try:
                out.append(float(v))
            except Exception:
                pass
        return out
    return []


def _resolve_explain_index_path(repo_root: Path, dataset_name: str) -> Path | None:
    candidates = [
        repo_root / "I_explainer_benchmark" / "resources" / "explainer" / "explain_index" / f"{dataset_name}.csv",
        repo_root / "I_explainer_benchmark" / "resources" / "datasets" / "explain_index" / f"{dataset_name}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_explain_index_events(repo_root: Path, dataset_name: str) -> list[int]:
    path = _resolve_explain_index_path(repo_root, dataset_name)
    if path is None:
        return []

    try:
        df = pd.read_csv(path)
        if df.empty:
            return []
        col = "event_idx" if "event_idx" in df.columns else df.columns[0]
        return [int(v) for v in pd.to_numeric(df[col], errors="coerce").dropna().astype(int).tolist()]
    except Exception:
        return []


def parse_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def latest_run_file(
    dataset_name: str,
    model_name: str,
    base_dir: Path,
    explainer_key: str,
    allow_missing: bool = False,
) -> Path | None:
    # Search both root and optional model subdir (e.g., official_tempme/tgn).
    search_roots = [base_dir]
    model_subdir = base_dir / str(model_name)
    if model_subdir.exists():
        search_roots.append(model_subdir)

    file_pattern = f"{dataset_name}_{model_name}_official_{explainer_key}_*.jsonl"
    dir_pattern = f"{dataset_name}_{model_name}_official_{explainer_key}_*"
    for root in search_roots:
        # Layout A: flat jsonl files in search root.
        files = sorted(root.glob(file_pattern))
        if files:
            return files[-1]

        # Layout B: run directory with results.jsonl inside.
        run_dirs = sorted([p for p in root.glob(dir_pattern) if p.is_dir()])
        for run_dir in reversed(run_dirs):
            candidate = run_dir / "results.jsonl"
            if candidate.exists():
                return candidate

    if allow_missing:
        return None

    raise FileNotFoundError(
        f"No run found for dataset={dataset_name}, model={model_name}, explainer={explainer_key} "
        f"in {base_dir} (file_pattern={file_pattern}, dir_pattern={dir_pattern})"
    )


def list_run_files(
    dataset_name: str,
    model_name: str,
    base_dir: Path,
    explainer_key: str,
) -> list[Path]:
    """Return all matching run files ordered from oldest to newest."""
    search_roots = [base_dir]
    model_subdir = base_dir / str(model_name)
    if model_subdir.exists():
        search_roots.append(model_subdir)

    file_pattern = f"{dataset_name}_{model_name}_official_{explainer_key}_*.jsonl"
    dir_pattern = f"{dataset_name}_{model_name}_official_{explainer_key}_*"
    out: list[Path] = []

    for root in search_roots:
        out.extend(sorted(root.glob(file_pattern)))
        run_dirs = sorted([p for p in root.glob(dir_pattern) if p.is_dir()])
        for run_dir in run_dirs:
            candidate = run_dir / "results.jsonl"
            if candidate.exists():
                out.append(candidate)

    # Preserve deterministic chronological-ish ordering while deduplicating.
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in sorted(out):
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def find_mcts_recorder_file(
    dataset_name: str,
    model_name: str,
    target_event: int,
    candidate_scores_dir: Path,
    threshold_num: int,
) -> Path | None:
    if not candidate_scores_dir.exists():
        return None

    patterns = [
        f"{model_name}_{dataset_name}_{int(target_event)}_mcts_recorder_*_th{int(threshold_num)}.csv",
        f"{model_name}_{dataset_name}_{int(target_event)}_mcts_recorder_*.csv",
    ]

    for pattern in patterns:
        matches = sorted(candidate_scores_dir.glob(pattern))
        if matches:
            return matches[-1]

    return None


def find_mcts_node_info_file(
    dataset_name: str,
    model_name: str,
    target_event: int,
    mcts_saved_dir: Path,
    threshold_num: int,
) -> Path | None:
    """Locate a TGNNExplainer MCTS node-info artifact for one target event."""
    if not mcts_saved_dir.exists():
        return None

    # Prefer explicit known suffixes first, then broader wildcards.
    preferred_suffixes = [
        "pg_false",
        "mlp_true_pg_positive",
        "mlp_true_pg_negative",
        "pg_true_pg_positive",
        "pg_true_pg_negative",
        "dot_true_pg_positive",
        "dot_true_pg_negative",
        "mlp_true",
        "pg_true",
        "dot_true",
    ]

    patterns = [
        f"{model_name}_{dataset_name}_{int(target_event)}_mcts_node_info_{suffix}_th{int(threshold_num)}.pt"
        for suffix in preferred_suffixes
    ]
    patterns.extend(
        [
            f"{model_name}_{dataset_name}_{int(target_event)}_mcts_node_info_*_th{int(threshold_num)}.pt",
            f"{model_name}_{dataset_name}_{int(target_event)}_mcts_node_info_*.pt",
        ]
    )

    for pattern in patterns:
        matches = sorted(mcts_saved_dir.glob(pattern))
        if matches:
            return matches[-1]

    return None


def empty_run_table() -> pd.DataFrame:
    df = pd.DataFrame(
        columns=[
            "run_id",
            "anchor_idx",
            "event_idx",
            "candidate_eidx",
            "selected_eidx",
            "importance_edges",
            "num_candidates",
            "num_selected",
        ]
    )
    if not df.empty:
        df["event_idx"] = df["event_idx"].astype(int)
    return df


def flatten_official_records(records: list[dict]) -> pd.DataFrame:
    flat = []
    for rec in records:
        result = rec.get("result") or {}
        extras = result.get("extras") or {}

        event_idx_raw = extras.get("event_idx")
        if event_idx_raw is None:
            event_idx_raw = rec.get("event_idx")
        try:
            event_idx = int(event_idx_raw)
        except Exception:
            continue

        candidate_eidx = _as_int_list(
            extras.get("candidate_eidx")
            or extras.get("candidate_event_ids")
            or extras.get("candidate_events")
        )

        selected_eidx = _as_int_list(
            extras.get("selected_eidx")
            or extras.get("coalition_eidx")
            or extras.get("explanation_event_ids")
            or extras.get("cf_event_ids")
            or extras.get("counterfactual_event_ids")
        )

        importance_edges = _as_float_list(
            result.get("importance_edges")
            or result.get("edge_importance")
        )

        # Fallback for explainers that encode selection only via positive importance.
        if not selected_eidx and candidate_eidx and importance_edges:
            selected_eidx = [
                int(eid)
                for eid, score in zip(candidate_eidx, importance_edges)
                if float(score) > 0.0
            ]

        if not candidate_eidx and selected_eidx:
            candidate_eidx = list(selected_eidx)
            if not importance_edges:
                importance_edges = [1.0 for _ in candidate_eidx]

        flat.append(
            {
                "run_id": rec.get("run_id"),
                "anchor_idx": rec.get("anchor_idx"),
                "event_idx": event_idx,
                "candidate_eidx": candidate_eidx,
                "selected_eidx": selected_eidx,
                "importance_edges": list(importance_edges),
                "extras": extras if isinstance(extras, dict) else {},
            }
        )

    if not flat:
        return empty_run_table()

    df = pd.DataFrame(flat).sort_values("event_idx").reset_index(drop=True)
    df["num_candidates"] = df["candidate_eidx"].apply(len)
    df["num_selected"] = df["selected_eidx"].apply(len)
    return df


def choose_target_events(
    table: pd.DataFrame,
    preferred_events: list[int] | None,
    max_targets: int,
) -> tuple[list[int], str]:
    if max_targets <= 0 or table.empty:
        return [], "none"

    ranked_events = (
        table.sort_values(["num_selected", "event_idx"], ascending=[False, False])["event_idx"]
        .astype(int)
        .tolist()
    )

    chosen: list[int] = []
    preferred_included: list[int] = []

    for event_idx in _as_int_list(preferred_events):
        if event_idx in ranked_events and event_idx not in chosen:
            chosen.append(int(event_idx))
            preferred_included.append(int(event_idx))
        if len(chosen) >= max_targets:
            break

    for event_idx in ranked_events:
        if event_idx not in chosen:
            chosen.append(int(event_idx))
        if len(chosen) >= max_targets:
            break

    mode = f"preferred+topk({max_targets})" if preferred_included else f"topk_selected({max_targets})"
    return chosen, mode


def align_case_to_reference(
    compare_row: pd.Series | None,
    reference_candidate_eidx: list[int],
) -> tuple[list[int], list[float], bool]:
    if compare_row is None:
        return [], [0.0 for _ in reference_candidate_eidx], False

    own_candidate_eidx = _as_int_list(compare_row.get("candidate_eidx"))
    if not own_candidate_eidx:
        own_candidate_eidx = list(reference_candidate_eidx)

    own_selected_eidx = _as_int_list(compare_row.get("selected_eidx"))
    own_selected_set = set(own_selected_eidx)

    own_importance_edges = _as_float_list(compare_row.get("importance_edges"))
    used_binary_fallback = False

    # Prefer explicit per-candidate scores whenever available.
    if len(own_importance_edges) == len(own_candidate_eidx):
        importance_by_eid = {
            int(eid): float(score)
            for eid, score in zip(own_candidate_eidx, own_importance_edges)
        }
    # Some runs provide scores for selected edges only.
    elif own_candidate_eidx and own_importance_edges and len(own_importance_edges) == len(own_selected_eidx):
        score_by_selected = {
            int(eid): float(score)
            for eid, score in zip(own_selected_eidx, own_importance_edges)
        }
        importance_by_eid = {
            int(eid): float(score_by_selected.get(int(eid), 0.0))
            for eid in own_candidate_eidx
        }
    else:
        # Last-resort fallback: binary selected mask.
        used_binary_fallback = True
        importance_by_eid = {
            int(eid): 1.0 if int(eid) in own_selected_set else 0.0
            for eid in own_candidate_eidx
        }

    own_selected_set = {int(eid) for eid in own_selected_eidx}

    aligned_selected_eidx = [
        int(eid) for eid in reference_candidate_eidx if int(eid) in own_selected_set
    ]
    aligned_importance_edges = [
        float(
            importance_by_eid.get(
                int(eid),
                (1.0 if int(eid) in own_selected_set else 0.0) if used_binary_fallback else 0.0,
            )
        )
        for eid in reference_candidate_eidx
    ]
    return aligned_selected_eidx, aligned_importance_edges, True


def extract_native_case_from_row(
    compare_row: pd.Series | None,
) -> tuple[list[int], list[int], list[float], bool]:
    """Return a case in the explainer's native candidate order for one target."""
    if compare_row is None:
        return [], [], [], False

    candidate_eidx = _as_int_list(compare_row.get("candidate_eidx"))
    selected_eidx = _as_int_list(compare_row.get("selected_eidx"))
    selected_set = {int(eid) for eid in selected_eidx}

    if not candidate_eidx and selected_eidx:
        candidate_eidx = list(selected_eidx)

    importance_edges = _as_float_list(compare_row.get("importance_edges"))

    # Defensive fallback: build binary importance in candidate order.
    if len(importance_edges) != len(candidate_eidx):
        if candidate_eidx and importance_edges and len(importance_edges) == len(selected_eidx):
            # Some runs provide scores for selected edges only.
            score_by_selected = {
                int(eid): float(score)
                for eid, score in zip(selected_eidx, importance_edges)
            }
            importance_edges = [
                float(score_by_selected.get(int(eid), 0.0))
                for eid in candidate_eidx
            ]
        else:
            importance_edges = [
                1.0 if int(eid) in selected_set else 0.0
                for eid in candidate_eidx
            ]

    return candidate_eidx, selected_eidx, importance_edges, True


def _normalize_sparsity_levels(levels: Sequence[float] | None) -> list[float]:
    if levels is None:
        return [round(0.05 * i, 2) for i in range(21)]
    out: list[float] = []
    for raw in levels:
        try:
            val = float(raw)
        except Exception:
            continue
        if not np.isfinite(val):
            continue
        out.append(float(max(0.0, min(1.0, val))))
    if not out:
        return [round(0.05 * i, 2) for i in range(21)]
    return sorted(set(out))


def _ranking_from_importance(candidate_eidx: list[int], importance_edges: list[float]) -> list[int]:
    if not candidate_eidx:
        return []
    n = min(len(candidate_eidx), len(importance_edges))
    if n <= 0:
        return list(candidate_eidx)
    order = sorted(
        range(n),
        key=lambda i: (
            -abs(float(importance_edges[i])),
            -float(importance_edges[i]),
            -int(candidate_eidx[i]),
            i,
        ),
    )
    ranked = [int(candidate_eidx[i]) for i in order]
    if n < len(candidate_eidx):
        ranked.extend([int(e) for e in candidate_eidx[n:]])
    return ranked


def _selected_by_ranking_for_levels(
    *,
    ranked_eidx: list[int],
    levels: Sequence[float],
    ensure_min_one: bool = True,
) -> tuple[list[list[int]], list[int]]:
    n = len(ranked_eidx)
    selected_levels: list[list[int]] = []
    ks: list[int] = []
    for level in levels:
        k = int(np.floor(float(level) * float(n)))
        if ensure_min_one and n > 0:
            k = k + 1
        k = max(0, min(int(k), int(n)))
        ks.append(int(k))
        selected_levels.append([int(e) for e in ranked_eidx[:k]])
    return selected_levels, ks


def _load_mcts_node_info(path: Path) -> list[dict[str, Any]]:
    try:
        import torch
    except Exception:
        return []

    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return []
    if not isinstance(payload, Mapping):
        return []

    rows = payload.get("saved_MCTSInfo_list")
    if not isinstance(rows, list):
        return []

    out: list[dict[str, Any]] = []
    for item in rows:
        if isinstance(item, Mapping):
            out.append(dict(item))
    return out


def _best_mcts_selection_by_levels(
    *,
    mcts_nodes: Sequence[Mapping[str, Any]],
    levels: Sequence[float],
    candidate_eidx: Sequence[int],
) -> tuple[list[list[int]], list[float]]:
    if not mcts_nodes:
        return [], []

    cand_len = max(1, len([int(e) for e in candidate_eidx]))
    parsed: list[tuple[float, float, list[int]]] = []
    for node in mcts_nodes:
        coalition = _as_int_list(node.get("coalition"))
        reward_raw = node.get("P")
        sparsity_raw = node.get("Sparsity")
        try:
            reward = float(reward_raw)
        except Exception:
            continue
        if not np.isfinite(reward):
            continue
        if sparsity_raw is None:
            sparsity = float(len(coalition)) / float(cand_len)
        else:
            try:
                sparsity = float(sparsity_raw)
            except Exception:
                sparsity = float(len(coalition)) / float(cand_len)
        if not np.isfinite(sparsity):
            continue
        parsed.append((float(max(0.0, min(1.0, sparsity))), reward, coalition))

    if not parsed:
        return [], []

    parsed.sort(key=lambda x: x[0])
    running_best_reward = -np.inf
    running_best_coalition: list[int] = []
    sparsity_sorted: list[float] = []
    reward_best_sorted: list[float] = []
    coalition_best_sorted: list[list[int]] = []
    for sparsity, reward, coalition in parsed:
        if float(reward) >= float(running_best_reward):
            running_best_reward = float(reward)
            running_best_coalition = [int(e) for e in coalition]
        sparsity_sorted.append(float(sparsity))
        reward_best_sorted.append(float(running_best_reward))
        coalition_best_sorted.append([int(e) for e in running_best_coalition])

    selected_levels: list[list[int]] = []
    reward_levels: list[float] = []
    sparsity_arr = np.asarray(sparsity_sorted, dtype=float)
    for level in levels:
        eligible = np.where(sparsity_arr <= float(level))[0]
        pos = int(eligible.max()) if eligible.size > 0 else 0
        selected_levels.append([int(e) for e in coalition_best_sorted[pos]])
        reward_levels.append(float(reward_best_sorted[pos]))
    return selected_levels, reward_levels


def sparsity_selection_for_row(
    *,
    compare_row: pd.Series | Mapping[str, Any] | None,
    levels: Sequence[float] | None = None,
    dataset_name: str | None = None,
    model_name: str | None = None,
    target_event: int | None = None,
    mcts_saved_dir: Path | None = None,
    threshold_num: int = 20,
    ensure_min_one: bool = True,
) -> dict[str, Any]:
    """Resolve selected event sets at different sparsity levels.

    Resolution order:
    1) MCTS node-info artifact (`*.pt`) if available.
    2) Deterministic ranking from importance scores (top-k over levels).
    """
    row_obj: Mapping[str, Any] | None
    if compare_row is None:
        row_obj = None
    elif isinstance(compare_row, pd.Series):
        row_obj = compare_row.to_dict()
    elif isinstance(compare_row, Mapping):
        row_obj = compare_row
    else:
        row_obj = None

    use_levels = _normalize_sparsity_levels(levels)
    if row_obj is None:
        return {
            "levels": use_levels,
            "candidate_eidx": [],
            "selected_by_level": [[] for _ in use_levels],
            "k_by_level": [0 for _ in use_levels],
            "source": "missing_row",
            "mcts_node_info_path": None,
            "reward_by_level": [],
            "ranked_eidx": [],
        }

    case_series = pd.Series(dict(row_obj))
    candidate_eidx, selected_eidx, importance_edges, _ = extract_native_case_from_row(case_series)
    if not candidate_eidx and selected_eidx:
        candidate_eidx = list(selected_eidx)

    # 1) Prefer exact MCTS node selections when node info exists.
    node_info_path: Path | None = None
    if mcts_saved_dir is not None and dataset_name and model_name and target_event is not None:
        node_info_path = find_mcts_node_info_file(
            dataset_name=str(dataset_name),
            model_name=str(model_name),
            target_event=int(target_event),
            mcts_saved_dir=Path(mcts_saved_dir),
            threshold_num=int(threshold_num),
        )
    if node_info_path is None and row_obj.get("mcts_node_info_path"):
        possible = Path(str(row_obj.get("mcts_node_info_path")))
        if possible.exists():
            node_info_path = possible

    if node_info_path is not None:
        mcts_nodes = _load_mcts_node_info(node_info_path)
        selected_levels_mcts, reward_levels = _best_mcts_selection_by_levels(
            mcts_nodes=mcts_nodes,
            levels=use_levels,
            candidate_eidx=candidate_eidx,
        )
        if selected_levels_mcts:
            return {
                "levels": use_levels,
                "candidate_eidx": [int(e) for e in candidate_eidx],
                "selected_by_level": [[int(e) for e in selected] for selected in selected_levels_mcts],
                "k_by_level": [int(len(selected)) for selected in selected_levels_mcts],
                "source": "mcts_node_info",
                "mcts_node_info_path": str(node_info_path),
                "reward_by_level": [float(v) for v in reward_levels],
                "ranked_eidx": _ranking_from_importance(candidate_eidx, importance_edges),
            }

    # 2) Ranking fallback for runs where node-level MCTS artifacts are unavailable.
    ranked_eidx = _ranking_from_importance(candidate_eidx, importance_edges)
    selected_levels, ks = _selected_by_ranking_for_levels(
        ranked_eidx=ranked_eidx,
        levels=use_levels,
        ensure_min_one=bool(ensure_min_one),
    )
    source = "importance_ranking"
    if set({round(float(v), 9) for v in importance_edges}).issubset({0.0, 1.0}):
        source = "importance_ranking_binary"

    return {
        "levels": use_levels,
        "candidate_eidx": [int(e) for e in candidate_eidx],
        "selected_by_level": [[int(e) for e in selected] for selected in selected_levels],
        "k_by_level": [int(v) for v in ks],
        "source": source,
        "mcts_node_info_path": str(node_info_path) if node_info_path is not None else None,
        "reward_by_level": [],
        "ranked_eidx": [int(e) for e in ranked_eidx],
    }


def sparsity_selection_table(
    *,
    dataset_name: str,
    model_name: str,
    target_event: int,
    compare_row: pd.Series | Mapping[str, Any] | None,
    levels: Sequence[float] | None = None,
    mcts_saved_dir: Path | None = None,
    threshold_num: int = 20,
    ensure_min_one: bool = True,
) -> pd.DataFrame:
    """Tabular view of per-sparsity selected event ids for one target case."""
    payload = sparsity_selection_for_row(
        compare_row=compare_row,
        levels=levels,
        dataset_name=dataset_name,
        model_name=model_name,
        target_event=int(target_event),
        mcts_saved_dir=mcts_saved_dir,
        threshold_num=int(threshold_num),
        ensure_min_one=bool(ensure_min_one),
    )
    rows: list[dict[str, Any]] = []
    levels_use = [float(v) for v in payload.get("levels", [])]
    selected_levels = payload.get("selected_by_level", [])
    ks = payload.get("k_by_level", [])
    rewards = payload.get("reward_by_level", [])
    for idx, level in enumerate(levels_use):
        selected = selected_levels[idx] if idx < len(selected_levels) else []
        rows.append(
            {
                "dataset": str(dataset_name),
                "model": str(model_name),
                "target_event": int(target_event),
                "sparsity": float(level),
                "k_selected": int(ks[idx]) if idx < len(ks) else int(len(selected)),
                "selected_eidx": [int(e) for e in selected],
                "selection_source": str(payload.get("source", "unknown")),
                "mcts_node_info_path": payload.get("mcts_node_info_path"),
                "reward_best": float(rewards[idx]) if idx < len(rewards) else np.nan,
                "candidate_count": int(len(payload.get("candidate_eidx", []))),
            }
        )
    return pd.DataFrame(rows)


def native_case_for_target(
    compare_table: pd.DataFrame,
    target_event: int,
) -> dict[str, Any]:
    """Build a normalized dictionary for one explainer/target in native order."""
    match = compare_table.loc[compare_table["event_idx"] == int(target_event)]
    compare_row = match.iloc[0] if not match.empty else None

    candidate_eidx, selected_eidx, importance_edges, source_target_available = extract_native_case_from_row(compare_row)
    unique_vals = sorted({round(float(v), 6) for v in importance_edges})
    is_binary = set(unique_vals).issubset({0.0, 1.0})

    return {
        "target_event": int(target_event),
        "candidate_eidx": list(candidate_eidx),
        "selected_eidx": list(selected_eidx),
        "importance_edges": [float(v) for v in importance_edges],
        "importance_is_binary": bool(is_binary),
        "importance_unique_values": unique_vals,
        "num_candidates": int(len(candidate_eidx)),
        "num_selected": int(len(selected_eidx)),
        "source_target_available": bool(source_target_available),
    }


def _normalize_importance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.clip(values, 0.0, None)
    if values.size == 0:
        return values
    max_value = float(values.max())
    if max_value <= 0.0:
        return np.zeros_like(values)
    return values / max_value


def _load_mcts_recorder(path_str: str | None) -> pd.DataFrame | None:
    if not path_str:
        return None

    path = Path(path_str)
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    required_cols = {"rollout", "best_reward", "num_states"}
    if not required_cols.issubset(df.columns):
        return None

    return df.sort_values("rollout").reset_index(drop=True)


def discover_explainer_keys_from_notebooks(notebooks_dir: Path) -> list[str]:
    if not notebooks_dir.exists():
        return []

    keys: list[str] = []
    pattern = re.compile(r"^\d+_(.+)\.ipynb$")
    for path in sorted(notebooks_dir.glob("*.ipynb")):
        match = pattern.match(path.name)
        if not match:
            continue
        key = str(match.group(1)).strip().lower()
        if key and key not in keys:
            keys.append(key)
    return keys


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    slug = slug.strip("_")
    return slug or "unknown"


def case_plot_paths(
    output_root: Path,
    dataset_name: str,
    model_name: str,
    target_event: int,
    explainer_key: str,
) -> dict[str, Path]:
    """Deterministic plot paths grouped by dataset/target for easy browsing."""
    dataset_slug = _safe_slug(dataset_name)
    model_slug = _safe_slug(model_name)
    explainer_slug = _safe_slug(explainer_key)
    target_id = int(target_event)

    case_dir = Path(output_root) / dataset_slug / model_slug / f"target_{target_id:05d}"
    case_dir.mkdir(parents=True, exist_ok=True)

    return {
        "aligned": case_dir / f"{explainer_slug}_aligned.png",
        "native": case_dir / f"{explainer_slug}_native.png",
        "split": case_dir / f"{explainer_slug}_split.png",
        "all_explainers_split": case_dir / "all_views.png",
    }


def clear_dataset_plot_outputs(output_root: Path, dataset_name: str, model_name: str | None = None) -> int:
    """Remove old plot exports for one dataset/model so reruns start clean."""
    dataset_slug = _safe_slug(dataset_name)
    output_root = Path(output_root)
    dataset_dir = output_root / dataset_slug
    if model_name is not None:
        dataset_dir = dataset_dir / _safe_slug(model_name)
    file_suffixes = ("png", "pdf")
    if not dataset_dir.exists():
        removed = 0
    else:
        removed = 0
        for suffix in file_suffixes:
            for path in dataset_dir.rglob(f"*.{suffix}"):
                try:
                    path.unlink()
                    removed += 1
                except Exception:
                    continue

        # Prune empty directories from deepest to shallowest.
        for path in sorted(dataset_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if not path.is_dir():
                continue
            try:
                path.rmdir()
            except OSError:
                continue

    # Also remove legacy flat filenames from older notebook versions.
    legacy_prefix = f"{dataset_slug}_"
    for suffix in file_suffixes:
        for path in output_root.glob(f"{legacy_prefix}*.{suffix}"):
            try:
                path.unlink()
                removed += 1
            except Exception:
                continue

    return removed
