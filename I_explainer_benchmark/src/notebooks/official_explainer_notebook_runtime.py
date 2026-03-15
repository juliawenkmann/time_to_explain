from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..models.adapters.tg_model_adapter import TemporalGNNModelAdapter
from .notebook_runtime_common import event_triplet_from_events


@dataclass(frozen=True)
class CodyPreparedArtifacts:
    cody_events: pd.DataFrame
    edge_features_for_cody: np.ndarray
    model_event_ids: np.ndarray
    cody_dataset: Any
    num_hops: int
    tgnn_wrapper: Any
    event_id_offset: int


@dataclass(frozen=True)
class CounterfactualRecordArtifacts:
    rows: list[dict[str, object]]
    result_records: list[dict[str, object]]
    results_df: pd.DataFrame


def prepare_cody_runtime(
    *,
    events: pd.DataFrame,
    edge_features: np.ndarray,
    node_features: np.ndarray,
    dataset_name: str,
    backbone: Any,
    device: Any,
    dataset_cls: Any,
    tgnn_wrapper_cls: Any,
) -> CodyPreparedArtifacts:
    cody_events = pd.DataFrame(
        {
            "user_id": events["u"].to_numpy(dtype=int),
            "item_id": events["i"].to_numpy(dtype=int),
            "timestamp": events["ts"].to_numpy(dtype=float),
            "state_label": (
                events["label"].to_numpy(dtype=float)
                if "label" in events.columns
                else np.zeros(len(events), dtype=float)
            ),
            "idx": np.arange(len(events), dtype=int),
        }
    )

    if edge_features.shape[0] == len(events) + 1:
        edge_features_for_cody = edge_features[1:]
    elif edge_features.shape[0] == len(events):
        edge_features_for_cody = edge_features
    else:
        raise ValueError(
            f"Unexpected edge feature shape {edge_features.shape}; expected len(events) or len(events)+1 rows."
        )

    if "e_idx" in events.columns:
        model_event_ids = events["e_idx"].to_numpy(dtype=int)
    elif "idx" in events.columns:
        model_event_ids = events["idx"].to_numpy(dtype=int)
    else:
        model_event_ids = np.arange(1, len(events) + 1, dtype=int)

    cody_dataset = dataset_cls(
        cody_events,
        np.asarray(edge_features_for_cody),
        np.asarray(node_features),
        dataset_name,
        directed=False,
        bipartite=False,
    )

    num_hops = int(getattr(backbone, "num_layers", 2) or 2)
    tgnn_wrapper = tgnn_wrapper_cls(
        model=backbone,
        dataset=cody_dataset,
        num_hops=num_hops,
        model_name=type(backbone).__name__,
        device=device,
        batch_size=256,
        model_event_ids=model_event_ids,
    )
    event_id_offset = int(getattr(tgnn_wrapper, "event_id_offset", 0))

    return CodyPreparedArtifacts(
        cody_events=cody_events,
        edge_features_for_cody=np.asarray(edge_features_for_cody),
        model_event_ids=np.asarray(model_event_ids),
        cody_dataset=cody_dataset,
        num_hops=num_hops,
        tgnn_wrapper=tgnn_wrapper,
        event_id_offset=event_id_offset,
    )


def build_official_counterfactual_records(
    *,
    explainer: Any,
    cody_events: pd.DataFrame,
    target_event_ids: list[int],
    event_id_offset: int,
    backbone: Any,
    anchor_prefix: str,
    run_id: str,
    explainer_name: str,
    candidate_id_column: str,
    row_prefix_fields: Mapping[str, object] | None = None,
    row_suffix_fields: Mapping[str, object] | None = None,
) -> CounterfactualRecordArtifacts:
    rows: list[dict[str, object]] = []
    result_records: list[dict[str, object]] = []
    row_prefix = dict(row_prefix_fields or {})
    row_suffix = dict(row_suffix_fields or {})

    for anchor_idx, event_idx in enumerate(target_event_ids):
        internal_event_id = int(event_idx) - event_id_offset
        if internal_event_id < 0:
            raise ValueError(
                f"event_idx={event_idx} becomes negative internal id ({internal_event_id}) with offset={event_id_offset}."
            )

        candidate_subgraph = explainer.subgraph_generator.get_fixed_size_k_hop_temporal_subgraph(
            num_hops=explainer.num_hops,
            base_event_id=internal_event_id,
            size=explainer.candidates_size,
        )
        candidate_internal = (
            candidate_subgraph[candidate_id_column].to_numpy(dtype=int)
            if len(candidate_subgraph) > 0
            else np.array([], dtype=int)
        )
        candidate_event_ids_model = (candidate_internal + event_id_offset).astype(int).tolist()

        t0 = time.perf_counter()
        cf = explainer.explain(internal_event_id)
        elapsed_sec = float(time.perf_counter() - t0)

        cf_event_ids_model = (np.asarray(cf.event_ids, dtype=int) + event_id_offset).astype(int).tolist()

        raw_importances = list(np.asarray(cf.event_importances).tolist()) if len(cf.event_importances) else []
        if raw_importances and raw_importances[0] is not None:
            abs_importances = np.asarray(cf.get_absolute_importances(), dtype=float).tolist()
        else:
            abs_importances = [1.0] * len(cf_event_ids_model)

        importance_by_event = {
            int(eid): float(score)
            for eid, score in zip(cf_event_ids_model, abs_importances)
        }
        candidate_importances = [
            float(importance_by_event.get(int(eid), 0.0))
            for eid in candidate_event_ids_model
        ]

        event_row = cody_events.iloc[internal_event_id]
        target_u = int(event_row["user_id"])
        target_i = int(event_row["item_id"])
        target_ts = float(event_row["timestamp"])

        row = {
            "event_idx": int(event_idx),
            "internal_event_id": int(internal_event_id),
            **row_prefix,
            "candidate_size": int(len(candidate_event_ids_model)),
            **row_suffix,
            "elapsed_sec": elapsed_sec,
            "is_counterfactual": bool(cf.achieves_counterfactual_explanation),
            "original_prediction": float(cf.original_prediction),
            "counterfactual_prediction": float(cf.counterfactual_prediction),
            "cf_event_ids": cf_event_ids_model,
            "cf_event_importances": abs_importances,
            "cf_size": int(len(cf_event_ids_model)),
            "candidate_event_ids": candidate_event_ids_model,
        }
        rows.append(row)

        result_records.append(
            {
                "run_id": str(run_id),
                "anchor_idx": int(anchor_idx),
                "context_fp": f"{anchor_prefix}::{int(event_idx)}",
                "context": {
                    "target": {"event_idx": int(event_idx)},
                    "target_kind": "edge",
                    "window": None,
                    "k_hop": int(explainer.num_hops),
                    "num_neighbors": int(getattr(backbone, "num_neighbors", 20) or 20),
                },
                "result": {
                    "explainer": str(explainer_name),
                    "elapsed_sec": elapsed_sec,
                    "importance_edges": candidate_importances,
                    "importance_nodes": None,
                    "importance_time": None,
                    "extras": {
                        "event_idx": int(event_idx),
                        "candidate_eidx": candidate_event_ids_model,
                        "cf_event_ids": cf_event_ids_model,
                        "original_prediction": float(cf.original_prediction),
                        "counterfactual_prediction": float(cf.counterfactual_prediction),
                        "achieves_counterfactual_explanation": bool(cf.achieves_counterfactual_explanation),
                        "u": target_u,
                        "i": target_i,
                        "ts": target_ts,
                    },
                },
                "metrics": {},
                "metric_details": {},
            }
        )

    return CounterfactualRecordArtifacts(
        rows=rows,
        result_records=result_records,
        results_df=pd.DataFrame(rows),
    )


@dataclass(frozen=True)
class TGNNAggregationArtifacts:
    eval_df_official: pd.DataFrame
    summary_official: pd.DataFrame
    curve_official: pd.DataFrame
    summary: pd.DataFrame
    tab: pd.DataFrame
    summary_path: Path
    curve_path: Path
    official_curve_csv: Path


@dataclass(frozen=True)
class TGNNMetricRecordArtifacts:
    tgnn_rows: list[dict[str, object]]
    tgnn_result_records: list[dict[str, object]]
    tgnn_results_df: pd.DataFrame


def aggregate_official_tgnnexplainer_eval(
    *,
    eval_df: pd.DataFrame,
    target_event_idxs: list[int],
    model: Any,
    events: pd.DataFrame,
    device: Any,
    dataset_name: str,
    model_name: str,
    results_dir: str | Path,
    navigator_type: str,
    use_navigator: bool,
) -> TGNNAggregationArtifacts:
    eval_df_official = eval_df.copy()
    eval_df_official["explainer"] = "tgnnexplainer"

    if "anchor_idx" not in eval_df_official.columns:
        if "event_idx" in eval_df_official.columns and target_event_idxs:
            anchor_map = {int(eid): int(idx) for idx, eid in enumerate(target_event_idxs)}
            eval_df_official["anchor_idx"] = pd.to_numeric(eval_df_official["event_idx"], errors="coerce").map(anchor_map)
        elif "event_idx" in eval_df_official.columns:
            uniq_events = pd.unique(pd.to_numeric(eval_df_official["event_idx"], errors="coerce").dropna())
            anchor_map = {int(eid): int(idx) for idx, eid in enumerate(uniq_events.tolist())}
            eval_df_official["anchor_idx"] = pd.to_numeric(eval_df_official["event_idx"], errors="coerce").map(anchor_map)
        else:
            eval_df_official["anchor_idx"] = np.arange(len(eval_df_official), dtype=int)

    eval_df_official["anchor_idx"] = pd.to_numeric(eval_df_official["anchor_idx"], errors="coerce").astype("Int64")
    eval_df_official["sparsity"] = pd.to_numeric(eval_df_official["sparsity"], errors="coerce")
    eval_df_official["fid_inv"] = pd.to_numeric(eval_df_official["fid_inv"], errors="coerce")
    eval_df_official["fid_inv_best"] = pd.to_numeric(eval_df_official["fid_inv_best"], errors="coerce")

    if "prediction_full" not in eval_df_official.columns:
        if "event_idx" in eval_df_official.columns:
            score_model = TemporalGNNModelAdapter(
                backbone=model,
                events=events,
                device=device,
                return_logit=True,
            )
            event_ids = pd.to_numeric(eval_df_official["event_idx"], errors="coerce").dropna().astype(int).unique().tolist()
            full_score_map: dict[int, float] = {}
            for event_id in event_ids:
                try:
                    full_score_map[int(event_id)] = float(
                        score_model._score({"event_idx": int(event_id)}, subgraph=None, edge_mask=None)
                    )
                except Exception:
                    full_score_map[int(event_id)] = float("nan")
            eval_df_official["prediction_full"] = pd.to_numeric(eval_df_official["event_idx"], errors="coerce").map(full_score_map)
        else:
            eval_df_official["prediction_full"] = np.nan

    eval_df_official["prediction_full"] = pd.to_numeric(eval_df_official["prediction_full"], errors="coerce")
    eval_df_official = eval_df_official.dropna(subset=["anchor_idx", "sparsity", "fid_inv", "fid_inv_best"]).copy()
    eval_df_official["anchor_idx"] = eval_df_official["anchor_idx"].astype(int)
    eval_df_official = eval_df_official.sort_values(["explainer", "anchor_idx", "sparsity"]).reset_index(drop=True)

    def aggregate_eval(eval_curve_df: pd.DataFrame, *, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        summary_rows = []
        curve_tables = []

        for explainer_name, g in eval_curve_df.groupby("explainer", as_index=False):
            tab_local = g.groupby("sparsity", as_index=True).mean(numeric_only=True).sort_index()
            x = tab_local.index.to_numpy(dtype=float)
            y_raw = tab_local["fid_inv"].to_numpy(dtype=float)
            y_best = tab_local["fid_inv_best"].to_numpy(dtype=float)

            best_fid = float(np.max(y_best))
            best_fid_sparsity = float(x[int(np.argmax(y_best))])
            aufsc = float(np.trapz(y_best, x))

            flip_success_rate = float("nan")
            first_flip_sparsity = float("nan")
            if "prediction_full" in g.columns and "fid_inv" in g.columns:
                success_vals = []
                first_vals = []
                for _, g_anchor in g.groupby("anchor_idx", as_index=False):
                    g_anchor = g_anchor.sort_values("sparsity")
                    full_arr = pd.to_numeric(g_anchor["prediction_full"], errors="coerce").to_numpy(dtype=float)
                    full_arr = full_arr[np.isfinite(full_arr)]
                    if full_arr.size == 0:
                        continue
                    full_score = float(full_arr[0])
                    spars_arr = pd.to_numeric(g_anchor["sparsity"], errors="coerce").to_numpy(dtype=float)
                    fid_arr = pd.to_numeric(g_anchor["fid_inv"], errors="coerce").to_numpy(dtype=float)
                    keep_arr = full_score + fid_arr if full_score >= 0.0 else full_score - fid_arr

                    base_label = bool(full_score >= 0.0)
                    flip_mask = np.isfinite(spars_arr) & np.isfinite(keep_arr) & ((keep_arr >= 0.0) != base_label)
                    success = bool(np.any(flip_mask))
                    success_vals.append(1.0 if success else 0.0)
                    if success:
                        first_vals.append(float(spars_arr[np.where(flip_mask)[0][0]]))

                if success_vals:
                    flip_success_rate = float(np.mean(np.asarray(success_vals, dtype=float)))
                    first_flip_sparsity = float(np.mean(np.asarray(first_vals, dtype=float))) if first_vals else 1.0

            best_fid_raw = float(np.max(y_raw))
            best_fid_raw_sparsity = float(x[int(np.argmax(y_raw))])
            mask_lt1 = x < 1.0
            best_fid_raw_lt1 = float(np.max(y_raw[mask_lt1])) if mask_lt1.any() else np.nan
            best_fid_raw_lt1_sparsity = (
                float(x[mask_lt1][int(np.argmax(y_raw[mask_lt1]))]) if mask_lt1.any() else np.nan
            )

            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "explainer": str(explainer_name),
                    "navigator": navigator_type if use_navigator else "none",
                    "n_events": int(g["anchor_idx"].nunique()),
                    "variant": variant,
                    "best_fid": best_fid,
                    "best_fid_sparsity": best_fid_sparsity,
                    "aufsc": aufsc,
                    "best_minus_aufsc": float(best_fid - aufsc),
                    "fid_best_flat_curve": bool(np.allclose(y_best, y_best[0], rtol=1e-12, atol=1e-12)),
                    "best_fid_raw": best_fid_raw,
                    "best_fid_raw_sparsity": best_fid_raw_sparsity,
                    "best_fid_raw_lt1": best_fid_raw_lt1,
                    "best_fid_raw_lt1_sparsity": best_fid_raw_lt1_sparsity,
                    "flip_success_rate": float(flip_success_rate),
                    "first_flip_sparsity": float(first_flip_sparsity),
                }
            )

            tab_out = tab_local.reset_index().copy()
            tab_out.insert(0, "explainer", str(explainer_name))
            tab_out.insert(1, "variant", variant)
            curve_tables.append(tab_out)

        summary_local = pd.DataFrame(summary_rows).sort_values(["explainer", "variant"]).reset_index(drop=True)
        curve_local = pd.concat(curve_tables, axis=0, ignore_index=True)
        return summary_local, curve_local

    summary_official, curve_official = aggregate_eval(eval_df_official, variant="official")
    summary = summary_official.copy()
    tab = (
        curve_official.groupby("sparsity", as_index=True)[["fid_inv", "fid_inv_best"]]
        .mean(numeric_only=True)
        .sort_index()
    )

    results_dir = Path(results_dir).expanduser().resolve()
    summary_path = results_dir / f"{model_name}_{dataset_name}_official_tgnnexplainer_aufsc_bestfid_summary.csv"
    curve_path = results_dir / f"{model_name}_{dataset_name}_official_tgnnexplainer_fid_curve.csv"
    summary.to_csv(summary_path, index=False)
    curve_official.to_csv(curve_path, index=False)

    return TGNNAggregationArtifacts(
        eval_df_official=eval_df_official,
        summary_official=summary_official,
        curve_official=curve_official,
        summary=summary,
        tab=tab,
        summary_path=summary_path,
        curve_path=curve_path,
        official_curve_csv=curve_path,
    )


def build_tgnn_metric_records(
    *,
    explain_results: list[Any],
    target_event_idxs: list[int],
    events: pd.DataFrame,
    model: Any,
) -> TGNNMetricRecordArtifacts:
    def _candidate_from_tree_nodes(tree_nodes) -> list[int]:
        if not tree_nodes:
            return []
        coalitions = [list(getattr(node, "coalition", []) or []) for node in tree_nodes]
        if not coalitions:
            return []
        root_like = max(coalitions, key=len)
        return [int(e) for e in sorted(set(root_like))]

    tgnn_rows: list[dict[str, object]] = []
    tgnn_result_records: list[dict[str, object]] = []

    for anchor_idx, (event_idx, explain_result) in enumerate(zip(target_event_idxs, explain_results)):
        tree_nodes, tree_node_x = explain_result

        candidate_event_ids_model = _candidate_from_tree_nodes(tree_nodes)
        selected_event_ids_model = [int(e) for e in sorted(set(list(getattr(tree_node_x, "coalition", []) or [])))]
        selected_set = set(selected_event_ids_model)

        candidate_importances = [1.0 if int(eid) in selected_set else 0.0 for eid in candidate_event_ids_model]
        target_u, target_i, target_ts = event_triplet_from_events(events, int(event_idx))

        tgnn_rows.append(
            {
                "event_idx": int(event_idx),
                "candidate_size": int(len(candidate_event_ids_model)),
                "selected_size": int(len(selected_event_ids_model)),
                "selected_event_ids": selected_event_ids_model,
                "candidate_event_ids": candidate_event_ids_model,
            }
        )

        tgnn_result_records.append(
            {
                "run_id": "official_tgnnexplainer",
                "anchor_idx": int(anchor_idx),
                "context_fp": f"official_tgnnexplainer::{int(event_idx)}",
                "context": {
                    "target": {"event_idx": int(event_idx)},
                    "target_kind": "edge",
                    "window": None,
                    "k_hop": int(getattr(model, "num_layers", 2) or 2),
                    "num_neighbors": int(getattr(model, "num_neighbors", 20) or 20),
                },
                "result": {
                    "explainer": "tgnnexplainer",
                    "elapsed_sec": float("nan"),
                    "importance_edges": candidate_importances,
                    "importance_nodes": None,
                    "importance_time": None,
                    "extras": {
                        "event_idx": int(event_idx),
                        "candidate_eidx": candidate_event_ids_model,
                        "selected_eidx": selected_event_ids_model,
                        "explanation_event_ids": selected_event_ids_model,
                        "u": int(target_u),
                        "i": int(target_i),
                        "ts": float(target_ts),
                    },
                },
                "metrics": {},
                "metric_details": {},
            }
        )

    return TGNNMetricRecordArtifacts(
        tgnn_rows=tgnn_rows,
        tgnn_result_records=tgnn_result_records,
        tgnn_results_df=pd.DataFrame(tgnn_rows),
    )


__all__ = [
    "CodyPreparedArtifacts",
    "CounterfactualRecordArtifacts",
    "TGNNMetricRecordArtifacts",
    "TGNNAggregationArtifacts",
    "aggregate_official_tgnnexplainer_eval",
    "build_official_counterfactual_records",
    "build_tgnn_metric_records",
    "prepare_cody_runtime",
]
