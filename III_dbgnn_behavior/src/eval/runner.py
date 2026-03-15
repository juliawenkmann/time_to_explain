from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch

from config import ExperimentConfig
from data.registry import get_dataset_loader
from explainers.registry import build_explainer
from eval.metrics import (
    default_candidate_mask_from_triples,
    filter_edges,
    filter_edge_weights,
    keep_drop_masks,
    min_k_to_flip,
    ranked_edge_indices,
    target_prob,
)
from eval.shuffle_gt import precision_recall_over_k
from eval.summary import summarize_benchmark
from train import load_or_train
from utils import choose_indices, ensure_dir, get_device, set_seed


def _resolve_gt_label_attr(cfg: ExperimentConfig, data) -> Optional[str]:
    """Pick which ground-truth label attribute to use for GT metrics.

    Priority:
      1) cfg.gt_label_attr if present
      2) gt_stay_label_higher_order (deterministic cluster-stay GT)
      3) gt_label_higher_order (shuffle-time z>thr GT)

    Returns None if no GT labels are available.
    """
    gt_label_attr = getattr(cfg, "gt_label_attr", None)
    if isinstance(gt_label_attr, str) and hasattr(data, gt_label_attr):
        return gt_label_attr
    if hasattr(data, "gt_stay_label_higher_order"):
        return "gt_stay_label_higher_order"
    if hasattr(data, "gt_label_higher_order"):
        return "gt_label_higher_order"
    return None


def run_benchmark(cfg: ExperimentConfig) -> pd.DataFrame:
    """Train/load the model and evaluate explainers on node predictions.

    This benchmark is **transductive**:
    - the graph is fixed (single graph)
    - only node labels are split (train_mask / test_mask)
    - explainers are evaluated by perturbing the graph at inference time (no retraining)

    Returns:
        DataFrame with one row per (explainer, node, frac).
    """
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    # 1) Load data
    dataset_loader = get_dataset_loader(cfg.dataset_name)
    data, assets = dataset_loader(
        device=device,
        num_test=cfg.num_test,
        seed=cfg.seed,
        **dict(getattr(cfg, "dataset_kwargs", {}) or {}),
    )

    # 2) Train/load model checkpoint
    adapter = load_or_train(cfg, data=data, assets=assets, device=device)

    # 3) Pick nodes to explain (test nodes by default)
    nodes = choose_indices(data.test_mask, cfg.n_nodes, seed=cfg.seed)
    if not nodes:
        raise RuntimeError("No nodes found to explain. Is data.test_mask empty?")

    # Baseline logits once for speed
    logits0 = adapter.predict_logits(data)

    # Explain-space (higher-order edges by default for DBGNN)
    space = adapter.explain_space()
    edge_index_full = getattr(data, space.edge_index_attr)
    edge_weight_full = (
        getattr(data, space.edge_weight_attr)
        if space.edge_weight_attr and hasattr(data, space.edge_weight_attr)
        else None
    )
    E = int(edge_index_full.size(1))

    # Ground-truth labels (optional)
    gt_label_attr = _resolve_gt_label_attr(cfg, data)
    y_full = getattr(data, gt_label_attr) if gt_label_attr else None
    if y_full is not None and (not isinstance(y_full, torch.Tensor) or y_full.numel() != E):
        # Shape mismatch: disable GT metrics rather than failing the run.
        y_full = None
        gt_label_attr = None

    records: List[Dict] = []

    for explainer_name in cfg.explainer_names:
        explainer = build_explainer(
            explainer_name,
            seed=cfg.seed,
            **dict(cfg.explainer_kwargs.get(explainer_name, {})),
        )

        for node_idx in nodes:
            node_idx = int(node_idx)

            pred_class = int(logits0[node_idx].argmax().item())
            p0 = target_prob(logits0, node_idx, pred_class)

            # --- Explain once per node ---
            t0 = time.time()
            exp = explainer.explain_node(
                adapter=adapter,
                data=data,
                node_idx=node_idx,
                target_class=pred_class,
            )
            explain_time = time.time() - t0

            # Candidate set: explainer-provided or default (middle==node in ho_triples)
            cand_mask = getattr(exp, "candidate_mask", None)
            if cand_mask is None:
                cand_mask = default_candidate_mask_from_triples(data, E=E, node_idx=node_idx)

            # --- Counterfactual size metric: how many top-ranked edges must be dropped to flip? ---
            # This can be expensive when candidate sets are large, so it's capped by cfg.k_flip_max.
            k_flip = None
            flip_pred = None
            flip_success = 0
            frac_flip = float("nan")
            if int(getattr(cfg, "k_flip_max", 0)) > 0:
                ranked_idx = ranked_edge_indices(exp.edge_score, candidate_mask=cand_mask, descending=True)
                k_flip, flip_pred = min_k_to_flip(
                    adapter=adapter,
                    data=data,
                    node_idx=node_idx,
                    orig_class=pred_class,
                    edge_index_full=edge_index_full,
                    edge_weight_full=edge_weight_full,
                    ranked_edge_idx=ranked_idx,
                    max_k=int(getattr(cfg, "k_flip_max", 0)),
                )
                flip_success = int(k_flip is not None)
                if k_flip is not None and cand_mask is not None:
                    denom = float(cand_mask.sum().item())
                    frac_flip = float(k_flip) / denom if denom > 0 else float("nan")
            else:
                # Disabled: keep columns but fill with NaNs.
                flip_pred = int(pred_class)

            # GT ranking stats (per node; independent of frac)
            gt_ap = float("nan")
            gt_base_rate = float("nan")
            gt_pos = 0
            if y_full is not None and cand_mask is not None:
                y_cand = (y_full[cand_mask] > 0).detach().cpu().numpy().astype(int)
                s_cand = exp.edge_score[cand_mask].detach().cpu().numpy()
                gt_pos = int(y_cand.sum())
                gt_base_rate = float(y_cand.mean()) if y_cand.size > 0 else float("nan")
                if y_cand.size > 0:
                    gt_ap = float(precision_recall_over_k(s_cand, y_cand).ap)

            for frac in cfg.topk_fracs:
                frac = float(frac)

                keep_sel, keep_mask, drop_mask = keep_drop_masks(
                    exp.edge_score,
                    frac,
                    candidate_mask=cand_mask,
                )
                k_sel = int(keep_sel.sum().item())

                # --- Sufficiency graph (keep) ---
                ei_keep = filter_edges(edge_index_full, keep_mask)
                ew_keep = filter_edge_weights(edge_weight_full, keep_mask) if edge_weight_full is not None else None
                data_keep = adapter.clone_with_perturbed_edges(data, ei_keep, new_edge_weight=ew_keep)
                logits_keep = adapter.predict_logits(data_keep)
                pred_keep = int(logits_keep[node_idx].argmax().item())
                p_keep = target_prob(logits_keep, node_idx, pred_class)

                # --- Comprehensiveness graph (drop) ---
                ei_drop = filter_edges(edge_index_full, drop_mask)
                ew_drop = filter_edge_weights(edge_weight_full, drop_mask) if edge_weight_full is not None else None
                data_drop = adapter.clone_with_perturbed_edges(data, ei_drop, new_edge_weight=ew_drop)
                logits_drop = adapter.predict_logits(data_drop)
                pred_drop = int(logits_drop[node_idx].argmax().item())
                p_drop = target_prob(logits_drop, node_idx, pred_class)

                # Simple, checkable fidelity metrics (no logits/margins):
                keep_agree = int(pred_keep == pred_class)
                drop_flip = int(pred_drop != pred_class)

                # GT precision/recall at k (within candidate set)
                gt_prec_at_k = float("nan")
                gt_rec_at_k = float("nan")
                if y_full is not None and cand_mask is not None:
                    tp = float((y_full[keep_sel] > 0).sum().item()) if k_sel > 0 else 0.0
                    gt_prec_at_k = float(tp / max(1, k_sel)) if k_sel > 0 else float("nan")
                    gt_rec_at_k = float(tp / max(1, gt_pos)) if gt_pos > 0 else 0.0

                records.append(
                    {
                        "dataset": cfg.dataset_name,
                        "model": cfg.model_name,
                        "explainer": explainer_name,
                        "node": node_idx,
                        "pred_class": pred_class,
                        "frac": frac,
                        "k": k_sel,
                        "candidate_count": int(cand_mask.sum().item()) if cand_mask is not None else E,
                        "k_flip": (int(k_flip) if k_flip is not None else float("nan")),
                        "frac_flip": float(frac_flip),
                        "flip_pred": (int(flip_pred) if flip_pred is not None else -1),
                        "flip_success": int(flip_success),
                        "p0": float(p0),
                        "p_keep": float(p_keep),
                        "p_drop": float(p_drop),
                        "keep_agree": keep_agree,
                        "drop_flip": drop_flip,
                        "explain_time_s": float(explain_time),
                        "gt_label_attr": gt_label_attr or "",
                        "gt_ap": float(gt_ap),
                        "gt_base_rate": float(gt_base_rate),
                        "gt_pos": int(gt_pos),
                        "gt_precision_at_k": float(gt_prec_at_k),
                        "gt_recall_at_k": float(gt_rec_at_k),
                    }
                )

    df = pd.DataFrame.from_records(records)

    # Optional: save to disk (nice for notebooks)
    out_dir = ensure_dir(Path(cfg.run_dir) / cfg.run_name)
    out_path = out_dir / "benchmark.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Save a compact per-explainer summary (simple + checkable metrics).
    try:
        summary = summarize_benchmark(df)
        summary_path = out_dir / "summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Saved: {summary_path}")
    except Exception as e:
        print(f"Warning: failed to compute summary metrics ({type(e).__name__}: {e})")

    return df
