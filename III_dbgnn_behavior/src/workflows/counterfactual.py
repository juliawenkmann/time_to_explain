from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from explainers.counterfactual import CounterfactualConfig
from explainers.counterfactual_search import CounterfactualSearchResult, find_min_ho_edge_deletions_to_flip
from viz.counterfactual_viz import (
    plot_prob_curve,
    plot_probs_bar,
    prob_curve_by_k,
    prob_curve_keep_only_by_k,
    probs_before_after,
)
from workflows.train import (
    TrainingWorkflowConfig,
    TrainingWorkflowResult,
    run_training_workflow,
)


@dataclass(frozen=True)
class CounterfactualWorkflowConfig:
    training: TrainingWorkflowConfig = field(default_factory=TrainingWorkflowConfig)
    target_node_idx: Optional[int] = None
    max_k: int = 300
    drop_mode: str = "remove"
    k_hops: int = 2
    steps: int = 300
    lr: float = 0.1
    lambda_size: float = 0.02
    lambda_entropy: float = 0.001
    plot_dir: str = "notebooks/plots"


@dataclass
class CounterfactualWorkflowResult:
    training: TrainingWorkflowResult
    target_node_idx: int
    search_result: CounterfactualSearchResult
    p_before: np.ndarray
    p_after: Optional[np.ndarray]
    k_values: Optional[np.ndarray]
    probs: Optional[np.ndarray]
    k_values_keep: Optional[np.ndarray]
    probs_keep: Optional[np.ndarray]
    plot_paths: dict[str, Path]


def _pick_target_node(data) -> int:
    test_nodes = torch.where(data.test_mask)[0]
    if test_nodes.numel() == 0:
        return 0
    if not hasattr(data, "ho_triples"):
        return int(test_nodes[0].item())

    triples = data.ho_triples
    counts = torch.bincount(triples[:, 1], minlength=int(data.num_nodes))
    valid = test_nodes[counts[test_nodes] > 0]
    if valid.numel() > 0:
        return int(valid[0].item())
    return int(test_nodes[0].item())


def run_counterfactual_workflow(cfg: CounterfactualWorkflowConfig) -> CounterfactualWorkflowResult:
    training = run_training_workflow(cfg.training)
    adapter = training.adapter
    data = training.data
    assets = training.assets

    target_node_idx = int(cfg.target_node_idx) if cfg.target_node_idx is not None else _pick_target_node(data)
    cf_cfg = CounterfactualConfig(
        k_hops=int(cfg.k_hops),
        steps=int(cfg.steps),
        lr=float(cfg.lr),
        lambda_size=float(cfg.lambda_size),
        lambda_entropy=float(cfg.lambda_entropy),
        flip_to="non_targeted",
    )
    result = find_min_ho_edge_deletions_to_flip(
        adapter=adapter,
        data=data,
        assets=assets,
        target_node_idx=target_node_idx,
        cfg=cf_cfg,
        seed=int(training.cfg.seed),
        max_k=int(cfg.max_k),
        optimizer="counterfactual",
        drop_mode=str(cfg.drop_mode),
        search_mode="schedule",
        return_best_effort=True,
        best_effort_max_k=min(200, int(cfg.max_k)),
        verbose=False,
    )

    plot_dir = Path(cfg.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, Path] = {}

    p0, p1 = probs_before_after(
        adapter=adapter,
        data=data,
        node_idx=target_node_idx,
        removed_edge_indices=result.removed_edge_indices,
    )
    class_probs_path = plot_dir / f"target_{target_node_idx}_class_probs.pdf"
    plot_probs_bar(
        p_before=p0,
        p_after=p1,
        title=f"Node {target_node_idx}: probs before/after (k={result.n_removed})",
        backend="plotly",
        show=False,
        save_path=class_probs_path,
    )
    plot_paths["class_probs"] = class_probs_path

    k_values, probs = prob_curve_by_k(
        adapter=adapter,
        data=data,
        node_idx=target_node_idx,
        ranked_edge_indices=result.ranked_edge_indices,
        k_step=5,
        max_k=int(cfg.max_k),
        drop_mode=str(cfg.drop_mode),
    )
    if k_values is not None and probs is not None:
        prob_curve_path = plot_dir / f"target_{target_node_idx}_prob_curve.pdf"
        plot_prob_curve(
            k_values=k_values,
            probs=probs,
            vline_k=result.n_removed if result.success else None,
            title=f"Node {target_node_idx}: probability curve",
            backend="plotly",
            show=False,
            save_path=prob_curve_path,
        )
        plot_paths["prob_curve"] = prob_curve_path

    k_values_keep, probs_keep = prob_curve_keep_only_by_k(
        adapter=adapter,
        data=data,
        node_idx=target_node_idx,
        ranked_edge_indices=result.ranked_edge_indices,
        k_step=5,
        max_k=int(cfg.max_k),
        keep_non_ranked=False,
        drop_mode=str(cfg.drop_mode),
    )
    if k_values_keep is not None and probs_keep is not None:
        keep_curve_path = plot_dir / f"target_{target_node_idx}_prob_curve_keep_only.pdf"
        plot_prob_curve(
            k_values=k_values_keep,
            probs=probs_keep,
            title=f"Node {target_node_idx}: prob vs kept edges",
            x_label="Number of higher-order edges kept (k)",
            backend="plotly",
            show=False,
            save_path=keep_curve_path,
        )
        plot_paths["prob_curve_keep_only"] = keep_curve_path

    return CounterfactualWorkflowResult(
        training=training,
        target_node_idx=target_node_idx,
        search_result=result,
        p_before=p0,
        p_after=p1,
        k_values=k_values,
        probs=probs,
        k_values_keep=k_values_keep,
        probs_keep=probs_keep,
        plot_paths=plot_paths,
    )
