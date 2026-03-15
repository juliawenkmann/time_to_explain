from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import torch

from utils import clone_data
from workflows.train import (
    TrainingWorkflowConfig,
    TrainingWorkflowResult,
    run_training_workflow,
)


@dataclass(frozen=True)
class HOEffectWorkflowConfig:
    training: TrainingWorkflowConfig = field(default_factory=TrainingWorkflowConfig)
    target_node_idx: Optional[int] = None
    use_test_mask: bool = False
    max_nodes_eval: Optional[int] = None


@dataclass
class HOEffectWorkflowResult:
    training: TrainingWorkflowResult
    target_node_idx: int
    ho_impact: torch.Tensor
    summary: pd.DataFrame


def _margin_for_class(row: torch.Tensor, cls: int) -> float:
    cls = int(cls)
    if row.numel() <= 1:
        return float(row[cls].item())
    other = torch.cat([row[:cls], row[cls + 1 :]], dim=0)
    return float((row[cls] - other.max()).item())


def _summarize_ablation(logits_full: torch.Tensor, logits_ab: torch.Tensor, node_idx: torch.Tensor) -> dict[str, float]:
    lf = logits_full[node_idx]
    la = logits_ab[node_idx]

    pred_full = lf.argmax(dim=-1)
    pred_ab = la.argmax(dim=-1)
    agree_pct = float((pred_full == pred_ab).float().mean().item() * 100.0)

    p_full = torch.softmax(lf, dim=-1)
    p_ab = torch.softmax(la, dim=-1)
    p_full_orig = p_full[torch.arange(p_full.size(0)), pred_full]
    p_ab_orig = p_ab[torch.arange(p_ab.size(0)), pred_full]
    delta_p_orig_mean = float((p_ab_orig - p_full_orig).mean().item())

    margins_full = []
    margins_ab = []
    for i in range(lf.size(0)):
        cls = int(pred_full[i].item())
        margins_full.append(_margin_for_class(lf[i], cls))
        margins_ab.append(_margin_for_class(la[i], cls))
    delta_margin_mean = float(torch.tensor(margins_ab).mean().item() - torch.tensor(margins_full).mean().item())

    return {
        "agree_pct": agree_pct,
        "delta_p_orig_mean": delta_p_orig_mean,
        "delta_margin_mean": delta_margin_mean,
    }


def _node_index_subset(data, *, use_test_mask: bool, max_nodes_eval: Optional[int]) -> torch.Tensor:
    if use_test_mask:
        nodes = torch.where(data.test_mask)[0]
    else:
        nodes = torch.arange(int(data.num_nodes), device=data.y.device)
    if max_nodes_eval is not None:
        return nodes[: int(max_nodes_eval)]
    return nodes


def run_ho_effect_workflow(cfg: HOEffectWorkflowConfig) -> HOEffectWorkflowResult:
    training = run_training_workflow(cfg.training)
    adapter = training.adapter
    data = training.data

    with torch.no_grad():
        logits_full = adapter.predict_logits(data)

    data_no_ho = clone_data(data)
    data_no_ho.edge_index_higher_order = data.edge_index_higher_order[:, :0]
    data_no_ho.edge_weights_higher_order = data.edge_weights_higher_order[:0]
    with torch.no_grad():
        logits_no_ho = adapter.predict_logits(data_no_ho)

    ho_impact = (logits_full - logits_no_ho).abs().max(dim=1).values
    target_node_idx = int(cfg.target_node_idx) if cfg.target_node_idx is not None else int(ho_impact.argmax().item())

    data_feature_only = clone_data(data)
    for attr in ("edge_weight", "edge_weights", "edge_weights_higher_order"):
        if hasattr(data_feature_only, attr):
            value = getattr(data_feature_only, attr)
            if value is not None:
                setattr(data_feature_only, attr, torch.zeros_like(value))
    with torch.no_grad():
        logits_feature_only = adapter.predict_logits(data_feature_only)

    data_structure_only = clone_data(data)
    if hasattr(data_structure_only, "x") and data_structure_only.x is not None:
        data_structure_only.x = torch.ones_like(data_structure_only.x)
    if hasattr(data_structure_only, "x_h") and data_structure_only.x_h is not None:
        data_structure_only.x_h = torch.ones_like(data_structure_only.x_h)
    with torch.no_grad():
        logits_structure_only = adapter.predict_logits(data_structure_only)

    node_idx = _node_index_subset(
        data,
        use_test_mask=bool(cfg.use_test_mask),
        max_nodes_eval=cfg.max_nodes_eval,
    ).to(device=logits_full.device, dtype=torch.long)

    rows = [
        {
            "ablation": "feature_only (zero edges)",
            **_summarize_ablation(logits_full, logits_feature_only, node_idx),
        },
        {
            "ablation": "structure_only (ones features)",
            **_summarize_ablation(logits_full, logits_structure_only, node_idx),
        },
    ]

    return HOEffectWorkflowResult(
        training=training,
        target_node_idx=target_node_idx,
        ho_impact=ho_impact,
        summary=pd.DataFrame(rows),
    )
