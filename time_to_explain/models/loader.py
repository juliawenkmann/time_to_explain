from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from time_to_explain.adapters import TemporalGNNModelAdapter
from time_to_explain.data import load_processed_dataset
from time_to_explain.utils.constants import REPO_ROOT

from submodules.models.tgat.graph import NeighborFinder as TGATNeighborFinder
from submodules.models.tgat.module import TGAN
from submodules.models.tgn.model.tgn import TGN
from submodules.models.tgn.utils.data_processing import compute_time_statistics
from submodules.models.tgn.utils.utils import NeighborFinder as TGNNeighborFinder


MODEL_CONFIGS_DIR = REPO_ROOT / "configs" / "models"


def _model_config_path(model_type: str, dataset_name: str) -> Path:
    model_type = str(model_type).lower()
    dataset_name = str(dataset_name).lower()
    candidates = [
        MODEL_CONFIGS_DIR / f"infer_{model_type}_{dataset_name}.json",
        MODEL_CONFIGS_DIR / f"infer_{model_type}.json",
    ]
    return next(p for p in candidates if p.exists())


def _load_model_args(model_type: str, dataset_name: str) -> dict:
    cfg_path = _model_config_path(model_type, dataset_name)
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    return dict(config.get("args") or {})


def _infer_bipartite(events, dataset_name: str) -> bool:
    u_min, u_max = int(events["u"].min()), int(events["u"].max())
    i_min, i_max = int(events["i"].min()), int(events["i"].max())
    is_bipartite = i_min > u_max or u_min > i_max
    if str(dataset_name).lower() in {"stick_figure", "sticky_hips"}:
        is_bipartite = False
    return is_bipartite


def _build_adj_list(events):
    u = events["u"].to_numpy(dtype=int)
    v = events["i"].to_numpy(dtype=int)
    ts = events["ts"].to_numpy(dtype=float)
    if "e_idx" in events.columns:
        e_idx = events["e_idx"].to_numpy(dtype=int)
    elif "idx" in events.columns:
        e_idx = events["idx"].to_numpy(dtype=int)
    else:
        e_idx = np.arange(1, len(events) + 1, dtype=int)
    max_node = int(max(u.max(), v.max()))
    adj_list = [[] for _ in range(max_node + 1)]
    for src, dst, t, e in zip(u, v, ts, e_idx):
        adj_list[int(src)].append((int(dst), int(e), float(t)))
        adj_list[int(dst)].append((int(src), int(e), float(t)))
    return adj_list


def _build_backbone(
    model_type: str,
    dataset_name: str,
    events,
    node_feats,
    edge_feats,
    device,
    model_args: dict,
):
    model_type = str(model_type).lower()
    adj_list = _build_adj_list(events)

    if model_type == "tgat":
        if not _infer_bipartite(events, dataset_name):
            raise ValueError("TGAT expects bipartite datasets; set MODEL_TYPE='tgn' for stick_figure.")
        ngh_finder = TGATNeighborFinder(adj_list, uniform=False)
        return TGAN(ngh_finder, node_feats, edge_feats, device=device, **model_args)

    if model_type == "tgn":
        m_src, s_src, m_dst, s_dst = compute_time_statistics(
            events.u.values, events.i.values, events.ts.values
        )
        ngh_finder = TGNNeighborFinder(adj_list, uniform=False)
        return TGN(
            ngh_finder,
            node_feats,
            edge_feats,
            device=device,
            mean_time_shift_src=m_src,
            std_time_shift_src=s_src,
            mean_time_shift_dst=m_dst,
            std_time_shift_dst=s_dst,
            **model_args,
        )

    raise NotImplementedError(model_type)


def load_backbone_model(
    *,
    model_type: str,
    dataset_name: str,
    ckpt_path: str | Path,
    device,
):
    bundle = load_processed_dataset(dataset_name)
    events = bundle["interactions"]
    edge_feats = bundle.get("edge_features")
    node_feats = bundle.get("node_features")
    model_args = _load_model_args(model_type, dataset_name)
    backbone = _build_backbone(
        model_type,
        dataset_name,
        events,
        node_feats,
        edge_feats,
        device,
        model_args,
    )
    state_dict = torch.load(ckpt_path, map_location="cpu")
    _ = backbone.load_state_dict(state_dict, strict=False)
    _ = backbone.to(device).eval()
    model = TemporalGNNModelAdapter(backbone, events, device=device)
    return model, events
