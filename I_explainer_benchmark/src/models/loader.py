from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from .adapters.tg_model_adapter import TemporalGNNModelAdapter
from ..core.constants import ASSET_ROOT, REPO_ROOT, TGN_MODELS_ROOT
from ..data.io import load_processed_dataset

# Vendored TGN uses legacy absolute imports like `utils.*` and `modules.*`.
# Ensure submodules/models/tgn is importable as a top-level package root.
if str(ASSET_ROOT) not in sys.path:
    sys.path.insert(0, str(ASSET_ROOT))
if str(TGN_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(TGN_MODELS_ROOT))
_TGN_VENDOR_ROOT = (ASSET_ROOT / "submodules" / "models" / "tgn").resolve()
if str(_TGN_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_TGN_VENDOR_ROOT))

try:
    from submodules.models.tgat.graph import NeighborFinder as TGATNeighborFinder
    from submodules.models.tgat.module import TGAN
    from submodules.models.tgn.model.tgn import TGN
    from submodules.models.tgn.utils.data_processing import compute_time_statistics
    from submodules.models.tgn.utils.utils import NeighborFinder as TGNNeighborFinder
except ModuleNotFoundError:
    from tgat.graph import NeighborFinder as TGATNeighborFinder
    from tgat.module import TGAN
    from tgn.model.tgn import TGN
    from tgn.utils.data_processing import compute_time_statistics
    from tgn.utils.utils import NeighborFinder as TGNNeighborFinder


MODEL_CONFIG_DIRS = tuple(
    dict.fromkeys(
        (
            ASSET_ROOT / "configs" / "models",
            REPO_ROOT / "configs" / "models",
        )
    )
)
_TGAT_NON_BIPARTITE_ALLOWLIST = {
    "ucim",
    "uci",
    "uci_messages",
    "uci-messages",
    "ucimessages",
    "ucim_motif",
    "ucim-motif",
    "uci_motif",
    "uci-motif",
}


def _model_config_path(model_type: str, dataset_name: str) -> Path:
    model_type = str(model_type).lower()
    dataset_name = str(dataset_name).lower()
    candidates: list[Path] = []
    for cfg_dir in MODEL_CONFIG_DIRS:
        candidates.extend(
            (
                cfg_dir / f"infer_{model_type}_{dataset_name}.json",
                cfg_dir / f"infer_{model_type}.json",
            )
        )
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not find model infer config. Searched:\n"
        f"{searched}"
    )


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


def _metadata_bipartite_flag(metadata: Optional[dict[str, Any]]) -> Optional[bool]:
    if not isinstance(metadata, dict):
        return None
    if "bipartite" in metadata:
        return bool(metadata["bipartite"])
    config = metadata.get("config")
    if isinstance(config, dict) and "bipartite" in config:
        return bool(config["bipartite"])
    return None


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
    metadata: Optional[dict[str, Any]] = None,
):
    model_type = str(model_type).lower()
    adj_list = _build_adj_list(events)

    if model_type == "tgat":
        dataset_key = str(dataset_name).lower()
        inferred_bipartite = _infer_bipartite(events, dataset_name)
        metadata_bipartite = _metadata_bipartite_flag(metadata)
        allow_non_bipartite = dataset_key in _TGAT_NON_BIPARTITE_ALLOWLIST

        is_tgat_compatible = bool(
            inferred_bipartite
            or (metadata_bipartite is True)
            or allow_non_bipartite
        )
        if not is_tgat_compatible:
            raise ValueError(
                f"TGAT expects bipartite datasets; got dataset={dataset_name!r}. "
                "Use MODEL_TYPE='tgn' for non-bipartite datasets."
            )

        if (not inferred_bipartite) and allow_non_bipartite:
            logging.warning(
                "TGAT on non-bipartite dataset %s: proceeding due to allowlist override.",
                dataset_name,
            )

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
    metadata = bundle.get("metadata")
    model_args = _load_model_args(model_type, dataset_name)
    backbone = _build_backbone(
        model_type,
        dataset_name,
        events,
        node_feats,
        edge_feats,
        device,
        model_args,
        metadata=metadata,
    )
    state_dict = torch.load(ckpt_path, map_location="cpu")
    _ = backbone.load_state_dict(state_dict, strict=False)
    _ = backbone.to(device).eval()
    model = TemporalGNNModelAdapter(backbone, events, device=device)
    return model, events


__all__ = ["load_backbone_model"]
