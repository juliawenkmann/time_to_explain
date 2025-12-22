# time_to_explain/explainers/tempme_tgn_trainer.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import math
import os
import os.path as osp
import random
import sys
import importlib

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------------------
# IMPORTANT: update these imports to match your repo structure.
# They are the same utilities your original TempME script used.
# ---------------------------------------------------------------------------

# NOTE: TempME uses `from utils import ...` internally, so we must ensure the
#       official TempME utils module is visible during import to avoid pulling
#       in our own `time_to_explain.utils` by accident.
_TEMP_ME_BOUND = False
RandEdgeSampler = None
load_subgraph_margin = None
get_item = None
get_item_edge = None
NeighborFinder = None
TempME = None
TempME_TGAT = None


def _bind_official_tempme_modules() -> None:
    global _TEMP_ME_BOUND
    global RandEdgeSampler, load_subgraph_margin, get_item, get_item_edge, NeighborFinder
    global TempME, TempME_TGAT

    if _TEMP_ME_BOUND:
        return

    tempme_root = Path(__file__).resolve().parents[2] / "submodules" / "explainer" / "TempME"
    if str(tempme_root) not in sys.path:
        sys.path.insert(0, str(tempme_root))

    prev_utils = sys.modules.get("utils")
    try:
        import submodules.explainer.TempME.utils as tempme_utils
        sys.modules["utils"] = tempme_utils
        models_mod = importlib.import_module("submodules.explainer.TempME.models")
        models_mod = importlib.reload(models_mod)
        TempME_cls = models_mod.TempME
        TempME_TGAT_cls = models_mod.TempME_TGAT
    finally:
        if prev_utils is not None:
            sys.modules["utils"] = prev_utils
        else:
            sys.modules.pop("utils", None)

    RandEdgeSampler = tempme_utils.RandEdgeSampler
    load_subgraph_margin = tempme_utils.load_subgraph_margin
    get_item = tempme_utils.get_item
    get_item_edge = tempme_utils.get_item_edge
    NeighborFinder = tempme_utils.NeighborFinder
    TempME = TempME_cls
    TempME_TGAT = TempME_TGAT_cls
    _TEMP_ME_BOUND = True


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

_DEGREE_DEFAULTS = {
    "wikipedia": 20,
    "reddit": 20,
    "uci": 30,
    "mooc": 60,
    "enron": 30,
    "canparl": 30,
    "uslegis": 30,
}


@dataclass
class TempMETGNTrainingConfig:
    # Data / base model
    data: str                         # "wikipedia", "reddit", ...
    base_type: str = "tgn"            # "tgn" | "graphmixer" | "tgat"
    device: Optional[str] = "auto"

    # Batching
    bs: int = 500
    test_bs: int = 500

    # Model hyperparams (same as original script)
    n_degree: Optional[int] = None    # if None, use DEGREE_DEFAULTS[data]
    n_epoch: int = 150
    out_dim: int = 40
    hid_dim: int = 64
    temp: float = 0.07
    prior_p: float = 0.3
    lr: float = 1e-3
    drop_out: float = 0.1
    if_attn: bool = True
    if_bern: bool = True
    weight_decay: float = 0.0
    beta: float = 0.5
    lr_decay: float = 0.999
    use_lr_decay: bool = False
    verbose: int = 1                # how often to evaluate
    test_threshold: bool = True
    strict_base_match: bool = False
    mask_ratio: float = 0.1
    min_train: int = 100
    min_val: int = 10
    min_test: int = 10
    log_level: str = "summary"     # "silent" | "summary" | "full"
    show_progress: bool = False

    # Paths (assume same tree as your original script; override if needed)
    root_dir: str = "."             # directory containing "processed/" and "params/"
    gnn_ckpt_path: Optional[str] = None  # path to base TGN ckpt; if None, we don't load it here
    save_model: bool = True
    explainer_save_dir: Optional[str] = None  # default: <root_dir>/params/explainer/<base_type>/

    # Threshold evaluation ratios (same as original TempME code)
    ratios: List[float] = field(
        default_factory=lambda: [
            0.01,
            0.02,
            0.04,
            0.06,
            0.08,
            0.10,
            0.12,
            0.14,
            0.16,
            0.18,
            0.20,
            0.22,
            0.24,
            0.26,
            0.28,
            0.30,
        ]
    )

    def __post_init__(self):
        self.base_type = str(self.base_type).lower()
        if self.n_degree is None:
            self.n_degree = _DEGREE_DEFAULTS.get(self.data, 20)


# ---------------------------------------------------------------------------
# Helper: normalize importance (from original code)
# ---------------------------------------------------------------------------

def norm_imp(imp: torch.Tensor) -> torch.Tensor:
    imp = imp.clone()
    imp[imp < 0] = 0
    imp += 1e-16
    return imp / imp.sum()


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class TempMETGNTrainer:
    """
    Encapsulates the TempME training loop from the original script, but as a
    reusable class you can call from your code instead of `argparse`.

    Usage
    -----
    >>> cfg = TempMETGNTrainingConfig(
    ...     data="wikipedia",
    ...     base_type="tgn",
    ...     root_dir="/path/to/twitter_research",
    ...     device="cuda:0",
    ... )
    >>> base_model = torch.load("/path/to/params/tgnn/tgn_wikipedia.pt").to(cfg.device)
    >>> trainer = TempMETGNTrainer(cfg, base_model)
    >>> explainer = trainer.fit()    # trains TempME and returns the trained module
    >>> torch.save(explainer, "/path/to/params/explainer/tgn/wikipedia.pt")
    """

    def __init__(self, cfg: TempMETGNTrainingConfig, base_model: nn.Module):
        _bind_official_tempme_modules()
        from time_to_explain.adapters.tempme_base_adapter import TempMEBaseAdapter

        self.cfg = cfg
        self.device = self._resolve_device(cfg.device)
        self.root_dir = Path(cfg.root_dir).resolve()
        self.processed_dir = (
            self.root_dir / "processed" if (self.root_dir / "processed").is_dir() else self.root_dir
        )
        if not isinstance(base_model, TempMEBaseAdapter) and not self._looks_like_tempme_base(base_model):
            self._log("[TempME] Wrapping backbone with TempMEBaseAdapter for compatibility.", level="full")
            base_model = TempMEBaseAdapter(base_model)
        self.base_model = base_model.to(self.device)
        self._neighbor_finder_cls = self._select_neighbor_finder_cls(self.base_model)

        if self.cfg.n_degree is None:
            self.cfg.n_degree = _DEGREE_DEFAULTS.get(self.cfg.data, 20)

        if self.cfg.base_type == "tgn" and hasattr(self.base_model, "forbidden_memory_update"):
            self.base_model.forbidden_memory_update = True

        # Where to save explainer checkpoint
        if self.cfg.explainer_save_dir is None:
            self.cfg.explainer_save_dir = osp.join(
                str(self.root_dir), "params", f"explainer/{self.cfg.base_type}/"
            )
        os.makedirs(self.cfg.explainer_save_dir, exist_ok=True)

        # Will be set in fit()
        self.explainer: Optional[nn.Module] = None

        if not self._looks_like_tempme_base(self.base_model):
            msg = (
                "TempME trainer expects a base model compatible with the TempME "
                "submodule (embedding_module.embedding_update). Training will not "
                "match the official implementation unless the TempME base model is used."
            )
            if self.cfg.strict_base_match:
                raise ValueError(msg)
            self._log(f"[TempME][warn] {msg}", level="full")

        self._maybe_load_backbone_checkpoint()

    def _log(self, msg: str, *, level: str = "summary") -> None:
        cfg_level = str(getattr(self.cfg, "log_level", "summary")).lower()
        level = level.lower()
        if cfg_level in {"silent", "none", "off", "0"}:
            return
        if cfg_level in {"summary", "basic"} and level != "summary":
            return
        print(msg)

    def _maybe_load_backbone_checkpoint(self) -> None:
        ckpt_path = self.cfg.gnn_ckpt_path
        if not ckpt_path:
            return
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            self._log(f"[TempME][warn] Base checkpoint not found at {ckpt_path}", level="full")
            return

        target = getattr(self.base_model, "backbone", self.base_model)
        state = torch.load(ckpt_path, map_location="cpu")
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        try:
            target.load_state_dict(state, strict=False)
        except RuntimeError:
            if isinstance(state, dict):
                filtered = {
                    k: v
                    for k, v in state.items()
                    if "memory" not in k and "last_update" not in k
                }
                target.load_state_dict(filtered, strict=False)
            else:
                raise
        self._log(f"[TempME] Loaded base checkpoint: {ckpt_path}", level="full")

    @staticmethod
    def _safe_scores(y_true: torch.Tensor, y_score: torch.Tensor) -> Tuple[Optional[float], Optional[float]]:
        y_true_np = y_true.detach().cpu().numpy().reshape(-1)
        if np.unique(y_true_np).size < 2:
            return None, None
        y_score_np = y_score.detach().cpu().numpy().reshape(-1)
        aps = average_precision_score(y_true_np, y_score_np)
        auc = roc_auc_score(y_true_np, y_score_np)
        return float(aps), float(auc)

    @staticmethod
    def _sanitize_minima(n_events: int, min_train: int, min_val: int, min_test: int) -> Tuple[int, int, int]:
        if n_events <= 0:
            return 0, 0, 0
        min_train = max(1, int(min_train))
        min_val = max(0, int(min_val))
        min_test = max(0, int(min_test))
        if n_events == 1:
            return 1, 0, 0
        min_val = min(min_val, n_events - 1)
        min_test = min(min_test, n_events - 1)
        max_train = n_events - min_val - min_test
        if max_train < 1:
            deficit = 1 - max_train
            reduce_val = min(deficit, min_val)
            min_val -= reduce_val
            deficit -= reduce_val
            reduce_test = min(deficit, min_test)
            min_test -= reduce_test
        max_train = n_events - min_val - min_test
        min_train = min(min_train, max_train)
        if min_train < 1:
            min_train = 1
            if n_events - min_train - min_val - min_test < 0:
                min_val = max(0, n_events - min_train - min_test)
        return min_train, min_val, min_test

    @staticmethod
    def _fallback_time_split(
        ts: np.ndarray,
        *,
        min_train: int,
        min_val: int,
        min_test: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_events = len(ts)
        if n_events == 0:
            empty = np.zeros(0, dtype=bool)
            return empty, empty, empty
        min_train, min_val, min_test = TempMETGNTrainer._sanitize_minima(
            n_events, min_train, min_val, min_test
        )
        n_test = max(min_test, int(round(0.15 * n_events)))
        n_val = max(min_val, int(round(0.15 * n_events)))
        n_train = n_events - n_val - n_test
        if n_train < min_train:
            deficit = min_train - n_train
            reduce_val = min(deficit, n_val)
            n_val -= reduce_val
            deficit -= reduce_val
            reduce_test = min(deficit, n_test)
            n_test -= reduce_test
            deficit -= reduce_test
            n_train = n_events - n_val - n_test
        if n_train <= 0:
            n_train = max(1, n_events - n_test)
            n_val = max(0, n_events - n_train - n_test)
            n_test = n_events - n_train - n_val
        if n_events >= 2 and n_test == 0:
            n_test = 1
            if n_train > 1:
                n_train -= 1
            elif n_val > 0:
                n_val -= 1
        order = np.argsort(ts, kind="stable")
        train_idx = order[:n_train]
        val_idx = order[n_train : n_train + n_val]
        test_idx = order[n_train + n_val :]
        train_flag = np.zeros(n_events, dtype=bool)
        val_flag = np.zeros(n_events, dtype=bool)
        test_flag = np.zeros(n_events, dtype=bool)
        train_flag[train_idx] = True
        if len(val_idx) > 0:
            val_flag[val_idx] = True
        if len(test_idx) > 0:
            test_flag[test_idx] = True
        return train_flag, val_flag, test_flag

    def _resolve_device(self, requested: Optional[str]) -> torch.device:
        """
        Pick a device that works on the current machine.

        - If requested is None or "auto": prefer CUDA, then MPS (Apple), else CPU.
        - If a device string is provided but unavailable, fall back to CPU with a note.
        """
        if requested is None or str(requested).lower() == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            if torch.backends.mps.is_available():  # type: ignore[attr-defined]
                return torch.device("mps")
            return torch.device("cpu")

        try:
            dev = torch.device(requested)
        except Exception:
            return torch.device("cpu")

        if dev.type == "cuda" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():  # type: ignore[attr-defined]
                self._log(
                    f"Requested CUDA device '{requested}' but CUDA is unavailable; using MPS instead.",
                    level="full",
                )
                return torch.device("mps")
            self._log(
                f"Requested CUDA device '{requested}' but CUDA is unavailable; falling back to CPU.",
                level="full",
            )
            return torch.device("cpu")
        if dev.type == "mps" and not getattr(torch.backends, "mps", None):
            self._log(
                f"Requested MPS device '{requested}' but MPS is unavailable; falling back to CPU.",
                level="full",
            )
            return torch.device("cpu")
        return dev

    @staticmethod
    def _looks_like_tempme_base(model: nn.Module) -> bool:
        emb = getattr(model, "embedding_module", None)
        return emb is not None and hasattr(emb, "embedding_update")

    @staticmethod
    def _set_neighbor_sampler(model: nn.Module, neighbor_sampler: NeighborFinder) -> None:
        if hasattr(model, "set_neighbor_sampler"):
            model.set_neighbor_sampler(neighbor_sampler)
            return
        if hasattr(model, "set_neighbor_finder"):
            model.set_neighbor_finder(neighbor_sampler)
            return
        if hasattr(model, "ngh_finder"):
            model.ngh_finder = neighbor_sampler

    @staticmethod
    def _infer_pack_size(pack: Any) -> Optional[int]:
        try:
            subgraph_src = pack[0]
            node_records = subgraph_src[0]
            return int(node_records[0].shape[0])
        except Exception:
            return None

    def _resolve_instance_count(self, src: np.ndarray, pack: Any, edge: Any) -> int:
        sizes: List[int] = [len(src)]
        pack_size = self._infer_pack_size(pack)
        if pack_size is not None:
            sizes.append(pack_size)
        if edge is not None:
            sizes.append(int(edge.shape[0]))
        min_len = min(sizes) if sizes else 0
        return max(0, min_len - 1)

    @staticmethod
    def _select_neighbor_finder_cls(model: nn.Module):
        """
        Pick the NeighborFinder implementation that matches the base model API.
        TempME's NeighborFinder uses `num_neighbor`, while TGN expects `num_neighbors`.
        """
        try:
            from time_to_explain.adapters.tempme_base_adapter import TempMEBaseAdapter
        except Exception:
            TempMEBaseAdapter = None  # type: ignore

        if TempMEBaseAdapter is not None and isinstance(model, TempMEBaseAdapter):
            try:
                from submodules.models.tgn.tgn_utils.utils import NeighborFinder as TGNNeighborFinder
                return TGNNeighborFinder
            except Exception:
                return NeighborFinder

        module_name = type(model).__module__
        if "submodules.models.tgn" in module_name:
            try:
                from submodules.models.tgn.tgn_utils.utils import NeighborFinder as TGNNeighborFinder
                return TGNNeighborFinder
            except Exception:
                return NeighborFinder

        return NeighborFinder

    def _make_neighbor_finder(self, adj_list):
        nf_cls = self._neighbor_finder_cls or NeighborFinder
        try:
            return nf_cls(adj_list, uniform=False)
        except TypeError:
            return nf_cls(adj_list)

    # ------------------------------------------------------------------ #
    # Data loading (adapted from load_data in original script)          #
    # ------------------------------------------------------------------ #

    def _load_data(
        self,
        mode: str,
    ) -> Tuple[
        RandEdgeSampler,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        NeighborFinder,
    ]:
        """
        Load csv + build NeighborFinder for training or testing, as in the
        original TempME script.
        """
        data = self.cfg.data
        csv_path = self.processed_dir / f"ml_{data}.csv"
        g_df = pd.read_csv(csv_path)

        val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))
        src_l = g_df.u.values
        dst_l = g_df.i.values
        e_idx_l = g_df.idx.values
        label_l = g_df.label.values
        ts_l = g_df.ts.values

        max_src_index = src_l.max()
        max_idx = max(src_l.max(), dst_l.max())

        # Same masking logic as original code, with fallback for tiny splits
        random.seed(2023)
        total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
        num_total_unique_nodes = len(total_node_set)
        mask_candidates = list(
            set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time]))
        )
        mask_ratio = float(max(0.0, min(1.0, self.cfg.mask_ratio)))
        sample_size = int(mask_ratio * num_total_unique_nodes)
        if sample_size > len(mask_candidates):
            sample_size = len(mask_candidates)
        mask_node_set = set(random.sample(mask_candidates, sample_size)) if sample_size > 0 else set()

        mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
        mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
        none_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)

        valid_train_flag = (ts_l <= val_time) * (none_node_flag > 0)
        valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
        valid_test_flag = ts_l > test_time

        min_train, min_val, min_test = self._sanitize_minima(
            len(ts_l), self.cfg.min_train, self.cfg.min_val, self.cfg.min_test
        )
        if (
            valid_train_flag.sum() < min_train
            or valid_val_flag.sum() < min_val
            or valid_test_flag.sum() < min_test
        ):
            valid_train_flag, valid_val_flag, valid_test_flag = self._fallback_time_split(
                ts_l,
                min_train=min_train,
                min_val=min_val,
                min_test=min_test,
            )

        train_src_l = src_l[valid_train_flag]
        train_dst_l = dst_l[valid_train_flag]
        train_ts_l = ts_l[valid_train_flag]
        train_e_idx_l = e_idx_l[valid_train_flag]
        train_label_l = label_l[valid_train_flag]

        val_src_l = src_l[valid_val_flag]
        val_dst_l = dst_l[valid_val_flag]
        test_src_l = src_l[valid_test_flag]
        test_dst_l = dst_l[valid_test_flag]
        test_ts_l = ts_l[valid_test_flag]
        test_e_idx_l = e_idx_l[valid_test_flag]
        test_label_l = label_l[valid_test_flag]

        # Build neighbor finders
        adj_list = [[] for _ in range(max_idx + 1)]
        for s, d, eidx, ts in zip(
            train_src_l, train_dst_l, train_e_idx_l, train_ts_l
        ):
            adj_list[s].append((d, eidx, ts))
            adj_list[d].append((s, eidx, ts))
        train_ngh_finder = self._make_neighbor_finder(adj_list)

        full_adj_list = [[] for _ in range(max_idx + 1)]
        for s, d, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
            full_adj_list[s].append((d, eidx, ts))
            full_adj_list[d].append((s, eidx, ts))
        full_ngh_finder = self._make_neighbor_finder(full_adj_list)

        # Rand samplers
        train_rand_sampler = RandEdgeSampler((train_src_l,), (train_dst_l,))
        test_rand_sampler = RandEdgeSampler(
            (train_src_l, val_src_l, test_src_l),
            (train_dst_l, val_dst_l, test_dst_l),
        )

        if mode == "test":
            return (
                test_rand_sampler,
                test_src_l,
                test_dst_l,
                test_ts_l,
                test_label_l,
                test_e_idx_l,
                full_ngh_finder,
            )
        else:
            return (
                train_rand_sampler,
                train_src_l,
                train_dst_l,
                train_ts_l,
                train_label_l,
                train_e_idx_l,
                train_ngh_finder,
            )

    # ------------------------------------------------------------------ #
    # Threshold test (unchanged logic from original)                    #
    # ------------------------------------------------------------------ #

    def _threshold_test(
        self,
        explanation,
        src_l_cut,
        dst_l_cut,
        dst_l_fake,
        ts_l_cut,
        e_l_cut,
        pos_out_ori,
        neg_out_ori,
        y_ori,
        subgraph_src,
        subgraph_tgt,
        subgraph_bgd,
    ):
        """
        Compute AUC over ratios in [0, 0.3], same logic as original threshold_test.
        """
        args = self.cfg
        base_model = self.base_model

        AUC_aps, AUC_auc, AUC_acc = [], [], []
        AUC_fid_prob, AUC_fid_logit = [], []

        for ratio in args.ratios:
            if args.base_type == "tgn":
                batch = len(src_l_cut)

                edge_imp_src = torch.cat(
                    [explanation[0][:batch], explanation[1][:batch]], dim=1
                )
                edge_imp_tgt = torch.cat(
                    [explanation[0][batch : 2 * batch], explanation[1][batch : 2 * batch]],
                    dim=1,
                )
                edge_imp_bgd = torch.cat(
                    [explanation[0][2 * batch : 3 * batch], explanation[1][2 * batch : 3 * batch]], dim=1
                )

                num_edge = edge_imp_src.shape[1]
                topk = min(max(math.ceil(ratio * num_edge), 1), num_edge)

                selected_src = torch.topk(
                    edge_imp_src, k=num_edge - topk, dim=-1, largest=False
                ).indices
                selected_tgt = torch.topk(
                    edge_imp_tgt, k=num_edge - topk, dim=-1, largest=False
                ).indices
                selected_bgd = torch.topk(
                    edge_imp_bgd, k=num_edge - topk, dim=-1, largest=False
                ).indices

                node_records_src, eidx_records_src, t_records_src = subgraph_src
                n_degree = node_records_src[0].shape[1]
                node_records_src_cat = np.concatenate(node_records_src, axis=-1)
                np.put_along_axis(
                    node_records_src_cat,
                    selected_src.cpu().numpy(),
                    0,
                    axis=-1,
                )
                node_records_src = np.split(
                    node_records_src_cat, [n_degree], axis=1
                )
                subgraph_src_sub = node_records_src, eidx_records_src, t_records_src

                node_records_tgt, eidx_records_tgt, t_records_tgt = subgraph_tgt
                n_degree = node_records_tgt[0].shape[1]
                node_records_tgt_cat = np.concatenate(node_records_tgt, axis=-1)
                np.put_along_axis(
                    node_records_tgt_cat,
                    selected_tgt.cpu().numpy(),
                    0,
                    axis=-1,
                )
                node_records_tgt = np.split(
                    node_records_tgt_cat, [n_degree], axis=1
                )
                subgraph_tgt_sub = node_records_tgt, eidx_records_tgt, t_records_tgt

                node_records_bgd, eidx_records_bgd, t_records_bgd = subgraph_bgd
                n_degree = node_records_bgd[0].shape[1]
                node_records_bgd_cat = np.concatenate(node_records_bgd, axis=-1)
                np.put_along_axis(
                    node_records_bgd_cat,
                    selected_bgd.cpu().numpy(),
                    0,
                    axis=-1,
                )
                node_records_bgd = np.split(
                    node_records_bgd_cat, [n_degree], axis=1
                )
                subgraph_bgd_sub = node_records_bgd, eidx_records_bgd, t_records_bgd

            elif args.base_type == "graphmixer":
                batch = len(src_l_cut)
                edge_imp_src = explanation[0][:batch]
                edge_imp_tgt = explanation[0][batch : 2 * batch]
                edge_imp_bgd = explanation[0][2 * batch : 3 * batch]

                num_edge = edge_imp_src.shape[1]
                topk = min(max(math.ceil(ratio * num_edge), 1), num_edge)

                selected_src = torch.topk(
                    edge_imp_src, k=num_edge - topk, dim=-1, largest=False
                ).indices
                selected_tgt = torch.topk(
                    edge_imp_tgt, k=num_edge - topk, dim=-1, largest=False
                ).indices
                selected_bgd = torch.topk(
                    edge_imp_bgd, k=num_edge - topk, dim=-1, largest=False
                ).indices

                node_records_src, eidx_records_src, t_records_src = subgraph_src
                node_src_0 = node_records_src[0].copy()
                np.put_along_axis(
                    node_src_0, selected_src.cpu().numpy(), 0, axis=-1
                )
                node_records_src_sub = [node_src_0, node_records_src[1]]
                subgraph_src_sub = node_records_src_sub, eidx_records_src, t_records_src

                node_records_tgt, eidx_records_tgt, t_records_tgt = subgraph_tgt
                node_tgt_0 = node_records_tgt[0].copy()
                np.put_along_axis(
                    node_tgt_0, selected_tgt.cpu().numpy(), 0, axis=-1
                )
                node_records_tgt_sub = [node_tgt_0, node_records_tgt[1]]
                subgraph_tgt_sub = node_records_tgt_sub, eidx_records_tgt, t_records_tgt

                node_records_bgd, eidx_records_bgd, t_records_bgd = subgraph_bgd
                node_bgd_0 = node_records_bgd[0].copy()
                np.put_along_axis(
                    node_bgd_0, selected_bgd.cpu().numpy(), 0, axis=-1
                )
                node_records_bgd_sub = [node_bgd_0, node_records_bgd[1]]
                subgraph_bgd_sub = node_records_bgd_sub, eidx_records_bgd, t_records_bgd

            else:
                raise ValueError(f"Unsupported base_type={args.base_type!r} in threshold_test")

            with torch.no_grad():
                pos_logit, neg_logit = base_model.contrast(
                    src_l_cut,
                    dst_l_cut,
                    dst_l_fake,
                    ts_l_cut,
                    e_l_cut,
                    subgraph_src_sub,
                    subgraph_tgt_sub,
                    subgraph_bgd_sub,
                )

                y_pred = torch.cat([pos_logit, neg_logit], dim=0).sigmoid()
                pred_label = torch.where(y_pred > 0.5, 1.0, 0.0).view(y_pred.size(0), 1)

                fid_prob_batch = torch.cat(
                    [
                        pos_logit.sigmoid() - pos_out_ori.sigmoid(),
                        neg_out_ori.sigmoid() - neg_logit.sigmoid(),
                    ],
                    dim=0,
                )
                fid_prob = fid_prob_batch.mean()

                fid_logit_batch = torch.cat(
                    [pos_logit - pos_out_ori, neg_out_ori - neg_logit], dim=0
                )
                fid_logit = fid_logit_batch.mean()

                AUC_fid_prob.append(float(fid_prob))
                AUC_fid_logit.append(float(fid_logit))
                aps, auc = self._safe_scores(y_ori, y_pred)
                if aps is not None:
                    AUC_aps.append(aps)
                if auc is not None:
                    AUC_auc.append(auc)
                AUC_acc.append((pred_label.cpu() == y_ori.cpu()).float().mean())

        aps_AUC = float(np.mean(AUC_aps)) if AUC_aps else 0.0
        auc_AUC = float(np.mean(AUC_auc)) if AUC_auc else 0.0
        acc_AUC = float(np.mean(AUC_acc)) if AUC_acc else 0.0
        fid_prob_AUC = float(np.mean(AUC_fid_prob)) if AUC_fid_prob else 0.0
        fid_logit_AUC = float(np.mean(AUC_fid_logit)) if AUC_fid_logit else 0.0

        return aps_AUC, auc_AUC, acc_AUC, fid_prob_AUC, fid_logit_AUC

    # ------------------------------------------------------------------ #
    # Evaluation loop (base_type != tgat; TGN & GraphMixer)             #
    # ------------------------------------------------------------------ #

    def _eval_one_epoch(
        self,
        explainer: nn.Module,
        full_ngh_finder: NeighborFinder,
        src: np.ndarray,
        dst: np.ndarray,
        ts: np.ndarray,
        val_e_idx_l: np.ndarray,
        epoch: int,
        best_accuracy: float,
        test_pack: Any,
        test_edge: Any,
    ) -> float:
        cfg = self.cfg
        base_model = self.base_model
        explainer = explainer.eval()

        test_aps: List[float] = []
        test_auc: List[float] = []
        test_acc: List[float] = []
        test_fid_prob: List[float] = []
        test_fid_logit: List[float] = []
        test_loss: List[float] = []
        test_pred_loss: List[float] = []
        test_kl_loss: List[float] = []

        ratio_AUC_aps: List[float] = []
        ratio_AUC_auc: List[float] = []
        ratio_AUC_acc: List[float] = []
        ratio_AUC_prob: List[float] = []
        ratio_AUC_logit: List[float] = []

        base_model = base_model.eval()
        self._set_neighbor_sampler(base_model, full_ngh_finder)

        num_test_instance = self._resolve_instance_count(src, test_pack, test_edge)
        if num_test_instance <= 0:
            self._log(f"[TempME][warn] Eval epoch {epoch} skipped: empty test split.", level="full")
            return best_accuracy
        num_test_batch = math.ceil(num_test_instance / cfg.test_bs) - 1 if num_test_instance > 0 else 0
        if num_test_instance > 0 and num_test_batch <= 0:
            num_test_batch = 1
            self._log(
                f"[TempME][warn] Test split too small (instances={num_test_instance}); "
                "running a single eval batch.",
                level="full",
            )
        idx_list = np.arange(num_test_instance)
        criterion = torch.nn.BCEWithLogitsLoss()

        for k in tqdm(range(num_test_batch), desc=f"Eval epoch {epoch}", disable=not cfg.show_progress):
            s_idx = k * cfg.test_bs
            if num_test_instance <= cfg.test_bs:
                e_idx = min(num_test_instance, s_idx + cfg.test_bs)
            else:
                e_idx = min(num_test_instance - 1, s_idx + cfg.test_bs)
            if s_idx >= e_idx:
                continue
            batch_idx = idx_list[s_idx:e_idx]

            src_l_cut = src[batch_idx]
            dst_l_cut = dst[batch_idx]
            ts_l_cut = ts[batch_idx]
            e_l_cut = val_e_idx_l[batch_idx] if val_e_idx_l is not None else None

            subgraph_src, subgraph_tgt, subgraph_bgd, walks_src, walks_tgt, walks_bgd, dst_l_fake = get_item(
                test_pack, batch_idx
            )
            src_edge, tgt_edge, bgd_edge = get_item_edge(test_edge, batch_idx)

            with torch.no_grad():
                pos_out_ori, neg_out_ori = base_model.contrast(
                    src_l_cut,
                    dst_l_cut,
                    dst_l_fake,
                    ts_l_cut,
                    e_l_cut,
                    subgraph_src,
                    subgraph_tgt,
                    subgraph_bgd,
                )
                y_pred = torch.cat([pos_out_ori, neg_out_ori], dim=0).sigmoid()
                y_ori = torch.where(y_pred > 0.5, 1.0, 0.0).view(y_pred.size(0), 1)

            # Explainer forward
            graphlet_imp_src = explainer(walks_src, ts_l_cut, src_edge)
            graphlet_imp_tgt = explainer(walks_tgt, ts_l_cut, tgt_edge)
            graphlet_imp_bgd = explainer(walks_bgd, ts_l_cut, bgd_edge)

            explanation = explainer.retrieve_explanation(
                subgraph_src,
                graphlet_imp_src,
                walks_src,
                subgraph_tgt,
                graphlet_imp_tgt,
                walks_tgt,
                subgraph_bgd,
                graphlet_imp_bgd,
                walks_bgd,
                training=cfg.if_bern,
            )

            pos_logit, neg_logit = base_model.contrast(
                src_l_cut,
                dst_l_cut,
                dst_l_fake,
                ts_l_cut,
                e_l_cut,
                subgraph_src,
                subgraph_tgt,
                subgraph_bgd,
                explain_weights=explanation,
            )

            pred = torch.cat([pos_logit, neg_logit], dim=0).to(self.device)
            pred_loss = criterion(pred, y_ori)

            kl_loss = (
                explainer.kl_loss(graphlet_imp_src, walks_src, target=cfg.prior_p)
                + explainer.kl_loss(graphlet_imp_tgt, walks_tgt, target=cfg.prior_p)
                + explainer.kl_loss(graphlet_imp_bgd, walks_bgd, target=cfg.prior_p)
            )
            loss = pred_loss + cfg.beta * kl_loss

            with torch.no_grad():
                y_pred = torch.cat([pos_logit, neg_logit], dim=0).sigmoid()
                pred_label = torch.where(y_pred > 0.5, 1.0, 0.0).view(y_pred.size(0), 1)

                fid_prob_batch = torch.cat(
                    [
                        pos_logit.sigmoid() - pos_out_ori.sigmoid(),
                        neg_out_ori.sigmoid() - neg_logit.sigmoid(),
                    ],
                    dim=0,
                )
                fid_prob = fid_prob_batch.mean()

                fid_logit_batch = torch.cat(
                    [pos_logit - pos_out_ori, neg_out_ori - neg_logit],
                    dim=0,
                )
                fid_logit = fid_logit_batch.mean()

                test_fid_prob.append(float(fid_prob))
                test_fid_logit.append(float(fid_logit))
                aps, auc = self._safe_scores(y_ori, y_pred)
                if aps is not None:
                    test_aps.append(aps)
                if auc is not None:
                    test_auc.append(auc)
                test_acc.append((pred_label.cpu() == y_ori.cpu()).float().mean())
                test_loss.append(loss.item())
                test_pred_loss.append(pred_loss.item())
                test_kl_loss.append(kl_loss.item())

                # threshold_eval
                if cfg.test_threshold:
                    explanation_eval = explainer.retrieve_explanation(
                        subgraph_src,
                        graphlet_imp_src,
                        walks_src,
                        subgraph_tgt,
                        graphlet_imp_tgt,
                        walks_tgt,
                        subgraph_bgd,
                        graphlet_imp_bgd,
                        walks_bgd,
                        training=False,
                    )
                    aps_AUC, auc_AUC, acc_AUC, fid_prob_AUC, fid_logit_AUC = self._threshold_test(
                        explanation_eval,
                        src_l_cut,
                        dst_l_cut,
                        dst_l_fake,
                        ts_l_cut,
                        e_l_cut,
                        pos_out_ori,
                        neg_out_ori,
                        y_ori,
                        subgraph_src,
                        subgraph_tgt,
                        subgraph_bgd,
                    )
                    ratio_AUC_aps.append(aps_AUC)
                    ratio_AUC_auc.append(auc_AUC)
                    ratio_AUC_acc.append(acc_AUC)
                    ratio_AUC_prob.append(fid_prob_AUC)
                    ratio_AUC_logit.append(fid_logit_AUC)

        # Aggregate metrics
        aps_ratios_AUC = float(np.mean(ratio_AUC_aps)) if ratio_AUC_aps else 0.0
        auc_ratios_AUC = float(np.mean(ratio_AUC_auc)) if ratio_AUC_auc else 0.0
        acc_ratios_AUC = float(np.mean(ratio_AUC_acc)) if ratio_AUC_acc else 0.0
        prob_ratios_AUC = float(np.mean(ratio_AUC_prob)) if ratio_AUC_prob else 0.0
        logit_ratios_AUC = float(np.mean(ratio_AUC_logit)) if ratio_AUC_logit else 0.0

        aps_epoch = float(np.mean(test_aps)) if test_aps else 0.0
        auc_epoch = float(np.mean(test_auc)) if test_auc else 0.0
        acc_epoch = float(np.mean(test_acc)) if test_acc else 0.0
        fid_prob_epoch = float(np.mean(test_fid_prob)) if test_fid_prob else 0.0
        fid_logit_epoch = float(np.mean(test_fid_logit)) if test_fid_logit else 0.0
        loss_epoch = float(np.mean(test_loss)) if test_loss else 0.0
        pred_loss_epoch = float(np.mean(test_pred_loss)) if test_pred_loss else 0.0
        kl_loss_epoch = float(np.mean(test_kl_loss)) if test_kl_loss else 0.0

        self._log(
            f"[Eval] Epoch {epoch} | "
            f"Loss: {loss_epoch:.4f} "
            f"(pred {pred_loss_epoch:.4f}, kl {kl_loss_epoch:.4f}) | "
            f"APS: {aps_epoch:.4f} | AUC: {auc_epoch:.4f} | ACC: {acc_epoch:.4f} | "
            f"FID_prob: {fid_prob_epoch:.4f} | FID_logit: {fid_logit_epoch:.4f} | "
            f"Ratio APS: {aps_ratios_AUC:.4f} | Ratio AUC: {auc_ratios_AUC:.4f} | "
            f"Ratio ACC: {acc_ratios_AUC:.4f} | Ratio Prob: {prob_ratios_AUC:.4f} | "
            f"Ratio Logit: {logit_ratios_AUC:.4f}",
            level="full",
        )

        # Save best model by ratio APS
        if aps_ratios_AUC > best_accuracy and self.cfg.save_model:
            save_path = osp.join(
                self.cfg.explainer_save_dir,
                f"{self.cfg.data}.pt",
            )
            torch.save(explainer, save_path)
            self._log(
                f"[Eval] New best Ratio APS {aps_ratios_AUC:.4f}, saved explainer to {save_path}",
                level="full",
            )
            best_accuracy = aps_ratios_AUC

        return best_accuracy

    # ------------------------------------------------------------------ #
    # Fit: full training loop                                           #
    # ------------------------------------------------------------------ #

    def fit(self) -> nn.Module:
        """
        Train a TempME/TempME_TGAT explainer for cfg.n_epoch epochs.
        Returns the trained explainer module.
        """
        cfg = self.cfg
        device = self.device

        # Data packs / edges for random walks & subgraphs
        train_h5 = h5py.File(self.processed_dir / f"{cfg.data}_train_cat.h5", "r")
        test_h5 = h5py.File(self.processed_dir / f"{cfg.data}_test_cat.h5", "r")

        train_pack = load_subgraph_margin(cfg, train_h5)
        test_pack = load_subgraph_margin(cfg, test_h5)

        train_edge = np.load(self.processed_dir / f"{cfg.data}_train_edge.npy")
        test_edge = np.load(self.processed_dir / f"{cfg.data}_test_edge.npy")

        # Load training / test splits
        rand_sampler, src_l, dst_l, ts_l, label_l, e_idx_l, ngh_finder = self._load_data(
            mode="training"
        )
        (
            test_rand_sampler,
            test_src_l,
            test_dst_l,
            test_ts_l,
            test_label_l,
            test_e_idx_l,
            full_ngh_finder,
        ) = self._load_data(mode="test")

        num_instance = self._resolve_instance_count(src_l, train_pack, train_edge)
        num_batch = math.ceil(num_instance / cfg.bs) if num_instance > 0 else 0
        self._log(f"[TempME] num training instances: {num_instance}", level="full")
        self._log(f"[TempME] num batches per epoch: {num_batch}", level="full")

        idx_list = np.arange(num_instance)

        # Explainer
        if cfg.base_type == "tgat":
            explainer = TempME_TGAT(
                self.base_model,
                data=cfg.data,
                out_dim=cfg.out_dim,
                hid_dim=cfg.hid_dim,
                temp=cfg.temp,
                dropout_p=cfg.drop_out,
                device=device,
            )
        else:
            explainer = TempME(
                self.base_model,
                base_model_type=cfg.base_type,
                data=cfg.data,
                out_dim=cfg.out_dim,
                hid_dim=cfg.hid_dim,
                temp=cfg.temp,
                if_cat_feature=True,
                dropout_p=cfg.drop_out,
                device=device,
            )
        explainer = explainer.to(device)

        optimizer = torch.optim.Adam(
            explainer.parameters(),
            lr=cfg.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=cfg.weight_decay,
        )
        scheduler = None
        if cfg.use_lr_decay:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=cfg.lr_decay
            )
        criterion = torch.nn.BCEWithLogitsLoss()

        best_acc = 0.0

        for epoch in range(cfg.n_epoch):
            self._set_neighbor_sampler(self.base_model, ngh_finder)
            explainer.train()

            train_aps: List[float] = []
            train_auc: List[float] = []
            train_acc: List[float] = []
            train_fid_prob: List[float] = []
            train_fid_logit: List[float] = []
            train_loss: List[float] = []
            train_pred_loss: List[float] = []
            train_kl_loss: List[float] = []

            np.random.shuffle(idx_list)

            for k in tqdm(range(num_batch), desc=f"Train epoch {epoch}", disable=not cfg.show_progress):
                s_idx = k * cfg.bs
                if num_instance <= cfg.bs:
                    e_idx = min(num_instance, s_idx + cfg.bs)
                else:
                    e_idx = min(num_instance - 1, s_idx + cfg.bs)
                if s_idx >= e_idx:
                    continue
                batch_idx = idx_list[s_idx:e_idx]

                src_l_cut = src_l[batch_idx]
                dst_l_cut = dst_l[batch_idx]
                ts_l_cut = ts_l[batch_idx]
                e_l_cut = e_idx_l[batch_idx]

                subgraph_src, subgraph_tgt, subgraph_bgd, walks_src, walks_tgt, walks_bgd, dst_l_fake = get_item(
                    train_pack, batch_idx
                )
                src_edge, tgt_edge, bgd_edge = get_item_edge(train_edge, batch_idx)

                with torch.no_grad():
                    if cfg.base_type == "tgat":
                        pos_out_ori, neg_out_ori = self.base_model.contrast(
                            src_l_cut,
                            dst_l_cut,
                            dst_l_fake,
                            ts_l_cut,
                            e_l_cut,
                            subgraph_src,
                            subgraph_tgt,
                            subgraph_bgd,
                            test=True,
                            if_explain=False,
                        )
                    else:
                        pos_out_ori, neg_out_ori = self.base_model.contrast(
                            src_l_cut,
                            dst_l_cut,
                            dst_l_fake,
                            ts_l_cut,
                            e_l_cut,
                            subgraph_src,
                            subgraph_tgt,
                            subgraph_bgd,
                        )

                    y_pred = torch.cat([pos_out_ori, neg_out_ori], dim=0).sigmoid()
                    y_ori = torch.where(y_pred > 0.5, 1.0, 0.0).view(
                        y_pred.size(0), 1
                    )

                optimizer.zero_grad()

                # Explainer forward
                graphlet_imp_src = explainer(walks_src, ts_l_cut, src_edge)
                graphlet_imp_tgt = explainer(walks_tgt, ts_l_cut, tgt_edge)
                graphlet_imp_bgd = explainer(walks_bgd, ts_l_cut, bgd_edge)

                if cfg.base_type == "tgat":
                    edge_imp_src = explainer.retrieve_edge_imp(
                        subgraph_src, graphlet_imp_src, walks_src, training=cfg.if_bern
                    )
                    edge_imp_tgt = explainer.retrieve_edge_imp(
                        subgraph_tgt, graphlet_imp_tgt, walks_tgt, training=cfg.if_bern
                    )
                    edge_imp_bgd = explainer.retrieve_edge_imp(
                        subgraph_bgd, graphlet_imp_bgd, walks_bgd, training=cfg.if_bern
                    )
                    explain_weight = [[edge_imp_src, edge_imp_tgt], [edge_imp_src, edge_imp_bgd]]
                    pos_logit, neg_logit = self.base_model.contrast(
                        src_l_cut,
                        dst_l_cut,
                        dst_l_fake,
                        ts_l_cut,
                        e_l_cut,
                        subgraph_src,
                        subgraph_tgt,
                        subgraph_bgd,
                        test=True,
                        if_explain=True,
                        exp_weights=explain_weight,
                    )
                else:
                    explanation = explainer.retrieve_explanation(
                        subgraph_src,
                        graphlet_imp_src,
                        walks_src,
                        subgraph_tgt,
                        graphlet_imp_tgt,
                        walks_tgt,
                        subgraph_bgd,
                        graphlet_imp_bgd,
                        walks_bgd,
                        training=cfg.if_bern,
                    )
                    pos_logit, neg_logit = self.base_model.contrast(
                        src_l_cut,
                        dst_l_cut,
                        dst_l_fake,
                        ts_l_cut,
                        e_l_cut,
                        subgraph_src,
                        subgraph_tgt,
                        subgraph_bgd,
                        explain_weights=explanation,
                    )

                pred = torch.cat([pos_logit, neg_logit], dim=0).view(-1, 1).to(device)
                pred_loss = criterion(pred, y_ori)

                kl_loss = (
                    explainer.kl_loss(graphlet_imp_src, walks_src, target=cfg.prior_p)
                    + explainer.kl_loss(graphlet_imp_tgt, walks_tgt, target=cfg.prior_p)
                    + explainer.kl_loss(graphlet_imp_bgd, walks_bgd, target=cfg.prior_p)
                )

                loss = pred_loss + cfg.beta * kl_loss
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    y_pred = torch.cat([pos_logit, neg_logit], dim=0).sigmoid()
                    pred_label = torch.where(y_pred > 0.5, 1.0, 0.0).view(
                        y_pred.size(0), 1
                    )

                    fid_prob_batch = torch.cat(
                        [
                            pos_logit.sigmoid() - pos_out_ori.sigmoid(),
                            neg_out_ori.sigmoid() - neg_logit.sigmoid(),
                        ],
                        dim=0,
                    )
                    fid_prob = fid_prob_batch.mean()

                    fid_logit_batch = torch.cat(
                        [pos_logit - pos_out_ori, neg_out_ori - neg_logit],
                        dim=0,
                    )
                    fid_logit = fid_logit_batch.mean()

                    train_fid_prob.append(float(fid_prob))
                    train_fid_logit.append(float(fid_logit))
                    aps, auc = self._safe_scores(y_ori, y_pred)
                    if aps is not None:
                        train_aps.append(aps)
                    if auc is not None:
                        train_auc.append(auc)
                    train_acc.append((pred_label.cpu() == y_ori.cpu()).float().mean())
                    train_loss.append(loss.item())
                    train_pred_loss.append(pred_loss.item())
                    train_kl_loss.append(kl_loss.item())
                # LR scheduler step (optional)
                if scheduler is not None:
                    scheduler.step()

            # Epoch summary
            aps_epoch = float(np.mean(train_aps)) if train_aps else 0.0
            auc_epoch = float(np.mean(train_auc)) if train_auc else 0.0
            acc_epoch = float(np.mean(train_acc)) if train_acc else 0.0
            fid_prob_epoch = float(np.mean(train_fid_prob)) if train_fid_prob else 0.0
            fid_logit_epoch = float(np.mean(train_fid_logit)) if train_fid_logit else 0.0
            loss_epoch = float(np.mean(train_loss)) if train_loss else 0.0

            self._log(
                f"[TempME] Epoch {epoch + 1}/{cfg.n_epoch} | "
                f"Loss: {loss_epoch:.4f} | APS: {aps_epoch:.4f} | "
                f"AUC: {auc_epoch:.4f} | ACC: {acc_epoch:.4f}",
                level="summary",
            )

            # Evaluation
            if (epoch + 1) % cfg.verbose == 0:
                best_acc = self._eval_one_epoch(
                    explainer,
                    full_ngh_finder,
                    test_src_l,
                    test_dst_l,
                    test_ts_l,
                    test_e_idx_l,
                    epoch,
                    best_acc,
                    test_pack,
                    test_edge,
                )

        if cfg.save_model:
            save_path = osp.join(cfg.explainer_save_dir, f"{cfg.data}.pt")
            if not osp.exists(save_path):
                torch.save(explainer, save_path)
                self._log(
                    f"[TempME] Saved final explainer to {save_path} (no best checkpoint found)",
                    level="summary",
                )

        self.explainer = explainer
        return explainer
