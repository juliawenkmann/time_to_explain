# time_to_explain/adapters/temgx_adapter.py
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from time_to_explain.core.types import BaseExplainer, ExplanationContext, ExplanationResult


_TEMGX_MODULE = None


def _load_temgx_module() -> Any:
    global _TEMGX_MODULE
    if _TEMGX_MODULE is not None:
        return _TEMGX_MODULE

    root = Path(__file__).resolve().parents[2] / "submodules" / "explainer" / "TemGX" / "link"
    script = root / "scripts" / "temgx.py"
    if not script.exists():
        raise FileNotFoundError(f"TemGX script not found: {script}")

    for p in (root, root / "scripts"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    # Ensure TemGX resolves the local TGN implementation under submodules/models/tgn.
    tgn_root = Path(__file__).resolve().parents[2] / "submodules" / "models"
    if str(tgn_root) not in sys.path:
        sys.path.insert(0, str(tgn_root))
    if "TGN" not in sys.modules:
        try:
            import tgn as _tgn  # noqa: F401
        except Exception as exc:
            raise ModuleNotFoundError(
                "TemGX requires the TGN package from submodules/models/tgn. "
                "Ensure that path exists and is importable."
            ) from exc
        sys.modules["TGN"] = sys.modules["tgn"]

    # TemGX expects TGN.utils.* and tgn.tgn_utils.* but the local package uses tgn.utils.*.
    try:
        import importlib
        import types

        utils_pkg = sys.modules.get("TGN.utils")
        if utils_pkg is None:
            utils_pkg = types.ModuleType("TGN.utils")
            utils_pkg.__path__ = []  # mark as package
            sys.modules["TGN.utils"] = utils_pkg

        dp_mod = importlib.import_module("tgn.utils.data_processing")
        utils_mod = importlib.import_module("tgn.utils.utils")
        sys.modules["TGN.utils.data_processing"] = dp_mod
        sys.modules["TGN.utils.utils"] = utils_mod
        setattr(utils_pkg, "data_processing", dp_mod)
        setattr(utils_pkg, "utils", utils_mod)

        tgn_utils_pkg = sys.modules.get("tgn.tgn_utils")
        if tgn_utils_pkg is None:
            tgn_utils_pkg = types.ModuleType("tgn.tgn_utils")
            tgn_utils_pkg.__path__ = []
            sys.modules["tgn.tgn_utils"] = tgn_utils_pkg
        sys.modules["tgn.tgn_utils.data_processing"] = dp_mod
        sys.modules["tgn.tgn_utils.utils"] = utils_mod
        setattr(tgn_utils_pkg, "data_processing", dp_mod)
        setattr(tgn_utils_pkg, "utils", utils_mod)
    except Exception as exc:
        raise ModuleNotFoundError(
            "TemGX could not map TGN.utils or tgn.tgn_utils to submodules/models/tgn/utils. "
            "Check that data_processing.py and utils.py exist."
        ) from exc

    spec = importlib.util.spec_from_file_location("temgx_script", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load TemGX module from {script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _TEMGX_MODULE = mod
    return mod


def _resolve_events(dataset: Any) -> pd.DataFrame:
    events = dataset.get("events") if isinstance(dataset, dict) else dataset
    if isinstance(events, pd.DataFrame):
        df = events.copy()
    else:
        try:
            df = pd.DataFrame(events)
        except Exception as exc:
            raise TypeError("TemGXAdapter expects dataset['events'] to be a pandas DataFrame.") from exc

    # Try common column names; fall back to first three columns.
    col_map = {}
    for key, candidates in {
        "u": ["u", "src", "source", "user_id"],
        "i": ["i", "dst", "target", "item_id"],
        "ts": ["ts", "time", "timestamp"],
        "label": ["label", "state_label"],
    }.items():
        for cand in candidates:
            if cand in df.columns:
                col_map[key] = cand
                break

    if not all(k in col_map for k in ("u", "i", "ts")):
        if df.shape[1] < 3:
            raise ValueError("Events DataFrame must contain at least 3 columns for (u, i, ts).")
        col_map.setdefault("u", df.columns[0])
        col_map.setdefault("i", df.columns[1])
        col_map.setdefault("ts", df.columns[2])

    u = df[col_map["u"]].to_numpy(dtype=np.int64)
    i = df[col_map["i"]].to_numpy(dtype=np.int64)
    ts = df[col_map["ts"]].to_numpy(dtype=np.float64)
    e_id = np.arange(1, len(df) + 1, dtype=np.int64)
    label = (
        df[col_map["label"]].to_numpy(dtype=np.int64)
        if "label" in col_map
        else np.zeros(len(df), dtype=np.int64)
    )

    events_df = pd.DataFrame(
        {"u": u, "i": i, "ts": ts, "e_id": e_id, "label": label}
    )
    return events_df


class _TemGXWrapper:
    def __init__(self, model: Any, events_df: pd.DataFrame, logger: logging.Logger):
        self.model = model
        self.events_df = events_df
        self.logger = logger
        self.dataset = SimpleNamespace(
            events=pd.DataFrame(
                {
                    "user_id": events_df["u"].values,
                    "item_id": events_df["i"].values,
                    "timestamp": events_df["ts"].values,
                    "idx": events_df["e_id"].values,
                    "state_label": events_df["label"].values,
                }
            )
        )
        self._event_row = {int(eid): idx for idx, eid in enumerate(events_df["e_id"].values)}

    def _predict(self, src, dst, ts, preserve, result_as_logit: bool):
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        ts = np.asarray(ts, dtype=np.float64)

        prev = getattr(self.model, "forbidden_memory_update", None)
        if prev is not None:
            self.model.forbidden_memory_update = True
        try:
            return self.model.get_prob(
                src,
                dst,
                ts,
                edge_idx_preserve_list=preserve,
                logit=result_as_logit,
            )
        finally:
            if prev is not None:
                self.model.forbidden_memory_update = prev

    def _event_row_info(self, event_id: int) -> Tuple[int, int, float]:
        row_idx = self._event_row.get(int(event_id))
        if row_idx is None:
            raise KeyError(f"Event id {event_id} not found in events.")
        row = self.events_df.iloc[row_idx]
        return int(row["u"]), int(row["i"]), float(row["ts"])

    def get_raw_node_embedding(self, node_id: int) -> Optional[np.ndarray]:
        for attr in ("node_raw_features", "node_raw_embed", "node_embeddings"):
            emb = getattr(self.model, attr, None)
            if emb is None:
                continue
            try:
                import torch
                if isinstance(emb, torch.nn.Embedding):
                    return emb.weight[int(node_id)].detach().cpu().numpy()
                if isinstance(emb, torch.Tensor):
                    return emb[int(node_id)].detach().cpu().numpy()
            except Exception:
                pass
            try:
                arr = np.asarray(emb)
                return arr[int(node_id)]
            except Exception:
                continue
        return None

    def compute_edge_probabilities(
        self,
        source_nodes: np.ndarray,
        target_nodes: np.ndarray,
        edge_timestamps: np.ndarray,
        edge_ids: np.ndarray,
        negative_nodes: np.ndarray | None = None,
        result_as_logit: bool = False,
        perform_memory_update: bool = True,
    ):
        del edge_ids, negative_nodes, perform_memory_update
        return self._predict(source_nodes, target_nodes, edge_timestamps, None, result_as_logit)

    def compute_edge_probabilities_for_subgraph(
        self,
        event_id: int,
        edges_to_drop: np.ndarray,
        result_as_logit: bool = False,
        event_ids_to_rollout: np.ndarray | None = None,
    ):
        src, dst, ts = self._event_row_info(event_id)

        preserve = None
        if event_ids_to_rollout is not None and len(event_ids_to_rollout) > 0:
            preserve = [int(e) for e in event_ids_to_rollout]
        elif edges_to_drop is not None:
            target_ts = float(ts)
            past_ids = self.dataset.events[self.dataset.events["timestamp"] < target_ts]["idx"].values
            drop_set = set(int(e) for e in edges_to_drop)
            preserve = [int(e) for e in past_ids if int(e) not in drop_set]

        return self._predict([src], [dst], [ts], preserve, result_as_logit)


@dataclass
class TemGXAdapterConfig:
    l_hops: int = 2
    candidate_cap: int = 200
    sparsity: int = 5
    time_window: Optional[float] = None
    time_scale: Optional[str] = None

    icm_alpha: float = 1.0
    icm_beta: float = 0.1
    icm_lambda: float = 0.01
    icm_gamma: float = 0.5
    trd_scale: float = 1.0

    use_extractor_candidates: bool = False
    cache: bool = True
    seed: Optional[int] = None
    alias: Optional[str] = None
    debug_mode: bool = False


class TemGXAdapter(BaseExplainer):
    def __init__(self, cfg: TemGXAdapterConfig) -> None:
        super().__init__(name="temgx", alias=cfg.alias or "temgx")
        self.cfg = cfg
        self._prepared = False
        self._temgx = None
        self._events_df: Optional[pd.DataFrame] = None
        self._graph_index = None
        self._icm = None
        self._time_scale = None
        self._time_window = None
        self._wrapper: Optional[_TemGXWrapper] = None
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._event_ts: Dict[int, float] = {}
        self._event_endpoints: Dict[int, Tuple[int, int]] = {}

    def prepare(self, *, model: Any, dataset: Any) -> None:
        super().prepare(model=model, dataset=dataset)
        backbone = getattr(model, "backbone", model)
        backbone.eval()

        self._temgx = _load_temgx_module()
        self._events_df = _resolve_events(dataset)
        self._event_ts = {
            int(row["e_id"]): float(row["ts"]) for _, row in self._events_df.iterrows()
        }
        self._event_endpoints = {
            int(row["e_id"]): (int(row["u"]), int(row["i"]))
            for _, row in self._events_df.iterrows()
        }

        logger = logging.getLogger("TemGXAdapter")
        if self.cfg.debug_mode:
            logger.setLevel(logging.DEBUG)

        self._wrapper = _TemGXWrapper(backbone, self._events_df, logger)

        time_scale = self.cfg.time_scale
        if not time_scale:
            time_scale, _ = self._temgx.detect_time_scale(self._events_df, logger)
        self._time_scale = time_scale

        time_window = self.cfg.time_window
        if time_window is None:
            time_window = self._temgx.determine_time_window(self._events_df, time_scale, logger)
        self._time_window = float(time_window)

        self._graph_index = self._temgx.TemporalGraphIndex(self._events_df, self._wrapper, logger)
        self._icm = self._temgx.ICMInfluenceCalculator(
            self._graph_index,
            alpha=self.cfg.icm_alpha,
            beta=self.cfg.icm_beta,
            lambda_decay=self.cfg.icm_lambda,
            gamma=self.cfg.icm_gamma,
            trd_scale=self.cfg.trd_scale,
            time_scale=time_scale,
        )
        self._prepared = True

    def explain(self, context: ExplanationContext) -> ExplanationResult:
        assert self._prepared and self._temgx is not None
        assert self._graph_index is not None and self._icm is not None

        eidx = (
            context.target.get("event_idx")
            or context.target.get("index")
            or context.target.get("idx")
        )
        if eidx is None and context.subgraph and context.subgraph.payload:
            eidx = context.subgraph.payload.get("event_idx")
        if eidx is None:
            raise ValueError("TemGXAdapter requires target['event_idx'|'index'|'idx'].")
        eidx = int(eidx)

        if self.cfg.cache and eidx in self._cache:
            pack = self._cache[eidx]
            return self._pack_to_result(context, eidx, pack)

        if eidx not in self._event_ts:
            raise KeyError(f"Event id {eidx} not found in events.")

        candidate_pool: List[int] = []
        if self.cfg.use_extractor_candidates and context.subgraph and context.subgraph.payload:
            raw_candidates = context.subgraph.payload.get("candidate_eidx") or []
            target_ts = self._event_ts[eidx]
            candidate_pool = [
                int(e) for e in raw_candidates
                if int(e) in self._event_ts and self._event_ts[int(e)] < target_ts
            ]
        else:
            candidate_pool = self._graph_index.get_l_hop_temporal_neighbors(
                eidx, self.cfg.l_hops, self._time_window, self._time_scale
            )

        candidate_pool = [int(e) for e in candidate_pool if int(e) != eidx]
        candidate_pool = list(dict.fromkeys(candidate_pool))

        if self.cfg.candidate_cap and len(candidate_pool) > self.cfg.candidate_cap:
            rng = np.random.default_rng(self.cfg.seed)
            candidate_pool = rng.choice(candidate_pool, self.cfg.candidate_cap, replace=False).tolist()

        if not candidate_pool:
            pack = {
                "candidate_eidx": [],
                "importance_edges": [],
                "selected_eidx": [],
                "candidate_edge_index": [],
                "stats": {"error": "no candidates"},
            }
            if self.cfg.cache:
                self._cache[eidx] = pack
            return self._pack_to_result(context, eidx, pack)

        selected_edges, p_counterfactual, stats = self._temgx.genInstanceX(
            self._wrapper,
            self._graph_index,
            self._icm,
            eidx,
            int(self.cfg.sparsity),
            candidate_pool,
            float(self._time_window),
            str(self._time_scale),
            logging.getLogger("TemGXAdapter"),
        )

        scores = self._icm.normalize_icm_scores(candidate_pool, eidx, self._event_ts[eidx])
        score_map = {int(e): float(s) for e, s in scores}
        candidate_eidx = list(sorted(candidate_pool))
        importance_edges = [score_map.get(int(e), 0.0) for e in candidate_eidx]
        candidate_edge_index = [
            list(self._event_endpoints.get(int(e), (0, 0)))
            for e in candidate_eidx
        ]

        pack = {
            "candidate_eidx": candidate_eidx,
            "importance_edges": importance_edges,
            "selected_eidx": [int(e) for e in selected_edges],
            "candidate_edge_index": candidate_edge_index,
            "counterfactual_prediction": p_counterfactual,
            "stats": stats,
        }
        if self.cfg.cache:
            self._cache[eidx] = pack

        return self._pack_to_result(context, eidx, pack)

    def _pack_to_result(
        self,
        context: ExplanationContext,
        eidx: int,
        pack: Dict[str, Any],
    ) -> ExplanationResult:
        if context.subgraph is not None:
            if context.subgraph.payload is None:
                context.subgraph.payload = {}
            context.subgraph.payload["candidate_eidx"] = list(pack.get("candidate_eidx") or [])
            if pack.get("candidate_edge_index") is not None:
                context.subgraph.payload["candidate_edge_index"] = list(pack["candidate_edge_index"])

        extras = {
            "event_idx": eidx,
            "candidate_eidx": list(pack.get("candidate_eidx") or []),
            "candidate_edge_index": list(pack.get("candidate_edge_index") or []),
            "selected_eidx": list(pack.get("selected_eidx") or []),
            "counterfactual_prediction": pack.get("counterfactual_prediction"),
            "temgx_stats": pack.get("stats"),
        }

        return ExplanationResult(
            run_id=context.run_id,
            explainer=self.alias,
            context_fp=context.fingerprint(),
            importance_edges=list(pack.get("importance_edges") or []),
            importance_nodes=None,
            importance_time=None,
            elapsed_sec=0.0,
            extras=extras,
        )


__all__ = ["TemGXAdapter", "TemGXAdapterConfig"]
