# time_to_explain/adapters/tempme_neural_tg_adapter.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

from time_to_explain.core.types import (
    BaseExplainer,
    ExplanationContext,
    ExplanationResult,
)


def _inject_tempme_utils():
    tempme_root = Path(__file__).resolve().parents[2] / "submodules" / "explainer" / "TempME"
    if str(tempme_root) not in sys.path:
        sys.path.insert(0, str(tempme_root))
    prev_utils = sys.modules.get("utils")
    try:
        import submodules.explainer.TempME.utils as tempme_utils
    except Exception:
        return prev_utils, False
    sys.modules["utils"] = tempme_utils
    return prev_utils, True


def _restore_tempme_utils(prev_utils, active):
    if not active:
        return
    if prev_utils is not None:
        sys.modules["utils"] = prev_utils
    else:
        sys.modules.pop("utils", None)


@dataclass
class TempMENeuralAdapterConfig:
    """
    Config for the learned TempME explainer adapter.

    Assumptions
    -----------
    - You have already trained a TempME (or TempME_TGAT) explainer using the
      original code you pasted, and saved it with `torch.save(explainer, path)`.
    - For each target interaction (event), you can provide the same inputs that
      the original eval code uses:
        * subgraph_src, subgraph_tgt, (optional) subgraph_bgd
        * walks_src, walks_tgt, (optional) walks_bgd
        * src_edge, tgt_edge, (optional) bgd_edge
      Typically these come out of your `get_item(...)` and `get_item_edge(...)`
      for a single batch index.

    Integration with time_to_explain
    --------------------------------
    - The adapter produces `importance_edges` aligned to a `candidate_eidx`
      derived from the subgraph edge indices:
          [src 1-hop, src 2-hop, tgt 1-hop, tgt 2-hop, (optional bgd ...)]
    - It *also* writes this into `context.subgraph.payload["candidate_eidx"]`
      so that your fidelity metrics will interpret the mask in exactly
      this order.
    - Your model / model adapter must implement:
          predict_proba(subgraph, target)
          predict_proba_with_mask(subgraph, target, edge_mask)
      expecting `edge_mask` with the same ordering as `candidate_eidx`.
    """

    base_type: str  # "tgn" | "graphmixer" | "tgat"
    dataset_name: str

    # Where to get the trained TempME module from
    explainer: Optional[nn.Module] = None          # already-instantiated module
    explainer_ckpt: Optional[str] = None           # torch.load checkpoint path

    # Training hooks (optional)
    train_if_missing: bool = True
    force_retrain: bool = False
    trainer_overrides: Optional[Dict[str, Any]] = None
    processed_dir: Optional[str] = None            # defaults to resources/datasets/processed
    preprocess_if_missing: bool = True
    preprocess_overwrite: bool = False
    preprocess_validate: bool = False

    # Device + misc
    device: Optional[Union[str, torch.device]] = None
    alias: Optional[str] = None
    debug_mode: bool = False

    # Whether to include background node’s neighborhood as candidate edges
    use_background: bool = False


class TempMENeuralAdapter(BaseExplainer):
    """
    Adapter that wraps a *trained* TempME explainer (GRU + attention + classifier)
    and exposes it through the time_to_explain BaseExplainer API.

    It uses the exact random-walk + subgraph structures from your original
    TempME code, and returns a dense importance vector over local edges
    around the source and target.
    """

    def __init__(self, cfg: TempMENeuralAdapterConfig) -> None:
        super().__init__(name="tempme_neural", alias=cfg.alias or "tempme_neural")
        self.cfg = cfg
        self.device: Optional[torch.device] = (
            torch.device(cfg.device) if cfg.device is not None else None
        )

        self._model: Any = None
        self._dataset: Any = None
        self._events: Optional[Sequence[Any]] = None
        self._prepared: bool = False

        self._explainer: Optional[nn.Module] = cfg.explainer

        # Optional: cache per-event explanation if you want
        self._cache: Dict[int, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Lifecycle                                                         #
    # ------------------------------------------------------------------ #

    def prepare(self, *, model: Any, dataset: Any) -> None:
        """
        Attach base model + dataset and load the trained TempME explainer.
        """
        super().prepare(model=model, dataset=dataset)
        self._model = model
        self._dataset = dataset

        # Device detection
        if self.device is None:
            try:
                p = next(model.parameters())
                self.device = p.device
            except Exception:
                self.device = torch.device(
                    "cuda:0" if torch.cuda.is_available() else "cpu"
                )

        # Load events if dataset provides them
        if isinstance(dataset, dict) and "events" in dataset:
            self._events = dataset["events"]
            if "dataset_name" in dataset:
                self.cfg.dataset_name = dataset["dataset_name"]
        else:
            self._events = dataset

        # Load/attach the trained TempME module (train if missing)
        if self._explainer is None:
            ckpt_path = self._resolve_ckpt_path()
            needs_train = self.cfg.force_retrain or not (ckpt_path and ckpt_path.exists())

            if needs_train:
                if not self.cfg.train_if_missing:
                    raise FileNotFoundError(
                        f"TempME checkpoint not found at {ckpt_path} and train_if_missing=False."
                    )
                self._explainer = self._train_explainer(model, ckpt_path)
            else:
                prev_utils, active = _inject_tempme_utils()
                try:
                    self._explainer = torch.load(ckpt_path, map_location=self.device)
                finally:
                    _restore_tempme_utils(prev_utils, active)

        assert self._explainer is not None
        self._explainer.to(self.device)
        self._explainer.eval()

        self._prepared = True

    # ------------------------------------------------------------------ #
    # Single explanation                                                #
    # ------------------------------------------------------------------ #

    def explain(self, context: ExplanationContext) -> ExplanationResult:
        assert self._prepared and self._explainer is not None, "Call .prepare() first."

        t0 = time.perf_counter()

        # --- 1) event index (if available) -----------------------------
        eidx = (
            context.target.get("event_idx")
            or context.target.get("index")
            or context.target.get("idx")
        )
        if eidx is None and context.subgraph and context.subgraph.payload:
            eidx = context.subgraph.payload.get("event_idx")
        eidx = int(eidx) if eidx is not None else -1

        # Cache
        if eidx >= 0 and eidx in self._cache:
            pack = self._cache[eidx]
            return self._pack_to_result(context, eidx, pack)

        # --- 2) fetch TempME inputs from context.subgraph.payload ------
        if not context.subgraph or not context.subgraph.payload:
            if not self._ensure_tempme_payload(context):
                raise ValueError(
                    "TempMENeuralAdapter expects TempME inputs in "
                    "context.subgraph.payload (subgraphs + walks + edge ids)."
                )

        payload = context.subgraph.payload or {}

        def _payload_get(primary: str, fallback: str):
            if primary in payload and payload[primary] is not None:
                return payload[primary]
            if fallback in payload and payload[fallback] is not None:
                return payload[fallback]
            return None

        subgraph_src = _payload_get("tempme_subgraph_src", "subgraph_src")
        subgraph_tgt = _payload_get("tempme_subgraph_tgt", "subgraph_tgt")
        walks_src = _payload_get("tempme_walks_src", "walks_src")
        walks_tgt = _payload_get("tempme_walks_tgt", "walks_tgt")
        src_edge = _payload_get("tempme_edge_src", "edge_src")
        tgt_edge = _payload_get("tempme_edge_tgt", "edge_tgt")

        missing = [
            name
            for name, value in (
                ("tempme_subgraph_src/subgraph_src", subgraph_src),
                ("tempme_subgraph_tgt/subgraph_tgt", subgraph_tgt),
                ("tempme_walks_src/walks_src", walks_src),
                ("tempme_walks_tgt/walks_tgt", walks_tgt),
                ("tempme_edge_src/edge_src", src_edge),
                ("tempme_edge_tgt/edge_tgt", tgt_edge),
            )
            if value is None
        ]
        if missing:
            if self._ensure_tempme_payload(context):
                payload = context.subgraph.payload or {}
                subgraph_src = _payload_get("tempme_subgraph_src", "subgraph_src")
                subgraph_tgt = _payload_get("tempme_subgraph_tgt", "subgraph_tgt")
                walks_src = _payload_get("tempme_walks_src", "walks_src")
                walks_tgt = _payload_get("tempme_walks_tgt", "walks_tgt")
                src_edge = _payload_get("tempme_edge_src", "edge_src")
                tgt_edge = _payload_get("tempme_edge_tgt", "edge_tgt")
                missing = [
                    name
                    for name, value in (
                        ("tempme_subgraph_src/subgraph_src", subgraph_src),
                        ("tempme_subgraph_tgt/subgraph_tgt", subgraph_tgt),
                        ("tempme_walks_src/walks_src", walks_src),
                        ("tempme_walks_tgt/walks_tgt", walks_tgt),
                        ("tempme_edge_src/edge_src", src_edge),
                        ("tempme_edge_tgt/edge_tgt", tgt_edge),
                    )
                    if value is None
                ]
            if missing:
                raise KeyError(
                    "Missing keys in context.subgraph.payload for TempME: "
                    + ", ".join(missing)
                    + "."
                )

        subgraph_bgd = _payload_get("tempme_subgraph_bgd", "subgraph_bgd")
        walks_bgd = _payload_get("tempme_walks_bgd", "walks_bgd")
        bgd_edge = _payload_get("tempme_edge_bgd", "edge_bgd")

        # Timestamp for this event
        ts_val = (
            context.target.get("ts")
            or context.target.get("time")
            or payload.get("ts")
            or payload.get("time")
        )
        if ts_val is None and self._events is not None and eidx >= 0:
            try:
                ts_val = self._parse_event(self._events[eidx])[2]
            except Exception:
                ts_val = 0.0
        ts_l_cut = np.asarray([float(ts_val)], dtype=np.float32)

        # --- 3) run TempME neural explainer on walks -------------------
        expl = self._explainer
        assert expl is not None

        with torch.no_grad():
            # graphlet_imp_* : [B=1, n_walks, 1]
            graphlet_imp_src = expl(walks_src, ts_l_cut, src_edge)
            graphlet_imp_tgt = expl(walks_tgt, ts_l_cut, tgt_edge)

            graphlet_imp_bgd = None
            if (
                self.cfg.use_background
                and walks_bgd is not None
                and bgd_edge is not None
                and subgraph_bgd is not None
            ):
                graphlet_imp_bgd = expl(walks_bgd, ts_l_cut, bgd_edge)

            # Edge-level importance for src / tgt / (optional bgd)
            if hasattr(expl, "retrieve_edge_imp_node"):
                # TempME (TGN / GraphMixer case)
                src_e0, src_e1 = expl.retrieve_edge_imp_node(
                    subgraph_src, graphlet_imp_src, walks_src, training=False
                )
                tgt_e0, tgt_e1 = expl.retrieve_edge_imp_node(
                    subgraph_tgt, graphlet_imp_tgt, walks_tgt, training=False
                )
                if graphlet_imp_bgd is not None and subgraph_bgd is not None:
                    bgd_e0, bgd_e1 = expl.retrieve_edge_imp_node(
                        subgraph_bgd, graphlet_imp_bgd, walks_bgd, training=False
                    )
                else:
                    bgd_e0 = bgd_e1 = None

            elif hasattr(expl, "retrieve_edge_imp"):
                # TempME_TGAT case
                src_e0, src_e1 = expl.retrieve_edge_imp(
                    subgraph_src, graphlet_imp_src, walks_src, training=False
                )
                tgt_e0, tgt_e1 = expl.retrieve_edge_imp(
                    subgraph_tgt, graphlet_imp_tgt, walks_tgt, training=False
                )
                if graphlet_imp_bgd is not None and subgraph_bgd is not None:
                    bgd_e0, bgd_e1 = expl.retrieve_edge_imp(
                        subgraph_bgd, graphlet_imp_bgd, walks_bgd, training=False
                    )
                else:
                    bgd_e0 = bgd_e1 = None
            else:
                raise RuntimeError(
                    "TempME explainer has neither `retrieve_edge_imp_node` "
                    "nor `retrieve_edge_imp`."
                )

        # Shapes: [B, n_deg] / [B, n_deg^2] → we assume B=1
        src_e0 = src_e0[0].detach().cpu().numpy()  # [n_deg]
        src_e1 = src_e1[0].detach().cpu().numpy()  # [n_deg^2]
        tgt_e0 = tgt_e0[0].detach().cpu().numpy()
        tgt_e1 = tgt_e1[0].detach().cpu().numpy()

        if bgd_e0 is not None:
            bgd_e0_np = bgd_e0[0].detach().cpu().numpy()
            bgd_e1_np = bgd_e1[0].detach().cpu().numpy()
        else:
            bgd_e0_np = bgd_e1_np = None

        # --- 4) derive candidate_eidx from subgraphs -------------------
        # subgraph_* = (node_records, eidx_records, t_records)
        node_src, eidx_src, _ = subgraph_src
        node_tgt, eidx_tgt, _ = subgraph_tgt

        # For k=2, eidx_src[0]: [B, n_deg], eidx_src[1]: [B, n_deg*n_deg]
        src_ids_0 = eidx_src[0][0]  # np.ndarray of edge/event ids
        src_ids_1 = eidx_src[1][0]
        tgt_ids_0 = eidx_tgt[0][0]
        tgt_ids_1 = eidx_tgt[1][0]

        candidate_eidx_src = list(map(int, src_ids_0)) + list(map(int, src_ids_1))
        candidate_eidx_tgt = list(map(int, tgt_ids_0)) + list(map(int, tgt_ids_1))

        imp_src = np.concatenate([src_e0, src_e1], axis=0)
        imp_tgt = np.concatenate([tgt_e0, tgt_e1], axis=0)

        candidate_eidx: List[int] = candidate_eidx_src + candidate_eidx_tgt
        importance_edges_list: List[float] = (
            imp_src.tolist() + imp_tgt.tolist()
        )

        candidate_eidx_bgd: List[int] = []
        imp_bgd_list: List[float] = []
        if self.cfg.use_background and subgraph_bgd is not None and bgd_e0_np is not None:
            _, eidx_bgd, _ = subgraph_bgd
            bgd_ids_0 = eidx_bgd[0][0]
            bgd_ids_1 = eidx_bgd[1][0]
            candidate_eidx_bgd = list(map(int, bgd_ids_0)) + list(map(int, bgd_ids_1))
            imp_bgd_list = np.concatenate([bgd_e0_np, bgd_e1_np], axis=0).tolist()

            candidate_eidx += candidate_eidx_bgd
            importance_edges_list += imp_bgd_list

        # Normalize (optional): let metrics handle normalization; we just pass raw
        importance_edges = importance_edges_list

        # Ensure context payload has candidate_eidx in same order
        if context.subgraph and context.subgraph.payload is not None:
            context.subgraph.payload["candidate_eidx"] = candidate_eidx

        elapsed = time.perf_counter() - t0

        pack = {
            "candidate_eidx": candidate_eidx,
            "importance_edges": importance_edges,
            "candidate_eidx_src": candidate_eidx_src,
            "candidate_eidx_tgt": candidate_eidx_tgt,
            "candidate_eidx_bgd": candidate_eidx_bgd,
            "src_edge_importance_0": src_e0.tolist(),
            "src_edge_importance_1": src_e1.tolist(),
            "tgt_edge_importance_0": tgt_e0.tolist(),
            "tgt_edge_importance_1": tgt_e1.tolist(),
            "bgd_edge_importance_0": bgd_e0_np.tolist() if bgd_e0_np is not None else None,
            "bgd_edge_importance_1": bgd_e1_np.tolist() if bgd_e1_np is not None else None,
            "elapsed_sec": elapsed,
            "ts": float(ts_val),
        }

        if eidx >= 0:
            self._cache[eidx] = pack

        return self._pack_to_result(context, eidx, pack)

    # ------------------------------------------------------------------ #
    # Optional: bulk explain                                             #
    # ------------------------------------------------------------------ #

    def explain_many(self, contexts: Sequence[ExplanationContext]) -> Dict[int, Dict[str, Any]]:
        """
        Convenience wrapper: run explanations for multiple contexts and
        return the internal packed dicts for each event_idx.

        This does NOT return ExplanationResult objects, just the internal
        representation (candidate_eidx + importance_edges + extras).
        """
        results: Dict[int, Dict[str, Any]] = {}
        for ctx in contexts:
            eidx = (
                ctx.target.get("event_idx")
                or ctx.target.get("index")
                or ctx.target.get("idx")
            )
            if eidx is None:
                continue
            eidx = int(eidx)
            res = self.explain(ctx)
            pack = {
                "candidate_eidx": res.extras["candidate_eidx"],
                "importance_edges": res.importance_edges,
                "elapsed_sec": res.elapsed_sec,
                "ts": res.extras.get("ts"),
            }
            results[eidx] = pack
        return results

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _resolve_ckpt_path(self) -> Optional[Path]:
        if self.cfg.explainer_ckpt:
            return Path(self.cfg.explainer_ckpt).expanduser()
        if not self.cfg.dataset_name:
            return None
        repo_root = Path(__file__).resolve().parents[2]
        default_dir = repo_root / "resources" / "explainer" / "tempme"
        filename = f"{self.cfg.base_type}_{self.cfg.dataset_name}_tempme_ckpt.pt"
        return default_dir / filename

    def _ensure_tempme_payload(self, context: ExplanationContext) -> bool:
        if self._model is None or self._events is None:
            return False
        eidx = (
            context.target.get("event_idx")
            or context.target.get("index")
            or context.target.get("idx")
        )
        if eidx is None:
            return False
        try:
            from time_to_explain.extractors.tempme_extractor import TempMEExtractor
            extractor = TempMEExtractor(
                model=self._model,
                events=self._events,
                num_neighbors=context.num_neighbors,
            )
            subgraph = extractor.extract(
                self._dataset,
                context.target,
                k_hop=context.k_hop,
                num_neighbors=context.num_neighbors,
                window=context.window,
            )
        except Exception:
            return False
        if context.subgraph is None:
            context.subgraph = subgraph
            return True
        if context.subgraph.payload is None:
            context.subgraph.payload = {}
        if subgraph.payload:
            context.subgraph.payload.update(subgraph.payload)
        return True

    def _resolve_processed_dir(self) -> Path:
        if self.cfg.processed_dir:
            return Path(self.cfg.processed_dir).expanduser()
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / "resources" / "datasets" / "processed"

    def _ensure_preprocessed(self, processed_dir: Path) -> None:
        if not self.cfg.preprocess_if_missing:
            return
        dataset_name = self.cfg.dataset_name
        if not dataset_name:
            return
        required = [
            processed_dir / f"ml_{dataset_name}.csv",
            processed_dir / f"{dataset_name}_train_cat.h5",
            processed_dir / f"{dataset_name}_test_cat.h5",
            processed_dir / f"{dataset_name}_train_edge.npy",
            processed_dir / f"{dataset_name}_test_edge.npy",
        ]
        if all(p.exists() for p in required):
            return
        from time_to_explain.data.tempme_preprocess import TempMEPreprocessConfig, prepare_tempme_dataset
        prep_cfg = TempMEPreprocessConfig(
            dataset_name=dataset_name,
            processed_dir=processed_dir,
            output_dir=processed_dir,
            overwrite=self.cfg.preprocess_overwrite,
            validate_existing=self.cfg.preprocess_validate,
        )
        prepare_tempme_dataset(prep_cfg)

    def _train_explainer(self, model: Any, ckpt_path: Optional[Path]) -> nn.Module:
        from time_to_explain.explainer.tempme_tgn_trainer import (
            TempMETGNTrainingConfig,
            TempMETGNTrainer,
        )

        if not self.cfg.dataset_name:
            raise ValueError("TempMENeuralAdapter requires dataset_name for training.")

        base_model = getattr(model, "backbone", model)
        processed_dir = self._resolve_processed_dir()
        self._ensure_preprocessed(processed_dir)

        trainer_kwargs: Dict[str, Any] = {
            "data": self.cfg.dataset_name,
            "base_type": self.cfg.base_type,
            "device": str(self.device) if self.device is not None else "auto",
            "root_dir": str(processed_dir),
            "save_model": True,
            "log_level": "summary",
            "show_progress": False,
        }
        if self.cfg.trainer_overrides:
            trainer_kwargs.update(self.cfg.trainer_overrides)

        train_cfg = TempMETGNTrainingConfig(**trainer_kwargs)
        trainer = TempMETGNTrainer(train_cfg, base_model)
        explainer = trainer.fit()

        trained_ckpt = Path(train_cfg.explainer_save_dir) / f"{train_cfg.data}.pt"
        if trained_ckpt.exists():
            prev_utils, active = _inject_tempme_utils()
            try:
                explainer = torch.load(trained_ckpt, map_location=self.device)
            finally:
                _restore_tempme_utils(prev_utils, active)

        if ckpt_path is not None:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                torch.save(explainer, ckpt_path)
            except Exception:
                if trained_ckpt.exists():
                    try:
                        import shutil
                        shutil.copy2(trained_ckpt, ckpt_path)
                    except Exception:
                        pass

        return explainer

    def _pack_to_result(
        self,
        context: ExplanationContext,
        eidx: int,
        pack: Dict[str, Any],
    ) -> ExplanationResult:
        extras = {
            "event_idx": eidx,
            "candidate_eidx": list(pack["candidate_eidx"]),
            "candidate_eidx_src": pack.get("candidate_eidx_src"),
            "candidate_eidx_tgt": pack.get("candidate_eidx_tgt"),
            "candidate_eidx_bgd": pack.get("candidate_eidx_bgd"),
            "src_edge_importance_0": pack.get("src_edge_importance_0"),
            "src_edge_importance_1": pack.get("src_edge_importance_1"),
            "tgt_edge_importance_0": pack.get("tgt_edge_importance_0"),
            "tgt_edge_importance_1": pack.get("tgt_edge_importance_1"),
            "bgd_edge_importance_0": pack.get("bgd_edge_importance_0"),
            "bgd_edge_importance_1": pack.get("bgd_edge_importance_1"),
            "ts": pack.get("ts"),
        }

        return ExplanationResult(
            run_id=context.run_id,
            explainer=self.alias,
            context_fp=context.fingerprint(),
            importance_edges=list(pack["importance_edges"]),
            importance_nodes=None,
            importance_time=None,
            elapsed_sec=float(pack.get("elapsed_sec", 0.0)),
            extras=extras,
        )

    def _parse_event(self, ev: Any) -> Tuple[int, int, float]:
        """
        Heuristic parser used only as a fallback to get timestamp if it's not
        in context or payload.

        Supported shapes (same as in previous TempMEAdapter I gave you):
        - dict with keys: (u,v,t) or (src,dst,time) or (source,target,timestamp)
        - tuple/list: (u,v,t, ...)
        - object with attributes: .u,.v,.t or .src,.dst,.time or .source,.target,.timestamp
        """
        if isinstance(ev, dict):
            if all(k in ev for k in ("u", "v", "t")):
                u, v, t = ev["u"], ev["v"], ev["t"]
            elif all(k in ev for k in ("src", "dst", "time")):
                u, v, t = ev["src"], ev["dst"], ev["time"]
            elif all(k in ev for k in ("source", "target", "timestamp")):
                u, v, t = ev["source"], ev["target"], ev["timestamp"]
            else:
                raise KeyError(
                    "Unsupported event dict keys in TempMENeuralAdapter; "
                    "expected ('u','v','t') or ('src','dst','time') or "
                    "('source','target','timestamp')."
                )
            return int(u), int(v), float(t)

        if isinstance(ev, (list, tuple)) and len(ev) >= 3:
            u, v, t = ev[0], ev[1], ev[2]
            return int(u), int(v), float(t)

        for (u_name, v_name, t_name) in [
            ("u", "v", "t"),
            ("src", "dst", "time"),
            ("source", "target", "timestamp"),
        ]:
            if hasattr(ev, u_name) and hasattr(ev, v_name) and hasattr(ev, t_name):
                u = getattr(ev, u_name)
                v = getattr(ev, v_name)
                t = getattr(ev, t_name)
                return int(u), int(v), float(t)

        raise TypeError(
            "Unsupported event object type for TempMENeuralAdapter. "
            "Override `_parse_event` if your dataset has a different format."
        )
