# time_to_explain/adapters/tempme_adapter.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List, Union
from pathlib import Path
import sys
import importlib

import torch
import numpy as np

from ..core.types import BaseExplainer, ExplanationContext, ExplanationResult
from .tempme_base_adapter import TempMEBaseAdapter


def _load_official_tempme_classes():
    """
    Import TempME classes from the official submodule, ensuring its local
    `utils` package is visible for absolute imports inside the submodule.
    """
    tempme_root = Path(__file__).resolve().parents[2] / "submodules" / "explainer" / "TempME"
    if str(tempme_root) not in sys.path:
        sys.path.insert(0, str(tempme_root))

    prev_utils = sys.modules.get("utils")
    try:
        import submodules.explainer.TempME.utils as tempme_utils
        sys.modules["utils"] = tempme_utils
        models_mod = importlib.import_module("submodules.explainer.TempME.models")
        models_mod = importlib.reload(models_mod)
        OfficialTempME = models_mod.TempME
        OfficialTempME_TGAT = models_mod.TempME_TGAT
    finally:
        if prev_utils is not None:
            sys.modules["utils"] = prev_utils
        else:
            sys.modules.pop("utils", None)

    return OfficialTempME, OfficialTempME_TGAT


def _load_tempme_classes(prefer_official: bool = True):
    if prefer_official:
        return _load_official_tempme_classes()
    from ..explainer.tempme.tempme import TempME, TempME_TGAT  # fallback to local copy
    return TempME, TempME_TGAT


def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x


class TempMEExplainer(BaseExplainer):
    """
    Adapter to run TempME under the unified BaseExplainer -> ExplanationResult API.

    Expected payload in `context.subgraph.payload` (produced by the extractor below):
      {
        "subgraph_src": (node_records_src, eidx_records_src, t_records_src),
        "subgraph_tgt": (node_records_tgt, eidx_records_tgt, t_records_tgt),
        "subgraph_bgd": (node_records_bgd, eidx_records_bgd, t_records_bgd),
        "walks_src":  ...,
        "walks_tgt":  ...,
        "walks_bgd":  ...,
        "edge_src":   ...,
        "edge_tgt":   ...,
        "edge_bgd":   ...,
        # optional: "dst_l_fake": tensor or array
      }

    Returns:
      ExplanationResult with a single flattened `importance_edges` vector, plus
      a structured copy (per set: src/tgt/bgd) in `extras["per_set_importances"]`.
    """

    def __init__(
        self,
        *,
        base_type: str = "tgn",            # "tgn" | "tgat" | "graphmixer"
        checkpoint: Optional[str] = None,  # torch.save(...) of the trained TempME
        out_dim: int = 40,
        hid_dim: int = 64,
        temp: float = 0.07,
        dropout_p: float = 0.1,
        if_bern: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dataset_name: Optional[str] = None,
        use_official: bool = True,
        alias: Optional[str] = None,
    ) -> None:
        super().__init__(name="tempme", alias=alias or f"tempme_{base_type}")
        self.base_type = base_type.lower()
        self.checkpoint = checkpoint
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.temp = temp
        self.dropout_p = dropout_p
        self.if_bern = if_bern
        self.device = torch.device(device) if device is not None else None
        self.dataset_name = dataset_name
        self.use_official = use_official

        self.explainer = None
        self._prepared = False

    # Called once by the runner
    def prepare(self, *, model: Any, dataset: Any) -> None:
        super().prepare(model=model, dataset=dataset)

        # Wrap canonical TGN/TGAT into TempME-compatible base if needed
        if not isinstance(model, TempMEBaseAdapter) and not self._looks_like_tempme_base(model):
            model = TempMEBaseAdapter(model)
        self.base_wrapper = model
        # Decide device: prefer the base model's device
        if self.device is None:
            try:
                # Heuristic: find a tensor parameter
                p = next(model.parameters())
                self.device = p.device
            except Exception:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if isinstance(dataset, dict) and "dataset_name" in dataset:
            self.dataset_name = dataset["dataset_name"]
        data_name = self.dataset_name
        if data_name is None:
            raise ValueError("TempMEExplainer requires dataset_name to construct the official TempME model.")

        TempME, TempME_TGAT = _load_tempme_classes(self.use_official)

        # Build the TempME module
        if self.base_type == "tgat":
            if TempME_TGAT is None:
                raise ImportError("TempME_TGAT not found. Ensure it's exported from models.")
            self.explainer = TempME_TGAT(
                model, data=data_name,
                out_dim=self.out_dim, hid_dim=self.hid_dim, temp=self.temp,
                dropout_p=self.dropout_p, device=self.device,
            )
        else:
            if TempME is None:
                raise ImportError("TempME not found. Ensure it's exported from models.")
            self.explainer = TempME(
                model, base_model_type=self.base_type, data=data_name,
                out_dim=self.out_dim, hid_dim=self.hid_dim, temp=self.temp,
                if_cat_feature=True, dropout_p=self.dropout_p, device=self.device,
            )

        self.explainer = self.explainer.to(self.device)
        # keep a handle to the wrapper for payload building
        self.explainer.base_model = self.base_wrapper  # type: ignore[attr-defined]

        # Load checkpoint if provided
        if self.checkpoint is not None:
            ckpt = torch.load(self.checkpoint, map_location=self.device)
            # Allow either a full module file or state_dict
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                self.explainer.load_state_dict(ckpt["state_dict"], strict=False)
            else:
                try:
                    self.explainer.load_state_dict(ckpt.state_dict(), strict=False)  # type: ignore[attr-defined]
                except Exception:
                    # If user saved the module directly via torch.save(explainer, ...)
                    self.explainer = ckpt.to(self.device)
        self.explainer.eval()
        self._prepared = True

    # One explanation per anchor/context
    def explain(self, context: ExplanationContext) -> ExplanationResult:
        assert self._prepared, "Call .prepare(model=..., dataset=...) before explain()."

        # --- Pull payload constructed by the extractor
        payload: Dict[str, Any] = {}
        if context.subgraph is not None and context.subgraph.payload is not None:
            payload = context.subgraph.payload
        required = {
            "subgraph_src",
            "subgraph_tgt",
            "subgraph_bgd",
            "walks_src",
            "walks_tgt",
            "walks_bgd",
            "edge_src",
            "edge_tgt",
            "edge_bgd",
        }
        if not required.issubset(payload.keys()):
            # minimal on-the-fly payload using the adapter's grab_subgraph (no precomputed HDF5 needed)
            payload = self._build_minimal_payload(context)

        subgraph_src = payload["subgraph_src"]
        subgraph_tgt = payload["subgraph_tgt"]
        subgraph_bgd = payload["subgraph_bgd"]
        walks_src = payload["walks_src"]
        walks_tgt = payload["walks_tgt"]
        walks_bgd = payload["walks_bgd"]
        edge_src = payload["edge_src"]
        edge_tgt = payload["edge_tgt"]
        edge_bgd = payload["edge_bgd"]

        # Build a (batch=1) timestamp tensor from the anchor ts
        ts = float(context.target.get("ts", 0.0))
        ts_t = torch.as_tensor([ts], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # Forward pass to get graphlet importances
            if self.base_type == "tgat":
                raise NotImplementedError("TempME_TGAT path not fully supported in the unified runner.")
            else:
                gimp_src = self.explainer(_to_device(walks_src, self.device), ts_t, _to_device(edge_src, self.device))
                gimp_tgt = self.explainer(_to_device(walks_tgt, self.device), ts_t, _to_device(edge_tgt, self.device))
                gimp_bgd = self.explainer(_to_device(walks_bgd, self.device), ts_t, _to_device(edge_bgd, self.device))

            # Convert graphlet importances to edge importances
            if self.base_type == "tgat":
                # For TGAT we go through retrieve_edge_imp per set and then flatten
                eimp_src = self.explainer.retrieve_edge_imp(subgraph_src, gimp_src, walks_src, training=self.if_bern)
                eimp_tgt = self.explainer.retrieve_edge_imp(subgraph_tgt, gimp_tgt, walks_tgt, training=self.if_bern)
                eimp_bgd = self.explainer.retrieve_edge_imp(subgraph_bgd, gimp_bgd, walks_bgd, training=self.if_bern)

                # Each eimp_* is typically a list[tensor] (levels); concatenate along last dim
                def _cat_levels(x):
                    if isinstance(x, list):
                        x = [xi if isinstance(xi, torch.Tensor) else torch.as_tensor(xi) for xi in x]
                        return torch.cat([xi.squeeze(0) for xi in x], dim=-1)  # drop batch dim
                    return torch.as_tensor(x).squeeze(0)

                e_src = _cat_levels(eimp_src)
                e_tgt = _cat_levels(eimp_tgt)
                e_bgd = _cat_levels(eimp_bgd)

                per_set = {
                    "src": e_src.detach().cpu().tolist(),
                    "tgt": e_tgt.detach().cpu().tolist(),
                    "bgd": e_bgd.detach().cpu().tolist(),
                }
                importance_edges = torch.cat([e_src, e_tgt, e_bgd], dim=-1).detach().cpu().tolist()

            else:
                # Non-TGAT variants often provide a joint retrieve_explanation
                explanation = self.explainer.retrieve_explanation(
                    subgraph_src, gimp_src, walks_src,
                    subgraph_tgt, gimp_tgt, walks_tgt,
                    subgraph_bgd, gimp_bgd, walks_bgd,
                    training=self.if_bern
                )
                # Normalize into a single vector + per-set
                per_set, importance_edges = self._normalize_explanation(explanation)

        return ExplanationResult(
            run_id=context.run_id,
            explainer=self.alias,
            context_fp=context.fingerprint(),
            importance_edges=importance_edges,
            importance_nodes=None,
            importance_time=None,
            extras={"per_set_importances": per_set, "base_type": self.base_type},
        )

    @staticmethod
    def _looks_like_tempme_base(model: Any) -> bool:
        emb = getattr(model, "embedding_module", None)
        return emb is not None and hasattr(emb, "embedding_update")

    # --- helpers ---
    def _normalize_explanation(self, explanation: Any) -> Tuple[Dict[str, List[float]], List[float]]:
        """
        Makes a best-effort conversion of TempME's explanation outputs into a flat edge vector.
        We also return a structured per-set version (src/tgt/bgd) for inspection.
        """
        # Common patterns seen in TempME code:
        # - A list/tuple of tensors for hops/levels per-set, e.g., edge_imp_src is a list of tensors
        # - Sometimes explanation is an object or dict; we try to be permissive.
        def _to_tensor_list(x) -> List[torch.Tensor]:
            if isinstance(x, list) or isinstance(x, tuple):
                out = []
                for xi in x:
                    if isinstance(xi, torch.Tensor):
                        out.append(xi)
                    else:
                        out.append(torch.as_tensor(xi))
                return out
            elif isinstance(x, torch.Tensor):
                return [x]
            else:
                return [torch.as_tensor(x)]

        # Try common keys first
        if isinstance(explanation, dict) and all(k in explanation for k in ("src", "tgt", "bgd")):
            src_list = _to_tensor_list(explanation["src"])
            tgt_list = _to_tensor_list(explanation["tgt"])
            bgd_list = _to_tensor_list(explanation["bgd"])
        else:
            # Fallback heuristic: some implementions return a list like [src, tgt, bgd] or nested lists
            if isinstance(explanation, (list, tuple)) and len(explanation) >= 3:
                src_list = _to_tensor_list(explanation[0])
                tgt_list = _to_tensor_list(explanation[1])
                bgd_list = _to_tensor_list(explanation[2])
            else:
                # Last resort: treat everything as "src"
                src_list = _to_tensor_list(explanation)
                tgt_list = []
                bgd_list = []

        def _cat_and_squeeze(tl: List[torch.Tensor]) -> torch.Tensor:
            if len(tl) == 0:
                return torch.empty(0)
            tl = [t.squeeze(0) for t in tl]  # drop batch dim if present
            return torch.cat(tl, dim=-1)

        e_src = _cat_and_squeeze(src_list)
        e_tgt = _cat_and_squeeze(tgt_list)
        e_bgd = _cat_and_squeeze(bgd_list)
        per_set = {
            "src": e_src.detach().cpu().tolist(),
            "tgt": e_tgt.detach().cpu().tolist(),
            "bgd": e_bgd.detach().cpu().tolist(),
        }
        importance_edges = torch.cat([e_src, e_tgt, e_bgd], dim=-1).detach().cpu().tolist()
        return per_set, importance_edges

    # --------- minimal payload construction (fallback) ----------
    def _build_minimal_payload(self, context: ExplanationContext) -> Dict[str, Any]:
        """
        Build a lightweight TempME payload from events + ngh_finder without relying on precomputed HDF5.
        Shapes are chosen to satisfy TempME forward: walks tensors with n_walk=1, len_walk=3.
        """
        events = self.dataset["events"] if isinstance(self.dataset, dict) else self.dataset
        eidx = int(context.target.get("event_idx") or context.target.get("idx") or context.target.get("index"))
        row = events.iloc[eidx - 1]
        src = int(row[0]); dst = int(row[1]); ts = float(row[2])

        # sample a negative destination (background) different from dst
        n_nodes = int(max(events.iloc[:, 0].max(), events.iloc[:, 1].max())) + 1
        bgd = dst
        if n_nodes > 1:
            bgd = (dst + 1) % n_nodes

        # grab 2-hop subgraphs for src/dst/bgd
        base = getattr(self, "base_wrapper", None)
        sub_src = base.grab_subgraph([src], [ts])
        sub_tgt = base.grab_subgraph([dst], [ts])
        sub_bgd = base.grab_subgraph([bgd], [ts])

        # minimal walks: (node_idx, edge_idx, time_idx, cat_feat, extra)
        def make_walk(node_id):
            node_idx = np.array([[[node_id, dst, 0, 0, 0, 0]]])  # shape [1,1,6]
            edge_idx = np.zeros((1, 1, 3), dtype=np.int64)
            time_idx = np.zeros((1, 1, 3), dtype=np.float32)
            cat_feat = np.zeros((1, 1, 1), dtype=np.int64)
            out_anony = np.zeros((1, 1, 3), dtype=np.int64)
            return (node_idx, edge_idx, time_idx, cat_feat, out_anony)

        walks_src = make_walk(src)
        walks_tgt = make_walk(dst)
        walks_bgd = make_walk(bgd)

        edge_src = np.zeros((1, 1), dtype=np.int64)
        edge_tgt = np.zeros((1, 1), dtype=np.int64)
        edge_bgd = np.zeros((1, 1), dtype=np.int64)

        return {
            "subgraph_src": sub_src,
            "subgraph_tgt": sub_tgt,
            "subgraph_bgd": sub_bgd,
            "walks_src": walks_src,
            "walks_tgt": walks_tgt,
            "walks_bgd": walks_bgd,
            "edge_src": edge_src,
            "edge_tgt": edge_tgt,
            "edge_bgd": edge_bgd,
            "event_idx": eidx,
            "ts": ts,
        }
