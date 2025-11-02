# time_to_explain/adapters/tempme_adapter.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List, Union

import torch
import numpy as np

from ..core.types import BaseExplainer, ExplanationContext, ExplanationResult

# Your repo should provide these:
# - TempME / TempME_TGAT with .forward(walks, ts, edge_ids) -> graphlet_importances
# - .retrieve_explanation(...) or .retrieve_edge_imp(...)

from ..explainer.tempme.tempme import TempME, TempME_TGAT  


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

        self.explainer = None
        self._prepared = False

    # Called once by the runner
    def prepare(self, *, model: Any, dataset: Any) -> None:
        super().prepare(model=model, dataset=dataset)

        # Decide device: prefer the base model's device
        if self.device is None:
            try:
                # Heuristic: find a tensor parameter
                p = next(model.parameters())
                self.device = p.device
            except Exception:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Build the TempME module
        if self.base_type == "tgat":
            if TempME_TGAT is None:
                raise ImportError("TempME_TGAT not found. Ensure it's exported from models.")
            self.explainer = TempME_TGAT(
                model, data=getattr(self, "data", None),
                out_dim=self.out_dim, hid_dim=self.hid_dim, temp=self.temp,
                dropout_p=self.dropout_p, device=self.device,
            )
        else:
            if TempME is None:
                raise ImportError("TempME not found. Ensure it's exported from models.")
            self.explainer = TempME(
                model, base_model_type=self.base_type, data=getattr(self, "data", None),
                out_dim=self.out_dim, hid_dim=self.hid_dim, temp=self.temp,
                if_cat_feature=True, dropout_p=self.dropout_p, device=self.device,
            )

        self.explainer = self.explainer.to(self.device)

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
        if context.subgraph is None or context.subgraph.payload is None:
            raise ValueError(
                "TempMEExplainer expects context.subgraph.payload with the TempME fields. "
                "Use the TempMEExtractor or build the payload yourself."
            )
        payload: Dict[str, Any] = context.subgraph.payload

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
