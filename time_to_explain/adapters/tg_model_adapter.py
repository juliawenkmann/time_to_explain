"""
Adapter that wraps TGNN backbones (TGN / TGAT) and exposes the predict_proba
API expected by the evaluation framework and fidelity metrics.

Extended to optionally support a *differentiable* soft edge mask for
gradient-based explainers: if `edge_mask` is a torch.Tensor, the adapter
builds a full-length mask over all historical events, injects the candidate
mask values, multiplies the backbone's edge features by this mask, and runs
the forward pass without `torch.no_grad()`, allowing gradients to flow back
to the mask.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from time_to_explain.core.types import ModelProtocol, Subgraph


@dataclass
class TemporalEvent:
    src: int
    dst: int
    ts: float
    eid: Optional[int] = None


class TemporalGNNModelAdapter(ModelProtocol):
    """
    Wraps a temporal GNN backbone (TGN/TGAT) so that it satisfies the
    ModelProtocol expected by EvaluationRunner and fidelity metrics.

    The underlying backbone must expose a ``get_prob`` method matching the
    original implementation: it receives source ids, destination ids, timestamps
    (numpy arrays) and optionally ``edge_idx_preserve_list`` that determines
    which historical edges are allowed to influence the prediction.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        events: Any,
        *,
        device: Optional[torch.device] = None,
        return_logit: bool = True,
    ) -> None:
        self.backbone = backbone
        self.events = events
        self.return_logit = return_logit

        if device is not None:
            self.device = torch.device(device)
        else:
            try:
                self.device = next(backbone.parameters()).device
            except StopIteration:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if hasattr(self.backbone, "to"):
            self.backbone.to(self.device)
        self.backbone.eval()

        # Ensure the backbone advertises a concrete neighbor count; several
        # explainers assume this is an int when building temporal windows.
        backbone_neighbors = getattr(self.backbone, "num_neighbors", None)
        if backbone_neighbors is None:
            fallback = getattr(getattr(self.backbone, "ngh_finder", None), "num_neighbors", 20)
            self.backbone.num_neighbors = fallback
            backbone_neighbors = fallback
        self.num_neighbors = backbone_neighbors

    # ------------------------------------------------------------------ public #
    def predict_proba(self, subgraph: Subgraph, target: dict) -> Any:
        return self._score(target, subgraph=subgraph, edge_mask=None)

    def predict_proba_with_mask(
        self,
        subgraph: Subgraph,
        target: dict,
        edge_mask: Optional[Sequence[float]] = None,
        node_mask: Optional[Sequence[float]] = None,
    ) -> Any:
        del node_mask  # node masks are not used by TGNN models
        return self._score(target, subgraph=subgraph, edge_mask=edge_mask)

    # ---------------------------------------------------------------- helpers #
    def _score(
        self,
        target: dict,
        *,
        subgraph: Optional[Subgraph],
        edge_mask: Optional[Sequence[float]],
    ) -> float:
        event = self._resolve_event(target, subgraph=subgraph)

        # Soft mask path: expect a torch.Tensor aligned to candidate_eidx.
        if isinstance(edge_mask, torch.Tensor):
            if subgraph is None or not subgraph.payload:
                raise ValueError("Tensor edge_mask provided but subgraph payload is missing.")
            candidate = subgraph.payload.get("candidate_eidx")
            if candidate is None:
                raise ValueError("Subgraph payload does not contain 'candidate_eidx'; cannot apply edge mask.")
            if len(candidate) != len(edge_mask):
                raise ValueError(
                    f"Edge mask length ({len(edge_mask)}) does not match candidate_eidx length ({len(candidate)})."
                )
            if not hasattr(self.backbone, "edge_raw_features"):
                raise ValueError("Backbone is missing 'edge_raw_features'; cannot apply differentiable mask.")

            edge_features = getattr(self.backbone, "edge_raw_features")
            if not isinstance(edge_features, torch.Tensor):
                raise ValueError("'edge_raw_features' must be a torch.Tensor for differentiable masking.")

            if edge_mask.ndim != 1:
                raise ValueError("edge_mask must be 1D over candidate edges.")

            soft_mask = edge_mask.to(device=self.device, dtype=edge_features.dtype)
            num_edges, feat_dim = edge_features.shape

            base = torch.ones(num_edges, device=self.device, dtype=edge_features.dtype)
            candidate_idx = torch.as_tensor(candidate, device=self.device, dtype=torch.long)
            candidate_idx = torch.clamp(candidate_idx.long() - 1, min=0)

            full_mask = base.scatter(0, candidate_idx, soft_mask)
            masked_edge_features = edge_features * full_mask.view(-1, 1)

            src = np.asarray([event.src], dtype=np.int64)
            dst = np.asarray([event.dst], dtype=np.int64)
            ts = np.asarray([event.ts], dtype=np.float32)

            kwargs = {"logit": self.return_logit}
            original_feats = getattr(self.backbone, "edge_raw_features")
            try:
                self.backbone.edge_raw_features = masked_edge_features
                score = self.backbone.get_prob(src, dst, ts, **kwargs)
            finally:
                self.backbone.edge_raw_features = original_feats

            if not isinstance(score, torch.Tensor):
                score = torch.as_tensor(score, device=self.device, dtype=edge_features.dtype)
            return score.squeeze()

        edge_idx_preserve = self._build_preserve_list(edge_mask, subgraph=subgraph)

        src = np.asarray([event.src], dtype=np.int64)
        dst = np.asarray([event.dst], dtype=np.int64)
        ts = np.asarray([event.ts], dtype=np.float32)

        kwargs = {"logit": self.return_logit}
        if edge_idx_preserve is not None:
            kwargs["edge_idx_preserve_list"] = edge_idx_preserve

        with torch.no_grad():
            score = self.backbone.get_prob(src, dst, ts, **kwargs)

        return self._to_scalar(score)

    def _resolve_event(
        self,
        target: dict,
        *,
        subgraph: Optional[Subgraph],
    ) -> TemporalEvent:
        if subgraph and subgraph.payload:
            payload = subgraph.payload
            if {"u", "i", "ts"}.issubset(payload.keys()):
                return TemporalEvent(
                    src=int(payload["u"]),
                    dst=int(payload["i"]),
                    ts=float(payload["ts"]),
                    eid=int(payload.get("event_idx") or payload.get("idx") or payload.get("index", 0)) or None,
                )

        eidx = (
            target.get("event_idx")
            or target.get("index")
            or target.get("idx")
            or (subgraph.payload.get("event_idx") if subgraph and subgraph.payload else None)
        )
        if eidx is None:
            raise ValueError("TemporalGNNModelAdapter requires an 'event_idx' in target or subgraph payload.")

        eidx = int(eidx)
        # The loader returns a pandas DataFrame; fall back to positional columns [u, i, ts].
        row = self.events.iloc[eidx - 1]
        # Support both attribute access and dict-style access.
        src = int(row["u"] if "u" in row else row.iloc[0])
        dst = int(row["i"] if "i" in row else row.iloc[1])
        ts = float(row["ts"] if "ts" in row else row.iloc[2])
        return TemporalEvent(src=src, dst=dst, ts=ts, eid=eidx)

    def _build_preserve_list(
        self,
        edge_mask: Optional[Sequence[float]],
        *,
        subgraph: Optional[Subgraph],
    ) -> Optional[List[int]]:
        if edge_mask is None:
            return None
        if subgraph is None or not subgraph.payload:
            raise ValueError("Edge mask provided but subgraph payload is missing; cannot align mask to candidate edges.")

        candidate = subgraph.payload.get("candidate_eidx")
        if candidate is None:
            raise ValueError("Subgraph payload does not contain 'candidate_eidx'; cannot apply edge mask.")

        if len(candidate) != len(edge_mask):
            raise ValueError(
                f"Edge mask length ({len(edge_mask)}) does not match candidate_eidx length ({len(candidate)})."
            )

        preserve = [int(e) for e, m in zip(candidate, edge_mask) if m and m > 0]
        return preserve

    @staticmethod
    def _to_scalar(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return float(value.item())
            return float(value.reshape(-1)[0].item())
        if isinstance(value, (np.ndarray, list, tuple)):
            arr = np.asarray(value).reshape(-1)
            return float(arr[0])
        return float(value)

    # Delegate attribute/parameter lookups to the underlying backbone so that
    # existing code (extractors, explainers) can keep using the original API.
    def __getattr__(self, item: str) -> Any:
        return getattr(self.backbone, item)

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.backbone.parameters()
