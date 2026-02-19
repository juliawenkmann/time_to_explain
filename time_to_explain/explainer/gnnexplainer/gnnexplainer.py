# time_to_explain/explainer/gnn_explainer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import random

import numpy as np
import torch

from time_to_explain.core.types import ExplanationContext


_Tensor = torch.Tensor


@dataclass
class GNNExplainer:
    """
    Thin wrapper around PyTorch Geometric's GNNExplainer with robust context parsing
    and candidate-edge alignment to your framework's `ExplanationContext`.

    This engine prefers the new PyG API when available and falls back to legacy:
        - New:    torch_geometric.explain(.*)
        - Legacy: torch_geometric.nn.models.GNNExplainer

    Returned `extras` are strictly JSON-serializable (ints/floats/strings/lists).
    """

    seed: Optional[int] = None
    epochs: int = 100
    lr: float = 0.01
    feat_mask_type: str = "scalar"
    allow_new_pyg_api: bool = True
    log: bool = False
    default_scope: str = "graph"  # or "node"
    return_type: Optional[str] = None  # (new API hint) "log_probs" | "probs" | "raw" | None
    force_scope: Optional[str] = None  # "node" | "graph" | None

    # Runtime
    _model: Optional[torch.nn.Module] = None
    _dataset: Any = None
    _device: Optional[torch.device] = None

    # ------------------------------ lifecycle
    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = seed
        if self.seed is None:
            return
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def attach(self, *, model: Any, dataset: Any) -> None:
        self._model = self._unwrap_model(model)
        self._dataset = dataset
        self._device = self._infer_device(self._model)

    # ------------------------------ helpers
    def _infer_device(self, model: Any) -> torch.device:
        try:
            p = next(iter(model.parameters()))
            return p.device
        except Exception:
            return torch.device("cpu")

    def _unwrap_model(self, model: Any) -> Any:
        def _find_module(obj: Any, depth: int = 2) -> Optional[torch.nn.Module]:
            if isinstance(obj, torch.nn.Module):
                return obj
            if depth <= 0 or obj is None:
                return None
            for attr in ("backbone", "model", "module", "net", "gnn", "encoder"):
                cand = getattr(obj, attr, None)
                mod = _find_module(cand, depth - 1)
                if mod is not None:
                    return mod
            return None

        found = _find_module(model, depth=2)
        return found if found is not None else model

    def _ensure_model_compatible(self) -> None:
        if not isinstance(self._model, torch.nn.Module):
            raise RuntimeError(
                f"GNNExplainer requires a torch.nn.Module; got {type(self._model).__name__}."
            )
        base_forward = torch.nn.Module.forward
        if getattr(self._model.__class__, "forward", base_forward) is base_forward:
            raise RuntimeError(
                "GNNExplainer requires a model with an implemented forward() method."
            )

    def _to_long_edge_index(self, edge_index: Any, device: torch.device) -> _Tensor:
        ei = torch.as_tensor(edge_index, dtype=torch.long, device=device)
        if ei.dim() != 2:
            raise ValueError(f"edge_index must be 2D (got shape {tuple(ei.shape)})")
        if ei.size(0) == 2:
            return ei
        if ei.size(1) == 2:
            return ei.t().contiguous()
        raise ValueError(f"edge_index must be of shape [2, E] or [E, 2] (got {tuple(ei.shape)})")

    def _maybe_tensor(self, obj: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> Optional[_Tensor]:
        if obj is None:
            return None
        if isinstance(obj, torch.Tensor):
            return obj.to(device=device, dtype=dtype if dtype is not None else obj.dtype)
        return torch.as_tensor(obj, device=device, dtype=dtype)

    def _infer_num_nodes(self, edge_index: Any, node_ids: Optional[List[int]] = None) -> int:
        if node_ids:
            try:
                return int(max(node_ids)) + 1
            except Exception:
                pass
        if isinstance(edge_index, torch.Tensor):
            if edge_index.numel() == 0:
                return 0
            return int(edge_index.max().item()) + 1
        try:
            ei = np.asarray(edge_index)
            if ei.size == 0:
                return 0
            return int(ei.max()) + 1
        except Exception:
            return 0

    def _resolve_node_features(
        self,
        subgraph: Any,
        payload: Dict[str, Any],
        edge_index: Any,
    ) -> Optional[Any]:
        x = (
            getattr(subgraph, "node_features", None)
            or payload.get("node_features")
            or payload.get("x")
        )
        if x is None:
            dataset = self._dataset
            if isinstance(dataset, dict):
                x = dataset.get("node_features") or dataset.get("node_feats")
            else:
                x = getattr(dataset, "node_features", None) or getattr(dataset, "node_feats", None)

        if x is None:
            num_nodes = self._infer_num_nodes(edge_index, getattr(subgraph, "node_ids", None))
            if num_nodes <= 0:
                return None
            return np.zeros((num_nodes, 1), dtype=np.float32)

        num_nodes = self._infer_num_nodes(edge_index, getattr(subgraph, "node_ids", None))
        if isinstance(x, torch.Tensor):
            if x.dim() == 1:
                x = x.view(-1, 1)
            if num_nodes > 0 and x.size(0) < num_nodes:
                pad = torch.zeros(
                    (num_nodes - x.size(0), x.size(1)),
                    dtype=x.dtype,
                    device=x.device,
                )
                x = torch.cat([x, pad], dim=0)
            return x

        arr = np.asarray(x)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if num_nodes > 0 and arr.shape[0] < num_nodes:
            pad = np.zeros((num_nodes - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
            arr = np.vstack([arr, pad])
        return arr

    def _extract_payload(self, context: ExplanationContext) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extracts fields from ExplanationContext/subgraph in a defensive manner.

        Returns
        -------
        (graph_dict, meta_dict)

        graph_dict keys (best-effort):
            x:            Node features tensor [N, F]
            edge_index:   LongTensor [2, E]
            edge_attr:    Optional edge features
            edge_weight:  Optional scalar weights per edge [E]
            batch:        Optional graph batch vector [N]
            node_idx:     Optional int (node-level)
            target:       Optional int for classification target

        meta_dict keys:
            candidate_eidx, event_idx, num_nodes, num_edges
        """
        subgraph = getattr(context, "subgraph", None)
        if subgraph is None:
            raise ValueError("ExplanationContext has no `subgraph` attribute; GNNExplainer requires a graph or subgraph.")

        payload = getattr(subgraph, "payload", None) or {}

        x = getattr(subgraph, "x", None) or payload.get("x")
        edge_index = getattr(subgraph, "edge_index", None) or payload.get("edge_index")
        edge_attr = getattr(subgraph, "edge_attr", None) or payload.get("edge_attr") or payload.get("edge_weight")
        batch = getattr(subgraph, "batch", None) or payload.get("batch")

        # Node- vs graph-level scope hints:
        node_idx = payload.get("node_idx", None)
        if node_idx is None:
            node_idx = getattr(subgraph, "node_idx", None)

        # Optional target label for the explanation objective:
        target = payload.get("target", None)

        candidate = payload.get("candidate_eidx", None)
        event_idx = payload.get("event_idx", None)

        # Sanity checks:
        if edge_index is None:
            raise ValueError("`edge_index` missing from subgraph/payload.")
        if x is None:
            x = self._resolve_node_features(subgraph, payload, edge_index)
        if x is None:
            raise ValueError("`x` (node feature matrix) missing from subgraph/payload.")

        # Convert shapes/dtypes on device:
        device = self._device or torch.device("cpu")
        x = self._maybe_tensor(x, device=device, dtype=torch.float32)
        edge_index = self._to_long_edge_index(edge_index, device=device)
        edge_attr = self._maybe_tensor(edge_attr, device=device) if edge_attr is not None else None
        batch = self._maybe_tensor(batch, device=device, dtype=torch.long) if batch is not None else None

        # Normalize candidate index list (if any):
        if candidate is not None and not isinstance(candidate, list):
            try:
                candidate = list(candidate)  # works for numpy arrays / tensors
            except TypeError:
                candidate = [int(candidate)]

        graph_d = {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "batch": batch,
            "node_idx": node_idx,
            "target": target,
        }
        meta_d = {
            "candidate_eidx": candidate,
            "event_idx": event_idx,
            "num_nodes": int(x.size(0)),
            "num_edges": int(edge_index.size(1)),
        }
        return graph_d, meta_d

    def _scope_from(self, graph_d: Dict[str, Any]) -> str:
        if self.force_scope in ("node", "graph"):
            return self.force_scope
        if graph_d.get("node_idx", None) is not None:
            return "node"
        return self.default_scope

    @torch.no_grad()
    def _infer_target_if_missing(
        self,
        graph_d: Dict[str, Any],
    ) -> Optional[int]:
        """
        If no explicit target is provided, infer one by taking the argmax of the
        model prediction for the node/graph in question.
        """
        if self._model is None:
            return None

        x: _Tensor = graph_d["x"]
        ei: _Tensor = graph_d["edge_index"]
        batch: Optional[_Tensor] = graph_d.get("batch", None)
        node_idx: Optional[int] = graph_d.get("node_idx", None)

        try:
            self._model.eval()
            if batch is not None:
                out = self._model(x, ei, batch=batch)  # type: ignore[call-arg]
            else:
                out = self._model(x, ei)  # type: ignore[call-arg]

            # Node-level: out.shape ~ [N, C]; Graph-level: [G, C] or [C]
            if out is None:
                return None
            if isinstance(out, (_Tensor,)):
                if out.dim() == 1:
                    # Single graph logits: [C]
                    return int(out.argmax().item())
                if out.dim() == 2:
                    if node_idx is not None and out.size(0) == x.size(0):
                        return int(out[node_idx].argmax().item())
                    # Assume [G, C] (first graph in batch)
                    return int(out[0].argmax().item())
            return None
        except Exception:
            return None

    def _align_edge_scores(
        self,
        full_edge_mask: _Tensor,
        candidate: Optional[List[int]],
    ) -> List[float]:
        full = full_edge_mask.detach().float().cpu().tolist()
        if candidate is None:
            return [float(v) for v in full]
        aligned: List[float] = []
        E = len(full)
        for idx in candidate:
            try:
                j = int(idx)
                aligned.append(float(full[j]) if 0 <= j < E else 0.0)
            except Exception:
                aligned.append(0.0)
        return aligned

    # ------------------------------ main API
    def generate(self, context: ExplanationContext) -> Tuple[List[float], Optional[List[float]], Dict[str, Union[int, float, str, List[int]]]]:
        """
        Returns
        -------
        (edge_scores, node_scores, extras)

        - `edge_scores` is aligned with `payload['candidate_eidx']` if provided,
          otherwise covers all edges in the given subgraph order.
        - `node_scores` is typically `None` for the legacy GNNExplainer which
          returns a node *feature* mask, not per-node importance. If the new API
          yields a node mask, it will be returned (best-effort).
        """
        if self._model is None:
            raise RuntimeError("GNNExplainerEngine: no model attached. Call `attach(model=..., dataset=...)` via adapter.prepare().")

        self._ensure_model_compatible()

        graph_d, meta = self._extract_payload(context)
        scope = self._scope_from(graph_d)

        # If no target was provided, infer from the model prediction:
        if graph_d.get("target", None) is None:
            graph_d["target"] = self._infer_target_if_missing(graph_d)

        # ---------------- Prefer NEW API if available, fallback to LEGACY
        used_api = None
        node_scores: Optional[List[float]] = None
        edge_mask_tensor: Optional[_Tensor] = None
        final_loss: Optional[float] = None
        new_error: Optional[Exception] = None
        legacy_error: Optional[Exception] = None

        if self.allow_new_pyg_api:
            try:
                # New API shapes & configs:
                #   from torch_geometric.explain import Explainer, GNNExplainer
                # Some versions expose algorithm at `torch_geometric.explain.algorithm`
                used_api = "new"
                try:
                    from torch_geometric.explain import Explainer, GNNExplainer as NewGNNExplainer  # type: ignore
                except Exception:  # pragma: no cover
                    from torch_geometric.explain.algorithm import GNNExplainer as NewGNNExplainer  # type: ignore
                    from torch_geometric.explain import Explainer  # type: ignore

                # Infer return_type/mode conservatively; many models emit logits.
                model_return_type = self.return_type or "raw"
                # task_level is "node" or "graph"
                task_level = "node" if scope == "node" else "graph"

                explainer_kwargs = dict(
                    model=self._model,
                    algorithm=NewGNNExplainer(epochs=int(self.epochs), lr=float(self.lr)),
                    explanation_type="model",  # follow the model's decision boundary (newer PyG)
                    node_mask_type="attributes",
                    edge_mask_type="object",
                    model_config=dict(
                        mode="classification",
                        task_level=task_level,
                        return_type=model_return_type,  # "raw" | "log_probs" | "probs" (version-dependent)
                    ),
                )
                def _filter_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
                    try:
                        import inspect
                        sig = inspect.signature(Explainer)
                        allowed = set(sig.parameters.keys())
                        return {k: v for k, v in kwargs.items() if k in allowed}
                    except Exception:
                        return kwargs

                def _build_explainer(kwargs: Dict[str, Any]):
                    try:
                        return Explainer(**_filter_kwargs(kwargs))
                    except TypeError as exc:
                        msg = str(exc)
                        for key in ("explanation_type", "node_mask_type", "edge_mask_type", "model_config"):
                            if key in msg and key in kwargs:
                                trimmed = dict(kwargs)
                                trimmed.pop(key, None)
                                return _build_explainer(trimmed)
                        raise

                explainer = _build_explainer(explainer_kwargs)

                if scope == "node":
                    exp = explainer(
                        graph_d["x"],
                        graph_d["edge_index"],
                        target=graph_d.get("target", None),
                        index=int(graph_d["node_idx"]),
                        batch=graph_d.get("batch", None),
                        edge_attr=graph_d.get("edge_attr", None),
                    )
                else:
                    exp = explainer(
                        graph_d["x"],
                        graph_d["edge_index"],
                        target=graph_d.get("target", None),
                        batch=graph_d.get("batch", None),
                        edge_attr=graph_d.get("edge_attr", None),
                    )

                # New API returns an Explanation object:
                edge_mask_tensor = getattr(exp, "edge_mask", None)
                node_mask_tensor = getattr(exp, "node_mask", None)
                if node_mask_tensor is not None:
                    node_scores = [float(v) for v in node_mask_tensor.detach().float().cpu().view(-1).tolist()]

                # The new API stores loss in `exp` for some versions; best-effort:
                try:
                    final_loss = float(getattr(exp, "loss", math.nan))
                except Exception:
                    final_loss = None

            except Exception as exc:
                new_error = exc
                edge_mask_tensor = None
                node_scores = None
                final_loss = None

        if edge_mask_tensor is None:
            try:
                from torch_geometric.nn.models import GNNExplainer as LegacyGNNExplainer  # type: ignore
                used_api = "legacy"
                explainer = LegacyGNNExplainer(
                    self._model,
                    epochs=int(self.epochs),
                    lr=float(self.lr),
                    feat_mask_type=self.feat_mask_type,
                    log=self.log,
                )

                if scope == "node":
                    node_idx = int(graph_d["node_idx"])
                    # Legacy signatures accept optional kwargs (edge_attr/batch):
                    kwargs = {}
                    if graph_d.get("edge_attr", None) is not None:
                        kwargs["edge_attr"] = graph_d["edge_attr"]
                    if graph_d.get("batch", None) is not None:
                        kwargs["batch"] = graph_d["batch"]
                    node_feat_mask, edge_mask = explainer.explain_node(
                        node_idx,
                        graph_d["x"],
                        graph_d["edge_index"],
                        **kwargs,
                    )
                    # Legacy API returns (node_feature_mask, edge_mask)
                    edge_mask_tensor = edge_mask
                    # GNNExplainer (legacy) does not produce per-node mask; we leave node_scores=None
                else:
                    kwargs = {}
                    if graph_d.get("edge_attr", None) is not None:
                        kwargs["edge_attr"] = graph_d["edge_attr"]
                    if graph_d.get("batch", None) is not None:
                        kwargs["batch"] = graph_d["batch"]

                    node_feat_mask, edge_mask = explainer.explain_graph(
                        graph_d["x"],
                        graph_d["edge_index"],
                        **kwargs,
                    )
                    edge_mask_tensor = edge_mask

                # Best-effort get final loss if present (not guaranteed):
                try:
                    final_loss = float(getattr(explainer, "loss", math.nan))
                except Exception:
                    final_loss = None

            except Exception as exc:
                legacy_error = exc

        if edge_mask_tensor is None:
            details = []
            if new_error is not None:
                details.append(f"new API error: {repr(new_error)}")
            if legacy_error is not None:
                details.append(f"legacy error: {repr(legacy_error)}")
            detail_txt = "\n".join(details) if details else "unknown error"
            raise RuntimeError(
                "GNNExplainerEngine could not import a compatible PyTorch Geometric GNNExplainer.\n"
                "Please install/upgrade PyG. Tried new API (`torch_geometric.explain`) and legacy "
                "(`torch_geometric.nn.models.GNNExplainer`).\n"
                f"Original error: {detail_txt}"
            )

        # --------------------------------- finalize
        assert edge_mask_tensor is not None, "Internal error: GNNExplainer didn't yield an edge mask."

        # Align to candidate set if provided, else return full mask order:
        edge_scores = self._align_edge_scores(edge_mask_tensor, meta.get("candidate_eidx"))

        extras: Dict[str, Union[int, float, str, List[int]]] = {
            "algorithm": "GNNExplainer",
            "pyg_api": used_api or "unknown",
            "epochs": int(self.epochs),
            "lr": float(self.lr),
            "feat_mask_type": str(self.feat_mask_type),
            "seed": self.seed if self.seed is not None else -1,
            "scope": scope,
            "target": int(graph_d["target"]) if graph_d.get("target") is not None else -1,
            "n_nodes": meta["num_nodes"],
            "n_edges_total": meta["num_edges"],
            "n_edges_returned": len(edge_scores),
            "candidate_eidx": meta.get("candidate_eidx") or [],
            "event_idx": meta.get("event_idx"),
        }
        if final_loss is not None and not math.isnan(final_loss):
            extras["final_loss"] = float(final_loss)

        return edge_scores, node_scores, extras


__all__ = ["GNNExplainer"]
