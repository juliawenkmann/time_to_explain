from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json, time, random, os, csv

from .types import ExplanationContext, ExplanationResult, BaseExplainer, ModelProtocol, SubgraphExtractorProtocol
from .registry import METRICS
from time_to_explain.metrics import ensure_builtin_metrics_loaded

# Ensure built-in metrics are registered before the runner looks them up.
ensure_builtin_metrics_loaded()

def set_global_seed(seed: int) -> None:
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)          # type: ignore
        torch.cuda.manual_seed_all(seed) # type: ignore
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False     # type: ignore
    except Exception:
        pass
    random.seed(seed)

@dataclass
class EvalConfig:
    out_dir: str = "runs"
    seed: int = 42
    metrics: List[str] = field(default_factory=lambda: ["sparsity"])
    save_jsonl: bool = True
    save_csv: bool = True

class EvaluationRunner:
    def __init__(self, *, model: ModelProtocol, dataset: Any, extractor: SubgraphExtractorProtocol = None,
                 explainers: Sequence[BaseExplainer], config: Optional[EvalConfig] = None,
                 extractor_map: Optional[Dict[str, SubgraphExtractorProtocol]] = None) -> None:
        """
        extractor: default extractor used for all explainers (backward compatible)
        extractor_map: optional mapping explainer.alias|name -> extractor to override per explainer
        """
        self.model = model
        self.dataset = dataset
        self.extractor = extractor
        self.extractor_map = extractor_map or {}
        self.explainers = list(explainers)
        self.config = config or EvalConfig()
        set_global_seed(self.config.seed)

        # ---- build metric objects from registry factories (one style only) ----
        self._metrics = []
        metrics_spec = self.config.metrics
        items = metrics_spec.items() if isinstance(metrics_spec, dict) else [(name, {}) for name in (metrics_spec or [])]
        for name, mcfg in items:
            factory = METRICS.get(name)  # MUST exist; else KeyError points to missing import/registration
            metric_obj = factory(dict(mcfg))  # factory -> BaseMetric
            if hasattr(metric_obj, "setup"):
                try:
                    metric_obj.setup(model=self.model, dataset=self.dataset)
                except TypeError:
                    metric_obj.setup(self.model, self.dataset)
            self._metrics.append(metric_obj)

    def _ensure_out(self, path: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)

    def run(self, anchors: Sequence[Dict[str, Any]], *, k_hop: int = 2,
            num_neighbors: int = 50, window: Optional[tuple] = None,
            run_id: Optional[str] = None) -> Dict[str, Any]:
        import os, json, csv, time as _time

        run_id = run_id or str(int(_time.time()))
        out_dir = os.path.join(self.config.out_dir, run_id)
        self._ensure_out(out_dir)

        for e in self.explainers:
            e.prepare(model=self.model, dataset=self.dataset)

        jsonl_path = os.path.join(out_dir, "results.jsonl")
        csv_path = os.path.join(out_dir, "metrics.csv")
        jsonl_f = open(jsonl_path, "w") if self.config.save_jsonl else None
        csv_f = open(csv_path, "w", newline="") if self.config.save_csv else None
        csv_writer = None
        csv_header = None
        total_anchors = len(anchors) if hasattr(anchors, "__len__") else None
        total_explainers = len(self.explainers)

        try:
            for idx, anchor in enumerate(anchors):
                ctx = ExplanationContext(
                    run_id=run_id, target_kind=anchor.get("target_kind","edge"),
                    target=anchor, window=window, k_hop=k_hop, num_neighbors=num_neighbors
                )
                target_label = (
                    anchor.get("event_idx")
                    or anchor.get("idx")
                    or anchor.get("index")
                    or anchor.get("target")
                )
                anchor_prefix = f"{idx + 1}" if total_anchors is None else f"{idx + 1}/{total_anchors}"
                print(f"\n[EvaluationRunner] Anchor {anchor_prefix} (target={target_label})")

                for expl_idx, explainer in enumerate(self.explainers):
                    # resolve extractor for this explainer
                    ext = self.extractor_map.get(getattr(explainer, "alias", explainer.name), None) \
                        or self.extractor_map.get(explainer.name, None) \
                        or self.extractor
                    if ext is None:
                        raise ValueError(f"No extractor provided for explainer '{explainer.alias}'.")

                    subg = ext.extract(self.dataset, anchor, k_hop=k_hop,
                                       num_neighbors=num_neighbors, window=window)
                    ctx.subgraph = subg

                    label = getattr(explainer, "alias", getattr(explainer, "name", explainer.__class__.__name__))
                    progress = f"{expl_idx + 1}/{total_explainers}"
                    print(f"[EvaluationRunner]   [{progress}] {label}: start")
                    t0 = _time.perf_counter()
                    res: ExplanationResult = explainer.explain(ctx)
                    res.elapsed_sec = _time.perf_counter() - t0
                    res.context_fp = ctx.fingerprint()
                    print(f"[EvaluationRunner]   [{progress}] {label}: done in {res.elapsed_sec:.2f}s")

                    # ---- ensure candidate alignment for metrics (fallback) ----
                    if ctx.subgraph is not None:
                        if ctx.subgraph.payload is None:
                            ctx.subgraph.payload = {}
                        payload = ctx.subgraph.payload
                        # If the explainer returns candidate ordering in extras but the payload is missing it,
                        # copy it over so fidelity_keep/drop can build masks.
                        cand_from_res = res.extras.get("candidate_eidx")
                        if cand_from_res and "candidate_eidx" not in payload:
                            payload["candidate_eidx"] = list(cand_from_res)
                        # Also propagate edge_index if provided
                        edge_idx_from_res = res.extras.get("candidate_edge_index") or res.extras.get("edge_index")
                        if edge_idx_from_res and not ctx.subgraph.edge_index:
                            ctx.subgraph.edge_index = edge_idx_from_res
                        if edge_idx_from_res and "candidate_edge_index" not in payload:
                            payload["candidate_edge_index"] = edge_idx_from_res

                        # Align importance vector to payload candidate ordering if lengths disagree
                        cand_payload = payload.get("candidate_eidx")
                        cand_res = cand_from_res
                        imp_edges = res.importance_edges
                        if (
                            cand_payload is not None
                            and imp_edges is not None
                            and cand_res is not None
                            and len(imp_edges) != len(cand_payload)
                            and len(cand_res) == len(imp_edges)
                        ):
                            mapping = {int(e): imp_edges[i] for i, e in enumerate(cand_res)}
                            aligned = [mapping.get(int(e), 0.0) for e in cand_payload]
                            res.extras.setdefault("importance_edges_raw", imp_edges)
                            res.importance_edges = aligned

                    # ---- compute metrics (object-style) ----
                    metrics_row: Dict[str, float] = {}
                    metric_details: Dict[str, List[Dict[str, Any]]] = {}
                    # debug prints for fidelity alignment
                    try:
                        if res.importance_edges is not None:
                            print("n_importance:", len(res.importance_edges))
                        if ctx.subgraph and ctx.subgraph.payload and "candidate_eidx" in ctx.subgraph.payload:
                            print("n_candidate:", len(ctx.subgraph.payload["candidate_eidx"]))
                    except Exception:
                        pass
                    # debug: candidate count for fidelity masks
                    try:
                        cand = None
                        if ctx.subgraph and ctx.subgraph.payload:
                            cand = ctx.subgraph.payload.get('candidate_eidx')
                        if cand is not None:
                            print(f"[Runner debug] explainer={res.explainer} candidates={len(cand)}")
                    except Exception:
                        pass
                    metric_details: Dict[str, List[Dict[str, Any]]] = {}
                    for metric in self._metrics:
                        mres = metric.compute(ctx, res)
                        mlist = mres if isinstance(mres, (list, tuple)) else [mres]
                        for r in mlist:
                            for k, v in (r.values or {}).items():
                                metrics_row[f"{r.name}.{k}"] = v
                            if getattr(r, "extras", None) or getattr(r, "values", None):
                                detail = {
                                    "values": dict(r.values or {}),
                                    "extras": dict(r.extras or {}),
                                }
                                if getattr(r, "direction", None):
                                    detail["direction"] = r.direction
                                metric_details.setdefault(r.name, []).append(detail)

                    # ---- write JSONL ----
                    if jsonl_f:
                        jsonl_f.write(json.dumps({
                            "context": {"target": ctx.target, "target_kind": ctx.target_kind,
                                        "window": ctx.window, "k_hop": k_hop, "num_neighbors": num_neighbors},
                            "result": {
                                "explainer": res.explainer, "elapsed_sec": res.elapsed_sec,
                                "importance_edges": res.importance_edges,
                                "importance_nodes": res.importance_nodes,
                                "importance_time": res.importance_time,
                                "extras": res.extras
                            },
                            "metrics": metrics_row,
                            "metric_details": metric_details
                        }) + "\n")

                    # ---- write CSV (first row sets header) ----
                    row = {"run_id": run_id, "anchor_idx": idx, "explainer": res.explainer,
                           "elapsed_sec": res.elapsed_sec}
                    row.update(metrics_row)

                    if csv_f:
                        if csv_writer is None:
                            csv_header = list(row.keys())
                            csv_writer = csv.DictWriter(csv_f, fieldnames=csv_header)
                            csv_writer.writeheader()
                        csv_writer.writerow(row)

        finally:
            if jsonl_f: jsonl_f.close()
            if csv_f: csv_f.close()

        return {"out_dir": out_dir,
                "jsonl": jsonl_path if self.config.save_jsonl else None,
                "csv": csv_path if self.config.save_csv else None}
