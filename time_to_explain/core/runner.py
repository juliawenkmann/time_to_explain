from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set, Iterable
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
    out_dir: str = "resources/results"
    seed: int = 42
    metrics: List[str] | Dict[str, Any] | List[Dict[str, Any]] = field(
        default_factory=lambda: ["sparsity", "fidelity_minus", "fidelity_plus", "aufsc"]
    )
    save_jsonl: bool = True
    save_csv: bool = True
    compute_metrics: bool = True
    resume: bool = False
    # None = auto (overwrite when rerunning a subset of explainers), True/False forces behavior.
    overwrite_explainers: Optional[bool] = None


def _normalize_metric_specs(metrics_spec: Any) -> List[Tuple[str, Dict[str, Any]]]:
    if not metrics_spec:
        return []
    if isinstance(metrics_spec, str):
        return [(metrics_spec, {})]
    if isinstance(metrics_spec, dict):
        items = []
        for name, cfg in metrics_spec.items():
            if cfg is None:
                cfg_dict: Dict[str, Any] = {}
            elif isinstance(cfg, dict):
                cfg_dict = dict(cfg)
            else:
                raise TypeError("Metric config must be a mapping.")
            items.append((str(name), cfg_dict))
        return items

    items = []
    for spec in metrics_spec:
        if isinstance(spec, str):
            items.append((spec, {}))
            continue
        if isinstance(spec, dict):
            name = spec.get("builder") or spec.get("name")
            if not name:
                raise ValueError("Metric config must contain 'builder' or 'name'.")
            cfg = spec.get("kwargs")
            if cfg is None:
                cfg = {k: v for k, v in spec.items() if k not in {"builder", "name", "kwargs"}}
            if cfg and not isinstance(cfg, dict):
                raise TypeError("Metric config 'kwargs' must be a mapping.")
            items.append((str(name), dict(cfg or {})))
            continue
        raise TypeError("Metric spec must be a string or mapping.")
    return items


def _build_metrics(
    metrics_spec: Any,
    *,
    model: ModelProtocol,
    dataset: Any,
) -> List[Any]:
    metrics = []
    items = _normalize_metric_specs(metrics_spec)
    for name, mcfg in items:
        factory = METRICS.get(name)
        metric_obj = factory(dict(mcfg))
        if hasattr(metric_obj, "setup"):
            try:
                metric_obj.setup(model=model, dataset=dataset)
            except TypeError:
                metric_obj.setup(model, dataset)
        metrics.append(metric_obj)
    return metrics


def _context_from_record(record: Dict[str, Any], *, fallback_run_id: str) -> ExplanationContext:
    ctx = record.get("context", {}) or {}
    run_id = record.get("run_id") or fallback_run_id
    return ExplanationContext(
        run_id=run_id,
        target_kind=ctx.get("target_kind", "edge"),
        target=ctx.get("target", {}),
        window=ctx.get("window"),
        k_hop=ctx.get("k_hop", 1),
        num_neighbors=ctx.get("num_neighbors", 50),
    )


def _result_from_record(
    record: Dict[str, Any],
    *,
    run_id: str,
    context_fp: str,
) -> ExplanationResult:
    res = record.get("result", {}) or {}
    return ExplanationResult(
        run_id=record.get("run_id") or run_id,
        explainer=res.get("explainer", "unknown"),
        context_fp=record.get("context_fp") or context_fp,
        importance_edges=res.get("importance_edges"),
        importance_nodes=res.get("importance_nodes"),
        importance_time=res.get("importance_time"),
        elapsed_sec=float(res.get("elapsed_sec", 0.0) or 0.0),
        extras=dict(res.get("extras") or {}),
    )


def _load_done_from_jsonl(jsonl_path: str) -> Set[Tuple[str, str]]:
    done: Set[Tuple[str, str]] = set()
    if not os.path.exists(jsonl_path):
        return done
    try:
        with open(jsonl_path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ctx_fp = rec.get("context_fp")
                if not ctx_fp:
                    ctx_fp = _context_from_record(rec, fallback_run_id="").fingerprint()
                explainer = (rec.get("result") or {}).get("explainer") or rec.get("explainer")
                if ctx_fp and explainer:
                    done.add((ctx_fp, str(explainer)))
    except Exception:
        return done
    return done


def _load_done_from_csv(csv_path: str) -> Set[Tuple[str, str, str]]:
    done: Set[Tuple[str, str, str]] = set()
    if not os.path.exists(csv_path):
        return done
    try:
        with open(csv_path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                run_id = row.get("run_id")
                anchor = row.get("anchor_idx")
                explainer = row.get("explainer")
                if run_id is None or anchor is None or explainer is None:
                    continue
                done.add((str(run_id), str(anchor), str(explainer)))
    except Exception:
        return done
    return done


def _explainer_label(obj: Any) -> str:
    return str(getattr(obj, "alias", None) or getattr(obj, "name", None) or obj.__class__.__name__)


def _collect_explainers_from_jsonl(jsonl_path: str) -> Set[str]:
    labels: Set[str] = set()
    if not os.path.exists(jsonl_path):
        return labels
    try:
        with open(jsonl_path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                label = (rec.get("result") or {}).get("explainer") or rec.get("explainer")
                if label:
                    labels.add(str(label))
    except Exception:
        return labels
    return labels


def _collect_explainers_from_csv(csv_path: str) -> Set[str]:
    labels: Set[str] = set()
    if not os.path.exists(csv_path):
        return labels
    try:
        with open(csv_path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                label = row.get("explainer")
                if label:
                    labels.add(str(label))
    except Exception:
        return labels
    return labels


def _filter_jsonl_by_explainer(jsonl_path: str, drop_labels: Iterable[str]) -> None:
    if not os.path.exists(jsonl_path):
        return
    drop_set = {str(l) for l in drop_labels}
    if not drop_set:
        return
    tmp_path = f"{jsonl_path}.tmp"
    try:
        with open(jsonl_path, "r") as src, open(tmp_path, "w") as dst:
            for line in src:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rec = json.loads(stripped)
                except Exception:
                    dst.write(line)
                    continue
                label = (rec.get("result") or {}).get("explainer") or rec.get("explainer")
                if label and str(label) in drop_set:
                    continue
                dst.write(line)
        os.replace(tmp_path, jsonl_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _filter_csv_by_explainer(csv_path: str, drop_labels: Iterable[str]) -> None:
    if not os.path.exists(csv_path):
        return
    drop_set = {str(l) for l in drop_labels}
    if not drop_set:
        return
    tmp_path = f"{csv_path}.tmp"
    try:
        with open(csv_path, "r", newline="") as src, open(tmp_path, "w", newline="") as dst:
            reader = csv.DictReader(src)
            fieldnames = reader.fieldnames
            if not fieldnames:
                return
            writer = csv.DictWriter(dst, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                label = row.get("explainer")
                if label and str(label) in drop_set:
                    continue
                writer.writerow(row)
        os.replace(tmp_path, csv_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _should_overwrite_explainers(
    *,
    overwrite_flag: Optional[bool],
    jsonl_path: str,
    csv_path: str,
    current_labels: Iterable[str],
) -> bool:
    if overwrite_flag is not None:
        return bool(overwrite_flag)
    current_set = {str(l) for l in current_labels}
    if not current_set:
        return False
    existing = _collect_explainers_from_jsonl(jsonl_path)
    if not existing:
        existing = _collect_explainers_from_csv(csv_path)
    if not existing:
        return False
    return current_set.issubset(existing) and current_set != existing


def _load_csv_header(csv_path: str) -> Optional[List[str]]:
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, "r", newline="") as fh:
            reader = csv.reader(fh)
            return next(reader, None)
    except Exception:
        return None


def _align_candidates(context: ExplanationContext, result: ExplanationResult) -> None:
    if context.subgraph is None:
        return
    if context.subgraph.payload is None:
        context.subgraph.payload = {}
    payload = context.subgraph.payload

    cand_from_res = result.extras.get("candidate_eidx") if result.extras else None
    if cand_from_res and "candidate_eidx" not in payload:
        payload["candidate_eidx"] = list(cand_from_res)

    edge_idx_from_res = None
    if result.extras:
        edge_idx_from_res = result.extras.get("candidate_edge_index") or result.extras.get("edge_index")
    if edge_idx_from_res and not context.subgraph.edge_index:
        context.subgraph.edge_index = edge_idx_from_res
    if edge_idx_from_res and "candidate_edge_index" not in payload:
        payload["candidate_edge_index"] = edge_idx_from_res

    cand_payload = payload.get("candidate_eidx")
    cand_res = cand_from_res
    imp_edges = result.importance_edges
    if (
        cand_payload is not None
        and imp_edges is not None
        and cand_res is not None
        and len(imp_edges) != len(cand_payload)
        and len(cand_res) == len(imp_edges)
    ):
        mapping = {int(e): imp_edges[i] for i, e in enumerate(cand_res)}
        aligned = [mapping.get(int(e), 0.0) for e in cand_payload]
        result.extras.setdefault("importance_edges_raw", imp_edges)
        result.importance_edges = aligned

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
        if self.config.compute_metrics:
            self._metrics = _build_metrics(self.config.metrics, model=self.model, dataset=self.dataset)

    def _ensure_out(self, path: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)

    def _compute_metrics(
        self,
        context: ExplanationContext,
        result: ExplanationResult,
        *,
        debug: bool = False,
    ) -> Tuple[Dict[str, float], Dict[str, List[Dict[str, Any]]]]:
        metrics_row: Dict[str, float] = {}
        metric_details: Dict[str, List[Dict[str, Any]]] = {}
        if not self._metrics:
            return metrics_row, metric_details

        if debug:
            try:
                if result.importance_edges is not None:
                    print("n_importance:", len(result.importance_edges))
                if context.subgraph and context.subgraph.payload and "candidate_eidx" in context.subgraph.payload:
                    print("n_candidate:", len(context.subgraph.payload["candidate_eidx"]))
            except Exception:
                pass
            try:
                cand = None
                if context.subgraph and context.subgraph.payload:
                    cand = context.subgraph.payload.get("candidate_eidx")
                if cand is not None:
                    print(f"[Runner debug] explainer={result.explainer} candidates={len(cand)}")
            except Exception:
                pass

        for metric in self._metrics:
            mres = metric.compute(context, result)
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

        return metrics_row, metric_details

    def run(self, anchors: Sequence[Dict[str, Any]], *, k_hop: int = 2,
            num_neighbors: int = 50, window: Optional[tuple] = None,
            run_id: Optional[str] = None) -> Dict[str, Any]:
        import os, json, csv, time as _time

        run_id = run_id or str(int(_time.time()))
        out_dir = os.path.join(self.config.out_dir, run_id)
        self._ensure_out(out_dir)

        jsonl_path = os.path.join(out_dir, "results.jsonl")
        csv_path = os.path.join(out_dir, "metrics.csv")
        explainer_labels = [_explainer_label(e) for e in self.explainers]
        overwrite = False
        if self.config.save_jsonl:
            overwrite = _should_overwrite_explainers(
                overwrite_flag=self.config.overwrite_explainers,
                jsonl_path=jsonl_path,
                csv_path=csv_path,
                current_labels=explainer_labels,
            )
            if overwrite:
                _filter_jsonl_by_explainer(jsonl_path, explainer_labels)
                if self.config.save_csv and self._metrics:
                    _filter_csv_by_explainer(csv_path, explainer_labels)
        jsonl_mode = "a" if (self.config.resume or overwrite) and os.path.exists(jsonl_path) else "w"
        csv_mode = "a" if (self.config.resume or overwrite) and os.path.exists(csv_path) else "w"
        jsonl_f = open(jsonl_path, jsonl_mode) if self.config.save_jsonl else None
        csv_f = None
        if self.config.save_csv and self._metrics:
            csv_f = open(csv_path, csv_mode, newline="")
        csv_writer = None
        csv_header = None
        total_anchors = len(anchors) if hasattr(anchors, "__len__") else None
        total_explainers = len(self.explainers)
        done: Set[Tuple[str, str]] = set()
        if self.config.resume and self.config.save_jsonl:
            done = _load_done_from_jsonl(jsonl_path)

        for e in self.explainers:
            e.prepare(model=self.model, dataset=self.dataset)

        try:
            for idx, anchor in enumerate(anchors):
                ctx = ExplanationContext(
                    run_id=run_id, target_kind=anchor.get("target_kind","edge"),
                    target=anchor, window=window, k_hop=k_hop, num_neighbors=num_neighbors
                )
                ctx_fp = ctx.fingerprint()
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
                    if self.config.resume and (ctx_fp, label) in done:
                        print(f"[EvaluationRunner]   [{expl_idx + 1}/{total_explainers}] {label}: skipped (resume)")
                        continue
                    progress = f"{expl_idx + 1}/{total_explainers}"
                    print(f"[EvaluationRunner]   [{progress}] {label}: start")
                    t0 = _time.perf_counter()
                    res: ExplanationResult = explainer.explain(ctx)
                    res.elapsed_sec = _time.perf_counter() - t0
                    res.context_fp = ctx.fingerprint()
                    print(f"[EvaluationRunner]   [{progress}] {label}: done in {res.elapsed_sec:.2f}s")

                    # ---- ensure candidate alignment for metrics (fallback) ----
                    metrics_row: Dict[str, float] = {}
                    metric_details: Dict[str, List[Dict[str, Any]]] = {}
                    if self._metrics:
                        _align_candidates(ctx, res)
                        metrics_row, metric_details = self._compute_metrics(ctx, res, debug=True)

                    # ---- write JSONL ----
                    if jsonl_f:
                        jsonl_f.write(json.dumps({
                            "run_id": run_id,
                            "anchor_idx": idx,
                            "context_fp": ctx_fp,
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
                    row = {"run_id": run_id, "anchor_idx": idx, "context_fp": res.context_fp,
                           "explainer": res.explainer,
                           "elapsed_sec": res.elapsed_sec}
                    row.update(metrics_row)

                    if csv_f:
                        if csv_writer is None:
                            if csv_mode == "a":
                                csv_header = _load_csv_header(csv_path)
                            if not csv_header:
                                csv_header = list(row.keys())
                            csv_writer = csv.DictWriter(csv_f, fieldnames=csv_header, extrasaction="ignore")
                            if csv_mode == "w" or os.path.getsize(csv_path) == 0:
                                csv_writer.writeheader()
                        csv_writer.writerow(row)

        finally:
            if jsonl_f: jsonl_f.close()
            if csv_f: csv_f.close()

        return {"out_dir": out_dir,
                "jsonl": jsonl_path if self.config.save_jsonl else None,
                "csv": csv_path if self.config.save_csv else None}

    def compute_metrics_from_results(
        self,
        results_path: str,
        *,
        out_dir: Optional[str] = None,
        resume: Optional[bool] = None,
    ) -> Dict[str, Any]:
        import os, json, csv

        if not self._metrics:
            self._metrics = _build_metrics(self.config.metrics, model=self.model, dataset=self.dataset)
        if not self._metrics:
            raise ValueError("No metrics configured for metric computation.")

        if os.path.isdir(results_path):
            results_path = os.path.join(results_path, "results.jsonl")
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Results JSONL not found: {results_path}")

        out_dir = out_dir or os.path.dirname(results_path)
        self._ensure_out(out_dir)

        csv_path = os.path.join(out_dir, "metrics.csv")
        resume = self.config.resume if resume is None else resume
        if resume and self.config.overwrite_explainers is not False:
            labels_in_results = _collect_explainers_from_jsonl(results_path)
            overwrite = _should_overwrite_explainers(
                overwrite_flag=self.config.overwrite_explainers,
                jsonl_path=results_path,
                csv_path=csv_path,
                current_labels=labels_in_results,
            )
            if overwrite:
                _filter_csv_by_explainer(csv_path, labels_in_results)
        csv_mode = "a" if resume and os.path.exists(csv_path) else "w"

        done: Set[Tuple[str, str, str]] = set()
        if resume:
            done = _load_done_from_csv(csv_path)

        csv_f = open(csv_path, csv_mode, newline="")
        csv_writer = None
        csv_header = None
        subgraph_cache: Dict[Tuple[str, str], Any] = {}

        try:
            with open(results_path, "r") as fh:
                for line_idx, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue

                    run_id = rec.get("run_id") or os.path.basename(out_dir)
                    anchor_idx = rec.get("anchor_idx")
                    if anchor_idx is None:
                        anchor_idx = line_idx
                    ctx = _context_from_record(rec, fallback_run_id=str(run_id))
                    ctx_fp = rec.get("context_fp") or ctx.fingerprint()

                    res = _result_from_record(rec, run_id=str(run_id), context_fp=ctx_fp)
                    label = res.explainer
                    if not label:
                        continue

                    if resume and (str(run_id), str(anchor_idx), str(label)) in done:
                        continue

                    ext = self.extractor_map.get(label, None) or self.extractor_map.get(str(label), None) or self.extractor
                    if ext is None:
                        raise ValueError(f"No extractor provided for explainer '{label}'.")

                    cache_key = (ctx_fp, str(id(ext)))
                    if cache_key in subgraph_cache:
                        subg = subgraph_cache[cache_key]
                    else:
                        subg = ext.extract(
                            self.dataset,
                            ctx.target,
                            k_hop=ctx.k_hop,
                            num_neighbors=ctx.num_neighbors,
                            window=ctx.window,
                        )
                        subgraph_cache[cache_key] = subg

                    ctx.subgraph = subg
                    _align_candidates(ctx, res)
                    metrics_row, metric_details = self._compute_metrics(ctx, res, debug=False)

                    row = {
                        "run_id": run_id,
                        "anchor_idx": anchor_idx,
                        "context_fp": ctx_fp,
                        "explainer": res.explainer,
                        "elapsed_sec": res.elapsed_sec,
                    }
                    row.update(metrics_row)

                    if csv_writer is None:
                        if csv_mode == "a":
                            csv_header = _load_csv_header(csv_path)
                        if not csv_header:
                            csv_header = list(row.keys())
                        csv_writer = csv.DictWriter(csv_f, fieldnames=csv_header, extrasaction="ignore")
                        if csv_mode == "w" or os.path.getsize(csv_path) == 0:
                            csv_writer.writeheader()
                    csv_writer.writerow(row)
        finally:
            csv_f.close()

        return {"out_dir": out_dir, "jsonl": results_path, "csv": csv_path}
