from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set, Iterable
import json, time, random, os, csv

from .types import (
    ExplanationContext,
    ExplanationResult,
    BaseExplainer,
    ModelProtocol,
    SubgraphExtractorProtocol,
)
from .registry import METRICS
from ..metrics.builtin import ensure_builtin_metrics_loaded

# Ensure built-in metrics are registered before the runner looks them up.
ensure_builtin_metrics_loaded()
# Register extended metric builders (best_fid/tgnn_aufsc/tempme_acc_auc/temgx/cody).
from ..metrics import fidelity as _fidelity  # noqa: F401

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
    show_progress: bool = True
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


def _rewrite_csv_with_header(csv_path: str, header: Sequence[str]) -> None:
    rows: List[Dict[str, Any]] = []
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            with open(csv_path, "r", newline="") as src:
                reader = csv.DictReader(src)
                for row in reader:
                    rows.append(dict(row))
        except Exception:
            rows = []

    with open(csv_path, "w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=list(header), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _ensure_csv_header_if_empty(csv_path: str, *, header: Sequence[str]) -> None:
    """Ensure a CSV exists and has at least a header row.

    This avoids downstream ``pandas.read_csv`` EmptyDataError when a run emits
    zero rows (e.g., empty/filtered inputs, all records skipped on resume).
    """
    if not header:
        return
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        return
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(header), extrasaction="ignore")
        writer.writeheader()


def _ensure_csv_writer_and_header(
    *,
    csv_path: str,
    csv_mode: str,
    csv_f: Any,
    csv_writer: Any,
    csv_header: Optional[List[str]],
    row: Dict[str, Any],
) -> Tuple[Any, Any, List[str]]:
    if csv_f is None:
        return csv_f, csv_writer, list(csv_header or [])

    if csv_writer is None:
        header = list(csv_header or [])
        loaded_existing = False
        if not header and csv_mode == "a":
            loaded = _load_csv_header(csv_path)
            header = list(loaded or [])
            loaded_existing = bool(header)
        if not header:
            header = list(row.keys())
        missing_init = [k for k in row.keys() if k not in header]
        if missing_init:
            header.extend(missing_init)
            if loaded_existing:
                try:
                    csv_f.flush()
                except Exception:
                    pass
                try:
                    csv_f.close()
                except Exception:
                    pass
                _rewrite_csv_with_header(csv_path, header)
                csv_f = open(csv_path, "a", newline="")
        csv_writer = csv.DictWriter(csv_f, fieldnames=header, extrasaction="ignore")
        if csv_mode == "w" or os.path.getsize(csv_path) == 0:
            csv_writer.writeheader()
        return csv_f, csv_writer, header

    header = list(csv_header or [])
    missing = [k for k in row.keys() if k not in header]
    if not missing:
        return csv_f, csv_writer, header

    header.extend(missing)
    try:
        csv_f.flush()
    except Exception:
        pass
    try:
        csv_f.close()
    except Exception:
        pass
    _rewrite_csv_with_header(csv_path, header)
    csv_f = open(csv_path, "a", newline="")
    csv_writer = csv.DictWriter(csv_f, fieldnames=header, extrasaction="ignore")
    return csv_f, csv_writer, header


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


def _report_metric_sanity(csv_path: str) -> None:
    """Print lightweight sanity diagnostics for degenerate metrics.

    This is intentionally non-fatal and best-effort.
    """
    try:
        import pandas as pd
        import numpy as np
    except Exception:
        return

    if not os.path.exists(csv_path):
        return
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return
    if df.empty:
        return
    if not {"anchor_idx", "explainer"}.issubset(df.columns):
        return

    skip = {"anchor_idx", "elapsed_sec"}
    metric_cols = [
        c
        for c in df.columns
        if c not in {"run_id", "context_fp", "explainer"}
        and pd.api.types.is_numeric_dtype(df[c])
        and not str(c).startswith("prediction_")
        and ".prediction_" not in str(c)
        and "@s=" not in str(c)
        and c not in skip
    ]
    if not metric_cols:
        return

    degenerate: list[tuple[str, float]] = []
    n_anchors = int(df["anchor_idx"].nunique())
    for metric in metric_cols:
        try:
            piv = df.pivot_table(index="anchor_idx", columns="explainer", values=metric, aggfunc="first")
        except Exception:
            continue
        if piv.empty or piv.shape[1] < 2:
            continue
        frac_identical = float((piv.nunique(axis=1, dropna=True) <= 1).mean())
        if metric == "tempme_acc_auc.ratio_acc" and n_anchors < 20:
            # Ratio ACC is label-quantized and often saturates on tiny N.
            continue
        if np.isfinite(frac_identical) and frac_identical >= 0.95:
            degenerate.append((metric, frac_identical))

    if degenerate:
        print("[metrics sanity] Potentially degenerate metrics:")
        for metric, frac in degenerate:
            print(f"  - {metric}: {frac:.0%} anchors identical across explainers")

    if "random" in set(df["explainer"].astype(str)):
        # Lightweight random baseline check on common headline metrics.
        checks = [
            m
            for m in (
                "best_fid.value",
                "tgnn_aufsc.value",
                "temgx_aufsc.value",
                "tempme_acc_auc",
                "tempme_acc_auc.ratio_acc",
                "seed_stability.value",
                "perturbation_robustness.value",
            )
            if m in df.columns
        ]
        for metric in checks:
            means = df.groupby("explainer", dropna=False)[metric].mean()
            if "random" not in means.index:
                continue
            others = means.drop(index="random", errors="ignore").dropna()
            if others.empty:
                continue
            random_v = float(means.loc["random"])
            higher_better = metric != "elapsed_sec"
            beat = int((others > random_v).sum()) if higher_better else int((others < random_v).sum())
            print(
                f"[metrics sanity] Random baseline check {metric}: "
                f"{beat}/{len(others)} explainers {'beat' if higher_better else 'are faster than'} random."
            )

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
            run_id: Optional[str] = None, show_progress: Optional[bool] = None) -> Dict[str, Any]:
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
        show_progress_resolved = bool(self.config.show_progress if show_progress is None else show_progress)
        pbar = None
        if show_progress_resolved:
            try:
                from tqdm.auto import tqdm
                total_steps = (
                    int(total_anchors) * int(total_explainers)
                    if total_anchors is not None
                    else None
                )
                pbar = tqdm(
                    total=total_steps,
                    desc="Evaluation",
                    unit="expl",
                    dynamic_ncols=True,
                    leave=True,
                )
            except Exception:
                pbar = None

        def _log(message: str) -> None:
            if pbar is None:
                print(message)

        def _progress_step(*, count: int = 1, postfix: Optional[str] = None) -> None:
            if pbar is None:
                return
            if postfix is not None:
                pbar.set_postfix_str(str(postfix), refresh=False)
            pbar.update(int(count))

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
                if pbar is not None:
                    pbar.set_description(f"Anchor {anchor_prefix}")
                else:
                    _log(f"\n[EvaluationRunner] Anchor {anchor_prefix} (target={target_label})")

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
                    _progress_step(count=0, postfix=label)
                    if self.config.resume and (ctx_fp, label) in done:
                        _log(f"[EvaluationRunner]   [{expl_idx + 1}/{total_explainers}] {label}: skipped (resume)")
                        _progress_step()
                        continue
                    progress = f"{expl_idx + 1}/{total_explainers}"
                    _log(f"[EvaluationRunner]   [{progress}] {label}: start")
                    t0 = _time.perf_counter()
                    res: ExplanationResult = explainer.explain(ctx)
                    res.elapsed_sec = _time.perf_counter() - t0
                    res.context_fp = ctx.fingerprint()
                    _log(f"[EvaluationRunner]   [{progress}] {label}: done in {res.elapsed_sec:.2f}s")
                    _progress_step()

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
                        csv_f, csv_writer, csv_header = _ensure_csv_writer_and_header(
                            csv_path=csv_path,
                            csv_mode=csv_mode,
                            csv_f=csv_f,
                            csv_writer=csv_writer,
                            csv_header=csv_header,
                            row=row,
                        )
                        csv_writer.writerow(row)

        finally:
            if pbar is not None:
                pbar.close()
            if jsonl_f: jsonl_f.close()
            if csv_f: csv_f.close()

        if self.config.save_csv:
            _ensure_csv_header_if_empty(
                csv_path,
                header=["run_id", "anchor_idx", "context_fp", "explainer", "elapsed_sec"],
            )
            _report_metric_sanity(csv_path)
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

                    csv_f, csv_writer, csv_header = _ensure_csv_writer_and_header(
                        csv_path=csv_path,
                        csv_mode=csv_mode,
                        csv_f=csv_f,
                        csv_writer=csv_writer,
                        csv_header=csv_header,
                        row=row,
                    )
                    csv_writer.writerow(row)
        finally:
            csv_f.close()

        _ensure_csv_header_if_empty(
            csv_path,
            header=["run_id", "anchor_idx", "context_fp", "explainer", "elapsed_sec"],
        )
        _report_metric_sanity(csv_path)
        return {"out_dir": out_dir, "jsonl": results_path, "csv": csv_path}
