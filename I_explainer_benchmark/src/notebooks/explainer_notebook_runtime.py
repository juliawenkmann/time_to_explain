from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from ..core.runner import EvalConfig, EvaluationRunner
from ..explainers.extractors.khop import KHopCandidatesExtractor
from ..models.loader import load_backbone_model
from .notebook_helpers import build_edge_anchors, select_explain_event_ids
from .notebook_runtime_common import (
    event_triplet_from_events,
    jsonl_contains_targets,
    jsonl_has_records,
    load_jsonl_records,
)
from .tempme_run_helpers import (
    fallback_candidates as _fallback_candidates,
    load_stage_cache_metrics as _load_stage_cache_metrics,
    parse_learn_base_metrics,
    parse_testing_epoch_metrics,
    run_with_batch_fallback as _run_with_batch_fallback,
    runs_dir_for_context,
    save_metrics_and_logs as _save_metrics_and_logs,
    save_stage_cache as _save_stage_cache,
)


@dataclass(frozen=True)
class StandardExplainerRunContext:
    model: Any
    events: pd.DataFrame
    all_event_idxs: list[int]
    target_event_idxs: list[int]
    anchors: list[dict[str, int | str]]
    extractor: KHopCandidatesExtractor
    runner: EvaluationRunner


def select_edge_targets(
    explain_index_path: str | Path,
    *,
    n_eval_events: int,
    start: int = 0,
    include_event_ids: Sequence[int] | None = None,
    explicit_target_event_ids: Sequence[int] | None = None,
    dataset_name: str | None = None,
) -> tuple[list[int], list[int], list[dict[str, int | str]]]:
    all_event_idxs, default_target_event_idxs = select_explain_event_ids(
        str(explain_index_path),
        n_eval_events=int(n_eval_events),
        start=int(start),
        include_event_ids=include_event_ids,
    )
    if explicit_target_event_ids:
        target_event_idxs = [int(v) for v in explicit_target_event_ids]
        missing = [event_idx for event_idx in target_event_idxs if event_idx not in set(all_event_idxs)]
        if missing and dataset_name:
            print(f"Warning: target events not in explain index for {dataset_name}: {missing}")
    else:
        target_event_idxs = list(default_target_event_idxs)
    anchors = build_edge_anchors(target_event_idxs)
    return all_event_idxs, target_event_idxs, anchors


def build_standard_runner(
    *,
    model: Any,
    events: pd.DataFrame,
    dataset_name: str,
    explainer: Any,
    out_dir: str | Path,
    candidates_size: int,
    seed: int,
    num_hops: int | None = None,
) -> tuple[KHopCandidatesExtractor, EvaluationRunner]:
    extractor = KHopCandidatesExtractor(
        model=model,
        events=events,
        candidates_size=int(candidates_size),
        num_hops=int(num_hops if num_hops is not None else (getattr(model, "num_layers", 2) or 2)),
    )
    cfg = EvalConfig(
        out_dir=str(Path(out_dir).expanduser().resolve()),
        metrics={},
        compute_metrics=False,
        save_jsonl=True,
        save_csv=False,
        seed=int(seed),
        resume=False,
    )
    runner = EvaluationRunner(
        model=model,
        dataset={"events": events, "dataset_name": str(dataset_name)},
        extractor=extractor,
        explainers=[explainer],
        config=cfg,
    )
    return extractor, runner


def prepare_standard_edge_explainer_run(
    *,
    dataset_name: str,
    model_name: str,
    ckpt_path: str | Path,
    device: Any,
    explain_index_path: str | Path,
    n_eval_events: int,
    out_dir: str | Path,
    explainer: Any,
    candidates_size: int,
    seed: int,
    start: int = 0,
    include_event_ids: Sequence[int] | None = None,
    explicit_target_event_ids: Sequence[int] | None = None,
) -> StandardExplainerRunContext:
    model, events = load_backbone_model(
        model_type=str(model_name),
        dataset_name=str(dataset_name),
        ckpt_path=Path(ckpt_path),
        device=device,
    )
    all_event_idxs, target_event_idxs, anchors = select_edge_targets(
        explain_index_path,
        n_eval_events=int(n_eval_events),
        start=int(start),
        include_event_ids=include_event_ids,
        explicit_target_event_ids=explicit_target_event_ids,
        dataset_name=str(dataset_name),
    )
    extractor, runner = build_standard_runner(
        model=model,
        events=events,
        dataset_name=str(dataset_name),
        explainer=explainer,
        out_dir=out_dir,
        candidates_size=int(candidates_size),
        seed=int(seed),
    )
    return StandardExplainerRunContext(
        model=model,
        events=events,
        all_event_idxs=all_event_idxs,
        target_event_idxs=target_event_idxs,
        anchors=anchors,
        extractor=extractor,
        runner=runner,
    )


def find_cached_run_dirs(
    *,
    cache_roots: Iterable[str | Path],
    run_prefix: str,
    target_event_idxs: Sequence[int],
) -> list[Path]:
    cached_run_dirs: list[Path] = []
    for root in cache_roots:
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists():
            continue
        for path in root_path.glob("*"):
            jsonl_path = path / "results.jsonl"
            if not path.is_dir() or not jsonl_path.exists():
                continue
            if not path.name.lower().startswith(str(run_prefix).lower()):
                continue
            if not jsonl_has_records(jsonl_path):
                continue
            if not jsonl_contains_targets(jsonl_path, target_event_idxs):
                continue
            cached_run_dirs.append(path)
    return sorted(cached_run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)


def run_or_reuse_explanations(
    *,
    runner: EvaluationRunner,
    anchors: Sequence[Mapping[str, Any]],
    dataset_name: str,
    model_name: str,
    explainer_name: str,
    target_event_idxs: Sequence[int],
    use_cached: bool,
    cache_roots: Sequence[str | Path] | None = None,
    run_label: str | None = None,
    k_hop: int | None = None,
    num_neighbors: int | None = None,
    show_progress: bool = True,
) -> tuple[dict[str, Any], Path, Path, str]:
    out_dir = Path(str(runner.config.out_dir)).expanduser().resolve()
    run_label_slug = str(run_label or "").strip().lower().replace(" ", "_")
    run_prefix = f"{str(dataset_name)}_{str(model_name)}_official_{str(explainer_name)}"
    if run_label_slug:
        run_prefix = f"{run_prefix}_{run_label_slug}"

    cached_run_dirs = find_cached_run_dirs(
        cache_roots=cache_roots or [out_dir],
        run_prefix=run_prefix,
        target_event_idxs=target_event_idxs,
    )

    if use_cached and cached_run_dirs:
        run_dir = cached_run_dirs[0]
        out = {
            "out_dir": str(run_dir),
            "jsonl": str(run_dir / "results.jsonl"),
            "csv": None,
        }
        print(f"Using cached {explainer_name} explanations from:", run_dir)
    else:
        model = runner.model
        run_id = f"{run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out = runner.run(
            anchors,
            k_hop=int(k_hop if k_hop is not None else (getattr(model, "num_layers", 2) or 2)),
            num_neighbors=int(
                num_neighbors if num_neighbors is not None else (getattr(model, "num_neighbors", 20) or 20)
            ),
            run_id=run_id,
            show_progress=bool(show_progress),
        )
        print(f"Stored new {explainer_name} explanations under:", out["out_dir"])

    run_dir = Path(out["out_dir"]).expanduser().resolve()
    out_jsonl = Path(out["jsonl"]).expanduser().resolve()
    return out, run_dir, out_jsonl, run_dir.name


def enrich_results_jsonl_with_candidates(
    path: str | Path,
    *,
    extractor: Any,
    events: pd.DataFrame,
    dataset_name: str,
    model: Any,
) -> int:
    resolved = Path(path).expanduser().resolve()
    records = load_jsonl_records(resolved)
    updated = 0
    for rec in records:
        context = rec.get("context") or {}
        target = context.get("target") or {}
        event_idx = target.get("event_idx")
        if event_idx is None:
            continue
        event_idx = int(event_idx)
        anchor = {"target_kind": "edge", "event_idx": event_idx}
        subg = extractor.extract(
            {"events": events, "dataset_name": str(dataset_name)},
            anchor,
            k_hop=int(context.get("k_hop", int(getattr(model, "num_layers", 2) or 2))),
            num_neighbors=int(context.get("num_neighbors", int(getattr(model, "num_neighbors", 20) or 20))),
            window=None,
        )

        cand = []
        if getattr(subg, "payload", None) and "candidate_eidx" in subg.payload:
            cand = [int(x) for x in list(subg.payload.get("candidate_eidx") or [])]

        result = rec.get("result") or {}
        imp = list(result.get("importance_edges") or [])
        if cand:
            if len(imp) < len(cand):
                imp = [float(x) for x in imp] + [0.0] * (len(cand) - len(imp))
            elif len(imp) > len(cand):
                imp = [float(x) for x in imp[: len(cand)]]
            result["importance_edges"] = imp

        extras = dict(result.get("extras") or {})
        extras["event_idx"] = event_idx
        extras["candidate_eidx"] = cand
        src, dst, ts = event_triplet_from_events(events, event_idx)
        extras["u"] = int(src)
        extras["i"] = int(dst)
        extras["ts"] = float(ts)
        result["extras"] = extras
        rec["result"] = result
        updated += 1

    with resolved.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return updated


@dataclass(frozen=True)
class TempMENotebookRuntimeConfig:
    project_root: Path
    benchmark_root: Path
    temp_me_root: Path
    dataset_name: str
    base_type: str
    python_bin: str
    quick_run: bool
    base_overrides: dict[str, object]
    explainer_overrides: dict[str, object]
    force_rerun_base_effective: bool
    force_rerun_explainer_effective: bool
    base_ckpt: Path
    expl_ckpt: Path


@dataclass(frozen=True)
class TempMENotebookRuntimeResult:
    base_logs: list[str]
    base_metrics: dict[str, float]
    expl_logs: list[str]
    expl_metrics: dict[str, float]
    metrics_df: pd.DataFrame
    metrics_path: Path
    logs_path: Path
def _runs_dir(cfg: TempMENotebookRuntimeConfig) -> Path:
    return runs_dir_for_context(
        project_root=cfg.project_root,
        dataset_name=cfg.dataset_name,
        base_type=cfg.base_type,
    )


def save_stage_cache(
    cfg: TempMENotebookRuntimeConfig,
    stage: str,
    metrics: dict[str, float],
    logs: list[str],
) -> Path:
    return _save_stage_cache(stage, metrics, logs, _runs_dir(cfg))


def load_stage_cache_metrics(cfg: TempMENotebookRuntimeConfig, stage: str) -> dict[str, float]:
    return _load_stage_cache_metrics(stage, _runs_dir(cfg))


def save_metrics_and_logs(
    cfg: TempMENotebookRuntimeConfig,
    metrics_rows: list[dict[str, object]],
    logs: dict[str, list[str]],
) -> tuple[Path, Path]:
    return _save_metrics_and_logs(metrics_rows, logs, _runs_dir(cfg))


def run_tempme_training_pipeline(cfg: TempMENotebookRuntimeConfig) -> TempMENotebookRuntimeResult:
    base_logs: list[str] = []
    base_metrics: dict[str, float] = {}

    base_script = cfg.temp_me_root / "learn_base.py"
    base_run_args = {
        "base_type": cfg.base_type,
        "data": cfg.dataset_name,
        **cfg.base_overrides,
    }

    if cfg.base_ckpt.exists() and not cfg.force_rerun_base_effective:
        print(f"Skipping learn_base.py (checkpoint exists): {cfg.base_ckpt}")
        base_metrics = load_stage_cache_metrics(cfg, "learn_base")
    else:
        start_bs = int(base_run_args.get("bs", 1024))
        floor = 512 if cfg.quick_run else 256
        candidates = _fallback_candidates(start=start_bs, floor=floor)
        used_args, base_logs = _run_with_batch_fallback(
            base_script,
            base_run_args,
            batch_keys=("bs",),
            candidates=candidates,
            cwd=cfg.temp_me_root,
            python_bin=cfg.python_bin,
            project_root=cfg.project_root,
            progress_title="TempME base model",
        )
        print("learn_base.py args used:", used_args)

        base_metrics = parse_learn_base_metrics(base_logs)
        cache_path = save_stage_cache(cfg, "learn_base", base_metrics, base_logs)
        print("Saved learn_base cache:", cache_path)

    if not base_metrics and base_logs:
        base_metrics = parse_learn_base_metrics(base_logs)

    expl_logs: list[str] = []
    expl_metrics: dict[str, float] = {}

    exp_script = cfg.temp_me_root / "temp_exp_main.py"
    exp_run_args = {
        "base_type": cfg.base_type,
        "data": cfg.dataset_name,
        **cfg.explainer_overrides,
    }

    if cfg.expl_ckpt.exists() and not cfg.force_rerun_explainer_effective:
        print(f"Skipping temp_exp_main.py (checkpoint exists): {cfg.expl_ckpt}")
        expl_metrics = load_stage_cache_metrics(cfg, "temp_exp_main")
    else:
        start_bs = int(exp_run_args.get("bs", 512))
        floor = 256 if cfg.quick_run else 128
        candidates = _fallback_candidates(start=start_bs, floor=floor)

        used_args, expl_logs = _run_with_batch_fallback(
            exp_script,
            exp_run_args,
            batch_keys=("bs", "test_bs"),
            candidates=candidates,
            cwd=cfg.temp_me_root,
            python_bin=cfg.python_bin,
            project_root=cfg.project_root,
            progress_title="TempME explainer",
        )
        print("temp_exp_main.py args used:", used_args)

        expl_metrics = parse_testing_epoch_metrics(expl_logs)
        cache_path = save_stage_cache(cfg, "temp_exp_main", expl_metrics, expl_logs)
        print("Saved temp_exp_main cache:", cache_path)

    if not expl_metrics and expl_logs:
        expl_metrics = parse_testing_epoch_metrics(expl_logs)

    rows: list[dict[str, object]] = []
    if base_metrics:
        rows.append({"stage": "learn_base", **base_metrics})
    if expl_metrics:
        rows.append({"stage": "temp_exp_main", **expl_metrics})

    metrics_df = pd.DataFrame(rows)
    if "ratio_acc" in metrics_df.columns and "acc_auc" not in metrics_df.columns:
        metrics_df["acc_auc"] = metrics_df["ratio_acc"]

    logs_payload = {
        "learn_base": base_logs,
        "temp_exp_main": expl_logs,
    }
    metrics_path, logs_path = save_metrics_and_logs(cfg, metrics_df.to_dict(orient="records"), logs_payload)

    return TempMENotebookRuntimeResult(
        base_logs=base_logs,
        base_metrics=base_metrics,
        expl_logs=expl_logs,
        expl_metrics=expl_metrics,
        metrics_df=metrics_df,
        metrics_path=metrics_path,
        logs_path=logs_path,
    )


__all__ = [
    "StandardExplainerRunContext",
    "TempMENotebookRuntimeConfig",
    "TempMENotebookRuntimeResult",
    "build_standard_runner",
    "enrich_results_jsonl_with_candidates",
    "event_triplet_from_events",
    "find_cached_run_dirs",
    "jsonl_contains_targets",
    "jsonl_has_records",
    "load_jsonl_records",
    "load_stage_cache_metrics",
    "parse_learn_base_metrics",
    "parse_testing_epoch_metrics",
    "prepare_standard_edge_explainer_run",
    "run_or_reuse_explanations",
    "run_tempme_training_pipeline",
    "save_metrics_and_logs",
    "save_stage_cache",
    "select_edge_targets",
]
