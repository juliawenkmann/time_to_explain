from __future__ import annotations

import json
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from I_explainer_benchmark.src.core.cli import find_repo_root

from .common import (
    ExplainerSelection,
    _deduplicate_preserve_order,
    _derive_selected_event_idxs,
    _extract_record_explainer,
    _extract_record_target_idx,
    _normalize_name,
)

def discover_latest_explainer_results(
    results_root: Path | str,
    dataset_name: str,
    explainer_name: str,
    model_type: Optional[str] = None,
) -> Path:
    """
    Find the latest result file path for a dataset/explainer combination.

    The search supports both:
    - nested `.../results.jsonl`
    - flat `<run_id>.jsonl`
    """
    root = Path(results_root)
    if not root.exists():
        raise FileNotFoundError(f"Results root does not exist: {root}")

    ds = _normalize_name(dataset_name)
    explainer = _normalize_name(explainer_name)
    model = _normalize_name(model_type) if model_type else None

    candidates = sorted(set(root.glob("**/results.jsonl")).union(set(root.glob("**/*.jsonl"))))
    filtered: List[Path] = []
    for path in candidates:
        s = path.as_posix().lower().replace("-", "_")
        if ds not in s:
            continue
        if model and model not in s:
            continue
        if explainer not in s and f"official_{explainer}" not in s:
            continue
        filtered.append(path)

    if not filtered:
        model_txt = f", model='{model_type}'" if model_type else ""
        raise FileNotFoundError(
            f"No results found under {root} for dataset='{dataset_name}', "
            f"explainer='{explainer_name}'{model_txt}."
        )

    timestamp_re = re.compile(r"(20\d{6}_\d{6})")

    def _sort_key(path: Path) -> Tuple[str, float]:
        match = timestamp_re.search(path.as_posix())
        stamp = match.group(1) if match else ""
        return stamp, path.stat().st_mtime

    filtered.sort(key=_sort_key, reverse=True)
    return filtered[0]


def discover_explainer_result_files(
    results_root: Path | str,
    dataset_name: str,
    explainer_name: str,
    model_type: Optional[str] = None,
) -> List[Path]:
    """
    Find all matching result files for a dataset/explainer/model, newest first.
    """
    root = Path(results_root)
    if not root.exists():
        raise FileNotFoundError(f"Results root does not exist: {root}")

    ds = _normalize_name(dataset_name)
    explainer = _normalize_name(explainer_name)
    model = _normalize_name(model_type) if model_type else None

    candidates = sorted(set(root.glob("**/results.jsonl")).union(set(root.glob("**/*.jsonl"))))
    filtered: List[Path] = []
    for path in candidates:
        s = path.as_posix().lower().replace("-", "_")
        if ds not in s:
            continue
        if model and model not in s:
            continue
        if explainer not in s and f"official_{explainer}" not in s:
            continue
        filtered.append(path)

    if not filtered:
        model_txt = f", model='{model_type}'" if model_type else ""
        raise FileNotFoundError(
            f"No results found under {root} for dataset='{dataset_name}', "
            f"explainer='{explainer_name}'{model_txt}."
        )

    timestamp_re = re.compile(r"(20\d{6}_\d{6})")

    def _sort_key(path: Path) -> Tuple[str, float]:
        match = timestamp_re.search(path.as_posix())
        stamp = match.group(1) if match else ""
        return stamp, path.stat().st_mtime

    filtered.sort(key=_sort_key, reverse=True)
    return filtered


def load_explainer_selection_from_results(
    results_jsonl_path: Path | str,
    explainer_name: str,
    target_idx: int,
    top_k_fallback: int = 8,
) -> ExplainerSelection:
    """
    Load selected edge indices for a given explainer/target from a JSONL results file.
    """
    path = Path(results_jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    explainer = _normalize_name(explainer_name)
    selected_candidate: Optional[ExplainerSelection] = None
    available_targets: set[int] = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            if not isinstance(record, dict):
                continue

            record_explainer = _extract_record_explainer(record)
            if record_explainer and record_explainer != explainer:
                continue

            event_idx = _extract_record_target_idx(record)
            if event_idx is None:
                continue
            available_targets.add(event_idx)
            if event_idx != int(target_idx):
                continue

            selected, candidate, source = _derive_selected_event_idxs(record, top_k_fallback=top_k_fallback)
            run_id = record.get("run_id") if isinstance(record.get("run_id"), str) else None

            current = ExplainerSelection(
                explainer_name=explainer,
                target_idx=int(target_idx),
                selected_event_idxs=_deduplicate_preserve_order(selected),
                candidate_event_idxs=_deduplicate_preserve_order(candidate),
                source=source,
                results_path=path,
                run_id=run_id,
            )
            if current.selected_event_idxs:
                return current
            if selected_candidate is None:
                selected_candidate = current

    if selected_candidate is not None:
        return selected_candidate

    available_preview = sorted(available_targets)[:15]
    raise KeyError(
        f"No record found for target_idx={target_idx} and explainer='{explainer_name}' in {path}. "
        f"Available target indices (first 15): {available_preview}"
    )


def list_available_targets_in_results(
    results_jsonl_path: Path | str,
    explainer_name: str,
) -> List[int]:
    """
    List target event indices available for an explainer in a JSONL file.
    """
    path = Path(results_jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    explainer = _normalize_name(explainer_name)
    targets: set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            if not isinstance(record, dict):
                continue

            record_explainer = _extract_record_explainer(record)
            if record_explainer and record_explainer != explainer:
                continue

            target = _extract_record_target_idx(record)
            if target is not None:
                targets.add(int(target))
    return sorted(targets)


def resolve_explainer_selection(
    dataset_name: str,
    explainer_name: str,
    target_idx: int,
    results_root: Path | str,
    model_type: Optional[str] = None,
    results_jsonl_path: Optional[Path | str] = None,
    top_k_fallback: int = 8,
    allow_fallback_target: bool = True,
) -> ExplainerSelection:
    """
    Resolve explainer selection by either direct JSONL path or auto-discovery.
    """
    if results_jsonl_path is not None:
        path = Path(results_jsonl_path)
    else:
        try:
            path = discover_latest_explainer_results(
                results_root=results_root,
                dataset_name=dataset_name,
                explainer_name=explainer_name,
                model_type=model_type,
            )
        except FileNotFoundError:
            if model_type:
                warnings.warn(
                    f"No results found for model_type='{model_type}'. Retrying discovery without model filter.",
                    stacklevel=2,
                )
                path = discover_latest_explainer_results(
                    results_root=results_root,
                    dataset_name=dataset_name,
                    explainer_name=explainer_name,
                    model_type=None,
                )
            else:
                raise
    try:
        return load_explainer_selection_from_results(
            results_jsonl_path=path,
            explainer_name=explainer_name,
            target_idx=target_idx,
            top_k_fallback=top_k_fallback,
        )
    except KeyError:
        if not allow_fallback_target:
            raise

        available = list_available_targets_in_results(path, explainer_name=explainer_name)
        if not available:
            raise
        fallback_target = min(available, key=lambda x: abs(int(x) - int(target_idx)))
        warnings.warn(
            f"target_idx={target_idx} not found in explainer results. "
            f"Using nearest available target_idx={fallback_target}.",
            stacklevel=2,
        )
        return load_explainer_selection_from_results(
            results_jsonl_path=path,
            explainer_name=explainer_name,
            target_idx=int(fallback_target),
            top_k_fallback=top_k_fallback,
        )
def _ensure_project_import_paths() -> None:
    import sys

    repo_root = find_repo_root(start=Path(__file__).resolve(), marker="I_explainer_benchmark")
    bench_root = repo_root / "I_explainer_benchmark"

    ordered_paths = [
        bench_root,
        bench_root / "submodules",
        bench_root / "submodules" / "models",
        repo_root,
        bench_root / "src",
    ]
    for path in ordered_paths:
        if path.exists():
            s = str(path)
            if s not in sys.path:
                sys.path.insert(0, s)


def _resolve_model_checkpoint(
    dataset_name: str,
    model_type: str,
    checkpoint_path: Optional[Path | str] = None,
) -> Path:
    if checkpoint_path is not None:
        path = Path(checkpoint_path).expanduser()
        if not path.is_absolute():
            path = (find_repo_root(start=Path(__file__).resolve(), marker="I_explainer_benchmark") / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Provided checkpoint_path does not exist: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Provided checkpoint_path is not a file: {path}")
        return path

    repo_root = find_repo_root(start=Path(__file__).resolve(), marker="I_explainer_benchmark")
    model = _normalize_name(model_type)
    dataset = _normalize_name(dataset_name)
    model_roots = [
        repo_root / "I_explainer_benchmark" / "resources" / "models",
        repo_root / "resources" / "models",
    ]
    model_roots = [root for root in model_roots if root.exists()]
    if not model_roots:
        raise FileNotFoundError(
            f"Could not find model roots under {repo_root}. "
            "Expected I_explainer_benchmark/resources/models or resources/models."
        )

    preferred = f"{model}_{dataset}_best.pth"
    exact_candidates: List[Path] = []
    for root in model_roots:
        exact_candidates.extend(
            [
                root / dataset / model / preferred,
                root / dataset / "checkpoints" / preferred,
                root / "checkpoints" / preferred,
            ]
        )
    for candidate in exact_candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    search_dirs: List[Path] = []
    for root in model_roots:
        search_dirs.extend(
            [
                root / dataset / model,
                root / dataset / "checkpoints",
                root / "checkpoints",
                root / dataset,
                root,
            ]
        )
    search_dirs = [p for p in search_dirs if p.exists() and p.is_dir()]

    patterns = [
        f"{model}_{dataset}*.pth",
        f"{model}_{dataset}*.pt",
        f"{model}-{dataset}*.pth",
        f"{model}-{dataset}*.pt",
        f"{model}*{dataset}*.pth",
        f"{model}*{dataset}*.pt",
    ]
    discovered: List[Path] = []
    seen: set[Path] = set()
    for search_dir in search_dirs:
        for pattern in patterns:
            for hit in search_dir.rglob(pattern):
                resolved = hit.resolve()
                if resolved in seen or not resolved.is_file():
                    continue
                seen.add(resolved)
                discovered.append(resolved)

    if not discovered:
        # Fall back to case-insensitive matching (CoDy often stores checkpoints with uppercase model names).
        for search_dir in search_dirs:
            for ext in ("*.pth", "*.pt"):
                for hit in search_dir.rglob(ext):
                    resolved = hit.resolve()
                    if resolved in seen or not resolved.is_file():
                        continue
                    low_name = resolved.name.lower()
                    if model in low_name and dataset in low_name:
                        seen.add(resolved)
                        discovered.append(resolved)

    if not discovered:
        checked = "\n".join(str(p) for p in exact_candidates[:12])
        raise FileNotFoundError(
            "No suitable model checkpoint found for on-demand explainer run.\n"
            f"dataset={dataset_name}, model_type={model_type}\n"
            f"Tried exact paths:\n{checked}"
        )

    def _score(path: Path) -> Tuple[int, float]:
        name = path.name.lower()
        rank = 0
        if "best" in name:
            rank += 4
        if dataset in name:
            rank += 2
        if model in name:
            rank += 1
        return rank, path.stat().st_mtime

    discovered.sort(key=_score, reverse=True)
    return discovered[0]


def run_explainer_for_target(
    dataset_name: str,
    explainer_name: str,
    target_idx: int,
    *,
    results_root: Path | str,
    model_type: str,
    checkpoint_path: Optional[Path | str] = None,
    num_hops: int = 2,
    candidates_size: int = 75,
    num_neighbors: int = 20,
    window: Optional[Tuple[float, float]] = None,
    seed: int = 42,
    top_k_fallback: int = 8,
    run_id: Optional[str] = None,
    explainer_overrides: Optional[Dict[str, object]] = None,
    show_progress: bool = True,
) -> ExplainerSelection:
    """
    Execute one explainer run for a single target event and return parsed selection.
    """
    if not model_type:
        raise ValueError("`model_type` is required to run an explainer on-demand.")

    _ensure_project_import_paths()

    import torch
    from I_explainer_benchmark.src.core.runner import EvalConfig, EvaluationRunner
    from I_explainer_benchmark.src.explainers.builder import build_explainer
    from I_explainer_benchmark.src.explainers.extractors.khop import KHopCandidatesExtractor
    from I_explainer_benchmark.src.models.loader import load_backbone_model

    checkpoint = _resolve_model_checkpoint(
        dataset_name=dataset_name,
        model_type=model_type,
        checkpoint_path=checkpoint_path,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, events = load_backbone_model(
        model_type=model_type,
        dataset_name=dataset_name,
        ckpt_path=checkpoint,
        device=device,
    )

    try:
        target_int = int(target_idx)
    except Exception as exc:
        raise ValueError(f"Invalid target_idx={target_idx!r}") from exc

    if isinstance(events, pd.DataFrame) and "idx" in events.columns:
        valid_target_ids = set(int(v) for v in events["idx"].astype(int).tolist())
        if target_int not in valid_target_ids:
            raise KeyError(
                f"target_idx={target_int} not found in loaded events for dataset='{dataset_name}'."
            )

    explainer = build_explainer(
        explainer_name,
        dataset_name=dataset_name,
        model_type=model_type,
        device=device,
        seed=int(seed),
        overrides=dict(explainer_overrides or {}),
        verbose=False,
    )
    if explainer is None:
        raise RuntimeError(
            f"Could not build explainer='{explainer_name}' for dataset='{dataset_name}', model='{model_type}'."
        )

    extractor = KHopCandidatesExtractor(
        model=model,
        events=events,
        candidates_size=max(1, int(candidates_size)),
        num_hops=max(1, int(num_hops)),
    )

    out_root = Path(results_root)
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_run_id = run_id or (
        f"{_normalize_name(dataset_name)}_{_normalize_name(model_type)}_"
        f"{_normalize_name(explainer_name)}_target_{target_int}_{stamp}"
    )

    cfg = EvalConfig(
        out_dir=str(out_root),
        seed=int(seed),
        save_jsonl=True,
        save_csv=False,
        compute_metrics=False,
        resume=True,
        show_progress=bool(show_progress),
        overwrite_explainers=False,
    )
    runner = EvaluationRunner(
        model=model,
        dataset=events,
        extractor=extractor,
        explainers=[explainer],
        config=cfg,
    )
    run_out = runner.run(
        anchors=[{"target_kind": "edge", "event_idx": target_int}],
        k_hop=max(1, int(num_hops)),
        num_neighbors=max(1, int(num_neighbors)),
        window=window,
        run_id=resolved_run_id,
        show_progress=bool(show_progress),
    )
    jsonl_path = run_out.get("jsonl")
    if not jsonl_path:
        raise RuntimeError(f"On-demand run did not produce JSONL output. Runner output: {run_out}")

    return load_explainer_selection_from_results(
        results_jsonl_path=Path(jsonl_path),
        explainer_name=explainer_name,
        target_idx=target_int,
        top_k_fallback=int(top_k_fallback),
    )


def resolve_or_run_explainer_selection(
    dataset_name: str,
    explainer_name: str,
    target_idx: int,
    results_root: Path | str,
    *,
    model_type: Optional[str] = None,
    results_jsonl_path: Optional[Path | str] = None,
    top_k_fallback: int = 8,
    allow_fallback_target: bool = True,
    run_if_missing: bool = False,
    run_checkpoint_path: Optional[Path | str] = None,
    run_num_hops: int = 2,
    run_candidates_size: int = 75,
    run_num_neighbors: int = 20,
    run_window: Optional[Tuple[float, float]] = None,
    run_seed: int = 42,
    run_explainer_overrides: Optional[Dict[str, object]] = None,
    run_show_progress: bool = True,
) -> ExplainerSelection:
    """
    Resolve explainer selection from saved results, or run the explainer for this target if missing.
    """
    candidate_paths: List[Path] = []
    if results_jsonl_path is not None:
        candidate_paths = [Path(results_jsonl_path)]
    else:
        try:
            candidate_paths = discover_explainer_result_files(
                results_root=results_root,
                dataset_name=dataset_name,
                explainer_name=explainer_name,
                model_type=model_type,
            )
        except FileNotFoundError:
            if model_type:
                try:
                    candidate_paths = discover_explainer_result_files(
                        results_root=results_root,
                        dataset_name=dataset_name,
                        explainer_name=explainer_name,
                        model_type=None,
                    )
                except FileNotFoundError:
                    candidate_paths = []
            else:
                candidate_paths = []

    for path in candidate_paths:
        try:
            return load_explainer_selection_from_results(
                results_jsonl_path=path,
                explainer_name=explainer_name,
                target_idx=int(target_idx),
                top_k_fallback=top_k_fallback,
            )
        except KeyError:
            continue
        except FileNotFoundError:
            continue

    try:
        return resolve_explainer_selection(
            dataset_name=dataset_name,
            explainer_name=explainer_name,
            target_idx=int(target_idx),
            results_root=results_root,
            model_type=model_type,
            results_jsonl_path=results_jsonl_path,
            top_k_fallback=top_k_fallback,
            allow_fallback_target=False,
        )
    except (FileNotFoundError, KeyError) as exact_err:
        if not run_if_missing:
            if allow_fallback_target:
                warnings.warn(
                    "Exact target was not found; falling back to nearest available target from saved results.",
                    stacklevel=2,
                )
                for path in candidate_paths:
                    try:
                        return resolve_explainer_selection(
                            dataset_name=dataset_name,
                            explainer_name=explainer_name,
                            target_idx=int(target_idx),
                            results_root=results_root,
                            model_type=model_type,
                            results_jsonl_path=path,
                            top_k_fallback=top_k_fallback,
                            allow_fallback_target=True,
                        )
                    except Exception:
                        continue
                return resolve_explainer_selection(
                    dataset_name=dataset_name,
                    explainer_name=explainer_name,
                    target_idx=int(target_idx),
                    results_root=results_root,
                    model_type=model_type,
                    results_jsonl_path=results_jsonl_path,
                    top_k_fallback=top_k_fallback,
                    allow_fallback_target=True,
                )
            raise

        if not model_type:
            raise ValueError(
                "run_if_missing=True requires `model_type` so the model/explainer can be loaded."
            ) from exact_err

        warnings.warn(
            f"No exact cached explanation for target_idx={int(target_idx)}. "
            f"Running explainer='{explainer_name}' on-demand.",
            stacklevel=2,
        )
        try:
            return run_explainer_for_target(
                dataset_name=dataset_name,
                explainer_name=explainer_name,
                target_idx=int(target_idx),
                results_root=results_root,
                model_type=model_type,
                checkpoint_path=run_checkpoint_path,
                num_hops=run_num_hops,
                candidates_size=run_candidates_size,
                num_neighbors=run_num_neighbors,
                window=run_window,
                seed=run_seed,
                top_k_fallback=top_k_fallback,
                explainer_overrides=run_explainer_overrides,
                show_progress=run_show_progress,
            )
        except Exception:
            if allow_fallback_target:
                warnings.warn(
                    "On-demand run failed; falling back to nearest available cached target.",
                    stacklevel=2,
                )
                return resolve_explainer_selection(
                    dataset_name=dataset_name,
                    explainer_name=explainer_name,
                    target_idx=int(target_idx),
                    results_root=results_root,
                    model_type=model_type,
                    results_jsonl_path=results_jsonl_path,
                    top_k_fallback=top_k_fallback,
                    allow_fallback_target=True,
                )
            raise


def ensure_explainer_results_for_targets(
    dataset_name: str,
    explainer_name: str,
    target_idxs: Sequence[int],
    results_root: Path | str,
    *,
    model_type: Optional[str] = None,
    results_jsonl_path: Optional[Path | str] = None,
    top_k_fallback: int = 8,
    max_success: Optional[int] = None,
    continue_on_error: bool = True,
    run_if_missing: bool = True,
    run_checkpoint_path: Optional[Path | str] = None,
    run_num_hops: int = 2,
    run_candidates_size: int = 75,
    run_num_neighbors: int = 20,
    run_window: Optional[Tuple[float, float]] = None,
    run_seed: int = 42,
    run_explainer_overrides: Optional[Dict[str, object]] = None,
    run_show_progress: bool = True,
) -> Dict[str, object]:
    """
    Ensure selections exist for multiple targets and return a summary.
    """
    unique_targets = _deduplicate_preserve_order(int(v) for v in target_idxs)  # type: ignore[arg-type]
    selections: Dict[int, ExplainerSelection] = {}
    successful_targets: List[int] = []
    failed_targets: Dict[int, str] = {}

    for target_idx in unique_targets:
        if max_success is not None and len(successful_targets) >= int(max_success):
            break
        try:
            selection = resolve_or_run_explainer_selection(
                dataset_name=dataset_name,
                explainer_name=explainer_name,
                target_idx=int(target_idx),
                results_root=results_root,
                model_type=model_type,
                results_jsonl_path=results_jsonl_path,
                top_k_fallback=top_k_fallback,
                allow_fallback_target=False,
                run_if_missing=run_if_missing,
                run_checkpoint_path=run_checkpoint_path,
                run_num_hops=run_num_hops,
                run_candidates_size=run_candidates_size,
                run_num_neighbors=run_num_neighbors,
                run_window=run_window,
                run_seed=run_seed,
                run_explainer_overrides=run_explainer_overrides,
                run_show_progress=run_show_progress,
            )
            resolved_target = int(selection.target_idx)
            selections[resolved_target] = selection
            successful_targets.append(resolved_target)
        except Exception as exc:
            failed_targets[int(target_idx)] = str(exc)
            if not continue_on_error:
                raise

    return {
        "successful_targets": _deduplicate_preserve_order(successful_targets),
        "failed_targets": failed_targets,
        "selections": selections,
    }

__all__ = [
    "discover_latest_explainer_results",
    "discover_explainer_result_files",
    "load_explainer_selection_from_results",
    "list_available_targets_in_results",
    "resolve_explainer_selection",
    "run_explainer_for_target",
    "resolve_or_run_explainer_selection",
    "ensure_explainer_results_for_targets",
]
