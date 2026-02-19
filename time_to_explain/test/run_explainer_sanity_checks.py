# time_to_explain/test/run_explainer_sanity_checks.py
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np
import torch

from time_to_explain.core.types import ExplanationContext, ExplanationResult, Subgraph

_TEMGX_VENDOR = Path(__file__).resolve().parents[2] / "submodules" / "explainer" / "TemGX" / "link"
if str(_TEMGX_VENDOR) not in sys.path:
    sys.path.insert(0, str(_TEMGX_VENDOR))

from temgxlib.data import ContinuousTimeDynamicGraphDataset
from time_to_explain.data import load_explain_idx
from time_to_explain.data import load_processed_dataset
from time_to_explain.extractors.base_extractor import BaseExtractor
from time_to_explain.models.adapter.wrapper import TGNNWrapper
from time_to_explain.utils.device import pick_device
from time_to_explain.utils.graph import NeighborFinder

from submodules.models.tgat.module import TGAN
from submodules.models.tgn.model.tgn import TGN
from submodules.explainer.tgnnexplainer.tgnnexplainer.xgraph.models.ext.tgn.utils.data_processing import (
    compute_time_statistics,
)

from time_to_explain.adapters import (
    AttnAdapter,
    AttnAdapterConfig,
    CoDyAdapter,
    CoDyAdapterConfig,
    DegreeAdapter,
    DegreeAdapterConfig,
    GNNExplainerAdapter,
    GNNExplainerAdapterConfig,
    PGAdapter,
    PGAdapterConfig,
    PerturbOneAdapter,
    PerturbOneAdapterConfig,
    RandomAdapter,
    RandomAdapterConfig,
    TGNNExplainerAdapter,
    TGNNExplainerAdapterConfig,
    TemGXAdapter,
    TemGXAdapterConfig,
    TemporalGNNModelAdapter,
)
from time_to_explain.explainer.cody_tgn_impl import CoDyTGNImplAdapter, CoDyTGNImplAdapterConfig
from time_to_explain.explainer.gradient import GradientAdapterConfig, GradientExplainer
from time_to_explain.explainer.shap import ShapAdapterConfig, ShapExplainer
from time_to_explain.explainer.tempme import TempMEAdapter, TempMEAdapterConfig
from time_to_explain.explainer.tempme_tgn_impl import TempMETGNImplAdapter, TempMETGNImplAdapterConfig


CONFIGS_DIR = REPO_ROOT / "configs" / "explainer"
MODEL_CONFIGS_DIR = REPO_ROOT / "configs" / "models"
RESOURCES_DIR = REPO_ROOT / "resources"


@dataclass
class SanityReport:
    explainer: str
    passed: bool
    warnings: List[str]
    errors: List[str]

@dataclass
class ExplainerOutcome:
    report: SanityReport
    importances: List[Optional[List[float]]]
    candidates: List[Optional[List[int]]]


def main() -> int:
    args = _parse_args()
    device = pick_device(args.device)
    _set_seed(args.seed)

    bundle = load_processed_dataset(args.dataset)
    events = bundle["interactions"]
    edge_feats = bundle.get("edge_features")
    node_feats = bundle.get("node_features")
    metadata = bundle.get("metadata") or {}

    is_bipartite = _infer_bipartite(events, metadata)
    backbone = _build_backbone(
        model_type=args.model,
        dataset_name=args.dataset,
        events=events,
        edge_feats=edge_feats,
        node_feats=node_feats,
        device=device,
        is_bipartite=is_bipartite,
    )
    model = TemporalGNNModelAdapter(backbone, events, device=device)

    extractor_main = BaseExtractor(
        model=model,
        events=events,
        threshold_num=args.threshold_num,
        keep_order="last-N-then-sort",
    )

    anchors = _load_anchors(args.dataset, events, args.num_anchors)
    explainer_names = args.explainers or list(EXPLAINER_BUILDERS.keys())

    reports: List[SanityReport] = []
    outcomes: Dict[str, ExplainerOutcome] = {}
    reports.append(
        _check_wrapper_model_mapping(
            backbone=backbone,
            events=events,
            edge_feats=edge_feats,
            node_feats=node_feats,
            anchors=anchors,
            device=device,
            tol=args.wrapper_tol,
        )
    )
    reports.append(
        _check_subgraph_oracle_correctness(
            backbone=backbone,
            events=events,
            edge_feats=edge_feats,
            node_feats=node_feats,
            anchors=anchors,
            device=device,
            tol=args.oracle_tol,
        )
    )
    for name in explainer_names:
        try:
            explainer = build_explainer(name, dataset_name=args.dataset, model_type=args.model, device=device, seed=args.seed)
        except ModuleNotFoundError as exc:
            report = SanityReport(name, False, [], [f"missing dependency: {exc}"])
            reports.append(report)
            outcomes[name] = ExplainerOutcome(report=report, importances=[], candidates=[])
            continue
        except Exception as exc:
            report = SanityReport(name, False, [], [f"build failed: {exc}"])
            reports.append(report)
            outcomes[name] = ExplainerOutcome(report=report, importances=[], candidates=[])
            continue

        outcome = _run_explainer_checks(
            explainer=explainer,
            model=model,
            extractor=extractor_main,
            anchors=anchors,
            dataset_name=args.dataset,
            dependence_tol=args.dependence_tol,
            run_id="sanity",
            k_hop=int(getattr(model, "num_layers", 5) or 5),
            num_neighbors=int(getattr(model, "num_neighbors", 20) or 20),
        )
        reports.append(outcome.report)
        outcomes[name] = outcome

    similarity_warnings = _check_explainer_similarity(
        outcomes,
        top_k=args.compare_top_k,
        min_anchors=args.compare_min_anchors,
    )

    _print_summary(reports, similarity_warnings)
    return 0 if all(r.passed for r in reports) else 1


# -----------------------------------------------------------------------------
# Notebook pipeline helpers
# -----------------------------------------------------------------------------

def _build_neighbor_finder(events) -> NeighborFinder:
    u = events["u"].to_numpy(dtype=int)
    v = events["i"].to_numpy(dtype=int)
    ts = events["ts"].to_numpy(dtype=float)
    if "e_idx" in events.columns:
        e_idx = events["e_idx"].to_numpy(dtype=int)
    elif "idx" in events.columns:
        e_idx = events["idx"].to_numpy(dtype=int)
    else:
        e_idx = np.arange(1, len(events) + 1, dtype=int)
    max_node = int(max(u.max(), v.max())) if len(events) else 0
    adj_list = [[] for _ in range(max_node + 1)]
    for src, dst, t, e in zip(u, v, ts, e_idx):
        adj_list[int(src)].append((int(dst), int(e), float(t)))
        adj_list[int(dst)].append((int(src), int(e), float(t)))
    return NeighborFinder(adj_list, uniform=False)


def _build_backbone(
    *,
    model_type: str,
    dataset_name: str,
    events,
    edge_feats,
    node_feats,
    device: torch.device,
    is_bipartite: bool,
):
    model_type = str(model_type).lower()
    cfg_path = _find_model_config(model_type, dataset_name)
    model_config = json.loads(cfg_path.read_text())
    model_args = dict(model_config.get("args") or {})

    if model_type == "tgat":
        if not is_bipartite:
            raise ValueError("TGAT expects bipartite datasets; use MODEL_TYPE='tgn' for non-bipartite data.")
        ngh_finder = _build_neighbor_finder(events)
        backbone = TGAN(
            ngh_finder,
            node_feats,
            edge_feats,
            device=device,
            **model_args,
        )
    elif model_type == "tgn":
        m_src, s_src, m_dst, s_dst = compute_time_statistics(events.u.values, events.i.values, events.ts.values)
        ngh_finder = _build_neighbor_finder(events)
        backbone = TGN(
            ngh_finder,
            node_feats,
            edge_feats,
            device=device,
            mean_time_shift_src=m_src,
            std_time_shift_src=s_src,
            mean_time_shift_dst=m_dst,
            std_time_shift_dst=s_dst,
            **model_args,
        )
    else:
        raise NotImplementedError(model_type)

    ckpt_path = _find_checkpoint(RESOURCES_DIR / "models", dataset_name=dataset_name, model_name=model_type)
    try:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(ckpt_path, map_location="cpu")
    _ = backbone.load_state_dict(state_dict, strict=False)
    _ = backbone.to(device).eval()
    return backbone


def _find_model_config(model_type: str, dataset_name: Optional[str] = None) -> Path:
    candidates = [
        MODEL_CONFIGS_DIR / f"compare_{model_type}_{dataset_name}.json" if dataset_name else None,
        MODEL_CONFIGS_DIR / f"compare_{model_type}.json",
        MODEL_CONFIGS_DIR / f"infer_{model_type}_{dataset_name}.json" if dataset_name else None,
        MODEL_CONFIGS_DIR / f"infer_{model_type}.json",
        MODEL_CONFIGS_DIR / f"train_{model_type}_{dataset_name}.json" if dataset_name else None,
        MODEL_CONFIGS_DIR / f"train_{model_type}.json",
    ]
    for cand in (c for c in candidates if c is not None):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Model config not found for {model_type}.")


def _find_checkpoint(models_root: Path, dataset_name: str, model_name: str) -> Path:
    model_name = model_name.lower()
    dataset_name = str(dataset_name)
    candidates = [
        models_root / dataset_name / model_name / f"{model_name}_{dataset_name}_best.pth",
        models_root / dataset_name / "checkpoints" / f"{model_name}_{dataset_name}_best.pth",
        models_root / "checkpoints" / f"{model_name}_{dataset_name}_best.pth",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    search_roots = [
        models_root / dataset_name / model_name,
        models_root / dataset_name,
        models_root / "checkpoints",
        models_root / "runs",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        matches = sorted(root.rglob(f"{model_name}*{dataset_name}*.pth"))
        if not matches:
            matches = sorted(root.rglob("*.pth"))
        for match in matches:
            if "best" in match.name:
                return match
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Checkpoint not found under {models_root} for {model_name}_{dataset_name}.")


def _infer_bipartite(events, metadata) -> bool:
    config_meta = metadata.get("config") if isinstance(metadata.get("config"), dict) else {}
    is_bipartite = metadata.get("bipartite", config_meta.get("bipartite"))
    if is_bipartite is None:
        if len(events) == 0:
            return False
        u_min, u_max = int(events["u"].min()), int(events["u"].max())
        i_min, i_max = int(events["i"].min()), int(events["i"].max())
        is_bipartite = i_min > u_max or u_min > i_max
    return bool(is_bipartite)


def _load_anchors(dataset_name: str, events, n: int) -> List[Dict[str, Any]]:
    explain_idx_csv = RESOURCES_DIR / "explainer" / "explain_index" / f"{dataset_name}.csv"
    if explain_idx_csv.exists():
        idxs = load_explain_idx(str(explain_idx_csv), start=0)[:n]
    else:
        total = len(events)
        start = min(10, max(0, total - 1))
        idxs = list(range(start + 1, min(total + 1, start + 1 + n)))
    return [{"target_kind": "edge", "event_idx": int(e)} for e in idxs]


def _column_or_fallback(events: pd.DataFrame, primary: str, fallback_idx: int) -> np.ndarray:
    if primary in events.columns:
        return events[primary].to_numpy()
    return events.iloc[:, fallback_idx].to_numpy()


def _normalize_events_for_wrapper(events: pd.DataFrame) -> pd.DataFrame:
    if {"user_id", "item_id", "timestamp", "state_label", "idx"}.issubset(events.columns):
        out = events.copy()
        if out["idx"].min() != 0 or out["idx"].max() != len(out) - 1:
            out["idx"] = np.arange(len(out), dtype=int)
        return out

    u = _column_or_fallback(events, "u", 0)
    v = _column_or_fallback(events, "i", 1)
    ts = _column_or_fallback(events, "ts", 2)
    if "label" in events.columns:
        label = events["label"].to_numpy()
    else:
        label = np.zeros(len(events), dtype=float)

    return pd.DataFrame(
        {
            "user_id": u.astype(int),
            "item_id": v.astype(int),
            "timestamp": ts.astype(float),
            "state_label": label.astype(float),
            "idx": np.arange(len(events), dtype=int),
        }
    )


def _resolve_event_row(events: pd.DataFrame, event_idx: Optional[int]) -> Optional[int]:
    if event_idx is None:
        return None
    try:
        event_idx_int = int(event_idx)
    except Exception:
        return None

    if "e_idx" in events.columns:
        matches = events.index[events["e_idx"] == event_idx_int]
        if len(matches) > 0:
            return int(matches[0])
    if "idx" in events.columns:
        matches = events.index[events["idx"] == event_idx_int]
        if len(matches) > 0:
            return int(matches[0])

    if 0 <= event_idx_int < len(events):
        return event_idx_int
    if 1 <= event_idx_int <= len(events):
        return event_idx_int - 1
    return None


def _build_wrapper_for_checks(
    *,
    backbone: torch.nn.Module,
    events: pd.DataFrame,
    edge_feats: Optional[Any],
    node_feats: Optional[Any],
    anchors: Sequence[Dict[str, Any]],
    device: torch.device,
) -> Tuple[Optional[TGNNWrapper], Optional[int], Optional[ContinuousTimeDynamicGraphDataset], List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []

    if edge_feats is None or node_feats is None:
        warnings.append("skipped: missing edge/node features for wrapper dataset")
        return None, None, None, warnings, errors

    if not isinstance(events, pd.DataFrame) or len(events) == 0:
        errors.append("missing or empty events for wrapper check")
        return None, None, None, warnings, errors

    if isinstance(edge_feats, torch.Tensor):
        edge_feats = edge_feats.detach().cpu().numpy()
    if isinstance(node_feats, torch.Tensor):
        node_feats = node_feats.detach().cpu().numpy()
    edge_feats = np.asarray(edge_feats)
    node_feats = np.asarray(node_feats)

    wrapper_events = _normalize_events_for_wrapper(events)

    event_id_offset = 0
    if "e_idx" in events.columns:
        min_eidx = int(events["e_idx"].min())
        event_id_offset = min_eidx if min_eidx >= 0 else 0
    elif "idx" in events.columns:
        min_idx = int(events["idx"].min())
        event_id_offset = min_idx if min_idx >= 0 else 0

    if edge_feats.shape[0] == len(events) + event_id_offset and event_id_offset > 0:
        edge_feats = edge_feats[event_id_offset:]

    if edge_feats.shape[0] != len(events):
        errors.append("edge features length does not match events for wrapper dataset")
        return None, None, None, warnings, errors

    try:
        cody_dataset = ContinuousTimeDynamicGraphDataset(
            wrapper_events,
            edge_feats,
            node_feats,
            "sanity_wrapper",
            directed=False,
            bipartite=False,
        )
    except Exception as exc:
        errors.append(f"wrapper dataset build failed: {exc}")
        return None, None, None, warnings, errors

    if "e_idx" in events.columns:
        model_event_ids = events["e_idx"].to_numpy(dtype=int)
    elif "idx" in events.columns:
        model_event_ids = events["idx"].to_numpy(dtype=int)
    else:
        model_event_ids = np.arange(1, len(events) + 1, dtype=int)

    wrapper = TGNNWrapper(
        model=backbone,
        dataset=cody_dataset,
        num_hops=int(getattr(backbone, "num_layers", 2) or 2),
        model_name=getattr(backbone, "__class__", type(backbone)).__name__,
        device=device,
        model_event_ids=model_event_ids,
    )

    event_idx = anchors[0].get("event_idx") if anchors else None
    event_row = _resolve_event_row(events, event_idx)
    if event_row is None:
        warnings.append("could not resolve anchor event; falling back to last event")
        event_row = len(events) - 1 if len(events) > 0 else None
    if event_row is None:
        errors.append("no valid event id for wrapper check")
        return None, None, None, warnings, errors

    return wrapper, event_row, cody_dataset, warnings, errors


# -----------------------------------------------------------------------------
# Explainer building (mirrors notebook)
# -----------------------------------------------------------------------------

def _format_value(val, *, model_type: str, dataset_name: str):
    if isinstance(val, str):
        lowered = val.lower()
        if lowered in {"inf", "infinity"}:
            return float("inf")
        try:
            return val.format(model_type=model_type, dataset_name=dataset_name)
        except Exception:
            return val
    if isinstance(val, dict):
        return {k: _format_value(v, model_type=model_type, dataset_name=dataset_name) for k, v in val.items()}
    if isinstance(val, list):
        return [_format_value(v, model_type=model_type, dataset_name=dataset_name) for v in val]
    return val


def _resolve_path(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path)


def load_explainer_config(name: str, *, dataset_name: str, model_type: str):
    candidates = [
        CONFIGS_DIR / f"{name.lower()}_{dataset_name}.json",
        CONFIGS_DIR / f"{name.lower()}.json",
    ]
    config_path = next((p for p in candidates if p.exists()), None)
    if config_path is None:
        raise FileNotFoundError(f"Explainer config not found for {name}.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    adapter = config.get("adapter") or config.get("name") or name
    args = _format_value(dict(config.get("args") or {}), model_type=model_type, dataset_name=dataset_name)
    for key in ("results_dir", "mcts_saved_dir", "explainer_ckpt_dir", "explainer_ckpt"):
        if key in args and args[key]:
            args[key] = _resolve_path(str(args[key]))
    nav_params = args.get("navigator_params")
    if isinstance(nav_params, dict) and nav_params.get("explainer_ckpt_dir"):
        nav_params["explainer_ckpt_dir"] = _resolve_path(str(nav_params["explainer_ckpt_dir"]))
    return adapter, args, config_path


def _apply_defaults(args: dict, fields: dict, *, model_type: str, dataset_name: str, device: torch.device, seed: int) -> dict:
    if "model_name" in fields:
        args.setdefault("model_name", model_type)
    if "dataset_name" in fields:
        args.setdefault("dataset_name", dataset_name)
    if "base_type" in fields:
        args.setdefault("base_type", model_type)
    if "device" in fields:
        args.setdefault("device", device)
    if "seed" in fields:
        args.setdefault("seed", seed)
    if "random_seed" in fields:
        args.setdefault("random_seed", seed)
    if "alias" in fields:
        args.setdefault("alias", None)
    return args


def build_explainer(name: str, *, dataset_name: str, model_type: str, device: torch.device, seed: int):
    adapter, args, config_path = load_explainer_config(name, dataset_name=dataset_name, model_type=model_type)
    if adapter not in EXPLAINER_BUILDERS:
        raise KeyError(f"No builder registered for explainer '{adapter}'.")
    cfg_cls, explainer_cls = EXPLAINER_BUILDERS[adapter]
    fields = getattr(cfg_cls, "__dataclass_fields__", {})
    args = _apply_defaults(args, fields, model_type=model_type, dataset_name=dataset_name, device=device, seed=seed)

    if adapter in {"tempme", "shap"} and isinstance(args.get("score_fn"), str):
        args["score_fn"] = _resolve_callable(args["score_fn"])
    if adapter == "grad" and isinstance(args.get("forward_fn"), str):
        args["forward_fn"] = _resolve_callable(args["forward_fn"])

    cfg = cfg_cls(**args)
    explainer = explainer_cls(cfg)
    print(f"Built explainer '{adapter}' from {config_path}")
    return explainer


EXPLAINER_BUILDERS = {
    "tgnnexplainer": (TGNNExplainerAdapterConfig, TGNNExplainerAdapter),
    "tempme": (TempMEAdapterConfig, TempMEAdapter),
    "tempme_tgn_impl": (TempMETGNImplAdapterConfig, TempMETGNImplAdapter),
    "temgx": (TemGXAdapterConfig, TemGXAdapter),
    "pg": (PGAdapterConfig, PGAdapter),
    "gnn": (GNNExplainerAdapterConfig, GNNExplainerAdapter),
    "perturb_one": (PerturbOneAdapterConfig, PerturbOneAdapter),
    "shap": (ShapAdapterConfig, ShapExplainer),
    "grad": (GradientAdapterConfig, GradientExplainer),
    "attn": (AttnAdapterConfig, AttnAdapter),
    "random": (RandomAdapterConfig, RandomAdapter),
    "degree": (DegreeAdapterConfig, DegreeAdapter),
    "cody": (CoDyAdapterConfig, CoDyAdapter),
    "cody_tgn_impl": (CoDyTGNImplAdapterConfig, CoDyTGNImplAdapter),
}


def _resolve_callable(name: str):
    if name in globals() and callable(globals()[name]):
        return globals()[name]
    raise NameError(f"Expected callable '{name}' to be defined before building explainer.")


# -----------------------------------------------------------------------------
# Explainer sanity checks
# -----------------------------------------------------------------------------

def _check_wrapper_model_mapping(
    *,
    backbone: torch.nn.Module,
    events: pd.DataFrame,
    edge_feats: Optional[Any],
    node_feats: Optional[Any],
    anchors: Sequence[Dict[str, Any]],
    device: torch.device,
    tol: float,
) -> SanityReport:
    warnings: List[str] = []
    errors: List[str] = []

    if not hasattr(backbone, "get_prob"):
        warnings.append("skipped: backbone has no get_prob()")
        return SanityReport("wrapper_model_mapping", True, warnings, errors)

    wrapper, event_row, cody_dataset, build_warnings, build_errors = _build_wrapper_for_checks(
        backbone=backbone,
        events=events,
        edge_feats=edge_feats,
        node_feats=node_feats,
        anchors=anchors,
        device=device,
    )
    warnings.extend(build_warnings)
    errors.extend(build_errors)
    if errors or wrapper is None or event_row is None or cody_dataset is None:
        return SanityReport("wrapper_model_mapping", len(errors) == 0, warnings, errors)

    u = cody_dataset.source_node_ids[event_row]
    v = cody_dataset.target_node_ids[event_row]
    ts = cody_dataset.timestamps[event_row]

    wrapper.reset_model()
    wrapper.initialize(int(event_row))
    with torch.no_grad():
        wrapper_pred, _ = wrapper.predict(int(event_row), result_as_logit=False)
        model_pred = backbone.get_prob(
            np.asarray([int(u)], dtype=np.int64),
            np.asarray([int(v)], dtype=np.int64),
            np.asarray([float(ts)], dtype=np.float32),
            logit=False,
        )

    if not (_is_finite(wrapper_pred) and _is_finite(model_pred)):
        errors.append("wrapper/model prediction returned NaN/Inf")
        return SanityReport("wrapper_model_mapping", False, warnings, errors)

    wrapper_val = _to_scalar_pred(wrapper_pred)
    model_val = _to_scalar_pred(model_pred)
    diff = abs(wrapper_val - model_val)
    if diff > float(tol):
        errors.append(
            f"wrapper/model mismatch (event_row={event_row}): "
            f"wrapper={wrapper_val:.6f}, model={model_val:.6f}, diff={diff:.6f}"
        )

    return SanityReport("wrapper_model_mapping", len(errors) == 0, warnings, errors)


def _check_subgraph_oracle_correctness(
    *,
    backbone: torch.nn.Module,
    events: pd.DataFrame,
    edge_feats: Optional[Any],
    node_feats: Optional[Any],
    anchors: Sequence[Dict[str, Any]],
    device: torch.device,
    tol: float,
) -> SanityReport:
    warnings: List[str] = []
    errors: List[str] = []

    wrapper, event_row, _, build_warnings, build_errors = _build_wrapper_for_checks(
        backbone=backbone,
        events=events,
        edge_feats=edge_feats,
        node_feats=node_feats,
        anchors=anchors,
        device=device,
    )
    warnings.extend(build_warnings)
    errors.extend(build_errors)
    if errors or wrapper is None or event_row is None:
        return SanityReport("subgraph_oracle", len(errors) == 0, warnings, errors)

    wrapper.reset_model()
    wrapper.initialize(int(event_row))
    with torch.no_grad():
        full_pred, _ = wrapper.predict(int(event_row), result_as_logit=False)
        sub_pred, _ = wrapper.compute_edge_probabilities_for_subgraph(
            int(event_row),
            edges_to_drop=np.asarray([], dtype=np.int64),
            result_as_logit=False,
        )

    if not (_is_finite(full_pred) and _is_finite(sub_pred)):
        errors.append("subgraph oracle returned NaN/Inf")
        return SanityReport("subgraph_oracle", False, warnings, errors)

    full_val = _to_scalar_pred(full_pred)
    sub_val = _to_scalar_pred(sub_pred)
    diff = abs(full_val - sub_val)
    if diff > float(tol):
        errors.append(
            f"subgraph oracle mismatch (event_row={event_row}): "
            f"full={full_val:.6f}, subgraph={sub_val:.6f}, diff={diff:.6f}"
        )

    return SanityReport("subgraph_oracle", len(errors) == 0, warnings, errors)


def _check_cody_counterfactual_flip(
    *,
    wrapper: TGNNWrapper,
    events: pd.DataFrame,
    anchor: Dict[str, Any],
    result: ExplanationResult,
) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []

    extras = result.extras or {}
    drop_ids = None
    expected = extras.get("achieves_counterfactual")
    if expected is None:
        expected = extras.get("is_counterfactual")
    for key in ("cf_event_ids", "omitted_edge_idxs"):
        vals = extras.get(key)
        if vals:
            try:
                drop_ids = [int(v) for v in vals]
            except Exception:
                drop_ids = None
            break

    if not drop_ids:
        if expected is not False:
            warnings.append("cody flip check skipped (missing explanation set)")
        return warnings, errors

    event_idx = (
        anchor.get("event_idx")
        or anchor.get("index")
        or anchor.get("idx")
        or extras.get("event_idx")
    )
    event_row = _resolve_event_row(events, event_idx)
    if event_row is None:
        warnings.append("cody flip check skipped (unable to resolve event row)")
        return warnings, errors

    try:
        drop_ids_ds = wrapper._to_dataset_event_ids(drop_ids)
    except Exception as exc:
        warnings.append(f"cody flip check skipped (id mapping failed: {exc})")
        return warnings, errors

    rollout_ids = None
    candidate_ids = extras.get("candidate_eidx")
    if candidate_ids:
        try:
            candidate_ids_ds = wrapper._to_dataset_event_ids(candidate_ids)
        except Exception:
            offset = int(getattr(wrapper, "event_id_offset", 0))
            candidate_ids_ds = np.asarray(candidate_ids, dtype=int) - offset
        candidate_ids_ds = np.asarray(candidate_ids_ds, dtype=np.int64)
        if candidate_ids_ds.size > 0:
            rollout_ids = candidate_ids_ds
            if drop_ids_ds is not None:
                rollout_ids = rollout_ids[~np.isin(rollout_ids, drop_ids_ds)]

    wrapper.reset_model()
    wrapper.initialize(int(event_row))
    with torch.no_grad():
        full_logit, _ = wrapper.predict(int(event_row), result_as_logit=True)

    wrapper.reset_model()
    wrapper.initialize(int(event_row))
    with torch.no_grad():
        drop_logit, _ = wrapper.compute_edge_probabilities_for_subgraph(
            int(event_row),
            edges_to_drop=np.asarray(drop_ids_ds, dtype=np.int64),
            result_as_logit=True,
            event_ids_to_rollout=rollout_ids,
        )

    if not (_is_finite(full_logit) and _is_finite(drop_logit)):
        errors.append("cody flip check failed (NaN/Inf predictions)")
        return warnings, errors

    full_val = _to_scalar_pred(full_logit)
    drop_val = _to_scalar_pred(drop_logit)
    flipped = (full_val >= 0.0) != (drop_val >= 0.0)

    if not flipped:
        if expected is True:
            errors.append(
                f"cody flip check failed (no label flip; full={full_val:.6f}, drop={drop_val:.6f})"
            )
        else:
            warnings.append(
                f"cody flip check: no label flip observed (full={full_val:.6f}, drop={drop_val:.6f})"
            )
    elif expected is False:
        warnings.append("cody flip check: label flipped even though explainer marked non-counterfactual")

    return warnings, errors


def _check_cody_candidate_overlap(
    *,
    explainer: Any,
    anchor: Dict[str, Any],
    candidate_eidx: Optional[List[int]],
) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []

    if not candidate_eidx:
        warnings.append("cody candidate overlap skipped (missing extractor candidates)")
        return warnings, errors

    cody_core = getattr(explainer, "_explainer", None)
    wrapper = getattr(explainer, "_wrapper", None)
    if cody_core is None or wrapper is None:
        warnings.append("cody candidate overlap skipped (missing cody explainer wrapper)")
        return warnings, errors

    event_idx = anchor.get("event_idx") or anchor.get("index") or anchor.get("idx")
    if event_idx is None:
        warnings.append("cody candidate overlap skipped (missing event id)")
        return warnings, errors

    try:
        internal_event_id = int(event_idx) - int(getattr(explainer, "_event_id_offset", 0))
    except Exception:
        warnings.append("cody candidate overlap skipped (invalid event id)")
        return warnings, errors

    if internal_event_id < 0:
        warnings.append("cody candidate overlap skipped (negative internal event id)")
        return warnings, errors

    try:
        _, sampler = cody_core.initialize_explanation(internal_event_id)
    except Exception as exc:
        warnings.append(f"cody candidate overlap skipped (init failed: {exc})")
        return warnings, errors

    subgraph = getattr(sampler, "subgraph", None)
    if subgraph is None or len(subgraph) == 0:
        warnings.append("cody candidate overlap skipped (empty sampler subgraph)")
        return warnings, errors

    if "idx" in subgraph.columns:
        cody_ids = subgraph["idx"].to_numpy()
    else:
        warnings.append("cody candidate overlap skipped (sampler subgraph missing idx column)")
        return warnings, errors

    try:
        cody_model_ids = wrapper._to_model_event_ids(cody_ids)
    except Exception:
        offset = int(getattr(explainer, "_event_id_offset", 0))
        cody_model_ids = np.asarray(cody_ids, dtype=int) + offset

    cody_set = {int(e) for e in cody_model_ids.tolist()}
    extractor_set = {int(e) for e in candidate_eidx}
    if not cody_set or not extractor_set:
        warnings.append("cody candidate overlap skipped (empty candidate sets)")
        return warnings, errors

    inter = cody_set & extractor_set
    if not inter:
        warnings.append("cody candidate overlap empty (no shared events)")
        return warnings, errors

    union = cody_set | extractor_set
    jaccard = len(inter) / len(union) if union else 0.0
    if jaccard < 0.1:
        warnings.append(
            f"cody candidate overlap low (jaccard={jaccard:.3f}, "
            f"cody={len(cody_set)}, extractor={len(extractor_set)})"
        )

    return warnings, errors

def _run_explainer_checks(
    *,
    explainer,
    model: TemporalGNNModelAdapter,
    extractor: BaseExtractor,
    anchors: Sequence[Dict[str, Any]],
    dataset_name: str,
    dependence_tol: float,
    run_id: str,
    k_hop: int,
    num_neighbors: int,
) -> ExplainerOutcome:
    warnings: List[str] = []
    errors: List[str] = []
    importances: List[Optional[List[float]]] = []
    candidates: List[Optional[List[int]]] = []
    pred_deltas: List[float] = []
    successes = 0
    cody_wrapper = None
    cody_wrapper_ready = False

    if explainer.alias in {"gnn", "gnnexplainer"}:
        backbone = getattr(model, "backbone", None)
        base_forward = torch.nn.Module.forward
        unsupported = (
            not isinstance(backbone, torch.nn.Module)
            or getattr(backbone.__class__, "forward", base_forward) is base_forward
        )
        if unsupported:
            warnings.append(
                "skipped: GNNExplainer requires a PyG model with a forward() method; "
                "temporal backbones are not supported."
            )
            report = SanityReport(explainer.alias, True, warnings, [])
            return ExplainerOutcome(report=report, importances=[], candidates=[])

    try:
        explainer.prepare(model=model, dataset={"events": extractor.events, "dataset_name": dataset_name})
    except Exception as exc:
        report = SanityReport(explainer.alias, False, warnings, [f"prepare failed: {exc}"])
        return ExplainerOutcome(report=report, importances=[], candidates=[])

    if explainer.alias in {"cody", "cody_tgn_impl"}:
        backbone = getattr(model, "backbone", None)
        if backbone is None:
            warnings.append("cody flip check skipped (model has no backbone)")
        else:
            edge_feats = getattr(backbone, "edge_raw_features", None)
            node_feats = getattr(backbone, "node_raw_features", None)
            wrapper, _, _, w, e = _build_wrapper_for_checks(
                backbone=backbone,
                events=extractor.events,
                edge_feats=edge_feats,
                node_feats=node_feats,
                anchors=anchors,
                device=model.device,
            )
            for msg in w:
                warnings.append(f"cody flip check: {msg}")
            for msg in e:
                errors.append(f"cody flip check: {msg}")
            if wrapper is not None and not e:
                cody_wrapper = wrapper
                cody_wrapper_ready = True

    for anchor in anchors:
        ctx = ExplanationContext(
            run_id=run_id,
            target_kind=anchor.get("target_kind", "edge"),
            target=anchor,
            k_hop=k_hop,
            num_neighbors=num_neighbors,
        )
        subg = extractor.extract({"events": extractor.events}, anchor, k_hop=k_hop, num_neighbors=num_neighbors, window=None)
        ctx.subgraph = subg

        try:
            res: ExplanationResult = explainer.explain(ctx)
        except Exception as exc:
            errors.append(f"explain failed: {exc}")
            importances.append(None)
            candidates.append(None)
            continue

        candidate = _resolve_candidate(ctx.subgraph, res)
        candidates.append(candidate)
        if not candidate:
            warnings.append("empty candidate list")
            importances.append(None)
            continue

        aligned, aligned_note = _align_importance(candidate, res)
        if aligned_note:
            warnings.append(aligned_note)
        consistency_error, consistency_warn = _check_candidate_consistency(candidate, res)
        if consistency_warn:
            warnings.append(consistency_warn)
        if consistency_error:
            errors.append(consistency_error)
            importances.append(None)
            continue

        if aligned is None:
            errors.append("missing importance edges")
            importances.append(None)
            continue

        if not _all_finite(aligned):
            errors.append("importance edges contain NaN/Inf")
            importances.append(None)
            continue

        importances.append(aligned)
        successes += 1

        ok, delta = _check_prediction_paths(model, ctx, aligned, top_k=min(5, len(aligned)))
        if not ok:
            warnings.append("predict_proba_with_mask returned NaN/Inf")
        elif delta is not None:
            pred_deltas.append(delta)

        mono_note = _monotonicity_check(model, ctx, aligned)
        if mono_note:
            warnings.append(mono_note)

        if cody_wrapper_ready and cody_wrapper is not None:
            c_warnings, c_errors = _check_cody_counterfactual_flip(
                wrapper=cody_wrapper,
                events=extractor.events,
                anchor=anchor,
                result=res,
            )
            warnings.extend(c_warnings)
            errors.extend(c_errors)

        if explainer.alias == "cody":
            c_warnings, c_errors = _check_cody_candidate_overlap(
                explainer=explainer,
                anchor=anchor,
                candidate_eidx=candidate,
            )
            warnings.extend(c_warnings)
            errors.extend(c_errors)

    if successes > 0:
        if pred_deltas:
            if all(d <= float(dependence_tol) for d in pred_deltas):
                errors.append(f"no model dependence detected (all deltas <= {dependence_tol})")
        else:
            warnings.append("no prediction deltas computed for model-dependence check")

    report = SanityReport(explainer.alias, len(errors) == 0, warnings, errors)
    return ExplainerOutcome(report=report, importances=importances, candidates=candidates)


def _resolve_candidate(subgraph: Optional[Subgraph], res: ExplanationResult) -> Optional[List[int]]:
    if subgraph and subgraph.payload and "candidate_eidx" in subgraph.payload:
        return list(subgraph.payload["candidate_eidx"])
    if res.extras and "candidate_eidx" in res.extras:
        return list(res.extras["candidate_eidx"])
    return None


def _align_importance(candidate: List[int], res: ExplanationResult) -> Tuple[Optional[List[float]], Optional[str]]:
    imp = res.importance_edges
    if imp is None:
        return None, None
    cand_res = res.extras.get("candidate_eidx") if res.extras else None
    candidate_ids = [int(e) for e in candidate]
    cand_res_ids = [int(e) for e in cand_res] if cand_res else None

    if len(imp) == len(candidate):
        if cand_res_ids is not None and len(cand_res_ids) == len(imp) and cand_res_ids != candidate_ids:
            mapping = {int(e): float(imp[i]) for i, e in enumerate(cand_res_ids)}
            aligned = [mapping.get(int(e), 0.0) for e in candidate_ids]
            return aligned, "importance edges aligned to payload candidate list"
        return list(imp), None

    if cand_res_ids is not None and len(cand_res_ids) == len(imp):
        mapping = {int(e): float(imp[i]) for i, e in enumerate(cand_res_ids)}
        aligned = [mapping.get(int(e), 0.0) for e in candidate_ids]
        return aligned, "importance edges aligned to payload candidate list"

    return None, "importance edges length mismatch"


def _check_candidate_consistency(
    candidate: List[int],
    res: ExplanationResult,
) -> Tuple[Optional[str], Optional[str]]:
    cand_res = res.extras.get("candidate_eidx") if res.extras else None
    if not cand_res:
        return None, None

    candidate_ids = [int(e) for e in candidate]
    cand_res_ids = [int(e) for e in cand_res]
    payload_set = set(candidate_ids)
    res_set = set(cand_res_ids)
    if not res_set:
        return None, None

    missing = res_set - payload_set
    if missing:
        sample = ", ".join(str(e) for e in sorted(list(missing))[:5])
        return f"explainer candidate ids missing from payload (sample: {sample})", None

    warnings: List[str] = []
    missing_from_explainer = payload_set - res_set
    if missing_from_explainer:
        sample = ", ".join(str(e) for e in sorted(list(missing_from_explainer))[:5])
        warnings.append(f"payload candidate ids missing from explainer (sample: {sample})")

    if len(candidate_ids) != len(cand_res_ids):
        warnings.append(f"candidate list size mismatch (payload={len(candidate_ids)}, explainer={len(cand_res_ids)})")
    elif candidate_ids != cand_res_ids:
        warnings.append("candidate order mismatch between payload and explainer")

    if warnings:
        return None, "; ".join(warnings)

    return None, None


def _check_prediction_paths(
    model: TemporalGNNModelAdapter,
    ctx: ExplanationContext,
    importance: List[float],
    top_k: int,
) -> Tuple[bool, Optional[float]]:
    subg = ctx.subgraph
    if subg is None:
        return True, None
    candidate = subg.payload.get("candidate_eidx") if subg.payload else None
    if not candidate:
        return True, None

    mask = _topk_mask(importance, top_k=top_k)
    pred_full = model.predict_proba(subg, ctx.target)
    pred_keep = model.predict_proba_with_mask(subg, ctx.target, edge_mask=mask)

    if not (_is_finite(pred_full) and _is_finite(pred_keep)):
        return False, None
    delta = abs(_to_scalar_pred(pred_full) - _to_scalar_pred(pred_keep))
    return True, float(delta)


def _monotonicity_check(
    model: TemporalGNNModelAdapter,
    ctx: ExplanationContext,
    importance: List[float],
    *,
    topk_values: Sequence[int] = (1, 2, 5, 10),
) -> Optional[str]:
    subg = ctx.subgraph
    if subg is None or not subg.payload:
        return None
    candidate = subg.payload.get("candidate_eidx")
    if not candidate:
        return None

    pred_full = model.predict_proba(subg, ctx.target)
    if not _is_finite(pred_full):
        return None
    full_val = _to_scalar_pred(pred_full)

    diffs: List[Tuple[int, float]] = []
    for k in topk_values:
        mask = _topk_mask(importance, top_k=min(int(k), len(importance)))
        pred_keep = model.predict_proba_with_mask(subg, ctx.target, edge_mask=mask)
        if not _is_finite(pred_keep):
            return "monotonicity check skipped (NaN/Inf predictions)"
        diff = abs(full_val - _to_scalar_pred(pred_keep))
        diffs.append((int(k), diff))

    if len(diffs) < 2:
        return None

    non_monotonic = False
    for i in range(len(diffs) - 1):
        if diffs[i + 1][1] > diffs[i][1] + 1e-8:
            non_monotonic = True
            break

    if non_monotonic:
        seq = ", ".join(f"k={k}: {d:.4f}" for k, d in diffs)
        return f"monotonicity warning (|full-keep| increases): {seq}"
    return None


def _topk_mask(scores: List[float], *, top_k: int) -> List[float]:
    if not scores:
        return []
    k = max(1, min(int(top_k), len(scores)))
    idx = np.argsort(np.asarray(scores))[-k:]
    keep = set(int(i) for i in idx)
    return [1.0 if i in keep else 0.0 for i in range(len(scores))]


def _all_finite(x: Iterable[float]) -> bool:
    return all(np.isfinite(float(v)) for v in x)


def _is_finite(val: Any) -> bool:
    try:
        if torch.is_tensor(val):
            val = val.detach().cpu().numpy()
    except Exception:
        pass
    arr = np.asarray(val)
    return np.all(np.isfinite(arr))


def _to_scalar_pred(val: Any) -> float:
    try:
        if torch.is_tensor(val):
            val = val.detach().cpu().numpy()
    except Exception:
        pass
    arr = np.asarray(val)
    if arr.size == 0:
        return float("nan")
    return float(arr.reshape(-1)[0])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sanity checks for all explainers.")
    parser.add_argument("--dataset", default="wikipedia")
    parser.add_argument("--model", default="tgn", choices=["tgn", "tgat"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-anchors", type=int, default=2)
    parser.add_argument("--threshold-num", type=int, default=5000)
    parser.add_argument("--explainers", nargs="*", default=None, help="Optional explainer names to run.")
    parser.add_argument("--compare-top-k", type=int, default=10)
    parser.add_argument("--compare-min-anchors", type=int, default=2)
    parser.add_argument("--dependence-tol", type=float, default=1e-5)
    parser.add_argument("--wrapper-tol", type=float, default=1e-4)
    parser.add_argument("--oracle-tol", type=float, default=1e-4)
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _print_summary(reports: Sequence[SanityReport], similarity_warnings: Sequence[str]) -> None:
    print("\nSanity check summary")
    for rep in reports:
        status = "PASS" if rep.passed else "FAIL"
        print(f"- {rep.explainer}: {status}")
        for w in rep.warnings:
            print(f"  warning: {w}")
        for e in rep.errors:
            print(f"  error: {e}")
    if similarity_warnings:
        print("\nExplainer similarity warnings")
        for msg in similarity_warnings:
            print(f"- {msg}")
    else:
        print("\nExplainer similarity warnings\n- none")


def _check_explainer_similarity(
    outcomes: Dict[str, ExplainerOutcome],
    *,
    top_k: int,
    min_anchors: int,
) -> List[str]:
    names = [name for name, out in outcomes.items() if out.report.passed]
    warnings: List[str] = []
    if len(names) < 2:
        return warnings

    from itertools import combinations

    for a, b in combinations(names, 2):
        out_a = outcomes[a]
        out_b = outcomes[b]
        pairs = _pairwise_topk_matches(out_a, out_b, top_k=top_k)
        if pairs is None:
            continue
        match_count, total = pairs
        if total < int(min_anchors):
            continue
        if match_count == total:
            warnings.append(f"{a} and {b} produced identical top-{top_k} selections on {total} anchors")
    return warnings


def _pairwise_topk_matches(
    out_a: ExplainerOutcome,
    out_b: ExplainerOutcome,
    *,
    top_k: int,
) -> Optional[Tuple[int, int]]:
    match_count = 0
    total = 0
    for imp_a, imp_b, cand_a, cand_b in zip(out_a.importances, out_b.importances, out_a.candidates, out_b.candidates):
        if imp_a is None or imp_b is None:
            continue
        if not cand_a or not cand_b:
            continue
        if len(cand_a) != len(cand_b):
            continue
        if len(imp_a) != len(cand_a) or len(imp_b) != len(cand_b):
            continue
        total += 1
        top_a = set(_topk_indices(imp_a, top_k=top_k))
        top_b = set(_topk_indices(imp_b, top_k=top_k))
        if top_a == top_b:
            match_count += 1
    if total == 0:
        return None
    return match_count, total


def _topk_indices(scores: List[float], *, top_k: int) -> List[int]:
    if not scores:
        return []
    k = max(1, min(int(top_k), len(scores)))
    idx = np.argsort(np.asarray(scores))[-k:]
    return [int(i) for i in idx]


# -----------------------------------------------------------------------------
# TempME/TGNN notebook helpers
# -----------------------------------------------------------------------------

from submodules.explainer.tgnnexplainer.tgnnexplainer.xgraph.method.tg_score import _set_tgat_data
from time_to_explain.core.types import Subgraph as TTESubgraph


def tempme_score_fn(model, dataset, target_event_idx, active_event_ids=None):
    events_df = dataset.get("events") if isinstance(dataset, dict) else dataset
    src_idx_l, target_idx_l, cut_time_l = _set_tgat_data(events_df, target_event_idx)
    preserve = list(active_event_ids) if active_event_ids else None
    out = model.get_prob(src_idx_l, target_idx_l, cut_time_l, logit=True, edge_idx_preserve_list=preserve)
    return float(out.detach().cpu().item()) if hasattr(out, "detach") else float(out)


def shap_score_fn(model, dataset, target_event_idx, active_event_ids):
    events_df = dataset.get("events") if isinstance(dataset, dict) else dataset
    src_idx_l, target_idx_l, cut_time_l = _set_tgat_data(events_df, target_event_idx)
    preserve = list(active_event_ids) if active_event_ids else None
    out = model.get_prob(src_idx_l, target_idx_l, cut_time_l, logit=True, edge_idx_preserve_list=preserve)
    return float(out.detach().cpu().item()) if hasattr(out, "detach") else float(out)


def grad_forward_fn(model, dataset, target_event_idx, candidate_eidx, mask):
    subgraph = TTESubgraph(node_ids=[], edge_index=[], payload={"candidate_eidx": list(candidate_eidx), "event_idx": int(target_event_idx)})
    score = model.predict_proba_with_mask(subgraph, {"event_idx": int(target_event_idx)}, edge_mask=mask)
    if isinstance(score, torch.Tensor):
        return score.squeeze()
    return torch.as_tensor(score, device=mask.device, dtype=mask.dtype).squeeze()


if __name__ == "__main__":
    sys.exit(main())
