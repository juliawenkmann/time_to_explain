from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from time_to_explain.adapters import (
    AttnAdapter,
    AttnAdapterConfig,
    CoDyAdapter,
    CoDyAdapterConfig,
    DegreeAdapter,
    DegreeAdapterConfig,
    GNNExplainerAdapter,
    GNNExplainerAdapterConfig,
    GreedyAdapter,
    GreedyAdapterConfig,
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
)
from time_to_explain.explainer.cody_tgn_impl import CoDyTGNImplAdapter, CoDyTGNImplAdapterConfig
from time_to_explain.explainer.gradient import GradientAdapterConfig, GradientExplainer
from time_to_explain.explainer.shap import ShapAdapterConfig, ShapExplainer
from time_to_explain.explainer.tempme import TempMEAdapter, TempMEAdapterConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs" / "explainer"

ExplainerBuilder = Tuple[type, type]

EXPLAINER_BUILDERS: Dict[str, ExplainerBuilder] = {
    "tgnnexplainer": (TGNNExplainerAdapterConfig, TGNNExplainerAdapter),
    "tempme": (TempMEAdapterConfig, TempMEAdapter),
    #"tempme_tgn_impl": (TempMETGNImplAdapterConfig, TempMETGNImplAdapter),
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
    "greedy": (GreedyAdapterConfig, GreedyAdapter),
}


def _format_value(val: Any, *, model_type: str, dataset_name: str) -> Any:
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


def _resolve_path(value: str, *, repo_root: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return str(path)


def load_explainer_config(
    name: str,
    *,
    dataset_name: str,
    model_type: str,
    allow_missing: bool = False,
    configs_dir: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Tuple[Optional[str], Optional[dict], Optional[Path]]:
    root = repo_root or REPO_ROOT
    cfg_dir = configs_dir or CONFIGS_DIR
    candidates = [
        cfg_dir / f"{name.lower()}_{dataset_name}.json",
        cfg_dir / f"{name.lower()}.json",
    ]
    config_path = next((p for p in candidates if p.exists()), None)
    if config_path is None:
        if allow_missing:
            return None, None, None
        raise FileNotFoundError(
            "Explainer config not found. Expected one of: "
            + ", ".join(str(p) for p in candidates)
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    adapter = config.get("adapter") or config.get("name") or name
    args = _format_value(dict(config.get("args") or {}), model_type=model_type, dataset_name=dataset_name)
    for key in ("results_dir", "mcts_saved_dir", "explainer_ckpt_dir", "explainer_ckpt"):
        if key in args and args[key]:
            args[key] = _resolve_path(str(args[key]), repo_root=root)
    nav_params = args.get("navigator_params")
    if isinstance(nav_params, dict) and nav_params.get("explainer_ckpt_dir"):
        nav_params["explainer_ckpt_dir"] = _resolve_path(str(nav_params["explainer_ckpt_dir"]), repo_root=root)
    return adapter, args, config_path


def _apply_defaults(
    args: dict,
    fields: dict,
    *,
    model_type: str,
    dataset_name: str,
    device: Any,
    seed: int,
) -> dict:
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


def build_explainer(
    name: str,
    *,
    dataset_name: str,
    model_type: str,
    device: Any,
    seed: int,
    resolve_callable: Optional[Callable[[str], Any]] = None,
    allow_missing: bool = False,
    overrides: Optional[Dict[str, Any]] = None,
    configs_dir: Optional[Path] = None,
    repo_root: Optional[Path] = None,
    verbose: bool = True,
):
    adapter, args, config_path = load_explainer_config(
        name,
        dataset_name=dataset_name,
        model_type=model_type,
        allow_missing=allow_missing,
        configs_dir=configs_dir,
        repo_root=repo_root,
    )
    if adapter is None:
        return None
    if adapter not in EXPLAINER_BUILDERS:
        raise KeyError(f"No builder registered for explainer '{adapter}'.")
    cfg_cls, explainer_cls = EXPLAINER_BUILDERS[adapter]
    fields = getattr(cfg_cls, "__dataclass_fields__", {})
    args = _apply_defaults(
        args,
        fields,
        model_type=model_type,
        dataset_name=dataset_name,
        device=device,
        seed=seed,
    )
    if overrides:
        args.update(overrides)

    if adapter in {"tempme", "shap"} and isinstance(args.get("score_fn"), str):
        if resolve_callable is None:
            raise NameError(f"Expected callable '{args['score_fn']}' to be defined.")
        args["score_fn"] = resolve_callable(args["score_fn"])
    if adapter == "grad" and isinstance(args.get("forward_fn"), str):
        if resolve_callable is None:
            raise NameError(f"Expected callable '{args['forward_fn']}' to be defined.")
        args["forward_fn"] = resolve_callable(args["forward_fn"])

    cfg = cfg_cls(**args)
    explainer = explainer_cls(cfg)
    if verbose:
        label = getattr(explainer, "alias", adapter)
        print(f"Built explainer '{label}' from {config_path}")
    return explainer


def make_explainer_builder(
    *,
    dataset_name: str,
    model_type: str,
    device: Any,
    seed: int,
    callable_scope: Optional[Mapping[str, Any]] = None,
    resolve_callable: Optional[Callable[[str], Any]] = None,
    configs_dir: Optional[Path] = None,
    repo_root: Optional[Path] = None,
):
    if resolve_callable is None and callable_scope is not None:
        def _resolve(name: str) -> Any:
            obj = callable_scope.get(name)
            if callable(obj):
                return obj
            raise NameError(f"Expected callable '{name}' to be defined.")

        resolve_callable = _resolve

    def _build(name: str, *, overrides: Optional[Dict[str, Any]] = None, allow_missing: bool = False):
        return build_explainer(
            name,
            dataset_name=dataset_name,
            model_type=model_type,
            device=device,
            seed=seed,
            resolve_callable=resolve_callable,
            allow_missing=allow_missing,
            overrides=overrides,
            configs_dir=configs_dir,
            repo_root=repo_root,
        )

    return _build


__all__ = [
    "CONFIGS_DIR",
    "EXPLAINER_BUILDERS",
    "build_explainer",
    "load_explainer_config",
    "make_explainer_builder",
]
