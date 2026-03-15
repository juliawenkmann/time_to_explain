from __future__ import annotations

"""Explainer construction helpers."""

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from .catalog import EXPLAINER_BUILDERS
from .config import load_explainer_config


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

    if adapter in {"tempme", "tempme_official", "shap"} and isinstance(args.get("score_fn"), str):
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


__all__ = ["build_explainer", "make_explainer_builder"]
