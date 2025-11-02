from __future__ import annotations

from typing import Any

from time_to_explain.core.registries import register_model
from time_to_explain.models.wrapper import _build_model


def _build_model(config: dict[str, Any], dataset, model_type: str) -> TGNNWrapper:
    cfg = dict(config)
    cfg.pop("builder", None)
    cfg.pop("base_model", None)
    cfg.setdefault("model_type", model_type.lower())
    alias = cfg.get("alias") or cfg.get("name") or model_type
    wrapper = TGNNWrapper(dataset=dataset, name=alias, config=cfg)
    checkpoint = cfg.get("checkpoint") or cfg.get("checkpoint_path")
    if checkpoint:
        wrapper.load_checkpoint(checkpoint)
    return wrapper


@register_model("graphmixer")
def build_graphmixer(config: dict[str, Any], dataset):
    return _build_model(config, dataset, model_type="graphmixer")


@register_model("tgn")
def build_tgn(config: dict[str, Any], dataset):
    return _build_model(config, dataset, model_type="tgn")


@register_model("tgat")
def build_tgat(config: dict[str, Any], dataset):
    return _build_model(config, dataset, model_type="tgat")

