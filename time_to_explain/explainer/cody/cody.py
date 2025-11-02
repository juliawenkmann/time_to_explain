from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from submodules.explainer.CoDy.cody.embedding import DynamicEmbedding, StaticEmbedding
from submodules.explainer.CoDy.cody.explainer.baseline.pgexplainer import FactualExplanation, TPGExplainer
from submodules.explainer.CoDy.cody.explainer.baseline.tgnnexplainer import TGNNExplainer, TGNNExplainerExplanation
from submodules.explainer.CoDy.cody.explainer.cody import CoDy
from submodules.explainer.CoDy.cody.explainer.greedy import CounterFactualExample, GreedyCFExplainer

from time_to_explain.core.registries import register_explainer
from time_to_explain.core.types import Explanation, ExplanationResult, ExplanationType
from time_to_explain.explainer.base import BaseExplainer

REQUIRED_WRAPPER_ATTRS = (
    "initialize",
    "get_candidate_events",
    "predict",
    "compute_edge_probabilities",
    "reset_model",
)


def _resolve_explainer_wrapper(model):
    candidate = None
    if hasattr(model, "raw"):
        raw = model.raw()
        candidate = raw
    if candidate is None:
        candidate = getattr(model, "wrapper", model)
    if callable(candidate):
        candidate = candidate()

    missing = [attr for attr in REQUIRED_WRAPPER_ATTRS if not hasattr(candidate, attr)]
    if missing:
        raise TypeError(
            "Explainer requires wrapper exposing methods: "
            f"{', '.join(REQUIRED_WRAPPER_ATTRS)}. Missing: {', '.join(missing)}"
        )
    return candidate


def _select_embedding(name: str, dataset, wrapper) -> Any:
    if name == "dynamic":
        return DynamicEmbedding(dataset, wrapper)
    return StaticEmbedding(dataset, wrapper)


def _explanation_from_factual(event: int, factual: FactualExplanation) -> Explanation:
    metadata = {
        "original_score": factual.original_score,
        "candidate_size": factual.statistics.get("candidate_size"),
    }
    metadata.update({f"stat.{k}": v for k, v in factual.statistics.items()})
    return Explanation(
        event_id=event,
        edge_ids=list(map(int, factual.event_ids)),
        edge_importance=list(map(float, factual.event_importances)),
        type=ExplanationType.FACTUAL,
        metadata=metadata,
    )


def _explanation_from_counterfactual(cf: CounterFactualExample) -> Explanation:
    metadata = {
        "original_prediction": cf.original_prediction,
        "counterfactual_prediction": cf.counterfactual_prediction,
        "achieves_counterfactual": cf.achieves_counterfactual_explanation,
    }
    return Explanation(
        event_id=cf.explained_event_id,
        edge_ids=list(map(int, cf.event_ids)),
        edge_importance=list(map(float, cf.get_absolute_importances() or [])),
        type=ExplanationType.COUNTERFACTUAL,
        metadata=metadata,
    )


class PGExplainerAdapter(BaseExplainer):
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        super().__init__(name="pgexplainer", alias=cfg.get("alias"), config=cfg)
        self.embedding: Any | None = None
        self.impl: TPGExplainer | None = None

    def setup(self, model, dataset):
        super().setup(model, dataset)
        wrapper = _resolve_explainer_wrapper(model)
        embedding_type = self.config.get("embedding", "static").lower()
        self.embedding = _select_embedding(embedding_type, dataset, wrapper)
        device = self.config.get("device", getattr(wrapper, "device", "cpu"))
        hidden_dim = int(self.config.get("hidden_dimension", 128))
        self.impl = TPGExplainer(wrapper, self.embedding, device=device, hidden_dimension=hidden_dim)

        checkpoint = self.config.get("checkpoint")
        if checkpoint:
            state_dict = torch.load(checkpoint, map_location=device)
            self.impl.explainer.load_state_dict(state_dict)
            self._is_trained = True

    def requires_training(self) -> bool:
        return True

    def fit(self, **kwargs: Any) -> None:
        if not self.model or not self.impl:
            raise RuntimeError("Explainer not initialized")
        wrapper = _resolve_explainer_wrapper(self.model)
        fit_cfg = {**{"epochs": 100, "learning_rate": 1e-4, "batch_size": 32}, **kwargs}
        out_dir = Path(fit_cfg.get("output_dir", "artifacts/pgexplainer"))
        out_dir.mkdir(parents=True, exist_ok=True)
        train_event_ids = fit_cfg.get("train_event_ids")
        self.impl.train(
            epochs=int(fit_cfg["epochs"]),
            learning_rate=float(fit_cfg["learning_rate"]),
            batch_size=int(fit_cfg["batch_size"]),
            model_name=self.model.name,
            save_directory=str(out_dir),
            train_event_ids=train_event_ids,
        )
        self._is_trained = True

    def explain(self, context) -> ExplanationResult:
        if not self.impl:
            raise RuntimeError("PGExplainer not initialised. Call setup first.")
        factual = self.impl.explain(context.event_id)
        primary = _explanation_from_factual(context.event_id, factual)
        result = ExplanationResult(
            event_id=context.event_id,
            explainer_name=self.alias,
            primary=primary,
            raw=factual,
            timings=factual.timings,
            statistics=factual.statistics,
        )
        return result


class TGNNExplainerAdapter(BaseExplainer):
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        super().__init__(name="tgnnexplainer", alias=cfg.get("alias"), config=cfg)
        self.pgexplainer: TPGExplainer | None = None
        self.impl: TGNNExplainer | None = None

    def setup(self, model, dataset):
        super().setup(model, dataset)
        wrapper = _resolve_explainer_wrapper(model)
        embedding_type = self.config.get("embedding", "static")
        embedding = _select_embedding(embedding_type, dataset, wrapper)
        device = self.config.get("device", getattr(wrapper, "device", "cpu"))
        hidden_dim = int(self.config.get("hidden_dimension", 128))
        self.pgexplainer = TPGExplainer(wrapper, embedding, device=device, hidden_dimension=hidden_dim)
        checkpoint = self.config.get("pgexplainer_checkpoint")
        if checkpoint:
            state_dict = torch.load(checkpoint, map_location=device)
            self.pgexplainer.explainer.load_state_dict(state_dict)
            self.pgexplainer.explainer.eval()
            self._is_trained = True

        results_dir = Path(self.config.get("results_dir", "artifacts/tgnnexplainer/results"))
        results_dir.mkdir(parents=True, exist_ok=True)
        mcts_dir = Path(self.config.get("mcts_dir", results_dir / "mcts"))
        mcts_dir.mkdir(parents=True, exist_ok=True)
        self.impl = TGNNExplainer(
            wrapper,
            embedding,
            self.pgexplainer,
            results_dir=str(results_dir),
            device=device,
            rollout=int(self.config.get("rollout", 20)),
            min_atoms=int(self.config.get("min_atoms", 1)),
            c_puct=float(self.config.get("c_puct", 10.0)),
            mcts_saved_dir=str(mcts_dir),
            save_results=bool(self.config.get("save_results", True)),
        )

    def requires_training(self) -> bool:
        return False

    def explain(self, context) -> ExplanationResult:
        if not self.impl:
            raise RuntimeError("TGNNExplainer not initialised. Call setup first.")
        explanation: TGNNExplainerExplanation = self.impl.explain(context.event_id)
        if explanation.results:
            top_candidate = max(explanation.results, key=lambda item: item["prediction"])
            primary = Explanation(
                event_id=context.event_id,
                edge_ids=list(map(int, top_candidate["event_ids_in_explanation"])),
                metadata={
                    "best_prediction": explanation.best_prediction,
                    "original_prediction": explanation.original_prediction,
                },
                type=ExplanationType.COUNTERFACTUAL,
            )
        else:
            primary = Explanation(event_id=context.event_id, edge_ids=[], type=ExplanationType.COUNTERFACTUAL)

        candidates: list[Explanation] = []
        for item in explanation.results:
            candidates.append(
                Explanation(
                    event_id=context.event_id,
                    edge_ids=list(map(int, item["event_ids_in_explanation"])),
                    metadata={"prediction": item["prediction"]},
                    type=ExplanationType.COUNTERFACTUAL,
                )
            )

        result = ExplanationResult(
            event_id=context.event_id,
            explainer_name=self.alias,
            primary=primary,
            candidates=candidates,
            raw=explanation,
            timings=explanation.timings,
            statistics=explanation.statistics,
        )
        return result


class CoDyExplainerAdapter(BaseExplainer):
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        super().__init__(name="cody", alias=cfg.get("alias"), config=cfg)
        self.impl: CoDy | None = None

    def setup(self, model, dataset):
        super().setup(model, dataset)
        wrapper = _resolve_explainer_wrapper(model)
        self.impl = CoDy(
            wrapper,
            candidates_size=int(self.config.get("candidates_size", 75)),
            selection_policy=self.config.get("selection_policy", "recent"),
            max_steps=int(self.config.get("max_steps", 200)),
            verbose=bool(self.config.get("verbose", False)),
            approximate_predictions=bool(self.config.get("approximate_predictions", True)),
            alpha=float(self.config.get("alpha", 2 / 3)),
        )

    def explain(self, context) -> ExplanationResult:
        if not self.impl:
            raise RuntimeError("CoDy explainer not initialised. Call setup first.")
        cf_example = self.impl.explain(context.event_id)
        primary = _explanation_from_counterfactual(cf_example)
        candidate_ids = primary.edge_ids
        metadata = {"candidate_ids": candidate_ids}
        result = ExplanationResult(
            event_id=context.event_id,
            explainer_name=self.alias,
            primary=primary,
            raw=cf_example,
            statistics=metadata,
        )
        return result


class GreedyCFExplainerAdapter(BaseExplainer):
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        super().__init__(name="greedycf", alias=cfg.get("alias"), config=cfg)
        self.impl: GreedyCFExplainer | None = None

    def setup(self, model, dataset):
        super().setup(model, dataset)
        wrapper = _resolve_explainer_wrapper(model)
        self.impl = GreedyCFExplainer(wrapper)

    def explain(self, context) -> ExplanationResult:
        if not self.impl:
            raise RuntimeError("GreedyCF explainer not initialised. Call setup first.")
        cf_example = self.impl.explain(context.event_id)
        primary = _explanation_from_counterfactual(cf_example)
        return ExplanationResult(
            event_id=context.event_id,
            explainer_name=self.alias,
            primary=primary,
            raw=cf_example,
        )


@register_explainer("pgexplainer")
def build_pgexplainer(cfg: dict[str, Any], model, dataset):
    explainer = PGExplainerAdapter(config=cfg)
    explainer.setup(model, dataset)
    return explainer


@register_explainer("tgnnexplainer")
def build_tgnnexplainer(cfg: dict[str, Any], model, dataset):
    explainer = TGNNExplainerAdapter(config=cfg)
    explainer.setup(model, dataset)
    return explainer


@register_explainer("cody")
def build_cody(cfg: dict[str, Any], model, dataset):
    explainer = CoDyExplainerAdapter(config=cfg)
    explainer.setup(model, dataset)
    return explainer


@register_explainer("greedycf")
def build_greedycf(cfg: dict[str, Any], model, dataset):
    explainer = GreedyCFExplainerAdapter(config=cfg)
    explainer.setup(model, dataset)
    return explainer
