from __future__ import annotations

from typing import Callable

from explainers.counterfactual import CounterfactualConfig, CounterfactualEdgeDeletionExplainer
from explainers.counterfactual_data_only import (
    DataOnlyCounterfactualConfig,
    DataOnlyCounterfactualEdgeDeletionExplainer,
)
from explainers.epsilon_sufficient import (
    EpsilonSufficientConfig,
    EpsilonSufficientEdgeExplainer,
)
from explainers.statistical_counterfactual import (
    StatisticalCounterfactualConfig,
    StatisticalCounterfactualEdgeDeletionExplainer,
)
from explainers.old.edge_weight import EdgeWeightExplainer
from explainers.grad_edge_weight import GradEdgeWeightExplainer
from explainers.gnnexplainer_debruijn import GNNExplainerDeBruijn
from explainers.integrated_gradients_edge_weight import IntegratedGradientsEdgeWeightExplainer
from explainers.old.lime import LIMEEdgeExplainer
from explainers.old.mask_optim import MaskOptimEdgeExplainer
from explainers.random_edges import RandomEdgeExplainer
from explainers.old.shuffle_gt_oracle import ShuffleGTOracleExplainer, ShuffleGTAttrs
from explainers.old.stay_gt_oracle import StayGTOracleExplainer, StayGTAttrs


ExplainerBuilder = Callable[..., object]


def _build_shuffle_gt_oracle(
    *,
    focus: str = "middle",
    score_mode: str = "z",
    z_thr: float = 2.0,
    positive_only: bool = False,
    irrelevant_score: float = -1e9,
    z_attr: str = "gt_z_higher_order",
    triples_attr: str = "ho_triples",
    **_: object,
) -> ShuffleGTOracleExplainer:
    return ShuffleGTOracleExplainer(
        attrs=ShuffleGTAttrs(z_attr=z_attr, triples_attr=triples_attr),
        focus=focus,  # type: ignore[arg-type]
        score_mode=score_mode,  # type: ignore[arg-type]
        z_thr=z_thr,
        positive_only=positive_only,
        irrelevant_score=irrelevant_score,
    )



def _build_stay_gt_oracle(
    *,
    focus: str = "middle",
    positive_only: bool = False,
    irrelevant_score: float = -1e9,
    score_attr: str = "gt_stay_score_higher_order",
    label_attr: str = "gt_stay_label_higher_order",
    triples_attr: str = "ho_triples",
    **_: object,
) -> StayGTOracleExplainer:
    return StayGTOracleExplainer(
        attrs=StayGTAttrs(score_attr=score_attr, label_attr=label_attr, triples_attr=triples_attr),
        focus=focus,  # type: ignore[arg-type]
        positive_only=positive_only,
        irrelevant_score=irrelevant_score,
    )


EXPLAINER_REGISTRY: dict[str, ExplainerBuilder] = {
    "random_edges": lambda *, seed=0, **_: RandomEdgeExplainer(seed=seed),
    "edge_weight": lambda *, abs_value=True, **_: EdgeWeightExplainer(abs_value=abs_value),
    "grad_edge_weight": lambda *, abs_value=True, grad_x_input=False, **_: GradEdgeWeightExplainer(
        abs_value=abs_value, grad_x_input=grad_x_input
    ),
    "integrated_gradients_edge_weight": lambda *, steps=32, abs_value=True, baseline="zeros", **_: IntegratedGradientsEdgeWeightExplainer(
        steps=steps, abs_value=abs_value, baseline=baseline
    ),
    # GNNExplainer-style optimization (edge mask over De Bruijn edges)
    "gnnexplainer": lambda *, steps=200, lr=0.05, lam_size=0.01, lam_ent=0.1, size_mode="mean", seed=0, **_: GNNExplainerDeBruijn(
        steps=steps,
        lr=lr,
        lam_size=lam_size,
        lam_ent=lam_ent,
        size_mode=size_mode,
        seed=seed,
    ),
    # LIME-style local surrogate over candidate higher-order edges (triples)
    "lime": lambda *, seed=0, **kwargs: LIMEEdgeExplainer(seed=seed, **kwargs),
    # KernelSHAP-like surrogate (same implementation, different kernel weighting)
    "kernel_shap": lambda *, seed=0, **kwargs: LIMEEdgeExplainer(seed=seed, kernel="shap", **kwargs),
    # Short alias (people often search for "shap")
    "shap": lambda *, seed=0, **kwargs: LIMEEdgeExplainer(seed=seed, kernel="shap", **kwargs),
    "mask_optim": lambda *, steps=150, lr=0.1, lam_size=0.01, lam_ent=0.1, seed=0, **_: MaskOptimEdgeExplainer(
        steps=steps, lr=lr, lam_size=lam_size, lam_ent=lam_ent, seed=seed
    ),
    # Counterfactual: learn an edge mask that flips the prediction with (approximately) minimal deletions.
    # Scores are "deletion priorities" (higher => remove first).
    "counterfactual": lambda *, seed=0, **kwargs: CounterfactualEdgeDeletionExplainer(
        cfg=CounterfactualConfig(**kwargs),
        seed=seed,
    ),
    # Epsilon-sufficient: keep a small edge set that preserves p(y|v).
    # Scores are "keep priorities" (higher => keep first).
    "epsilon_sufficient": lambda *, seed=0, **kwargs: EpsilonSufficientEdgeExplainer(
        cfg=EpsilonSufficientConfig(**kwargs),
        seed=seed,
    ),
    # Data-only counterfactual: rank HO edges by weights within the dominant source cluster (no model).
    "data_only_counterfactual": lambda *, seed=0, **kwargs: DataOnlyCounterfactualEdgeDeletionExplainer(
        cfg=DataOnlyCounterfactualConfig(**kwargs),
    ),
    # Statistical counterfactual: rank edges by deviation from shuffled-time null model (z-scores).
    # Requires data.<z_attr> (default "gt_z_higher_order") aligned with the explain-space HO edges.
    "statistical_counterfactual": lambda *, seed=0, **kwargs: StatisticalCounterfactualEdgeDeletionExplainer(
        cfg=StatisticalCounterfactualConfig(**kwargs),
    ),
    # Short alias
    "stat_cf": lambda *, seed=0, **kwargs: StatisticalCounterfactualEdgeDeletionExplainer(
        cfg=StatisticalCounterfactualConfig(**kwargs),
    ),
    # Oracle: deterministic cluster-stay GT (requires data.gt_stay_* + data.ho_triples)
    "stay_gt_oracle": _build_stay_gt_oracle,
    # Oracle: uses shuffle-time ground truth z-scores (requires data.gt_z_higher_order + data.ho_triples)
    "shuffle_gt_oracle": _build_shuffle_gt_oracle,
}


def build_explainer(name: str, *, seed: int = 0, **kwargs):
    if name not in EXPLAINER_REGISTRY:
        raise KeyError(f"Unknown explainer: {name}. Available: {list(EXPLAINER_REGISTRY)}")
    return EXPLAINER_REGISTRY[name](seed=seed, **kwargs)
