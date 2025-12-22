# time_to_explain/adapters/__init__.py

try:
    from .subgraphx_adapter import SubgraphXTGAdapter, SubgraphXTGAdapterConfig
    _SUBGRAPHX_AVAILABLE = True
except ModuleNotFoundError as exc:
    if "torch_sparse" not in str(exc) and "torch_geometric" not in str(exc):
        raise
    _SUBGRAPHX_AVAILABLE = False

    class SubgraphXTGAdapterConfig:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "SubgraphX-TG adapter requires torch_geometric + torch_sparse. "
                "Install the PyG deps or remove 'subgraphx_tg' from the explainer list."
            ) from exc

    class SubgraphXTGAdapter:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "SubgraphX-TG adapter requires torch_geometric + torch_sparse. "
                "Install the PyG deps or remove 'subgraphx_tg' from the explainer list."
            ) from exc
from .attn_adapter import AttnAdapter, AttnAdapterConfig
from .gnn_adapter import GNNExplainerAdapter, GNNExplainerAdapterConfig
from .pg_adapter import PGAdapter, PGAdapterConfig
from .perturb_one_adapter import PerturbOneAdapter, PerturbOneAdapterConfig
from .tg_model_adapter import TemporalGNNModelAdapter
from .random_adapter import RandomAdapter, RandomAdapterConfig
from .degree_adapter import DegreeAdapter, DegreeAdapterConfig
from .tempme_adapter import TempMEExplainer
from .tempme_neural_adapter import TempMENeuralAdapter, TempMENeuralAdapterConfig
from .cody_adapter import CoDyAdapter, CoDyAdapterConfig

__all__ = [
    "SubgraphXTGAdapter", "SubgraphXTGAdapterConfig",
    "AttnAdapter", "AttnAdapterConfig",
    "GNNExplainerAdapter", "GNNExplainerAdapterConfig",
    "PGAdapter", "PGAdapterConfig",
    "PerturbOneAdapter", "PerturbOneAdapterConfig",
    "TemporalGNNModelAdapter",
    "RandomAdapter", "RandomAdapterConfig",
    "DegreeAdapter", "DegreeAdapterConfig",
    "TempMEExplainer",
    "TempMENeuralAdapter", "TempMENeuralAdapterConfig",
    "CoDyAdapter", "CoDyAdapterConfig",
]
