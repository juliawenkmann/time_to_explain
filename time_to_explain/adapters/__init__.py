# time_to_explain/adapters/__init__.py

try:
    from .tgnnexplainer_adapter import TGNNExplainerAdapter, TGNNExplainerAdapterConfig
    _TGNNEXPLAINER_AVAILABLE = True
except ModuleNotFoundError as exc:
    if "torch_sparse" not in str(exc) and "torch_geometric" not in str(exc):
        raise
    _TGNNEXPLAINER_AVAILABLE = False

    class TGNNExplainerAdapterConfig:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "TGNNExplainer adapter requires torch_geometric + torch_sparse. "
                "Install the PyG deps or remove 'tgnnexplainer' from the explainer list."
            ) from exc

    class TGNNExplainerAdapter:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "TGNNExplainer adapter requires torch_geometric + torch_sparse. "
                "Install the PyG deps or remove 'tgnnexplainer' from the explainer list."
            ) from exc
from .attn_adapter import AttnAdapter, AttnAdapterConfig
from .gnn_adapter import GNNExplainerAdapter, GNNExplainerAdapterConfig
from .pg_adapter import PGAdapter, PGAdapterConfig
from .perturb_one_adapter import PerturbOneAdapter, PerturbOneAdapterConfig
from .tg_model_adapter import TemporalGNNModelAdapter
from .random_adapter import RandomAdapter, RandomAdapterConfig
from .degree_adapter import DegreeAdapter, DegreeAdapterConfig
from .tempme_adapter import TempMEAdapter, TempMEAdapterConfig
from .cody_adapter import CoDyAdapter, CoDyAdapterConfig
from .greedy_adapter import GreedyAdapter, GreedyAdapterConfig
from .temgx_adapter import TemGXAdapter, TemGXAdapterConfig

__all__ = [
    "TGNNExplainerAdapter", "TGNNExplainerAdapterConfig",
    "AttnAdapter", "AttnAdapterConfig",
    "GNNExplainerAdapter", "GNNExplainerAdapterConfig",
    "PGAdapter", "PGAdapterConfig",
    "PerturbOneAdapter", "PerturbOneAdapterConfig",
    "TemporalGNNModelAdapter",
    "RandomAdapter", "RandomAdapterConfig",
    "DegreeAdapter", "DegreeAdapterConfig",
    "TempMEAdapter", "TempMEAdapterConfig",
    "CoDyAdapter", "CoDyAdapterConfig",
    "GreedyAdapter", "GreedyAdapterConfig",
    "TemGXAdapter", "TemGXAdapterConfig",
]
