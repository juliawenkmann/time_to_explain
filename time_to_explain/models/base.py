from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from time_to_explain.data.data import ContinuousTimeDynamicGraphDataset


class BaseTemporalGNN(ABC):
    """Abstraction for temporal graph neural network backbones."""

    def __init__(
        self,
        *,
        name: str,
        dataset: ContinuousTimeDynamicGraphDataset,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.config = config or {}

    @abstractmethod
    def set_evaluation_mode(self, flag: bool) -> None:
        ...

    @abstractmethod
    def reset_state(self) -> None:
        ...

    @abstractmethod
    def predict_event(self, event_id: int, *, result_as_logit: bool = False) -> Any:
        ...

    @abstractmethod
    def compute_edge_probabilities(
        self,
        source_nodes,
        target_nodes,
        edge_timestamps,
        edge_ids,
        *,
        negative_nodes=None,
        result_as_logit: bool = False,
        perform_memory_update: bool = True,
        **kwargs: Any,
    ) -> Any:
        ...

    @abstractmethod
    def compute_edge_probabilities_for_subgraph(
        self,
        event_id: int,
        *,
        edges_to_drop,
        result_as_logit: bool = False,
        event_ids_to_rollout=None,
        **kwargs: Any,
    ) -> Any:
        ...

    def detach_memory(self) -> None:
        """Optional hook for models maintaining memory."""

    def restore_memory(self, snapshot: Any, *, event_id: int | None = None) -> None:
        """Optional hook to restore a memory snapshot."""

    def get_memory_snapshot(self) -> Any:
        """Optional hook to access memory snapshot."""

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"{self.__class__.__name__}(name={self.name!r})"


class WrapperTemporalGNN(BaseTemporalGNN):
    """Adapter turning legacy wrapper classes into BaseTemporalGNN instances."""

    def __init__(self, *, wrapper: Any, name: str | None = None, config: dict[str, Any] | None = None):
        if not hasattr(wrapper, "dataset"):
            raise AttributeError("Wrapper objects must expose a 'dataset' attribute.")
        dataset = wrapper.dataset
        model_name = name or getattr(wrapper, "name", wrapper.__class__.__name__)
        super().__init__(name=model_name, dataset=dataset, config=config)
        self.wrapper = wrapper

    # ----------------------------------------------------------------- delegates #
    def set_evaluation_mode(self, flag: bool) -> None:
        if hasattr(self.wrapper, "set_evaluation_mode"):
            self.wrapper.set_evaluation_mode(flag)

    def reset_state(self) -> None:
        if hasattr(self.wrapper, "reset_model"):
            self.wrapper.reset_model()

    def predict_event(self, event_id: int, *, result_as_logit: bool = False) -> Any:
        if hasattr(self.wrapper, "predict"):
            return self.wrapper.predict(event_id, result_as_logit=result_as_logit)
        raise NotImplementedError("Underlying wrapper does not implement 'predict'.")

    def compute_edge_probabilities(
        self,
        source_nodes,
        target_nodes,
        edge_timestamps,
        edge_ids,
        *,
        negative_nodes=None,
        result_as_logit: bool = False,
        perform_memory_update: bool = True,
        **kwargs: Any,
    ) -> Any:
        if hasattr(self.wrapper, "compute_edge_probabilities"):
            return self.wrapper.compute_edge_probabilities(
                source_nodes,
                target_nodes,
                edge_timestamps,
                edge_ids,
                negative_nodes=negative_nodes,
                result_as_logit=result_as_logit,
                perform_memory_update=perform_memory_update,
                **kwargs,
            )
        raise NotImplementedError("Underlying wrapper does not implement 'compute_edge_probabilities'.")

    def compute_edge_probabilities_for_subgraph(
        self,
        event_id: int,
        *,
        edges_to_drop,
        result_as_logit: bool = False,
        event_ids_to_rollout=None,
        **kwargs: Any,
    ) -> Any:
        if hasattr(self.wrapper, "compute_edge_probabilities_for_subgraph"):
            return self.wrapper.compute_edge_probabilities_for_subgraph(
                event_id,
                edges_to_drop=edges_to_drop,
                result_as_logit=result_as_logit,
                event_ids_to_rollout=event_ids_to_rollout,
                **kwargs,
            )
        raise NotImplementedError(
            "Underlying wrapper does not implement 'compute_edge_probabilities_for_subgraph'."
        )

    def detach_memory(self) -> None:
        if hasattr(self.wrapper, "detach_memory"):
            self.wrapper.detach_memory()

    def restore_memory(self, snapshot: Any, *, event_id: int | None = None) -> None:
        if hasattr(self.wrapper, "restore_memory"):
            if event_id is None:
                self.wrapper.restore_memory(snapshot)
            else:
                self.wrapper.restore_memory(snapshot, event_id)

    def get_memory_snapshot(self) -> Any:
        if hasattr(self.wrapper, "get_memory"):
            return self.wrapper.get_memory()
        return None

    # Convenience ----------------------------------------------------------- #
    def raw(self) -> Any:
        """Expose the underlying legacy wrapper."""
        return self.wrapper
