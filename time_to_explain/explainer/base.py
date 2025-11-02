from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from time_to_explain.core.types import ExplanationContext, ExplanationResult


class BaseExplainer(ABC):
    """Unified interface for third-party temporal graph explainers."""

    def __init__(
        self,
        *,
        name: str,
        alias: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.alias = alias or name
        self.config = config or {}
        self.model: "BaseTemporalGNN" | None = None
        self.dataset: "ContinuousTimeDynamicGraphDataset" | None = None
        self._is_setup = False
        self._is_trained = False

    # ------------------------------------------------------------------ hooks #
    def setup(
        self,
        model: "BaseTemporalGNN",
        dataset: "ContinuousTimeDynamicGraphDataset",
    ) -> None:
        """Attach model and dataset references."""

        self.model = model
        self.dataset = dataset
        self._is_setup = True

    def requires_training(self) -> bool:
        """Flag whether the explainer expects `fit` to be called before use."""

        return False

    def fit(self, *, context: ExplanationContext | None = None, **kwargs: Any) -> None:
        """Optional training hook."""

        self._is_trained = True

    def before_event(self, context: ExplanationContext) -> None:
        """Hook executed right before each `explain` call."""

    def after_event(self, context: ExplanationContext, result: ExplanationResult) -> None:
        """Hook executed directly after each `explain` call."""

    def teardown(self) -> None:
        """Optional cleanup hook."""

    # ----------------------------------------------------------------- helpers #
    @property
    def is_ready(self) -> bool:
        if not self._is_setup:
            return False
        if self.requires_training():
            return self._is_trained
        return True

    # ---------------------------------------------------------------- interface #
    @abstractmethod
    def explain(self, context: ExplanationContext) -> ExplanationResult:
        """Produce an explanation for a given event."""

    def __repr__(self) -> str:  # pragma: no cover - debug utility
        return f"{self.__class__.__name__}(alias={self.alias!r}, name={self.name!r})"


if True:  # pragma: no cover - mypy/pyright assistance block
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from time_to_explain.data.data import ContinuousTimeDynamicGraphDataset
        from time_to_explain.models.base import BaseTemporalGNN  # noqa: F401
