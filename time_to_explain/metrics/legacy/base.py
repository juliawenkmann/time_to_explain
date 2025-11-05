from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from time_to_explain.core.types import ExplanationContext, ExplanationResult, MetricResult


class MetricDirection(str, Enum):
    HIGHER_IS_BETTER = "higher-is-better"
    LOWER_IS_BETTER = "lower-is-better"


class BaseMetric(ABC):
    """Base class for metric adapters."""

    def __init__(
        self,
        *,
        name: str,
        direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.direction = direction
        self.config = config or {}
        self.model: "BaseTemporalGNN" | None = None
        self.dataset: "ContinuousTimeDynamicGraphDataset" | None = None
        self._is_setup = False

    def setup(
        self,
        model: "BaseTemporalGNN",
        dataset: "ContinuousTimeDynamicGraphDataset",
    ) -> None:
        self.model = model
        self.dataset = dataset
        self._is_setup = True

    @abstractmethod
    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        """Compute the metric for a given explanation."""

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"{self.__class__.__name__}(name={self.name!r}, direction={self.direction!r})"


