from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict

from time_to_explain.core.types import DatasetBundle

class DatasetRecipe(ABC):
    RECIPE_NAME = "base"

    def __init__(self, **config):
        self.config = config

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return {}

    @abstractmethod
    def generate(self) -> DatasetBundle:
        ...