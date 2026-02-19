"""Extractor package.

Importing this package registers all extractors via their @register_extractor
side effects, so a single `import ...extractors` is enough to make them
available in the registry.
"""

# NOTE: these imports are intentionally unused; they trigger registration.
from .base_extractor import TGEventCandidatesExtractor as TGEventCandidatesExtractor  # noqa: F401
from .khop_extractor import KHopCandidatesExtractor as KHopCandidatesExtractor  # noqa: F401
from .random_extractor import RandomExtractor as RandomExtractor  # noqa: F401
from .tempme_extractor import TempMEExtractor as TempMEExtractor  # noqa: F401

__all__ = [
    "TGEventCandidatesExtractor",
    "KHopCandidatesExtractor",
    "RandomExtractor",
    "TempMEExtractor",
]
