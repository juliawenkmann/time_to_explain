# Compatibility shim pointing to the relocated implementation under explainer.existing.
from time_to_explain.explainer.attention.attn_explainer_tg import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
