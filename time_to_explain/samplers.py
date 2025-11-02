from __future__ import annotations

from typing import Any

import numpy as np

from time_to_explain.core.registries import register_sampler


@register_sampler("random")
def build_random_sampler(config: dict[str, Any], dataset):
    cfg = dict(config)
    count = int(cfg.get("count", cfg.get("num_events", 32)))
    section = cfg.get("section", "test")
    seed = cfg.get("seed")

    def _sample() -> list[int]:
        rng = np.random.default_rng(seed)
        if hasattr(dataset, "extract_random_event_ids"):
            event_ids = dataset.extract_random_event_ids(section=section)
        else:
            edge_ids = getattr(dataset, "edge_ids")
            event_ids = rng.choice(edge_ids, size=min(count, len(edge_ids)), replace=False).tolist()
        if len(event_ids) > count:
            event_ids = sorted(rng.choice(event_ids, size=count, replace=False).tolist())
        else:
            event_ids = sorted(map(int, event_ids))
        return event_ids

    return _sample
