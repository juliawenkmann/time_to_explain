"""Shared plotting theme for CBM visualizations."""

from __future__ import annotations

from typing import List

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb

# Requested palette:
#   eventblue      = blue!65!black
#   snapshotorange = orange!85!black
#   edgegray       = black!55
EVENT_BLUE = "#0000A6"
SNAPSHOT_ORANGE = "#D98C00"
EDGE_GRAY = "#737373"


def _mix_hex(c1: str, c2: str, t: float) -> str:
    """Mix two colors in RGB space with interpolation factor t in [0, 1]."""
    t = float(np.clip(t, 0.0, 1.0))
    a = np.asarray(to_rgb(c1), dtype=float)
    b = np.asarray(to_rgb(c2), dtype=float)
    out = (1.0 - t) * a + t * b
    return "#{:02x}{:02x}{:02x}".format(
        int(np.clip(round(255.0 * out[0]), 0, 255)),
        int(np.clip(round(255.0 * out[1]), 0, 255)),
        int(np.clip(round(255.0 * out[2]), 0, 255)),
    )


def categorical_palette(n: int) -> List[str]:
    """Generate n categorical colors from the requested 3-color theme."""
    n = int(n)
    if n <= 0:
        return []
    if n == 1:
        return [EVENT_BLUE]
    if n == 2:
        return [EVENT_BLUE, SNAPSHOT_ORANGE]

    xs = np.linspace(0.0, 1.0, n)
    cols: List[str] = []
    for x in xs:
        if x < 0.5:
            cols.append(_mix_hex(EVENT_BLUE, EDGE_GRAY, x / 0.5))
        else:
            cols.append(_mix_hex(EDGE_GRAY, SNAPSHOT_ORANGE, (x - 0.5) / 0.5))
    return cols


DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "cbm_diverging_event_snapshot",
    [EVENT_BLUE, EDGE_GRAY, SNAPSHOT_ORANGE],
)
SEQ_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "cbm_seq_blue",
    [_mix_hex("#ffffff", EDGE_GRAY, 0.2), EVENT_BLUE],
)
SEQ_ORANGE_CMAP = LinearSegmentedColormap.from_list(
    "cbm_seq_orange",
    [_mix_hex("#ffffff", EDGE_GRAY, 0.2), SNAPSHOT_ORANGE],
)
SEQ_GRAY_CMAP = LinearSegmentedColormap.from_list(
    "cbm_seq_gray",
    [_mix_hex("#ffffff", EDGE_GRAY, 0.1), EDGE_GRAY],
)

