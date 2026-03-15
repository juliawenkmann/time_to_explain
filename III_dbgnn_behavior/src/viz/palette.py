"""Shared plotting colors for notebook scripts and visualization helpers."""

from __future__ import annotations

# xcolor equivalents requested by user:
#   \colorlet{eventblue}{blue!65!black}
#   \colorlet{snapshotorange}{orange!85!black}
#   \colorlet{edgegray}{black!55}
EVENT_BLUE = "#0000A6"
SNAPSHOT_ORANGE = "#D96C00"
EDGE_GRAY = "#737373"

BASE_COLORS: tuple[str, ...] = (EVENT_BLUE, SNAPSHOT_ORANGE, EDGE_GRAY)
DEFAULT_CLASS_COLORS: tuple[str, ...] = BASE_COLORS


def color_for_index(i: int) -> str:
    return BASE_COLORS[int(i) % len(BASE_COLORS)]


def class_colors(n: int) -> list[str]:
    n = int(max(0, n))
    return [color_for_index(i) for i in range(n)]


def continuous_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "dbg_event_snapshot_edge",
        [EDGE_GRAY, EVENT_BLUE, SNAPSHOT_ORANGE],
        N=256,
    )


def discrete_cmap(n: int):
    from matplotlib.colors import ListedColormap

    return ListedColormap(class_colors(int(max(1, n))), name="dbg_class_cycle")
