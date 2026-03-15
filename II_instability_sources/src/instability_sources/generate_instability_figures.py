from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
import math

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)


def mix(color_a, color_b, weight_a=0.5):
    """Linear color mixing: weight_a * A + (1-weight_a) * B."""
    a = np.array(mcolors.to_rgb(color_a))
    b = np.array(mcolors.to_rgb(color_b))
    return tuple(weight_a * a + (1 - weight_a) * b)


EVENTBLUE = mix("blue", "black", 0.65)
SNAPSHOTORANGE = mix("orange", "black", 0.85)
EDGEGRAY = mix("black", "white", 0.55)
PALEBLUE = mix(EVENTBLUE, "white", 0.24)
PALEORANGE = mix(SNAPSHOTORANGE, "white", 0.22)
PALEGRAY = mix(EDGEGRAY, "white", 0.28)
LIGHTGRID = mix(EDGEGRAY, "white", 0.14)


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "axes.edgecolor": EDGEGRAY,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": LIGHTGRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "legend.frameon": False,
        "xtick.color": EDGEGRAY,
        "ytick.color": EDGEGRAY,
        "text.color": mix("black", "white", 0.82),
        "axes.labelcolor": mix("black", "white", 0.82),
        "axes.titlecolor": mix("black", "white", 0.88),
        "mathtext.default": "regular",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    }
)


def save_pdf(fig: plt.Figure, filename: str) -> Path:
    path = OUTDIR / filename
    fig.savefig(path, bbox_inches="tight")
    return path


def save_gif(anim: FuncAnimation, filename: str, fps: int = 12) -> Path:
    path = OUTDIR / filename
    anim.save(path, writer=PillowWriter(fps=fps), dpi=140)
    return path


# -----------------------------------------------------------------------------
# Core GRU simulation
# -----------------------------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gru_run(messages, params, remove_idx=None):
    Wz, Uz, bz, Wr, Ur, br, Wn, Un, bn = params
    hidden_dim = Uz.shape[0]
    h = np.zeros(hidden_dim)
    states = [h.copy()]
    z_values = []
    r_values = []
    n_values = []

    for step, x in enumerate(messages, start=1):
        x = np.asarray(x, dtype=float)
        if remove_idx is not None and step == remove_idx:
            states.append(h.copy())
            z_values.append(np.full(hidden_dim, np.nan))
            r_values.append(np.full(hidden_dim, np.nan))
            n_values.append(np.full(hidden_dim, np.nan))
            continue

        z_t = sigmoid(Wz @ x + Uz @ h + bz)
        r_t = sigmoid(Wr @ x + Ur @ h + br)
        n_t = np.tanh(Wn @ x + Un @ (r_t * h) + bn)
        h = (1 - z_t) * n_t + z_t * h

        states.append(h.copy())
        z_values.append(z_t.copy())
        r_values.append(r_t.copy())
        n_values.append(n_t.copy())

    return {
        "states": np.array(states),
        "z": np.array(z_values),
        "r": np.array(r_values),
        "n": np.array(n_values),
    }


GRU_SEED = 2941
GRU_REMOVE_IDX = 6
GRU_STEPS = 18


def build_gru_case(seed=GRU_SEED, remove_idx=GRU_REMOVE_IDX, steps=GRU_STEPS):
    rng = np.random.default_rng(seed)
    hidden_dim = 2
    message_dim = 2
    scale_u = 1.5
    scale_w = 1.5

    Wz = rng.normal(size=(hidden_dim, message_dim)) * scale_w
    Uz = rng.normal(size=(hidden_dim, hidden_dim)) * scale_u
    bz = rng.normal(size=hidden_dim) * 0.2
    Wr = rng.normal(size=(hidden_dim, message_dim)) * scale_w
    Ur = rng.normal(size=(hidden_dim, hidden_dim)) * scale_u
    br = rng.normal(size=hidden_dim) * 0.2
    Wn = rng.normal(size=(hidden_dim, message_dim)) * scale_w
    Un = rng.normal(size=(hidden_dim, hidden_dim)) * scale_u
    bn = rng.normal(size=hidden_dim) * 0.2
    params = (Wz, Uz, bz, Wr, Ur, br, Wn, Un, bn)

    times = np.linspace(0, 3 * np.pi, steps)
    messages = np.stack(
        [
            0.9 * np.sin(times) + 0.25 * np.cos(2 * times),
            0.7 * np.cos(0.8 * times) + 0.35 * np.sin(1.7 * times),
        ],
        axis=1,
    )
    messages[remove_idx - 1] += np.array([1.3, -0.9])

    full = gru_run(messages, params, remove_idx=None)
    skipped = gru_run(messages, params, remove_idx=remove_idx)

    return {
        "messages": messages,
        "remove_idx": remove_idx,
        "params": params,
        "full": full,
        "skipped": skipped,
    }


# -----------------------------------------------------------------------------
# Figure helpers
# -----------------------------------------------------------------------------

def soften_axes(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(EDGEGRAY)
    ax.spines["bottom"].set_color(EDGEGRAY)
    ax.tick_params(colors=EDGEGRAY)



def annotate_panel(ax, text):
    # Panel labels intentionally disabled.
    return


def _gru_hidden_axis_limits(full_states, skipped_states, pad=0.18):
    points = np.vstack([full_states, skipped_states])
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, 0.2)
    pads = span * pad
    return (mins[0] - pads[0], maxs[0] + pads[0]), (mins[1] - pads[1], maxs[1] + pads[1])


def _setup_gru_hidden_axes(ax, full_states, skipped_states):
    soften_axes(ax)
    ax.grid(True, alpha=0.35)
    ax.axhline(0, color=LIGHTGRID, lw=1.0)
    ax.axvline(0, color=LIGHTGRID, lw=1.0)
    (xmin, xmax), (ymin, ymax) = _gru_hidden_axis_limits(full_states, skipped_states)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("hidden dim 1")
    ax.set_ylabel("hidden dim 2")


def _trajectory_until(states, progress, start_idx=0):
    progress = float(np.clip(progress, start_idx, len(states) - 1))
    base_idx = int(math.floor(progress))
    alpha = progress - base_idx
    pts = states[start_idx : base_idx + 1].copy()
    if pts.size == 0:
        pts = states[start_idx : start_idx + 1].copy()
    if alpha > 1e-9 and base_idx < len(states) - 1:
        interp = (1 - alpha) * states[base_idx] + alpha * states[base_idx + 1]
        if np.linalg.norm(pts[-1] - interp) > 1e-12:
            pts = np.vstack([pts, interp])
    return pts


def _striped_polyline(points, colors, linewidth=3.0, stripes=28, zorder=1):
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return None

    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    total = float(lengths.sum())
    if total <= 1e-12:
        return None

    stripe_count = max(int(stripes), len(points) - 1)
    distances = np.linspace(0.0, total, stripe_count + 1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    sampled = []
    for dist in distances:
        seg_idx = min(np.searchsorted(cumulative, dist, side="right") - 1, len(lengths) - 1)
        local_len = lengths[seg_idx]
        if local_len <= 1e-12:
            sampled.append(points[seg_idx].copy())
            continue
        frac = (dist - cumulative[seg_idx]) / local_len
        sampled.append(points[seg_idx] + frac * deltas[seg_idx])

    sampled = np.asarray(sampled)
    segments = np.stack([sampled[:-1], sampled[1:]], axis=1)
    segment_colors = [colors[i % len(colors)] for i in range(len(segments))]
    return LineCollection(
        segments,
        colors=segment_colors,
        linewidths=linewidth,
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
    )


def _gru_neighborhood_graph_spec():
    center = (2.1, 0.0)
    neighbors = {
        "A": (0.9, 1.25),
        "B": (3.45, 1.05),
        "C": (3.55, -0.95),
        "D": (0.95, -1.22),
        "E": (2.15, 1.85),
    }
    # (neighbor, temporal event index)
    events = [
        ("A", 1),
        ("B", 2),
        ("C", 3),
        ("D", 4),
        ("A", 5),
        ("E", 6),
        ("C", 7),
        ("B", 8),
        ("D", 9),
    ]
    return center, neighbors, events


def _gru_edge_label_xy(x0, y0, x1, y1, t, occ_idx=0, occ_total=1):
    dx, dy = x1 - x0, y1 - y0
    norm = math.hypot(dx, dy) + 1e-9
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux

    # Spread repeated timestamps on the same edge and alternate sides.
    if int(occ_total) > 1:
        span = np.linspace(-0.08, 0.08, int(occ_total))
        along_shift = float(span[int(np.clip(occ_idx, 0, int(occ_total) - 1))])
    else:
        along_shift = 0.0

    base_frac = 0.50
    side = -1.0 if ((int(t) + int(occ_idx)) % 2 == 0) else 1.0
    perp_mag = 0.085
    tiny_jitter = ((int(t) % 3) - 1) * 0.010

    bx = x0 + (base_frac + along_shift) * dx
    by = y0 + (base_frac + along_shift) * dy
    return (
        bx + px * perp_mag * side + ux * tiny_jitter,
        by + py * perp_mag * side + uy * tiny_jitter,
    )


# -----------------------------------------------------------------------------
# 1) GRU memory drift figure
# -----------------------------------------------------------------------------

def draw_gru_input_graph(ax, remove_idx=GRU_REMOVE_IDX, shown_steps=9):
    ax.set_xlim(0.0, 12.2)
    ax.set_ylim(-2.3, 2.3)
    ax.axis("off")

    center, neighbors, events = _gru_neighborhood_graph_spec()

    nbr_total = Counter(nbr for nbr, _ in events)
    nbr_seen = Counter()
    for nbr, t in events:
        x0, y0 = neighbors[nbr]
        x1, y1 = center
        is_removed = t == remove_idx
        edge_color = SNAPSHOTORANGE if is_removed else EVENTBLUE
        edge_style = (0, (4, 2)) if is_removed else "-"
        ax.plot([x0, x1], [y0, y1], color=edge_color, lw=2.2, ls=edge_style, zorder=1, alpha=0.95)
        occ_idx = int(nbr_seen[nbr])
        occ_total = int(nbr_total[nbr])
        nbr_seen[nbr] += 1
        lx, ly = _gru_edge_label_xy(x0, y0, x1, y1, t, occ_idx=occ_idx, occ_total=occ_total)
        ax.text(
            lx,
            ly,
            f"$t_{t}$",
            color=edge_color,
            fontsize=10.5,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.80),
            zorder=3,
        )
        if is_removed:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.plot([mx - 0.12, mx + 0.12], [my - 0.12, my + 0.12], color=SNAPSHOTORANGE, lw=2.0, zorder=2)
            ax.plot([mx - 0.12, mx + 0.12], [my + 0.12, my - 0.12], color=SNAPSHOTORANGE, lw=2.0, zorder=2)

    # Nodes.
    for node, (x, y) in neighbors.items():
        c = Circle((x, y), radius=0.17, facecolor="white", edgecolor=EVENTBLUE, lw=1.8, zorder=3)
        ax.add_patch(c)
        ax.text(x, y, node, ha="center", va="center", color=EVENTBLUE, fontsize=11, weight="bold", zorder=4)

    cu = Circle(center, radius=0.21, facecolor=PALEBLUE, edgecolor=EVENTBLUE, lw=2.2, zorder=3)
    ax.add_patch(cu)
    ax.text(center[0], center[1], "u", ha="center", va="center", color=EVENTBLUE, fontsize=12, weight="bold", zorder=4)

    ax.text(0.28, 2.08, "Temporal ego-neighborhood around target node $u$", color=EDGEGRAY, fontsize=11.5, ha="left")
    ax.text(0.28, -2.02, "timestamped interactions define message order", color=EDGEGRAY, fontsize=10.8, ha="left")

    # Arrow from graph to sequence.
    arrow = FancyArrowPatch((4.2, 0.0), (5.4, 0.0), arrowstyle="-|>", mutation_scale=13, lw=1.4, color=EDGEGRAY)
    ax.add_patch(arrow)
    ax.text(4.8, 0.28, "time sort", color=EDGEGRAY, fontsize=10.8, ha="center")

    # Sequence strip.
    xs = np.linspace(5.9, 9.7, shown_steps)
    y_box = 0.0
    w, h = 0.42, 0.32
    for step, x in enumerate(xs, start=1):
        is_removed = step == remove_idx
        face = PALEORANGE if is_removed else PALEBLUE
        edge = SNAPSHOTORANGE if is_removed else EVENTBLUE
        patch = FancyBboxPatch(
            (x - w / 2, y_box - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.4,
            edgecolor=edge,
            facecolor=face,
            alpha=0.97,
        )
        ax.add_patch(patch)
        ax.text(x, y_box, f"$m_{step}$", color=edge, fontsize=10.5, ha="center", va="center")
        if is_removed:
            ax.plot([x - 0.13, x + 0.13], [y_box - 0.12, y_box + 0.12], color=SNAPSHOTORANGE, lw=1.9)
            ax.plot([x - 0.13, x + 0.13], [y_box + 0.12, y_box - 0.12], color=SNAPSHOTORANGE, lw=1.9)
        if step < shown_steps:
            ax.plot([x + w / 2 + 0.04, xs[step] - w / 2 - 0.04], [y_box, y_box], color=EDGEGRAY, lw=1.0)

    ax.text(7.8, 0.62, "ordered message sequence", color=EDGEGRAY, fontsize=11, ha="center")

    # GRU update target.
    gru_x = 11.0
    gru_box = FancyBboxPatch(
        (gru_x - 0.64, -0.34),
        1.2,
        0.68,
        boxstyle="round,pad=0.03,rounding_size=0.1",
        linewidth=1.7,
        edgecolor=EVENTBLUE,
        facecolor=PALEBLUE,
        alpha=0.98,
    )
    ax.add_patch(gru_box)
    ax.text(gru_x - 0.05, 0.0, "GRU", color=EVENTBLUE, fontsize=12.2, weight="bold", ha="center", va="center")
    ax.plot([10.0, gru_x - 0.66], [0.0, 0.0], color=EDGEGRAY, lw=1.2)
    ax.text(10.0, 0.45, f"remove $m_{{{remove_idx}}}$", color=SNAPSHOTORANGE, fontsize=11, ha="center")
    ax.plot([xs[remove_idx - 1], xs[remove_idx - 1]], [0.22, 0.52], color=SNAPSHOTORANGE, lw=1.3)

    ax.text(10.95, -0.78, r"$h_t = GRU(h_{t-1}, m_t)$", color=EDGEGRAY, fontsize=12.2, ha="center")


def draw_gru_schematic(ax, remove_idx=GRU_REMOVE_IDX, shown_steps=9):
    ax.set_xlim(0.3, shown_steps + 1.65)
    ax.set_ylim(-1.65, 1.65)
    ax.axis("off")

    box_w, box_h = 0.66, 0.44
    xs = np.arange(1, shown_steps + 1, dtype=float)
    y_full = 0.38
    y_pruned = -0.86
    msg_y = 1.12

    ax.text(
        0.42,
        1.42,
        r"$h_t = GRU(h_{t-1}, m_t)$",
        fontsize=16,
        color=EDGEGRAY,
        ha="left",
        va="center",
    )

    # Shared prefix region before pruning.
    if remove_idx > 1:
        shared_left = 0.55
        shared_right = remove_idx - 0.52
        if shared_right > shared_left:
            ax.add_patch(
                Rectangle(
                    (shared_left, y_pruned - 0.38),
                    shared_right - shared_left,
                    (y_full - y_pruned) + 0.92,
                    facecolor=PALEBLUE,
                    edgecolor="none",
                    alpha=0.14,
                    zorder=0,
                )
            )
            ax.text(
                (shared_left + shared_right) / 2,
                0.88,
                "shared prefix",
                color=EDGEGRAY,
                fontsize=10.4,
                ha="center",
                va="center",
            )

    # GRU cells and incoming messages.
    for x in xs:
        rect = FancyBboxPatch(
            (x - box_w / 2, y_full - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.4,
            edgecolor=EVENTBLUE,
            facecolor=PALEBLUE,
            alpha=0.96,
        )
        ax.add_patch(rect)
        ax.text(x, y_full, "GRU", ha="center", va="center", color=EVENTBLUE, weight="bold", fontsize=10.6)
        if int(x) != int(remove_idx):
            ax.annotate(
                f"$m_{{{int(x)}}}$",
                xy=(x, msg_y),
                xytext=(x, msg_y + 0.18),
                ha="center",
                fontsize=10.2,
                color=EDGEGRAY,
            )
        else:
            ax.annotate(
                f"$m_{{{int(x)}}}$",
                xy=(x, msg_y),
                xytext=(x, msg_y + 0.18),
                ha="center",
                fontsize=10.2,
                color=SNAPSHOTORANGE,
            )
            ax.plot([x - 0.13, x + 0.13], [msg_y + 0.04, msg_y + 0.30], color=SNAPSHOTORANGE, lw=2.0)
            ax.plot([x - 0.13, x + 0.13], [msg_y + 0.30, msg_y + 0.04], color=SNAPSHOTORANGE, lw=2.0)
        arrow = FancyArrowPatch(
            (x, msg_y),
            (x, y_full + box_h / 2),
            arrowstyle="-|>",
            mutation_scale=11,
            lw=1.2,
            color=EDGEGRAY,
        )
        ax.add_patch(arrow)

    # Full run state sequence (top lane).
    state_full_x = np.r_[0.55, xs]
    state_full_y = np.full_like(state_full_x, y_full, dtype=float)
    ax.plot(state_full_x, state_full_y, color=EVENTBLUE, lw=2.8, solid_capstyle="round")
    ax.scatter(state_full_x, state_full_y, s=30, color=EVENTBLUE, zorder=3)

    # Pruned run: shared prefix + divergence + lower lane suffix.
    state_pruned_prefix_x = np.r_[0.55, np.arange(1, remove_idx, dtype=float)]
    if state_pruned_prefix_x.size > 1:
        state_pruned_prefix_y = np.full_like(state_pruned_prefix_x, y_full, dtype=float)
        ax.plot(state_pruned_prefix_x, state_pruned_prefix_y, color=SNAPSHOTORANGE, lw=2.1, ls=(0, (6, 2)))

    diverge = FancyArrowPatch(
        (remove_idx - 0.02, y_full - 0.02),
        (remove_idx + 0.2, y_pruned + 0.14),
        arrowstyle="-|>",
        mutation_scale=12,
        lw=2.0,
        color=SNAPSHOTORANGE,
        alpha=0.9,
    )
    ax.add_patch(diverge)

    state_pruned_suffix_x = np.arange(remove_idx, shown_steps + 1, dtype=float)
    state_pruned_suffix_y = np.full_like(state_pruned_suffix_x, y_pruned, dtype=float)
    ax.plot(state_pruned_suffix_x, state_pruned_suffix_y, color=SNAPSHOTORANGE, lw=2.5, ls=(0, (6, 2)))
    ax.scatter(state_pruned_suffix_x, state_pruned_suffix_y, s=30, color=SNAPSHOTORANGE, zorder=3)

    skip_x = float(remove_idx)
    ax.add_patch(
        Rectangle(
            (skip_x - 0.5, y_pruned - 0.48),
            1.04,
            2.1,
            facecolor=PALEORANGE,
            edgecolor="none",
            alpha=0.18,
            zorder=0,
        )
    )
    ax.axvline(skip_x, ymin=0.05, ymax=0.97, color=SNAPSHOTORANGE, lw=1.2, ls=(0, (3, 2)))

    ax.text(0.55, y_full + 0.26, "$h_0$", color=EVENTBLUE, fontsize=10.8, ha="center")
    ax.text(shown_steps + 0.08, y_full + 0.24, "$h_t$", color=EVENTBLUE, fontsize=10.8, ha="right")
    ax.text(shown_steps + 0.08, y_pruned - 0.26, r"$\tilde{h}_t$", color=SNAPSHOTORANGE, fontsize=10.8, ha="right")

    ax.annotate(
        "remove $m_{%d}$" % int(remove_idx),
        xy=(skip_x, msg_y + 0.14),
        xytext=(skip_x + 0.58, 1.34),
        color=SNAPSHOTORANGE,
        fontsize=10.8,
        arrowprops=dict(arrowstyle="-", color=SNAPSHOTORANGE, lw=1.2),
    )
    ax.text(
        remove_idx + 0.48,
        y_pruned - 0.48,
        "divergence starts here",
        color=SNAPSHOTORANGE,
        fontsize=10.8,
        ha="left",
    )

    ax.text(shown_steps + 0.28, y_full, "full run", color=EVENTBLUE, fontsize=10.8, va="center", ha="left")
    ax.text(shown_steps + 0.28, y_pruned, "pruned run", color=SNAPSHOTORANGE, fontsize=10.8, va="center", ha="left")


def make_gru_original_graph_figure(case=None) -> Path:
    if case is None:
        case = build_gru_case()
    remove_idx = int(case["remove_idx"])
    center, neighbors, events = _gru_neighborhood_graph_spec()

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw original graph in blue.
    nbr_total = Counter(nbr for nbr, _ in events)
    nbr_seen = Counter()
    for nbr, t in events:
        x0, y0 = neighbors[nbr]
        x1, y1 = center
        ax.plot([x0, x1], [y0, y1], color=EVENTBLUE, lw=2.5, zorder=1, solid_capstyle="round")
        occ_idx = int(nbr_seen[nbr])
        occ_total = int(nbr_total[nbr])
        nbr_seen[nbr] += 1
        lx, ly = _gru_edge_label_xy(x0, y0, x1, y1, t, occ_idx=occ_idx, occ_total=occ_total)
        ax.text(
            lx,
            ly,
            f"$t_{t}$",
            color=EVENTBLUE,
            fontsize=10.6,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.09", facecolor="white", edgecolor="none", alpha=0.82),
            zorder=3,
        )

        if int(t) == remove_idx:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.plot([x0, x1], [y0, y1], color=SNAPSHOTORANGE, lw=3.1, ls=(0, (4, 2)), zorder=2, alpha=0.95)
            ax.plot([mx - 0.1, mx + 0.1], [my - 0.1, my + 0.1], color=SNAPSHOTORANGE, lw=2.0, zorder=3)
            ax.plot([mx - 0.1, mx + 0.1], [my + 0.1, my - 0.1], color=SNAPSHOTORANGE, lw=2.0, zorder=3)

    # Nodes.
    for node, (x, y) in neighbors.items():
        c = Circle((x, y), radius=0.18, facecolor="white", edgecolor=EVENTBLUE, lw=1.9, zorder=4)
        ax.add_patch(c)
        ax.text(x, y, node, ha="center", va="center", color=EVENTBLUE, fontsize=11.2, weight="bold", zorder=5)

    cu = Circle(center, radius=0.22, facecolor=PALEBLUE, edgecolor=EVENTBLUE, lw=2.2, zorder=4)
    ax.add_patch(cu)
    ax.text(center[0], center[1], "u", ha="center", va="center", color=EVENTBLUE, fontsize=12.2, weight="bold", zorder=5)

    ax.text(
        0.02,
        0.98,
        "Original temporal neighborhood graph",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=EDGEGRAY,
        fontsize=12.2,
    )
    ax.text(
        0.02,
        0.04,
        f"Counterfactual removes event $t_{{{remove_idx}}}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=SNAPSHOTORANGE,
        fontsize=11.0,
        bbox=dict(boxstyle="round,pad=0.25", facecolor=PALEORANGE, edgecolor="none"),
    )

    return save_pdf(fig, "01a0_gru_original_graph.pdf")


def make_gru_input_setting_figure(case=None) -> Path:
    if case is None:
        case = build_gru_case()
    remove_idx = case["remove_idx"]

    fig, ax = plt.subplots(figsize=(12.8, 4.4))
    annotate_panel(ax, "A")
    draw_gru_input_graph(ax, remove_idx=remove_idx, shown_steps=9)
    return save_pdf(fig, "01a_gru_input_setting.pdf")


def make_gru_sequence_divergence_figure(case=None) -> Path:
    if case is None:
        case = build_gru_case()
    remove_idx = case["remove_idx"]

    fig, ax = plt.subplots(figsize=(13.2, 5.2))
    annotate_panel(ax, "B")
    draw_gru_schematic(ax, remove_idx=remove_idx, shown_steps=9)
    return save_pdf(fig, "01b_gru_sequence_divergence.pdf")


def make_gru_hidden_trajectory_figure(case=None) -> Path:
    if case is None:
        case = build_gru_case()
    remove_idx = case["remove_idx"]
    full_states = case["full"]["states"]
    skipped_states = case["skipped"]["states"]

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    annotate_panel(ax, "C")
    _setup_gru_hidden_axes(ax, full_states, skipped_states)

    ax.plot(
        full_states[:, 0],
        full_states[:, 1],
        color=EVENTBLUE,
        lw=2.6,
        marker="o",
        ms=4.8,
        label="full event stream",
    )
    ax.plot(
        skipped_states[:, 0],
        skipped_states[:, 1],
        color=SNAPSHOTORANGE,
        lw=2.4,
        ls=(0, (6, 2)),
        marker="o",
        ms=4.6,
        label=r"remove $m_{%d}$" % remove_idx,
    )
    # Explicit shared start marker.
    ax.scatter(full_states[0, 0], full_states[0, 1], s=80, color=EDGEGRAY, zorder=4)
    ax.annotate(
        "shared start $h_0$",
        xy=(full_states[0, 0], full_states[0, 1]),
        xytext=(full_states[0, 0] - 0.33, full_states[0, 1] + 0.2),
        color=EDGEGRAY,
        arrowprops=dict(arrowstyle="->", color=EDGEGRAY, lw=1.0),
    )

    ax.scatter(
        skipped_states[remove_idx, 0],
        skipped_states[remove_idx, 1],
        s=95,
        marker="X",
        facecolor=SNAPSHOTORANGE,
        edgecolor=SNAPSHOTORANGE,
        linewidth=1.2,
        zorder=4,
    )

    ax.annotate(
        "divergence point",
        xy=(skipped_states[remove_idx, 0], skipped_states[remove_idx, 1]),
        xytext=(skipped_states[remove_idx, 0] + 0.2, skipped_states[remove_idx, 1] - 0.3),
        color=SNAPSHOTORANGE,
        arrowprops=dict(arrowstyle="->", color=SNAPSHOTORANGE, lw=1.0),
    )
    ax.annotate(
        "drift keeps compounding",
        xy=(skipped_states[-1, 0], skipped_states[-1, 1]),
        xytext=(0.24, -0.95),
        color=SNAPSHOTORANGE,
        fontsize=11,
        arrowprops=dict(arrowstyle="->", color=SNAPSHOTORANGE, lw=1.1),
    )
    ax.legend(loc="upper left")
    return save_pdf(fig, "01c_gru_hidden_trajectory.pdf")


def make_gru_hidden_trajectory_gif(case=None) -> Path:
    if case is None:
        case = build_gru_case()
    remove_idx = case["remove_idx"]
    full_states = case["full"]["states"]
    skipped_states = case["skipped"]["states"]

    frames_per_transition = 6
    initial_hold_frames = 8
    final_hold_frames = 10
    fps = 12
    anchor_progress = float(remove_idx - 1)
    start_progress = float(remove_idx)
    progress_values = np.linspace(
        start_progress,
        full_states.shape[0] - 1,
        int(full_states.shape[0] - 1 - start_progress) * frames_per_transition + 1,
    )
    progress_values = np.concatenate(
        [
            np.repeat(anchor_progress, initial_hold_frames),
            progress_values,
            np.repeat(full_states.shape[0] - 1, final_hold_frames),
        ]
    )

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    annotate_panel(ax, "C")
    _setup_gru_hidden_axes(ax, full_states, skipped_states)

    legend_handles = [
        Line2D([0], [0], color=EVENTBLUE, lw=2.6, marker="o", ms=4.6, label="full event stream"),
        Line2D(
            [0],
            [0],
            color=SNAPSHOTORANGE,
            lw=2.4,
            ls=(0, (6, 2)),
            marker="o",
            ms=4.6,
            label=rf"remove $m_{{{remove_idx}}}$",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        columnspacing=1.6,
        handlelength=2.6,
    )

    shared_prefix = full_states[:remove_idx]
    shared_prefix_stripes = _striped_polyline(
        shared_prefix,
        colors=[SNAPSHOTORANGE, EVENTBLUE],
        linewidth=3.1,
        stripes=30,
        zorder=2,
    )
    if shared_prefix_stripes is not None:
        ax.add_collection(shared_prefix_stripes)

    full_line, = ax.plot([], [], color=EVENTBLUE, lw=2.6, marker="o", ms=4.6, zorder=3)
    skipped_line, = ax.plot(
        [],
        [],
        color=SNAPSHOTORANGE,
        lw=2.4,
        ls=(0, (6, 2)),
        marker="o",
        ms=4.6,
        zorder=3,
    )
    full_head, = ax.plot([], [], marker="o", ms=8.2, color=EVENTBLUE, ls="None", zorder=5)
    skipped_head, = ax.plot([], [], marker="o", ms=8.2, color=SNAPSHOTORANGE, ls="None", zorder=5)
    anchor_marker, = ax.plot([], [], marker="o", ms=7.0, color=EDGEGRAY, ls="None", zorder=4)
    divergence_marker, = ax.plot(
        [],
        [],
        marker="X",
        ms=9.6,
        color=SNAPSHOTORANGE,
        markeredgewidth=1.3,
        ls="None",
        zorder=6,
    )

    def init():
        for artist in [full_line, skipped_line, full_head, skipped_head, anchor_marker, divergence_marker]:
            artist.set_data([], [])
        return (
            full_line,
            skipped_line,
            full_head,
            skipped_head,
            anchor_marker,
            divergence_marker,
        )

    def update(progress):
        if progress <= anchor_progress + 1e-9:
            for artist in [full_line, skipped_line, full_head, skipped_head, anchor_marker, divergence_marker]:
                artist.set_data([], [])
            return (
                full_line,
                skipped_line,
                full_head,
                skipped_head,
                anchor_marker,
                divergence_marker,
            )

        full_pts = _trajectory_until(full_states, progress, start_idx=remove_idx - 1)
        skipped_pts = _trajectory_until(skipped_states, progress, start_idx=remove_idx - 1)
        full_line.set_data(full_pts[:, 0], full_pts[:, 1])
        skipped_line.set_data(skipped_pts[:, 0], skipped_pts[:, 1])
        full_head.set_data([full_pts[-1, 0]], [full_pts[-1, 1]])
        skipped_head.set_data([skipped_pts[-1, 0]], [skipped_pts[-1, 1]])
        anchor_marker.set_data([full_states[remove_idx - 1, 0]], [full_states[remove_idx - 1, 1]])
        divergence_marker.set_data([skipped_states[remove_idx, 0]], [skipped_states[remove_idx, 1]])
        return (
            full_line,
            skipped_line,
            full_head,
            skipped_head,
            anchor_marker,
            divergence_marker,
        )

    anim = FuncAnimation(
        fig,
        update,
        frames=progress_values,
        init_func=init,
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )
    path = save_gif(anim, "01c_gru_hidden_trajectory.gif", fps=fps)
    plt.close(fig)
    return path


def make_gru_state_drift_figure(case=None) -> Path:
    if case is None:
        case = build_gru_case()
    remove_idx = case["remove_idx"]
    full_states = case["full"]["states"]
    skipped_states = case["skipped"]["states"]
    drift = np.linalg.norm(full_states - skipped_states, axis=1)

    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    annotate_panel(ax, "D")
    soften_axes(ax)
    x_states = np.arange(drift.size)
    ax.fill_between(x_states, drift, color=PALEORANGE, alpha=0.55)
    ax.plot(x_states, drift, color=SNAPSHOTORANGE, lw=2.5)
    ax.axvline(remove_idx, color=EDGEGRAY, lw=1.2, ls=(0, (3, 2)))
    ax.scatter([remove_idx], [drift[remove_idx]], color=SNAPSHOTORANGE, s=52, zorder=3)
    ax.annotate(
        "divergence begins",
        xy=(remove_idx, drift[remove_idx]),
        xytext=(remove_idx + 0.7, drift[remove_idx] + 0.42),
        color=EDGEGRAY,
        fontsize=11,
        arrowprops=dict(arrowstyle="->", color=EDGEGRAY, lw=1.0),
    )
    ax.set_xlabel("event index")
    ax.set_ylabel("state drift")
    ax.set_xlim(0, drift.size - 1)
    ax.set_ylim(0, max(2.8, drift.max() + 0.2))
    return save_pdf(fig, "01d_gru_state_drift_curve.pdf")


def make_gru_gate_shift_figure(case=None) -> Path:
    if case is None:
        case = build_gru_case()
    remove_idx = case["remove_idx"]
    z_full = case["full"]["z"]
    z_skip = case["skipped"]["z"]
    event_idx = np.arange(1, z_full.shape[0] + 1)

    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    annotate_panel(ax, "E")
    soften_axes(ax)
    ax.plot(event_idx, z_full[:, 0], color=EVENTBLUE, lw=2.2, label=r"update gate $z_{t,1}$ (full)")
    ax.plot(
        event_idx,
        z_skip[:, 0],
        color=SNAPSHOTORANGE,
        lw=2.0,
        ls=(0, (6, 2)),
        label=r"after removing $m_{%d}$" % remove_idx,
    )
    ax.axvline(remove_idx, color=EDGEGRAY, lw=1.2, ls=(0, (3, 2)))
    ax.text(remove_idx + 0.1, 0.9, "regime shifts after divergence", color=SNAPSHOTORANGE, fontsize=10.8, ha="left")
    ax.set_xlabel("event index")
    ax.set_ylabel("update gate")
    ax.set_xlim(1, event_idx[-1])
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left")
    return save_pdf(fig, "01e_gru_gate_regime_shift.pdf")


def make_gru_memory_drift_figure():
    case = build_gru_case()
    paths = [
        make_gru_original_graph_figure(case=case),
        make_gru_input_setting_figure(case=case),
        make_gru_sequence_divergence_figure(case=case),
        make_gru_hidden_trajectory_figure(case=case),
        make_gru_hidden_trajectory_gif(case=case),
        make_gru_state_drift_figure(case=case),
        make_gru_gate_shift_figure(case=case),
    ]
    return paths


# -----------------------------------------------------------------------------
# 2) Aggregation normalization shift figure
# -----------------------------------------------------------------------------

def softmax(x):
    x = np.asarray(x, dtype=float)
    z = x - x.max()
    exp = np.exp(z)
    return exp / exp.sum()



def build_normalization_case():
    neighbors = np.array(["n1", "n2", "n3", "n4", "n5"])
    kept = np.array([0, 1, 2])
    removed = np.array([3, 4])
    values = np.array([0.84, 0.62, 0.46, -0.04, -0.22])
    full_mean = values.mean()
    kept_mean = values[kept].mean()

    logits = np.array([1.00, 0.62, 0.20, -0.10, -0.25])
    attn_full = softmax(logits)
    attn_kept = softmax(logits[kept])
    return {
        "neighbors": neighbors,
        "kept": kept,
        "removed": removed,
        "values": values,
        "full_mean": full_mean,
        "kept_mean": kept_mean,
        "attn_full": attn_full,
        "attn_kept": attn_kept,
    }


def make_normalization_shift_mean_figure(case=None) -> Path:
    if case is None:
        case = build_normalization_case()

    neighbors = case["neighbors"]
    kept = case["kept"]
    removed = case["removed"]
    values = case["values"]
    full_mean = case["full_mean"]
    kept_mean = case["kept_mean"]

    fig, ax_mean = plt.subplots(figsize=(6.4, 4.6))
    soften_axes(ax_mean)
    x = np.arange(len(neighbors))
    bar_colors = [EVENTBLUE if idx in kept else PALEGRAY for idx in range(len(neighbors))]
    edge_colors = [EVENTBLUE if idx in kept else EDGEGRAY for idx in range(len(neighbors))]
    bars = ax_mean.bar(x, values, color=bar_colors, edgecolor=edge_colors, linewidth=1.4, width=0.64)
    for idx in removed:
        bars[idx].set_hatch("//")
        bars[idx].set_alpha(0.7)

    ax_mean.axhline(full_mean, color=EVENTBLUE, lw=2.2, ls=(0, (1.5, 2)))
    ax_mean.axhline(kept_mean, color=SNAPSHOTORANGE, lw=2.4)
    ax_mean.fill_between([-0.6, 4.6], full_mean, kept_mean, color=PALEORANGE, alpha=0.28)
    ax_mean.set_xticks(x, neighbors)
    ax_mean.set_ylim(-0.38, 1.08)
    ax_mean.set_ylabel("projected neighbor feature")
    return save_pdf(fig, "02a_normalization_mean_shift.pdf")


def make_normalization_shift_attention_figure(case=None) -> Path:
    if case is None:
        case = build_normalization_case()

    neighbors = case["neighbors"]
    kept = case["kept"]
    removed = case["removed"]
    attn_full = case["attn_full"]
    attn_kept = case["attn_kept"]

    fig, ax_attn = plt.subplots(figsize=(6.4, 4.6))
    soften_axes(ax_attn)
    x = np.arange(len(neighbors))
    edge_colors = [EVENTBLUE if idx in kept else EDGEGRAY for idx in range(len(neighbors))]
    width = 0.34
    full_colors = [EVENTBLUE if idx in kept else PALEGRAY for idx in range(len(neighbors))]
    full_bars = ax_attn.bar(x - width / 2, attn_full, width=width, color=full_colors, edgecolor=edge_colors, linewidth=1.3)
    for idx in removed:
        full_bars[idx].set_hatch("//")
        full_bars[idx].set_alpha(0.7)

    ax_attn.bar(
        kept + width / 2,
        attn_kept,
        width=width,
        color=SNAPSHOTORANGE,
        edgecolor=SNAPSHOTORANGE,
        linewidth=1.3,
    )
    ax_attn.set_xticks(x, neighbors)
    ax_attn.set_ylim(0, 0.68)
    ax_attn.set_ylabel("attention weight")
    return save_pdf(fig, "02b_normalization_attention_shift.pdf")


def build_normalization_mask_scenarios(case=None):
    if case is None:
        case = build_normalization_case()

    neighbors = case["neighbors"]
    values = case["values"]
    full_mean = case["full_mean"]
    keep_count = max(2, len(neighbors) - 2)
    head_w, head_b = 2.4, -0.75
    full_logit = head_w * full_mean + head_b
    full_sign = np.sign(full_logit) if abs(full_logit) > 1e-12 else 1.0
    full_class = "positive" if full_logit >= 0 else "negative"

    scenarios = []
    for mask_idx, kept_tuple in enumerate(combinations(range(len(neighbors)), keep_count), start=1):
        kept = np.array(kept_tuple, dtype=int)
        removed = np.array([idx for idx in range(len(neighbors)) if idx not in kept_tuple], dtype=int)
        pruned_mean = float(values[kept].mean())
        logit = head_w * pruned_mean + head_b
        is_flip = np.sign(logit) != full_sign
        scenarios.append(
            {
                "mask_id": f"M{mask_idx}",
                "removed": removed,
                "mean": pruned_mean,
                "logit": logit,
                "flip": is_flip,
                "class": "positive" if logit >= 0 else "negative",
            }
        )

    scenarios.sort(key=lambda item: item["mean"])
    ylabels = [
        f"{item['mask_id']}: drop " + ",".join(str(neighbors[idx]) for idx in item["removed"])
        for item in scenarios
    ]
    return {
        "neighbors": neighbors,
        "full_mean": full_mean,
        "full_logit": full_logit,
        "full_class": full_class,
        "decision_mean": -head_b / head_w,
        "scenarios": scenarios,
        "ylabels": ylabels,
    }


def make_normalization_shift_instability_left_figure(case=None) -> Path:
    data = build_normalization_mask_scenarios(case=case)
    full_mean = data["full_mean"]
    decision_mean = data["decision_mean"]
    scenarios = data["scenarios"]
    ylabels = data["ylabels"]

    y = np.arange(len(scenarios))
    means = np.array([item["mean"] for item in scenarios], dtype=float)
    colors = [SNAPSHOTORANGE if item["flip"] else PALEBLUE for item in scenarios]

    fig, ax_shift = plt.subplots(figsize=(8.8, 6.4))
    soften_axes(ax_shift)

    mean_span = max(0.06, means.max() - means.min())
    xmin = means.min() - 0.12 * mean_span
    xmax = means.max() + 0.12 * mean_span
    if decision_mean >= xmin and decision_mean <= xmax:
        ax_shift.axvspan(xmin, decision_mean, color=PALEORANGE, alpha=0.10, zorder=0)

    for yi, item, color in zip(y, scenarios, colors):
        ax_shift.plot([full_mean, item["mean"]], [yi, yi], color=PALEGRAY, lw=1.5, zorder=1)
        ax_shift.scatter(full_mean, yi, color=EVENTBLUE, s=36, zorder=2, edgecolor="white", linewidth=0.7)
        ax_shift.scatter(item["mean"], yi, color=color, s=58, zorder=3, edgecolor="white", linewidth=0.8)

    ax_shift.axvline(full_mean, color=EVENTBLUE, lw=1.8, ls=(0, (1.5, 2)))
    ax_shift.axvline(decision_mean, color=EDGEGRAY, lw=1.6, ls=(0, (2, 2)))
    ax_shift.set_yticks(y, ylabels)
    ax_shift.set_xlabel("aggregate after normalization (same target prediction)")
    ax_shift.set_ylabel("one pruning mask M_i")
    ax_shift.set_xlim(xmin, xmax)
    ax_shift.invert_yaxis()
    ax_shift.set_title("Mask Sensitivity (Part 1): feature shift per pruning mask", fontsize=13, pad=10)

    fig.text(
        0.02,
        0.015,
        (
            f"Same target instance in every row. Baseline full-graph aggregate={full_mean:.3f}. "
            f"Each mask point shows the new aggregate after dropping the listed neighbors."
        ),
        ha="left",
        va="bottom",
        fontsize=10,
        color=EDGEGRAY,
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=EVENTBLUE, markeredgecolor="white", markersize=7, label="full graph"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALEBLUE, markeredgecolor="white", markersize=7, label="masked graph (no flip)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SNAPSHOTORANGE, markeredgecolor="white", markersize=7, label="masked graph (flip)"),
        Line2D([0], [0], color=EDGEGRAY, lw=1.6, ls=(0, (2, 2)), label="decision boundary"),
    ]
    ax_shift.legend(handles=legend_handles, loc="lower right", fontsize=8.8)

    return save_pdf(fig, "02c_normalization_mask_shift.pdf")


def make_normalization_shift_instability_right_figure(case=None) -> Path:
    data = build_normalization_mask_scenarios(case=case)
    full_logit = data["full_logit"]
    full_class = data["full_class"]
    scenarios = data["scenarios"]
    ylabels = data["ylabels"]

    y = np.arange(len(scenarios))
    logits = np.array([item["logit"] for item in scenarios], dtype=float)
    colors = [SNAPSHOTORANGE if item["flip"] else PALEBLUE for item in scenarios]
    logit_span = max(0.12, logits.max() - logits.min())

    fig, ax_logit = plt.subplots(figsize=(8.8, 6.4))
    soften_axes(ax_logit)
    ax_logit.barh(y, logits, color=colors, edgecolor=colors, alpha=0.92, height=0.66)
    ax_logit.axvline(0.0, color=EDGEGRAY, lw=1.8, ls=(0, (1.5, 2)))
    ax_logit.axvline(full_logit, color=EVENTBLUE, lw=1.6)
    ax_logit.set_xlabel("new logit for same target")
    ax_logit.set_yticks(y, ylabels)
    ax_logit.set_xlim(logits.min() - 0.1 * logit_span, logits.max() + 0.1 * logit_span)
    ax_logit.invert_yaxis()
    ax_logit.set_title("Mask Sensitivity (Part 2): prediction outcome per mask", fontsize=13, pad=10)

    for yi, item in zip(y, scenarios):
        offset = 0.03 * max(0.4, logit_span)
        x_text = item["logit"] + (offset if item["logit"] >= 0 else -offset)
        ha = "left" if item["logit"] >= 0 else "right"
        label = "flip" if item["flip"] else "keep"
        ax_logit.text(x_text, yi, label, fontsize=9, ha=ha, va="center", color=EDGEGRAY)

    flip_count = int(sum(item["flip"] for item in scenarios))
    fig.text(
        0.02,
        0.015,
        (
            f"Baseline full-graph logit={full_logit:.3f} ({full_class}). "
            f"Prediction flips when masked logit crosses 0 ({flip_count}/{len(scenarios)} masks)."
        ),
        ha="left",
        va="bottom",
        fontsize=10,
        color=EDGEGRAY,
    )

    return save_pdf(fig, "02d_normalization_mask_prediction_flip.pdf")


def make_normalization_shift_figure():
    case = build_normalization_case()
    paths = [
        make_normalization_shift_mean_figure(case=case),
        make_normalization_shift_attention_figure(case=case),
        make_normalization_shift_instability_left_figure(case=case),
        make_normalization_shift_instability_right_figure(case=case),
    ]
    return paths


# -----------------------------------------------------------------------------
# 3) Subgraph topology collapse figure
# -----------------------------------------------------------------------------

def draw_temporal_graph(ax, positions, events, active_times=None, title="", overlay_original=False):
    ax.set_aspect("equal")
    ax.axis("off")

    if active_times is None:
        active_times = {t for _, _, t in events}
    else:
        active_times = set(active_times)

    def _edge_label_xy(x0, y0, x1, y1, t, *, overlay=False):
        dx, dy = x1 - x0, y1 - y0
        norm = math.hypot(dx, dy) + 1e-9
        ux, uy = dx / norm, dy / norm
        px, py = -uy, ux

        # Distribute labels along edges and alternate side offsets by timestamp.
        frac_cycle = (0.36, 0.50, 0.64, 0.44)
        frac = frac_cycle[(int(t) - 1) % len(frac_cycle)]
        side = -1.0 if (int(t) % 2 == 0) else 1.0
        perp_mag = 0.085 if not overlay else 0.065
        along_jitter = ((int(t) % 3) - 1) * (0.018 if not overlay else 0.012)

        bx = x0 + frac * dx
        by = y0 + frac * dy
        return (
            bx + px * perp_mag * side + ux * along_jitter,
            by + py * perp_mag * side + uy * along_jitter,
        )

    # Draw all edges lightly if requested.
    if overlay_original:
        for u, v, t in events:
            x0, y0 = positions[u]
            x1, y1 = positions[v]
            ax.plot([x0, x1], [y0, y1], color=PALEGRAY, lw=1.5, ls=(0, (2, 2)), zorder=1)
            lx, ly = _edge_label_xy(x0, y0, x1, y1, t, overlay=True)
            ax.text(
                lx,
                ly,
                f"t{t}",
                color=PALEGRAY,
                fontsize=9,
                ha="center",
                va="center",
                zorder=2,
                bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.70),
            )

    # Draw active edges.
    for u, v, t in events:
        if t not in active_times and not (overlay_original and t not in active_times):
            continue
        if t in active_times:
            x0, y0 = positions[u]
            x1, y1 = positions[v]
            line_color = SNAPSHOTORANGE if overlay_original else EVENTBLUE
            lw = 3.2 if overlay_original else 2.6
            ax.plot([x0, x1], [y0, y1], color=line_color, lw=lw, zorder=2, solid_capstyle="round")
            lx, ly = _edge_label_xy(x0, y0, x1, y1, t, overlay=overlay_original)
            ax.text(
                lx,
                ly,
                f"t{t}",
                color=line_color,
                fontsize=10,
                ha="center",
                va="center",
                weight="bold",
                zorder=3,
                bbox=dict(boxstyle="round,pad=0.11", facecolor="white", edgecolor="none", alpha=0.82),
            )

    # Draw nodes on top.
    for node, (x, y) in positions.items():
        edge_color = SNAPSHOTORANGE if overlay_original else EVENTBLUE
        circle = Circle((x, y), radius=0.11, facecolor="white", edgecolor=edge_color, lw=2.0, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, node, ha="center", va="center", color=edge_color, weight="bold", zorder=4)

def count_components(nodes, events_subset):
    parent = {node: node for node in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for u, v, _ in events_subset:
        union(u, v)
    return len({find(n) for n in nodes})


def build_topology_case():
    positions = {
        "A": (-1.05, 0.62),
        "B": (-0.25, 0.95),
        "C": (0.55, 0.72),
        "D": (1.02, -0.05),
        "E": (-0.86, -0.28),
        "F": (-0.06, -0.82),
        "G": (0.92, -0.72),
    }
    events = [
        ("A", "B", 1),
        ("B", "C", 2),
        ("C", "D", 3),
        ("B", "E", 4),
        ("E", "F", 5),
        ("B", "F", 6),
        ("F", "G", 7),
        ("C", "G", 8),
        ("D", "G", 9),
    ]
    kept_times = {1, 4, 7, 9}
    kept_events = [edge for edge in events if edge[2] in kept_times]
    nodes = list(positions)

    full_degree = Counter()
    kept_degree = Counter()
    for u, v, _ in events:
        full_degree[u] += 1
        full_degree[v] += 1
    for u, v, _ in kept_events:
        kept_degree[u] += 1
        kept_degree[v] += 1

    full_components = count_components(nodes, events)
    kept_components = count_components(nodes, kept_events)

    return {
        "positions": positions,
        "events": events,
        "kept_times": kept_times,
        "kept_events": kept_events,
        "nodes": nodes,
        "full_degree": full_degree,
        "kept_degree": kept_degree,
        "full_components": full_components,
        "kept_components": kept_components,
    }


def make_subgraph_topology_panel_figures(case=None):
    if case is None:
        case = build_topology_case()

    positions = case["positions"]
    events = case["events"]
    kept_times = case["kept_times"]
    nodes = case["nodes"]
    full_degree = case["full_degree"]
    kept_degree = case["kept_degree"]
    full_components = case["full_components"]
    kept_components = case["kept_components"]

    paths = []

    # 03a: full temporal neighborhood graph.
    fig_a, ax_a = plt.subplots(figsize=(6.4, 5.4))
    draw_temporal_graph(ax_a, positions, events, active_times=None, title="")
    ax_a.text(
        0.02,
        0.04,
        f"components = {full_components}",
        transform=ax_a.transAxes,
        fontsize=11,
        color=EVENTBLUE,
        bbox=dict(boxstyle="round,pad=0.25", facecolor=PALEBLUE, edgecolor="none"),
    )
    paths.append(save_pdf(fig_a, "03a_topology_full_neighborhood.pdf"))

    # 03b: explanation-induced/pruned subgraph.
    fig_b, ax_b = plt.subplots(figsize=(6.4, 5.4))
    draw_temporal_graph(ax_b, positions, events, active_times=kept_times, title="", overlay_original=True)
    ax_b.text(
        0.02,
        0.04,
        f"components = {kept_components}",
        transform=ax_b.transAxes,
        fontsize=11,
        color=SNAPSHOTORANGE,
        bbox=dict(boxstyle="round,pad=0.25", facecolor=PALEORANGE, edgecolor="none"),
    )
    ax_b.annotate(
        "node C becomes isolated",
        xy=positions["C"],
        xytext=(0.68, 0.88),
        textcoords="axes fraction",
        color=SNAPSHOTORANGE,
        fontsize=11,
        arrowprops=dict(arrowstyle="->", color=SNAPSHOTORANGE, lw=1.0),
    )
    paths.append(save_pdf(fig_b, "03b_topology_explanation_subgraph.pdf"))

    # 03c: degree profile bars.
    fig_c, ax_c = plt.subplots(figsize=(6.4, 4.3))
    soften_axes(ax_c)
    order = np.arange(len(nodes))
    labels = np.array(nodes)
    full_vals = np.array([full_degree[n] for n in nodes])
    kept_vals = np.array([kept_degree[n] for n in nodes])
    w = 0.34
    ax_c.bar(order - w / 2, full_vals, width=w, color=EVENTBLUE, edgecolor=EVENTBLUE, alpha=0.92, label="full")
    ax_c.bar(order + w / 2, kept_vals, width=w, color=SNAPSHOTORANGE, edgecolor=SNAPSHOTORANGE, alpha=0.92, label="kept subgraph")
    ax_c.set_xticks(order, labels)
    ax_c.set_ylabel("degree")
    ax_c.legend(loc="upper right")
    paths.append(save_pdf(fig_c, "03c_topology_degree_profile.pdf"))

    # 03d: temporal-density strip.
    fig_d, ax_d = plt.subplots(figsize=(6.4, 4.3))
    soften_axes(ax_d)
    full_times = np.arange(1, 10)
    kept_times_sorted = np.array(sorted(kept_times))
    ax_d.hlines(1.0, full_times.min(), full_times.max(), color=PALEBLUE, lw=6, alpha=0.35)
    ax_d.hlines(0.0, full_times.min(), full_times.max(), color=PALEORANGE, lw=6, alpha=0.25)
    ax_d.scatter(full_times, np.ones_like(full_times), s=72, color=EVENTBLUE, zorder=3)
    ax_d.scatter(kept_times_sorted, np.zeros_like(kept_times_sorted), s=72, color=SNAPSHOTORANGE, zorder=3)
    for left, right in zip(kept_times_sorted[:-1], kept_times_sorted[1:]):
        if right - left > 1:
            ax_d.axvspan(left, right, ymin=0.0, ymax=0.42, color=PALEORANGE, alpha=0.23)
            ax_d.text((left + right) / 2, -0.25, f"gap {right-left}", ha="center", color=SNAPSHOTORANGE, fontsize=10)
    ax_d.set_yticks([1.0, 0.0], ["full stream", "kept events"])
    ax_d.set_xlabel("event index")
    ax_d.set_xlim(0.5, 9.5)
    ax_d.set_ylim(-0.4, 1.4)
    paths.append(save_pdf(fig_d, "03d_topology_temporal_density.pdf"))

    return paths



def make_subgraph_topology_figure() -> Path:
    case = build_topology_case()
    positions = case["positions"]
    events = case["events"]
    kept_times = case["kept_times"]
    nodes = case["nodes"]
    full_degree = case["full_degree"]
    kept_degree = case["kept_degree"]
    full_components = case["full_components"]
    kept_components = case["kept_components"]

    fig = plt.figure(figsize=(13.6, 7.6))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 0.75], wspace=0.28, hspace=0.32)

    ax_full = fig.add_subplot(gs[0, 0])
    annotate_panel(ax_full, "A")
    draw_temporal_graph(ax_full, positions, events, active_times=None, title="Full temporal neighborhood")
    ax_full.text(
        0.02,
        0.04,
        f"components = {full_components}",
        transform=ax_full.transAxes,
        fontsize=11,
        color=EVENTBLUE,
        bbox=dict(boxstyle="round,pad=0.25", facecolor=PALEBLUE, edgecolor="none"),
    )

    ax_sub = fig.add_subplot(gs[0, 1])
    annotate_panel(ax_sub, "B")
    draw_temporal_graph(
        ax_sub,
        positions,
        events,
        active_times=kept_times,
        title="Explanation-induced subgraph",
        overlay_original=True,
    )
    ax_sub.text(
        0.02,
        0.04,
        f"components = {kept_components}",
        transform=ax_sub.transAxes,
        fontsize=11,
        color=SNAPSHOTORANGE,
        bbox=dict(boxstyle="round,pad=0.25", facecolor=PALEORANGE, edgecolor="none"),
    )
    ax_sub.annotate(
        "node C becomes isolated",
        xy=positions["C"],
        xytext=(0.68, 0.88),
        textcoords="axes fraction",
        color=SNAPSHOTORANGE,
        fontsize=11,
        arrowprops=dict(arrowstyle="->", color=SNAPSHOTORANGE, lw=1.0),
    )

    ax_deg = fig.add_subplot(gs[1, 0])
    annotate_panel(ax_deg, "C")
    soften_axes(ax_deg)
    order = np.arange(len(nodes))
    labels = np.array(nodes)
    full_vals = np.array([full_degree[n] for n in nodes])
    kept_vals = np.array([kept_degree[n] for n in nodes])
    w = 0.34
    ax_deg.bar(order - w / 2, full_vals, width=w, color=EVENTBLUE, edgecolor=EVENTBLUE, alpha=0.92, label="full")
    ax_deg.bar(order + w / 2, kept_vals, width=w, color=SNAPSHOTORANGE, edgecolor=SNAPSHOTORANGE, alpha=0.92, label="kept subgraph")
    ax_deg.set_xticks(order, labels)
    ax_deg.set_ylabel("degree")
    ax_deg.legend(loc="upper right")

    ax_time = fig.add_subplot(gs[1, 1])
    annotate_panel(ax_time, "D")
    soften_axes(ax_time)
    full_times = np.arange(1, 10)
    kept_times_sorted = np.array(sorted(kept_times))
    ax_time.hlines(1.0, full_times.min(), full_times.max(), color=PALEBLUE, lw=6, alpha=0.35)
    ax_time.hlines(0.0, full_times.min(), full_times.max(), color=PALEORANGE, lw=6, alpha=0.25)
    ax_time.scatter(full_times, np.ones_like(full_times), s=72, color=EVENTBLUE, zorder=3)
    ax_time.scatter(kept_times_sorted, np.zeros_like(kept_times_sorted), s=72, color=SNAPSHOTORANGE, zorder=3)
    for left, right in zip(kept_times_sorted[:-1], kept_times_sorted[1:]):
        if right - left > 1:
            ax_time.axvspan(left, right, ymin=0.0, ymax=0.42, color=PALEORANGE, alpha=0.23)
            ax_time.text((left + right) / 2, -0.25, f"gap {right-left}", ha="center", color=SNAPSHOTORANGE, fontsize=10)
    ax_time.set_yticks([1.0, 0.0], ["full stream", "kept events"])
    ax_time.set_xlabel("event index")
    ax_time.set_xlim(0.5, 9.5)
    ax_time.set_ylim(-0.4, 1.4)
    return save_pdf(fig, "03_subgraph_topology_collapse.pdf")


# -----------------------------------------------------------------------------
# 4) Summary triptych
# -----------------------------------------------------------------------------

def make_summary_triptych() -> Path:
    case = build_gru_case()
    full_states = case["full"]["states"]
    skipped_states = case["skipped"]["states"]
    drift = np.linalg.norm(full_states - skipped_states, axis=1)

    fig = plt.figure(figsize=(14.0, 4.7))
    gs = GridSpec(1, 3, figure=fig, wspace=0.22)

    # Panel 1: normalization shift mini chart.
    ax1 = fig.add_subplot(gs[0, 0])
    soften_axes(ax1)
    annotate_panel(ax1, "1")
    vals = np.array([0.84, 0.62, 0.46, -0.04, -0.22])
    keep = np.array([0, 1, 2])
    x = np.arange(len(vals))
    colors = [EVENTBLUE if i in keep else PALEGRAY for i in range(len(vals))]
    bars = ax1.bar(x, vals, color=colors, edgecolor=[EVENTBLUE if i in keep else EDGEGRAY for i in range(len(vals))], linewidth=1.2)
    for idx in [3, 4]:
        bars[idx].set_hatch("//")
    ax1.axhline(vals.mean(), color=EVENTBLUE, lw=2.0, ls=(0, (1.5, 2)))
    ax1.axhline(vals[keep].mean(), color=SNAPSHOTORANGE, lw=2.2)
    ax1.set_xticks(x, [f"n{i+1}" for i in x])
    ax1.set_ylim(-0.35, 1.02)
    ax1.set_ylabel("feature")

    # Panel 2: GRU drift mini chart.
    ax2 = fig.add_subplot(gs[0, 1])
    soften_axes(ax2)
    annotate_panel(ax2, "2")
    ax2.plot(full_states[:, 0], full_states[:, 1], color=EVENTBLUE, lw=2.6)
    ax2.plot(skipped_states[:, 0], skipped_states[:, 1], color=SNAPSHOTORANGE, lw=2.3, ls=(0, (6, 2)))
    ax2.scatter(full_states[0, 0], full_states[0, 1], s=40, color=EDGEGRAY)
    ax2.scatter(full_states[-1, 0], full_states[-1, 1], s=50, color=EVENTBLUE)
    ax2.scatter(skipped_states[-1, 0], skipped_states[-1, 1], s=50, color=SNAPSHOTORANGE)
    ax2.set_xlabel("hidden dim 1")
    ax2.set_ylabel("hidden dim 2")

    # Panel 3: topology and temporal-gap mini summary.
    ax3 = fig.add_subplot(gs[0, 2])
    soften_axes(ax3)
    annotate_panel(ax3, "3")
    full_times = np.arange(1, 10)
    kept_times = np.array([1, 4, 7, 9])
    ax3.hlines(1.0, full_times.min(), full_times.max(), color=PALEBLUE, lw=5.5, alpha=0.35)
    ax3.hlines(0.0, full_times.min(), full_times.max(), color=PALEORANGE, lw=5.5, alpha=0.28)
    ax3.scatter(full_times, np.ones_like(full_times), s=58, color=EVENTBLUE, zorder=3)
    ax3.scatter(kept_times, np.zeros_like(kept_times), s=58, color=SNAPSHOTORANGE, zorder=3)
    for left, right in zip(kept_times[:-1], kept_times[1:]):
        if right - left > 1:
            ax3.axvspan(left, right, ymin=0.0, ymax=0.38, color=PALEORANGE, alpha=0.22)
    ax3.set_yticks([1.0, 0.0], ["full", "kept"])
    ax3.set_xlim(0.5, 9.5)
    ax3.set_ylim(-0.35, 1.35)
    ax3.set_xlabel("event index")
    ax3.set_ylabel("time strip")
    ax3.text(0.96, 0.95, "wider gaps + fewer links", transform=ax3.transAxes, ha="right", va="top", color=EDGEGRAY, fontsize=10)
    return save_pdf(fig, "04_three_instabilities_overview.pdf")


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def build_all_figures():
    paths = []
    paths.extend(make_gru_memory_drift_figure())
    paths.extend(make_normalization_shift_figure())
    paths.append(make_subgraph_topology_figure())
    paths.extend(make_subgraph_topology_panel_figures())
    paths.append(make_summary_triptych())
    print("Saved figures:")
    for path in paths:
        print(f" - {path}")
    return paths


if __name__ == "__main__":
    build_all_figures()
