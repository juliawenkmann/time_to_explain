"""Publication-quality local CBM explanation figure."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from cbm.core import _weights_for_class, node_embedding_to_plot2d
from viz.theme import EDGE_GRAY, EVENT_BLUE, SNAPSHOT_ORANGE


SIGNED_ORANGE_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "cbm_signed_orange_blue",
    ["#B2182B", SNAPSHOT_ORANGE, "#F6F6F6", "#6BAED6", EVENT_BLUE],
)


def _sanitize_basename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


def _quantile_scale(values: np.ndarray, q: float = 0.95, eps: float = 1e-8) -> float:
    vals = np.asarray(values, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    s = float(np.quantile(np.abs(vals), float(q)))
    if not np.isfinite(s) or s < eps:
        s = 1.0
    return s


def _standardize_embedding(emb2: np.ndarray) -> np.ndarray:
    x = np.asarray(emb2, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 2:
        raise RuntimeError(f"Expected 2D embedding with shape [N,2], got {x.shape}")
    mu = np.mean(x, axis=0, keepdims=True)
    sd = np.std(x, axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return ((x - mu) / sd).astype(np.float32)


def _short_concept_name(name: str, max_len: int = 40) -> str:
    s = str(name)
    if "__" in s:
        pref, rest = s.split("__", 1)
        if pref in {"g", "g2"}:
            s = rest
    if len(s) <= int(max_len):
        return s
    return s[: int(max_len) - 3] + "..."


def _active_concept_view(
    *,
    concept_names: List[str],
    use_both_layout_concepts: bool,
    spring_layout_on_g2: bool,
    edge_grp_only: bool,
) -> Tuple[np.ndarray, List[str], str]:
    active_prefix = "g2__" if bool(spring_layout_on_g2) else "g__"
    idx: List[int] = []
    names: List[str] = []
    for j, name in enumerate(concept_names):
        name_str = str(name)
        if bool(use_both_layout_concepts):
            if not name_str.startswith(active_prefix):
                continue
            body = name_str.split("__", 1)[1]
        else:
            body = name_str
        if bool(edge_grp_only) and re.search(r"_grp(\d+)$", body) is None:
            continue
        idx.append(int(j))
        names.append(body)
    return np.asarray(idx, dtype=int), names, active_prefix


def _build_local_masks(
    *,
    n_nodes: int,
    target: int,
    g_src: np.ndarray,
    g_dst: np.ndarray,
    x_nodes: np.ndarray,
    m_nodes: np.ndarray,
    v_nodes: np.ndarray,
    local_scope: str,
    local_hops: int,
    local_strict: bool,
    ho_incident_role: str,
    ho_incident_relax: bool,
    ho_incident_relax_hops: int,
    ho_min_local_candidates: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    adj = [[] for _ in range(int(n_nodes))]
    for u, v in zip(g_src.tolist(), g_dst.tolist()):
        uu, vv = int(u), int(v)
        adj[uu].append(vv)
        adj[vv].append(uu)

    def _build_khop_mask(hops: int) -> np.ndarray:
        m = np.zeros((int(n_nodes),), dtype=bool)
        m[int(target)] = True
        frontier = {int(target)}
        for _ in range(max(int(hops), 0)):
            nxt = set()
            for u in frontier:
                for v in adj[u]:
                    if not m[v]:
                        m[v] = True
                        nxt.add(v)
            frontier = nxt
            if not frontier:
                break
        return m

    local_scope = str(local_scope).strip().lower()
    ho_incident_role = str(ho_incident_role).strip().lower()

    khop_mask = _build_khop_mask(int(local_hops))
    ho_relax_mask = _build_khop_mask(int(ho_incident_relax_hops))
    incident_mask = np.zeros((int(n_nodes),), dtype=bool)
    incident_mask[int(target)] = True
    for v in adj[int(target)]:
        incident_mask[int(v)] = True

    if local_scope == "incident":
        fo_local_mask = (g_src == int(target)) | (g_dst == int(target))
        if ho_incident_role == "any":
            ho_local_mask = (x_nodes == int(target)) | (m_nodes == int(target)) | (v_nodes == int(target))
        elif ho_incident_role == "middle":
            ho_local_mask = m_nodes == int(target)
        elif ho_incident_role == "endpoints":
            ho_local_mask = (x_nodes == int(target)) | (v_nodes == int(target))
        else:
            raise ValueError(f"Unknown HO_INCIDENT_ROLE={ho_incident_role}")

        scope_txt = f"incident({ho_incident_role})"
        vis_local_mask = incident_mask
        if bool(ho_incident_relax) and int(ho_local_mask.sum()) < int(ho_min_local_candidates):
            ho_local_mask = ho_relax_mask[x_nodes] | ho_relax_mask[m_nodes] | ho_relax_mask[v_nodes]
            vis_local_mask = incident_mask | ho_relax_mask
            scope_txt = f"{scope_txt}->khop({int(ho_incident_relax_hops)})"
        return fo_local_mask, ho_local_mask, vis_local_mask, scope_txt

    if local_scope == "khop":
        if bool(local_strict):
            fo_local_mask = khop_mask[g_src] & khop_mask[g_dst]
            ho_local_mask = khop_mask[x_nodes] & khop_mask[m_nodes] & khop_mask[v_nodes]
        else:
            fo_local_mask = khop_mask[g_src] | khop_mask[g_dst]
            ho_local_mask = khop_mask[x_nodes] | khop_mask[m_nodes] | khop_mask[v_nodes]
        return fo_local_mask, ho_local_mask, khop_mask, f"khop({int(local_hops)})"

    if local_scope == "global":
        fo_local_mask = np.ones_like(g_src, dtype=bool)
        ho_local_mask = np.ones_like(x_nodes, dtype=bool)
        vis_local_mask = np.ones((int(n_nodes),), dtype=bool)
        return fo_local_mask, ho_local_mask, vis_local_mask, "global"

    raise ValueError(f"Unknown LOCAL_SCOPE={local_scope}")


def _select_local(scores: np.ndarray, local_mask: np.ndarray, topk: int, abs_thresh: float) -> np.ndarray:
    if scores.size == 0:
        return np.zeros((0,), dtype=int)
    thr = float(max(abs_thresh, 0.0))
    idx = np.where(local_mask & (np.abs(scores) >= thr))[0]
    if idx.size == 0:
        return np.zeros((0,), dtype=int)
    if int(topk) <= 0 or int(topk) >= idx.size:
        return idx.astype(int)
    keep_local = np.argsort(-np.abs(scores[idx]))[: int(topk)]
    return idx[keep_local].astype(int)


def run_local_edge_importance_plot(ns: Dict[str, Any]) -> None:
    """Render a publication-quality local explanation figure.

    Interface contract: accepts the same notebook namespace used by
    ``run_local_edge_importance(globals())``.
    """
    required = [
        "clf",
        "C_full",
        "concept_names",
        "group_of",
        "graphs",
        "node_emb",
        "y",
    ]
    missing = [k for k in required if k not in ns]
    if missing:
        raise RuntimeError(f"Missing required namespace keys for local plot: {missing}")

    clf = ns["clf"]
    C_full = np.asarray(ns["C_full"], dtype=np.float32)
    concept_names = [str(n) for n in ns["concept_names"]]
    group_of = np.asarray(ns["group_of"], dtype=int)
    graphs = ns["graphs"]
    y = np.asarray(ns["y"], dtype=int)

    if "node_internal" not in ns:
        ns["node_internal"] = int(np.asarray(ns["test_nodes_internal"], dtype=int)[0])
    target = int(ns["node_internal"])

    pred_label = int(clf.predict(C_full[target][None, :])[0])
    w_full, _ = _weights_for_class(clf, pred_label)
    w_full = np.asarray(w_full, dtype=np.float32)

    use_both_layout = bool(ns.get("USE_BOTH_LAYOUT_CONCEPTS", False))
    spring_layout_on_g2 = bool(ns.get("SPRING_LAYOUT_ON_G2", False))
    edge_grp_only_local = bool(ns.get("EDGE_GRP_ONLY_LOCAL", False))

    active_idx, active_names, active_prefix = _active_concept_view(
        concept_names=concept_names,
        use_both_layout_concepts=use_both_layout,
        spring_layout_on_g2=spring_layout_on_g2,
        edge_grp_only=edge_grp_only_local,
    )
    if active_idx.size == 0:
        raise RuntimeError("No active concepts selected for local edge map.")

    C_active = np.asarray(C_full[:, active_idx], dtype=np.float32)
    w_active = np.asarray(w_full[active_idx], dtype=np.float32)

    score_mode = str(
        ns.get(
            "EDGE_SCORE_MODE_LOCAL",
            ns.get("EDGE_SCORE_MODE", ns.get("SCORE_MODE", "target_overlap")),
        )
    ).strip().lower()
    if score_mode == "class_alignment":
        w_eff = w_active
        score_formula = "C[node] * w_class(pred)"
    elif score_mode == "target_overlap":
        w_eff = np.asarray(C_active[target], dtype=np.float32) * w_active
        score_formula = "C[node] * (C[target] * w_class(pred))"
    else:
        raise ValueError(f"Unknown EDGE_SCORE_MODE_LOCAL={score_mode}")

    # Normalize once using node-level score scale for cross-node comparability.
    node_raw_scores = np.asarray(C_active @ w_eff, dtype=np.float32)
    node_scale = _quantile_scale(node_raw_scores, q=0.95)
    node_scores = np.clip(node_raw_scores / node_scale, -1.0, 1.0)

    name_to_local = {str(n): i for i, n in enumerate(active_names)}

    def _idx(family: str, grp: int):
        return name_to_local.get(f"{family}_grp{int(grp)}", None)

    # First-order scores.
    g_src = np.asarray(graphs.g_edge_index[0], dtype=int).reshape(-1)
    g_dst = np.asarray(graphs.g_edge_index[1], dtype=int).reshape(-1)
    E1 = int(g_src.shape[0])
    fo_scores_raw = np.zeros((E1,), dtype=np.float32)
    for e, (u, v) in enumerate(zip(g_src.tolist(), g_dst.tolist())):
        gu = int(group_of[u])
        gv = int(group_of[v])
        terms = []
        j = _idx("direct_out", gv)
        if j is not None:
            terms.append(float(C_active[u, j] * w_eff[j]))
        j = _idx("direct_in", gu)
        if j is not None:
            terms.append(float(C_active[v, j] * w_eff[j]))
        fo_scores_raw[e] = float(np.mean(terms)) if len(terms) > 0 else 0.0

    # Higher-order transition scores.
    tok = np.asarray(graphs.g2_node_ids, dtype=int)
    g2s = np.asarray(graphs.g2_edge_index[0], dtype=int).reshape(-1)
    g2t = np.asarray(graphs.g2_edge_index[1], dtype=int).reshape(-1)
    E2 = int(g2s.shape[0])
    ho_scores_raw = np.zeros((E2,), dtype=np.float32)
    for e, (a, b) in enumerate(zip(g2s.tolist(), g2t.tolist())):
        x, _ = tok[a]
        m, v = tok[b]
        gx, gm, gv = int(group_of[x]), int(group_of[m]), int(group_of[v])
        terms = []
        j = _idx("out2_end", gv)
        if j is not None:
            terms.append(float(C_active[x, j] * w_eff[j]))
        j = _idx("out2_mid", gm)
        if j is not None:
            terms.append(float(C_active[x, j] * w_eff[j]))
        j = _idx("in2_start", gx)
        if j is not None:
            terms.append(float(C_active[v, j] * w_eff[j]))
        j = _idx("in2_mid", gm)
        if j is not None:
            terms.append(float(C_active[v, j] * w_eff[j]))
        ho_scores_raw[e] = float(np.mean(terms)) if len(terms) > 0 else 0.0

    fo_scores = np.clip(fo_scores_raw / node_scale, -1.0, 1.0)
    ho_scores = np.clip(ho_scores_raw / node_scale, -1.0, 1.0)

    # Locality and filtering.
    if E2 > 0:
        x_nodes = np.asarray(tok[g2s, 0], dtype=int)
        m_nodes = np.asarray(tok[g2t, 0], dtype=int)
        v_nodes = np.asarray(tok[g2t, 1], dtype=int)
    else:
        x_nodes = np.zeros((0,), dtype=int)
        m_nodes = np.zeros((0,), dtype=int)
        v_nodes = np.zeros((0,), dtype=int)

    fo_local_mask, ho_local_mask, vis_local_mask, ho_scope_txt = _build_local_masks(
        n_nodes=int(graphs.n_nodes),
        target=int(target),
        g_src=g_src,
        g_dst=g_dst,
        x_nodes=x_nodes,
        m_nodes=m_nodes,
        v_nodes=v_nodes,
        local_scope=str(ns.get("LOCAL_SCOPE", "incident")),
        local_hops=int(ns.get("LOCAL_HOPS", 1)),
        local_strict=bool(ns.get("LOCAL_STRICT", True)),
        ho_incident_role=str(ns.get("HO_INCIDENT_ROLE", "any")),
        ho_incident_relax=bool(ns.get("HO_INCIDENT_RELAX", False)),
        ho_incident_relax_hops=int(ns.get("HO_INCIDENT_RELAX_HOPS", 2)),
        ho_min_local_candidates=int(ns.get("HO_MIN_LOCAL_CANDIDATES", 20)),
    )

    default_thresh = float(ns.get("LOCAL_EDGE_ABS_THRESH", 0.06))
    fo_abs_thresh = float(ns.get("FO_ABS_THRESH_LOCAL", default_thresh))
    ho_abs_thresh = float(ns.get("HO_ABS_THRESH_LOCAL", default_thresh))
    fo_topk = int(ns.get("FO_TOPK_LOCAL", 20))
    ho_topk = int(ns.get("HO_TOPK_LOCAL", 14))

    fo_keep = _select_local(fo_scores, fo_local_mask, fo_topk, fo_abs_thresh)
    ho_keep = _select_local(ho_scores, ho_local_mask, ho_topk, ho_abs_thresh)

    # Standardize embedding once and use fixed axis limits for consistent layout.
    emb2 = node_embedding_to_plot2d(np.asarray(ns["node_emb"], dtype=np.float32), seed=int(ns.get("seed", 0)))
    emb_plot = _standardize_embedding(emb2)

    fo_segments = np.stack([emb_plot[g_src], emb_plot[g_dst]], axis=1) if E1 > 0 else np.zeros((0, 2, 2), dtype=np.float32)
    fo_draw_segments = fo_segments[fo_keep] if fo_keep.size else fo_segments[:0]
    fo_draw_scores = fo_scores[fo_keep]

    ho_render_mode = str(ns.get("HO_RENDER_MODE", "path")).strip().lower()
    if ho_render_mode == "token":
        tok_xy = 0.5 * (emb_plot[tok[:, 0]] + emb_plot[tok[:, 1]]) if tok.shape[0] > 0 else np.zeros((0, 2), dtype=np.float32)
        ho_segments_all = (
            np.stack([tok_xy[g2s], tok_xy[g2t]], axis=1)
            if E2 > 0
            else np.zeros((0, 2, 2), dtype=np.float32)
        )
        ho_draw_segments = ho_segments_all[ho_keep] if ho_keep.size else ho_segments_all[:0]
        ho_line_scores = ho_scores[ho_keep]
    else:
        segs = []
        seg_scores = []
        for e in ho_keep.tolist():
            a = int(g2s[e])
            b = int(g2t[e])
            x, _ = tok[a]
            m, v = tok[b]
            s = float(ho_scores[e])
            segs.append(np.asarray([emb_plot[int(x)], emb_plot[int(m)]], dtype=np.float32))
            seg_scores.append(s)
            segs.append(np.asarray([emb_plot[int(m)], emb_plot[int(v)]], dtype=np.float32))
            seg_scores.append(s)
        if len(segs) > 0:
            ho_draw_segments = np.stack(segs, axis=0).astype(np.float32)
            ho_line_scores = np.asarray(seg_scores, dtype=np.float32)
        else:
            ho_draw_segments = np.zeros((0, 2, 2), dtype=np.float32)
            ho_line_scores = np.zeros((0,), dtype=np.float32)

    # Concept contribution bars (normalized).
    contrib_raw_target = np.asarray(C_active[target] * w_active, dtype=np.float32)
    contrib_scale = _quantile_scale(C_active * w_active[None, :], q=0.95)
    contrib_norm_target = np.clip(contrib_raw_target / contrib_scale, -1.0, 1.0)

    topk_concepts = int(ns.get("LOCAL_CONTRIB_TOPK", 12))
    order_abs = sorted(
        range(contrib_norm_target.shape[0]),
        key=lambda j: (-abs(float(contrib_norm_target[j])), str(active_names[j])),
    )
    sel = np.asarray(order_abs[: max(1, topk_concepts)], dtype=int)
    sel = sel[np.argsort(contrib_norm_target[sel])] if sel.size else sel

    with plt.rc_context(
        {
            "font.size": 12.5,
            "axes.titlesize": 15,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 11,
            "ytick.labelsize": 10,
            "legend.fontsize": 10.5,
            "axes.facecolor": "#fbfbfb",
            "figure.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        fig, (ax_graph, ax_bar) = plt.subplots(
            1,
            2,
            figsize=(15.5, 7.2),
            gridspec_kw={"width_ratios": [1.8, 1.0]},
            constrained_layout=True,
        )

        norm_signed = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

        # Graph panel.
        ax_graph.scatter(
            emb_plot[:, 0],
            emb_plot[:, 1],
            s=16,
            c=EDGE_GRAY,
            alpha=0.18,
            edgecolor="none",
            zorder=1,
        )

        local_nodes = np.where(vis_local_mask)[0]
        if local_nodes.size > 0:
            node_size = 42 + 90 * np.abs(node_scores[local_nodes])
            ax_graph.scatter(
                emb_plot[local_nodes, 0],
                emb_plot[local_nodes, 1],
                c=node_scores[local_nodes],
                cmap=SIGNED_ORANGE_BLUE_CMAP,
                norm=norm_signed,
                s=node_size,
                alpha=0.92,
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )

        if fo_draw_segments.shape[0] > 0:
            lw_fo = 1.8 + 3.8 * np.abs(fo_draw_scores)
            lc_fo = LineCollection(
                fo_draw_segments,
                cmap=SIGNED_ORANGE_BLUE_CMAP,
                norm=norm_signed,
                linewidths=lw_fo,
                alpha=0.95,
                zorder=4,
            )
            lc_fo.set_array(fo_draw_scores)
            ax_graph.add_collection(lc_fo)

        if ho_draw_segments.shape[0] > 0:
            lw_ho = 1.4 + 3.4 * np.abs(ho_line_scores)
            lc_ho = LineCollection(
                ho_draw_segments,
                cmap=SIGNED_ORANGE_BLUE_CMAP,
                norm=norm_signed,
                linewidths=lw_ho,
                linestyles="dashed",
                alpha=0.92,
                zorder=5,
            )
            lc_ho.set_array(ho_line_scores)
            ax_graph.add_collection(lc_ho)

        ax_graph.scatter(
            emb_plot[target, 0],
            emb_plot[target, 1],
            marker="*",
            s=620,
            c=SNAPSHOT_ORANGE,
            edgecolor="black",
            linewidth=1.8,
            zorder=9,
        )

        axis_lim = float(ns.get("LOCAL_EXPLAIN_AXIS_LIM", 3.2))
        axis_lim = max(axis_lim, 1.0)
        ax_graph.set_xlim(-axis_lim, axis_lim)
        ax_graph.set_ylim(-axis_lim, axis_lim)
        ax_graph.set_aspect("equal")
        ax_graph.set_xlabel("Standardized layout dim 1")
        ax_graph.set_ylabel("Standardized layout dim 2")
        ax_graph.set_title(
            f"Local edge evidence (FO solid / HO dashed)\n"
            f"scope={ho_scope_txt}, FO={fo_keep.size}, HO={ho_keep.size}"
        )

        legend_handles = [
            Line2D([0], [0], marker="*", linestyle="None", markerfacecolor=SNAPSHOT_ORANGE, markeredgecolor="black", markersize=14, label=f"explained node {target}"),
            Line2D([0], [0], color=EDGE_GRAY, linewidth=2.4, label="first-order edges"),
            Line2D([0], [0], color=EDGE_GRAY, linewidth=2.0, linestyle="--", label="higher-order transitions"),
        ]
        ax_graph.legend(handles=legend_handles, loc="upper right", frameon=False)

        sm = ScalarMappable(norm=norm_signed, cmap=SIGNED_ORANGE_BLUE_CMAP)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_graph, fraction=0.046, pad=0.02)
        cbar.set_label("Normalized signed contribution")

        # Concept bar panel.
        if sel.size > 0:
            vals = contrib_norm_target[sel]
            labels = [_short_concept_name(active_names[int(j)]) for j in sel.tolist()]
            colors = [SIGNED_ORANGE_BLUE_CMAP(norm_signed(float(v))) for v in vals.tolist()]
            ypos = np.arange(sel.size)
            ax_bar.barh(
                ypos,
                vals,
                color=colors,
                edgecolor=EDGE_GRAY,
                linewidth=0.95,
                alpha=0.95,
            )
            ax_bar.axvline(0.0, color="black", linewidth=1.2)
            ax_bar.set_yticks(ypos)
            ax_bar.set_yticklabels(labels)
            ax_bar.set_xlim(-1.05, 1.05)
            ax_bar.set_xlabel("Normalized concept contribution")
            ax_bar.set_title(f"Top-{sel.size} concept contributions")
            ax_bar.grid(axis="x", linestyle=":", alpha=0.35)
            for yi, vv in zip(ypos.tolist(), vals.tolist()):
                ax_bar.text(
                    float(vv) + (0.02 if vv >= 0 else -0.02),
                    float(yi),
                    f"{vv:+.2f}",
                    va="center",
                    ha="left" if vv >= 0 else "right",
                    fontsize=9.5,
                )
        else:
            ax_bar.text(
                0.5,
                0.5,
                "No concepts selected",
                ha="center",
                va="center",
                transform=ax_bar.transAxes,
            )
            ax_bar.set_title("Top concept contributions")

        dataset_key = str(ns.get("dataset_key", "dataset"))
        true_label = int(y[target]) if target < y.shape[0] else -1
        fig.suptitle(
            f"CBM local explanation | dataset={dataset_key} | node={target} | true={true_label} | pred={pred_label}",
            fontsize=15,
        )

        saved_paths: List[str] = []
        save_fig = bool(ns.get("SAVE_LOCAL_EDGE_IMPORTANCE", True))
        save_pdf = bool(ns.get("SAVE_LOCAL_EDGE_IMPORTANCE_PDF", True))
        save_svg = bool(ns.get("SAVE_LOCAL_EDGE_IMPORTANCE_SVG", False))
        if save_fig and (save_pdf or save_svg):
            root = ns.get("ROOT", Path.cwd())
            if not isinstance(root, Path):
                root = Path(root)
            default_dir = root / "plots"
            out_dir = Path(ns.get("SAVE_LOCAL_EDGE_IMPORTANCE_DIR", default_dir))
            out_dir.mkdir(parents=True, exist_ok=True)
            dataset_tag = _sanitize_basename(dataset_key) or "dataset"
            base_default = f"{dataset_tag}_local_explanation_node{target}_pred{pred_label}"
            base = _sanitize_basename(ns.get("SAVE_LOCAL_EDGE_IMPORTANCE_BASENAME", base_default))
            if len(base) == 0:
                base = base_default
            if not base.startswith(f"{dataset_tag}_"):
                base = f"{dataset_tag}_{base}"
            if save_pdf:
                out_pdf = out_dir / f"{base}.pdf"
                fig.savefig(out_pdf, bbox_inches="tight")
                saved_paths.append(str(out_pdf.resolve()))
            if save_svg:
                out_svg = out_dir / f"{base}.svg"
                fig.savefig(out_svg, bbox_inches="tight")
                saved_paths.append(str(out_svg.resolve()))

        plt.show()

    ns["local_edge_importance_exports"] = saved_paths
    ns["local_edge_fo_scores"] = fo_scores
    ns["local_edge_ho_scores"] = ho_scores
    ns["local_edge_active_names"] = active_names
    ns["local_edge_selected_concepts"] = [active_names[int(j)] for j in sel.tolist()] if sel.size > 0 else []
    ns["local_edge_score_formula"] = score_formula
    ns["local_edge_node_scale"] = float(node_scale)
    ns["local_edge_contrib_scale"] = float(contrib_scale)

    print(f"Edge score mode: {score_mode}")
    print(f"Score formula: {score_formula}")
    print(
        "Active concept view:",
        (f"both-layout/{active_prefix[:-2]}" if use_both_layout else "single-layout"),
    )
    print(
        f"Edge filters (normalized): FO |score| >= {fo_abs_thresh:.3f}, "
        f"HO |score| >= {ho_abs_thresh:.3f}",
    )
    print(f"FO edges shown: {int(fo_keep.size)}/{E1} | HO transitions shown: {int(ho_keep.size)}/{E2}")
    if len(saved_paths) > 0:
        for p in saved_paths:
            print(f"[saved figure] {p}")
