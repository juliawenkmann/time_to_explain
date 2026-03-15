from __future__ import annotations

from ._fidelity_shared import *

class PredictionProfileMetric(BaseMetric):
    """Prediction profile as sparsity changes (keep/drop mode)."""

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        candidate = _candidate_eidx(context, result)
        n_candidates = len(candidate)
        imp = _importance(result, n_candidates)
        order = _ranked_order(self, context, result, candidate, imp)

        mode = str(self.config.get("mode", "keep"))
        levels = _resolve_levels(self.config, default=[0.0, 0.1, 0.2, 0.3])
        k_max = _resolve_k_max(self.config, n_candidates)
        threshold = float(self.config.get("label_threshold", 0.5))

        pred_full = self.model.predict_proba(context.subgraph, context.target)
        z_full = _score(pred_full)
        y_full = _label(pred_full, threshold=threshold)

        values: dict[str, float] = {
            "prediction_full": float(z_full),
            "label_full": float(y_full),
        }

        for level in levels:
            k = _k_from_level(
                float(level),
                n_candidates=n_candidates,
                k_max=k_max,
                ensure_min_one=False,
            )
            selected = [candidate[i] for i in order[:k]]
            pred_masked = self.model.predict_proba_with_mask(
                context.subgraph,
                context.target,
                edge_mask=_edge_mask(candidate, selected, mode=mode),
            )
            z_masked = _score(pred_masked)
            y_masked = _label(pred_masked, threshold=threshold)

            key = f"@s={float(level):g}"
            values[f"prediction_{mode}.{key}"] = float(z_masked)
            values[f"label_{mode}.{key}"] = float(y_masked)
            values[f"delta_{mode}.{key}"] = float(z_masked - z_full)
            values[f"delta_abs_{mode}.{key}"] = float(abs(z_masked - z_full))
            values[f"match_{mode}.{key}"] = float(1.0 if int(y_masked) == int(y_full) else 0.0)

        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=result.context_fp,
            extras={"mode": mode},
        )


class MonotonicityMetric(BaseMetric):
    """Spearman correlation between edge importance and single-edge impact."""

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        candidate = _candidate_eidx(context, result)
        n_candidates = len(candidate)
        imp = _importance(result, n_candidates)

        if n_candidates < 2:
            return MetricResult(
                name=self.name,
                values={"spearman_rho": float("nan")},
                direction=self.direction.value,
                run_id=context.run_id,
                explainer=result.explainer,
                context_fp=result.context_fp,
            )

        z_full = _score(self.model.predict_proba(context.subgraph, context.target))

        impacts = np.zeros((n_candidates,), dtype=float)
        for pos in range(n_candidates):
            dropped = [candidate[pos]]
            z_drop = _predict_masked(self, context, candidate, dropped, mode="drop")
            impacts[pos] = float(abs(z_full - z_drop))

        rank_imp = _rank_values(imp)
        rank_impact = _rank_values(impacts)

        if np.std(rank_imp) == 0.0 or np.std(rank_impact) == 0.0:
            rho = float("nan")
        else:
            rho = float(np.corrcoef(rank_imp, rank_impact)[0, 1])

        return MetricResult(
            name=self.name,
            values={"spearman_rho": rho},
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=result.context_fp,
            extras={"n_edges": float(n_candidates)},
        )


class SeedStabilityMetric(BaseMetric):
    """Top-k stability under deterministic score jitter (higher is better)."""

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        candidate = _candidate_eidx(context, result)
        n_candidates = len(candidate)
        if n_candidates <= 0:
            return MetricResult(
                name=self.name,
                values={"value": float("nan")},
                direction=self.direction.value,
                run_id=context.run_id,
                explainer=result.explainer,
                context_fp=result.context_fp,
                extras={"reason": "missing_candidates"},
            )

        k = _resolve_metric_topk(
            self,
            n_candidates=n_candidates,
            default_sparsity=float(self.config.get("sparsity_level", 0.2)),
            ensure_min_one=True,
        )
        if n_candidates <= 1 or k <= 1:
            return MetricResult(
                name=self.name,
                values={"value": 1.0},
                direction=self.direction.value,
                run_id=context.run_id,
                explainer=result.explainer,
                context_fp=result.context_fp,
                extras={"k": int(k), "n_candidates": int(n_candidates), "formula": "pairwise_jaccard(top-k)"},
            )

        n_restarts = max(2, int(self.config.get("n_restarts", 10)))
        noise_scale = max(0.0, float(self.config.get("noise_scale", 0.05)))
        imp = _importance(result, n_candidates)
        spread = float(np.nanstd(imp))
        if not np.isfinite(spread) or spread <= 0.0:
            spread = 1.0

        candidate_arr = np.asarray(candidate, dtype=int)
        tie_break = str(self.config.get("tie_break", "edge_id")).strip().lower()
        tie = np.arange(n_candidates, dtype=np.int64) if tie_break == "candidate_order" else candidate_arr.astype(np.int64)

        rng = np.random.default_rng(_deterministic_seed(self.name, context, result))
        topk_sets: list[set[int]] = []
        for _ in range(n_restarts):
            noise = rng.normal(0.0, noise_scale * spread, size=n_candidates)
            scores = np.asarray(imp, dtype=float) + noise
            scores = np.nan_to_num(scores, nan=-np.inf)
            order = np.lexsort((tie, -scores))
            chosen = {int(candidate_arr[pos]) for pos in order[:k]}
            topk_sets.append(chosen)

        jaccards: list[float] = []
        for i in range(len(topk_sets)):
            for j in range(i + 1, len(topk_sets)):
                a = topk_sets[i]
                b = topk_sets[j]
                union = a | b
                if not union:
                    jaccards.append(1.0)
                else:
                    jaccards.append(float(len(a & b)) / float(len(union)))

        value = float(np.mean(jaccards)) if jaccards else 1.0
        return MetricResult(
            name=self.name,
            values={"value": value},
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=result.context_fp,
            extras={
                "k": int(k),
                "n_candidates": int(n_candidates),
                "n_restarts": int(n_restarts),
                "noise_scale": float(noise_scale),
                "jaccard_std": float(np.std(jaccards)) if jaccards else 0.0,
                "formula": "mean pairwise jaccard of jittered top-k edge sets",
            },
        )


class PerturbationRobustnessMetric(BaseMetric):
    """Robustness of fidelity under small random swaps in the selected top-k set."""

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        candidate = _candidate_eidx(context, result)
        n_candidates = len(candidate)
        if n_candidates <= 0:
            return MetricResult(
                name=self.name,
                values={"value": float("nan")},
                direction=self.direction.value,
                run_id=context.run_id,
                explainer=result.explainer,
                context_fp=result.context_fp,
                extras={"reason": "missing_candidates"},
            )

        imp = _importance(result, n_candidates)
        order = _ranked_order(self, context, result, candidate, imp)
        k = _resolve_metric_topk(
            self,
            n_candidates=n_candidates,
            default_sparsity=float(self.config.get("sparsity_level", 0.2)),
            ensure_min_one=True,
        )
        selected = [int(candidate[pos]) for pos in order[:k]]
        if not selected:
            return MetricResult(
                name=self.name,
                values={"value": float("nan")},
                direction=self.direction.value,
                run_id=context.run_id,
                explainer=result.explainer,
                context_fp=result.context_fp,
                extras={"reason": "empty_selection"},
            )

        mode = str(self.config.get("mode", "keep")).strip().lower()
        mask_mode = "drop" if mode == "drop" else "keep"
        score_fn = fidelity_minus if mask_mode == "drop" else fidelity_plus

        z_full = _score(self.model.predict_proba(context.subgraph, context.target))
        z_base = _predict_masked(self, context, candidate, selected, mode=mask_mode)
        base_fid = float(score_fn(float(z_full), float(z_base)))

        perturb_frac = min(1.0, max(0.0, float(self.config.get("perturb_frac", 0.15))))
        n_perturbations = max(1, int(self.config.get("n_perturbations", 16)))

        selected_set = set(selected)
        unselected = [int(e) for e in candidate if int(e) not in selected_set]
        if perturb_frac <= 0.0 or not unselected:
            return MetricResult(
                name=self.name,
                values={"value": 1.0},
                direction=self.direction.value,
                run_id=context.run_id,
                explainer=result.explainer,
                context_fp=result.context_fp,
                extras={
                    "k": int(k),
                    "base_fidelity": float(base_fid),
                    "mean_abs_delta": 0.0,
                    "formula": "1 / (1 + mean_abs_delta_fidelity)",
                    "mode": mask_mode,
                },
            )

        swap_count = int(round(float(k) * float(perturb_frac)))
        swap_count = max(1, min(swap_count, len(selected), len(unselected)))
        rng = np.random.default_rng(_deterministic_seed(self.name, context, result))
        selected_arr = np.asarray(selected, dtype=int)
        unselected_arr = np.asarray(unselected, dtype=int)

        deltas: list[float] = []
        for _ in range(n_perturbations):
            drop_edges = rng.choice(selected_arr, size=swap_count, replace=False)
            add_edges = rng.choice(unselected_arr, size=swap_count, replace=False)
            drop_set = {int(e) for e in drop_edges.tolist()}
            perturbed_selected = [e for e in selected if int(e) not in drop_set]
            perturbed_selected.extend(int(e) for e in add_edges.tolist())

            z_perturbed = _predict_masked(self, context, candidate, perturbed_selected, mode=mask_mode)
            perturbed_fid = float(score_fn(float(z_full), float(z_perturbed)))
            deltas.append(float(abs(perturbed_fid - base_fid)))

        mean_abs_delta = float(np.mean(deltas)) if deltas else 0.0
        value = float(1.0 / (1.0 + max(0.0, mean_abs_delta)))
        return MetricResult(
            name=self.name,
            values={"value": value},
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=result.context_fp,
            extras={
                "k": int(k),
                "n_perturbations": int(n_perturbations),
                "perturb_frac": float(perturb_frac),
                "swap_count": int(swap_count),
                "base_fidelity": float(base_fid),
                "mean_abs_delta": float(mean_abs_delta),
                "formula": "1 / (1 + mean_abs_delta_fidelity)",
                "mode": mask_mode,
            },
        )


class TemGXFidelityMetric(BaseMetric):
    """TemGX fidelity variants over sparsity levels."""

    def __init__(self, *, name: str, mode: str, config: Mapping[str, Any] | None = None):
        cfg = dict(config or {})
        super().__init__(
            name=name,
            direction=MetricDirection.HIGHER_IS_BETTER,
            config=cfg,
        )
        self.mode = mode

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        candidate = _candidate_eidx(context, result)
        n_candidates = len(candidate)
        imp = _importance(result, n_candidates)
        order = _curve_order(self, context, result, candidate, imp, default="strict")

        levels = _resolve_levels(self.config, default=np.arange(0.0, 0.31, 0.02).tolist())
        k_max = _resolve_k_max(self.config, n_candidates)
        use_logit = bool(self.config.get("result_as_logit", False))

        z_full = _score(self.model.predict_proba(context.subgraph, context.target))
        triplet = _resolve_event_triplet(self.model, context)
        full_from_raw = False
        if triplet is not None:
            src, dst, ts = triplet
            z_full_raw = _predict_raw_pair_score(
                self.model,
                src=src,
                dst=dst,
                ts=ts,
                preserve_eidx=None,
                result_as_logit=use_logit,
            )
            if z_full_raw is not None and np.isfinite(z_full_raw):
                z_full = float(z_full_raw)
                full_from_raw = True
        if not use_logit and not full_from_raw:
            z_full = float(_to_probability(float(z_full), score_is_logit=True))

        if self.mode == "minus":
            mask_mode = "drop"
            value_fn = _temgx_fidelity_minus
        else:
            mask_mode = "keep"
            value_fn = _temgx_fidelity_plus

        values: dict[str, float] = {"prediction_full": float(z_full)}
        points: list[tuple[float, float]] = []

        for level in levels:
            k = _k_from_level(
                float(level),
                n_candidates=n_candidates,
                k_max=k_max,
                ensure_min_one=False,
            )
            selected = [candidate[i] for i in order[:k]]
            z_masked = _predict_masked(self, context, candidate, selected, mode=mask_mode)
            masked_from_raw = False
            if triplet is not None:
                src, dst, ts = triplet
                preserve = _preserve_from_selected(
                    context,
                    result,
                    candidate,
                    selected,
                    mode=mask_mode,
                )
                z_masked_raw = _predict_raw_pair_score(
                    self.model,
                    src=src,
                    dst=dst,
                    ts=ts,
                    preserve_eidx=preserve,
                    result_as_logit=use_logit,
                )
                if z_masked_raw is not None and np.isfinite(z_masked_raw):
                    z_masked = float(z_masked_raw)
                    masked_from_raw = True
            if not use_logit and not masked_from_raw:
                z_masked = float(_to_probability(float(z_masked), score_is_logit=True))

            fid = float(value_fn(float(z_full), float(z_masked)))

            key = f"@s={float(level):g}"
            values[key] = fid
            points.append((float(level), fid))

        aufsc_style = str(self.config.get("aufsc_style", "temgx")).strip().lower()
        if aufsc_style in {"cum", "cumulative", "cody"}:
            values["aufsc"] = float(
                aufsc(points, max_sparsity=float(self.config.get("max_sparsity", 1.0)))
            )
        else:
            values["aufsc"] = float(_temgx_aufsc(points))
        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=result.context_fp,
            extras={"mode": self.mode, "points": points, "result_as_logit": bool(use_logit)},
        )


class TemGXSparsityMetric(BaseMetric):
    """TemGX sparsity ratio |E_expl| / |E_candidates|."""

    def __init__(self, config: Mapping[str, Any] | None = None):
        cfg = dict(config or {})
        super().__init__(
            name="temgx_sparsity",
            direction=MetricDirection.LOWER_IS_BETTER,
            config=cfg,
        )

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        candidate = _candidate_eidx(context, result)
        n_candidates = len(candidate)
        imp = _importance(result, n_candidates)
        order = _curve_order(self, context, result, candidate, imp, default="strict")

        has_levels = self.config.get("sparsity_levels") is not None or self.config.get("levels") is not None
        if has_levels:
            levels = _resolve_levels(self.config, default=[0.0, 0.1, 0.2, 0.3])
            k_max = _resolve_k_max(self.config, n_candidates)
            values = {}
            for level in levels:
                k = _k_from_level(
                    float(level),
                    n_candidates=n_candidates,
                    k_max=k_max,
                    ensure_min_one=False,
                )
                selected = [candidate[i] for i in order[:k]]
                values[f"@s={float(level):g}"] = float(sparsity(len(selected), n_candidates))
        else:
            ks = _resolve_ks(self.config, n_candidates)
            k = int(ks[-1]) if ks else 0
            selected = [candidate[i] for i in order[:k]]
            values = {"ratio": float(sparsity(len(selected), n_candidates))}

        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=result.context_fp,
        )


class TemGXAufscMetric(BaseMetric):
    """TemGX-style AUFSC: trapezoidal area over fidelity-vs-sparsity points."""

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        candidate = _candidate_eidx(context, result)
        n_candidates = len(candidate)
        imp = _importance(result, n_candidates)
        # Keep AUFSC ordering explainer-driven by default (paper-style), while
        # still allowing explicit override via `order_strategy`.
        order = _curve_order(self, context, result, candidate, imp, default="strict")

        levels = _resolve_levels(
            self.config,
            default=np.arange(0.0, 1.05, 0.05).tolist(),
        )
        k_max = _resolve_k_max(self.config, n_candidates)
        use_logit = bool(self.config.get("result_as_logit", False))

        mode = str(self.config.get("mode", "minus")).strip().lower()
        if mode == "plus":
            mask_mode = "keep"
            value_fn = _temgx_fidelity_plus
        else:
            mask_mode = "drop"
            value_fn = _temgx_fidelity_minus

        z_full = _score(self.model.predict_proba(context.subgraph, context.target))
        triplet = _resolve_event_triplet(self.model, context)
        full_from_raw = False
        if triplet is not None:
            src, dst, ts = triplet
            z_full_raw = _predict_raw_pair_score(
                self.model,
                src=src,
                dst=dst,
                ts=ts,
                preserve_eidx=None,
                result_as_logit=use_logit,
            )
            if z_full_raw is not None and np.isfinite(z_full_raw):
                z_full = float(z_full_raw)
                full_from_raw = True
        if not use_logit and not full_from_raw:
            z_full = float(_to_probability(float(z_full), score_is_logit=True))

        values: dict[str, float] = {"prediction_full": float(z_full)}
        points: list[tuple[float, float]] = []

        for level in levels:
            k = _k_from_level(
                float(level),
                n_candidates=n_candidates,
                k_max=k_max,
                ensure_min_one=False,
            )
            selected = [candidate[i] for i in order[:k]]
            z_masked = _predict_masked(self, context, candidate, selected, mode=mask_mode)
            masked_from_raw = False
            if triplet is not None:
                src, dst, ts = triplet
                preserve = _preserve_from_selected(
                    context,
                    result,
                    candidate,
                    selected,
                    mode=mask_mode,
                )
                z_masked_raw = _predict_raw_pair_score(
                    self.model,
                    src=src,
                    dst=dst,
                    ts=ts,
                    preserve_eidx=preserve,
                    result_as_logit=use_logit,
                )
                if z_masked_raw is not None and np.isfinite(z_masked_raw):
                    z_masked = float(z_masked_raw)
                    masked_from_raw = True
            if not use_logit and not masked_from_raw:
                z_masked = float(_to_probability(float(z_masked), score_is_logit=True))

            fid = float(value_fn(float(z_full), float(z_masked)))
            key = f"@s={float(level):g}"
            values[key] = fid
            points.append((float(level), fid))

        area = float(_temgx_aufsc(points))
        values["value"] = area
        values["aufsc"] = area

        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=result.context_fp,
            extras={
                "definition": "TemGX AUFSC = trapz(fid, sparsity)",
                "mode": mode,
                "result_as_logit": bool(use_logit),
                "levels": [float(v) for v in levels],
                "points": points,
            },
        )


class SingularValueMetric(BaseMetric):
    """Largest singular value of explanation subgraph adjacency over levels."""

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        candidate = _candidate_eidx(context, result)
        n_candidates = len(candidate)
        imp = _importance(result, n_candidates)
        order = _ranked_order(self, context, result, candidate, imp)

        src, dst = _candidate_endpoints(context, result, candidate)
        n = min(n_candidates, src.size, dst.size)

        levels = _resolve_levels(self.config, default=[0.0, 0.1, 0.2, 0.3])
        k_max = _resolve_k_max(self.config, n)

        values: dict[str, float] = {}
        svs: list[float] = []

        for level in levels:
            k = _k_from_level(
                float(level),
                n_candidates=n,
                k_max=k_max,
                ensure_min_one=False,
            )
            keep_positions = order[:k]
            keep_positions = [idx for idx in keep_positions if idx < n]

            if not keep_positions:
                sv = 0.0
            else:
                src_sel = src[keep_positions]
                dst_sel = dst[keep_positions]
                nodes = np.unique(np.concatenate([src_sel, dst_sel]))
                if nodes.size == 0:
                    sv = 0.0
                else:
                    node_to_pos = {int(node): pos for pos, node in enumerate(nodes.tolist())}
                    adj = np.zeros((nodes.size, nodes.size), dtype=float)
                    for u, v in zip(src_sel.tolist(), dst_sel.tolist()):
                        iu = node_to_pos[int(u)]
                        iv = node_to_pos[int(v)]
                        adj[iu, iv] += 1.0
                        adj[iv, iu] += 1.0
                    singular_values = np.linalg.svd(adj, compute_uv=False)
                    sv = float(singular_values[0]) if singular_values.size > 0 else 0.0

            key = f"@s={float(level):g}"
            values[key] = float(sv)
            svs.append(float(sv))

        values["mean"] = float(np.mean(svs)) if svs else float("nan")
        values["max"] = float(np.max(svs)) if svs else float("nan")

        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=result.context_fp,
            extras={"n_endpoints": float(n)},
        )


# --------------------------------------------------------------------------- #
# Builders and conditional registration
# --------------------------------------------------------------------------- #
