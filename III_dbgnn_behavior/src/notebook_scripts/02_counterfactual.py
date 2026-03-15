from pathlib import Path
import sys

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


import numpy as np
import torch

from config import ExperimentConfig
from data.registry import get_dataset_loader
from utils import get_device, set_seed, make_run_name
from train import train_model, load_or_train, evaluate_balanced_accuracy

from models.dbgnn import DBGNNAdapter
from models.registry import get_model_builder

from explainers.counterfactual import CounterfactualConfig
from explainers.counterfactual_margin import greedy_drop_margin_order
from explainers.counterfactual_search import find_min_ho_edge_deletions_to_flip

from explainers.counterfactual_data_only import (
    DataOnlyCounterfactualConfig,
    DataOnlyCounterfactualEdgeDeletionExplainer,
)
from eval.metrics import ranked_edge_indices

from viz.counterfactual_viz import (
    probs_before_after,
    prob_curve_by_k,
    prob_curve_keep_only_by_k,
    plot_probs_bar,
    plot_prob_curve,
    plot_prob_curves_overlay,
    plot_removed_edges_graph,
    plot_de_bruijn_with_deleted_edges,
)
from viz.palette import DEFAULT_CLASS_COLORS


# --- Config ---
seed = int(globals().get("seed", 42))

# Pick any dataset registered in `data.registry.DATASET_REGISTRY`.
# Tip: any netzschleuder record name also works, e.g. dataset_name="sp_high_school".
# The netzschleuder loader in src applies the pandas string patch automatically.
dataset_name = str(globals().get("dataset_name", "sp_high_school"))  # e.g. "temporal_clusters", "covid", "sp_colocation"

# Optional: override the netzschleuder label attribute (e.g., "gender", "class").
# If None, the loader uses its built-in defaults for the record.
target_attr_override = globals().get("target_attr_override", "class")

# Dataset-specific kwargs (override netzschleuder defaults if needed).
# Example override:
# dataset_kwargs = dict(network="proximity", time_attr="time")
dataset_kwargs = dict(globals().get("dataset_kwargs", {}))

if target_attr_override is not None:
    dataset_kwargs["target_attr"] = str(target_attr_override)

# Train/test split fraction for node classification datasets
num_test = float(globals().get("num_test", 0.3))

# Training hyperparameters
epochs = int(globals().get("epochs", 200))
lr = float(globals().get("lr", 0.005))
p_dropout = float(globals().get("p_dropout", 0.4))
hidden_dims = tuple(globals().get("hidden_dims", (16, 32, 16)))

# Optional: drop a random subset of first-order edges each epoch (forces HO usage).
enable_fo_edge_drop = False
fo_edge_drop_frac = 0.5  # fraction (<=1) or absolute count (>1)
fo_edge_drop_seed = 0

# Optional: force HO branch by zeroing ALL FO edges every N epochs.
enable_force_ho_epochs = False
force_ho_every_n = 1  # every N epochs, use HO-only graph

# Train vs load checkpoint
train_from_scratch = bool(globals().get("train_from_scratch", True))

# Run bookkeeping (optional)
run_dir = ROOT / "notebooks" / "runs"
run_name = str(globals().get("run_name", make_run_name(dataset_name, dataset_kwargs, model_name="dbgnn")))

# Node to explain. If None, we pick a reasonable default (first test node if available).
target_node_idx = globals().get("target_node_idx", 4)

# Greedy deletion search cap
max_k = int(globals().get("max_k", 600))

# Drop mode for evaluations: "remove" or "zero" (zeroes edge weights)
drop_mode = str(globals().get("drop_mode", "remove"))

# When False, compute all artifacts but do not render/save plots.
render_all_plots = bool(globals().get("render_all_plots", True))



set_seed(seed)
device = get_device()
device


# --- Load dataset ---
# Note: netzschleuder loader in src applies the pandas string-attribute patch.
loader = get_dataset_loader(dataset_name)
data, assets = loader(device=device, num_test=num_test, seed=seed, **dataset_kwargs)

data

# --- Save dataset bundle for later inspection ---
from data.cache import save_dataset_bundle


save_dir = run_dir / str(run_name) / "dataset"
saved_to = save_dataset_bundle(
    out_dir=save_dir,
    data=data,
    assets=assets,
    extra_meta=dict(dataset_name=dataset_name, dataset_kwargs=dataset_kwargs),
)
print(f"Saved dataset bundle to: {saved_to}")



def pick_target_node(data):
    if hasattr(data, "test_mask") and data.test_mask is not None:
        idx = torch.where(data.test_mask)[0]
        if idx.numel() > 0:
            return int(idx[0])
    return 0

if target_node_idx is None:
    target_node_idx = pick_target_node(data)

target_node_idx

# or pick it yourself 
triples = data.ho_triples  # [E_ho, 3]
counts = torch.bincount(triples[:, 1], minlength=int(data.num_nodes))

# choose a test node with candidates
test_idx = torch.where(data.test_mask)[0]
valid = test_idx[counts[test_idx] > 0]
target_node_idx = int(valid[0])  # or counts.argmax()


# --- Choose node to explain ---

def _ho_candidate_counts(data):
    """Count how many higher-order edges (triples) are *eligible* for a node.

    We use `data.ho_triples` if available (shape [E_ho, 3] with triples (u,v,w)).
    For a node-focused explanation at node `v`, a common candidate set is
    transitions where the middle node equals `v`.

    This function returns per-node counts for `v = triples[:, 1]`.
    """
    triples = getattr(data, "ho_triples", None)
    if triples is None or not torch.is_tensor(triples):
        return None
    if triples.ndim != 2 or triples.size(1) != 3:
        return None

    num_nodes = int(getattr(data, "num_nodes", int(triples[:, 1].max().item()) + 1))
    v = triples[:, 1].to(torch.long)
    return torch.bincount(v, minlength=num_nodes)


def _margins_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return per-node logit margin (top1 - top2). Smaller => closer to decision boundary."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be [N,C], got {tuple(logits.shape)}")
    if logits.size(1) < 2:
        return torch.zeros(int(logits.size(0)), device=logits.device)
    top2 = torch.topk(logits, k=2, dim=-1).values
    return (top2[:, 0] - top2[:, 1]).detach()



print("E_ho:", data.edge_index_higher_order.size(1))
print("ho_triples:", getattr(data, "ho_triples", None).shape if hasattr(data, "ho_triples") else None)
#print("C_candidates:", result.meta.get("C_candidates"))


# --- Train model + build adapter ---
cfg = ExperimentConfig(
    dataset_name=dataset_name,
    model_name="dbgnn",
    dataset_kwargs=dataset_kwargs,
    run_dir=str(run_dir),
    run_name=str(run_name),
    seed=seed,
    epochs=200,
    lr=lr,
    p_dropout=p_dropout,
    hidden_dims=hidden_dims,
    num_test=num_test,
)

if train_from_scratch:
    if enable_fo_edge_drop or enable_force_ho_epochs:
        model_builder = get_model_builder(cfg.model_name)
        adapter = model_builder(data=data, assets=assets, device=device, hidden_dims=cfg.hidden_dims, p_dropout=cfg.p_dropout)
        model = adapter.model

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        loss_function = torch.nn.CrossEntropyLoss()

        losses = []
        rng = torch.Generator(device=data.edge_index.device)
        rng.manual_seed(int(fo_edge_drop_seed))

        for epoch in range(int(cfg.epochs)):
            model.train()
            data_train = data

            # Force HO-only epochs by removing ALL first-order edges.
            if enable_force_ho_epochs and force_ho_every_n and (epoch % int(force_ho_every_n) == 0):
                data_train = data.clone()
                data_train.edge_index = data.edge_index[:, :0]
                if getattr(data, "edge_weights", None) is not None:
                    data_train.edge_weights = data.edge_weights[:0]
                data_train.x = torch.zeros_like(data.x)
            # Otherwise, optionally drop a random subset of FO edges.
            elif enable_fo_edge_drop and fo_edge_drop_frac and fo_edge_drop_frac > 0:
                E = int(data.edge_index.size(1))
                k = int(fo_edge_drop_frac * E) if fo_edge_drop_frac <= 1 else int(fo_edge_drop_frac)
                k = max(0, min(k, E))
                if k > 0:
                    drop_idx = torch.randperm(E, generator=rng, device=data.edge_index.device)[:k]
                    keep_mask = torch.ones(E, dtype=torch.bool, device=data.edge_index.device)
                    keep_mask[drop_idx] = False

                    data_train = data.clone()
                    data_train.edge_index = data.edge_index[:, keep_mask]
                    if getattr(data, "edge_weights", None) is not None:
                        data_train.edge_weights = data.edge_weights[keep_mask]

            output = model(data_train)
            loss = loss_function(output[data_train.train_mask], data_train.y[data_train.train_mask])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(float(loss.detach().cpu().item()))

            if epoch % 10 == 0:
                train_ba, test_ba = evaluate_balanced_accuracy(model, data)
                print(
                    f"Epoch: {epoch}, Loss: {loss.detach().item():.4f}, Train balanced accuracy: {train_ba:.4f}, Test balanced accuracy: {test_ba:.4f}"
                )

        train_info = {"losses": losses}
    else:
        adapter, train_info = train_model(cfg, data=data, assets=assets, device=device)
else:
    adapter = load_or_train(cfg, data=data, assets=assets, device=device)

model = adapter.model


# HO edge weights + features
ew = data.edge_weights_higher_order
print("ew nonzero:", int((ew != 0).sum()), "min/mean/max:", float(ew.min()), float(ew.mean()), float(ew.max()))
print("x_h std:", float(data.x_h.float().std()))

# Does HO branch ever matter if FO is removed?
d = data.clone()
d.edge_index = data.edge_index[:, :0]
if getattr(data, "edge_weights", None) is not None:
    d.edge_weights = data.edge_weights[:0]
logits_ho_only = adapter.predict_logits(d)[target_node_idx]
logits_full = adapter.predict_logits(data)[target_node_idx]
print("Δlogit HO-only vs full:", float((logits_full - logits_ho_only).abs().max()))


logits = model(data)
d = data.clone()
d.edge_index_higher_order = data.edge_index_higher_order[:, :0]
d.edge_weights_higher_order = data.edge_weights_higher_order[:0]
logits_no = model(d)

score = (logits - logits_no).abs().max(dim=1).values
target_node_idx = int(score.argmax())
target_node_idx



logits = adapter.predict_logits(data)
d = data.clone()
d.edge_index_higher_order = data.edge_index_higher_order[:, :0]
d.edge_weights_higher_order = data.edge_weights_higher_order[:0]
logits_no = adapter.predict_logits(d)

delta = (logits - logits_no).abs().max(dim=1).values
print("max/mean HO impact:", float(delta.max()), float(delta.mean()))
target_node_idx = int(delta.argmax())


# --- Counterfactual search (ranking + flip search) ---
#
# Tip (netzschleuder): k_hops=4 can easily make the candidate set cover most of the HO graph.
# Start with k_hops=2 (often enough for a 2-layer HO message passing stack) and increase only if needed.
cf_cfg = CounterfactualConfig(
    k_hops=2,
    steps=300,
    lr=0.1,
    lambda_size=0.02,
    lambda_entropy=0.001,
    flip_to="non_targeted",
)

# target_node_idx = 7
result = find_min_ho_edge_deletions_to_flip(
    adapter=adapter,
    data=data,
    assets=assets,
    target_node_idx=target_node_idx,
    cfg=cf_cfg,
    seed=seed,
    max_k=max_k,
    optimizer="counterfactual",
    drop_mode=drop_mode,
    # Fast notebook default: probe a schedule of k values + local refinement (much faster than linear scan).
    search_mode="schedule",
    # If no flip occurs, still return a compact set of deletions that most reduces the original margin (for plotting).
    return_best_effort=True,
    best_effort_max_k=min(200, int(max_k)),
    verbose=True,
)

result.to_dict()


# --- Plots (Plotly-first) ---
plot_dir = ROOT / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

# Shared class palette (starts with eventblue / snapshotorange / edgegray).
class_colors = list(globals().get("class_colors", DEFAULT_CLASS_COLORS))

# Before/after class probabilities
p0, p1 = probs_before_after(
    adapter=adapter,
    data=data,
    node_idx=target_node_idx,
    removed_edge_indices=result.removed_edge_indices,
)
if render_all_plots:
    plot_probs_bar(
        p_before=p0,
        p_after=p1,
        title=f"Node {target_node_idx}: probs before/after (k={result.n_removed})",
        backend="plotly",
        save_path=plot_dir / f"target_{int(target_node_idx)}_class_probs.pdf",
    )

# Probability curve as we delete more and more ranked HO edges
k_vals, probs = prob_curve_by_k(
    adapter=adapter,
    data=data,
    node_idx=target_node_idx,
    ranked_edge_indices=result.ranked_edge_indices,
    k_step=5,
    max_k=max_k,
    drop_mode=drop_mode,
)
if render_all_plots and k_vals is not None:
    plot_prob_curve(
        k_values=k_vals,
        probs=probs,
        vline_k=result.n_removed if result.success else None,
        title=f"Node {target_node_idx}: probability curve",
        backend="plotly",
        class_colors=class_colors,
        save_path=plot_dir / f"target_{int(target_node_idx)}_prob_curve.pdf",
    )

# Probability curve when we ONLY keep the ranked HO edges (cf order)
k_keep, probs_keep = prob_curve_keep_only_by_k(
    adapter=adapter,
    data=data,
    node_idx=target_node_idx,
    ranked_edge_indices=result.ranked_edge_indices,
    k_step=5,
    max_k=max_k,
    keep_non_ranked=False,
    drop_mode=drop_mode,
)
if render_all_plots and k_keep is not None:
    plot_prob_curve(
        k_values=k_keep,
        probs=probs_keep,
        title=f"Node {target_node_idx}: prob vs kept edges (cf order)",
        x_label="Number of higher-order edges kept (k)",
        backend="plotly",
        class_colors=class_colors,
        save_path=plot_dir / f"target_{int(target_node_idx)}_prob_curve_keep_only_cf.pdf",
    )

k_vals_margin = None
probs_margin = None
k_keep_margin = None
probs_keep_margin = None

# Probability curve for margin-based counterfactual order (computed on the fly)
edge_order_margin = []
try:
    edge_order_margin, margin_orig, margin_new = greedy_drop_margin_order(
        adapter=adapter,
        data=data,
        node_idx=target_node_idx,
        max_steps=200,
        stop_on_flip=False,
    )
except Exception as e:
    print(f"[warn] Margin-based order failed: {type(e).__name__}: {e}")

if edge_order_margin:
    k_vals_margin, probs_margin = prob_curve_by_k(
        adapter=adapter,
        data=data,
        node_idx=target_node_idx,
        ranked_edge_indices=edge_order_margin,
        k_step=5,
        max_k=min(200, len(edge_order_margin)),
        drop_mode=drop_mode,
    )
    if render_all_plots and k_vals_margin is not None:
        plot_prob_curve(
            k_values=k_vals_margin,
            probs=probs_margin,
            title=f"Node {target_node_idx}: prob curve (margin-based order)",
            backend="plotly",
            class_colors=class_colors,
            save_path=plot_dir / f"target_{int(target_node_idx)}_prob_curve_margin_order.pdf",
        )

    # Keep-only curve for margin-based order
    k_keep_margin, probs_keep_margin = prob_curve_keep_only_by_k(
        adapter=adapter,
        data=data,
        node_idx=target_node_idx,
        ranked_edge_indices=edge_order_margin,
        k_step=5,
        max_k=min(200, len(edge_order_margin)),
        keep_non_ranked=False,
        drop_mode=drop_mode,
    )
    if render_all_plots and k_keep_margin is not None:
        plot_prob_curve(
            k_values=k_keep_margin,
            probs=probs_keep_margin,
            title=f"Node {target_node_idx}: prob vs kept edges (margin order)",
            x_label="Number of higher-order edges kept (k)",
            backend="plotly",
            class_colors=class_colors,
            save_path=plot_dir / f"target_{int(target_node_idx)}_prob_curve_keep_only_margin_order.pdf",
        )
else:
    print("[warn] Margin-based order empty.")

k_vals_data_only = None
probs_data_only = None
k_keep_data_only = None
probs_keep_data_only = None

# Probability curve for data-only counterfactual order (from HO weights + labels)
edge_order_data_only = []
try:
    data_only_explainer = DataOnlyCounterfactualEdgeDeletionExplainer(
        cfg=DataOnlyCounterfactualConfig(),
    )
    target_class = getattr(result, "orig_pred", None)
    if target_class is None:
        with torch.no_grad():
            logits_row = adapter.predict_logits(data)[int(target_node_idx)]
        target_class = int(logits_row.argmax().item())
    exp_data_only = data_only_explainer.explain_node(
        adapter=adapter,
        data=data,
        node_idx=target_node_idx,
        target_class=target_class,
    )
    edge_order_data_only = ranked_edge_indices(
        exp_data_only.edge_score,
        candidate_mask=exp_data_only.candidate_mask,
        descending=True,
    ).tolist()
except Exception as e:
    print(f"[warn] Data-only order failed: {type(e).__name__}: {e}")

if edge_order_data_only:
    k_vals_data_only, probs_data_only = prob_curve_by_k(
        adapter=adapter,
        data=data,
        node_idx=target_node_idx,
        ranked_edge_indices=edge_order_data_only,
        k_step=5,
        max_k=min(int(max_k), len(edge_order_data_only)),
        drop_mode=drop_mode,
    )
    if render_all_plots and k_vals_data_only is not None:
        plot_prob_curve(
            k_values=k_vals_data_only,
            probs=probs_data_only,
            title=f"Node {target_node_idx}: prob curve (data-only order)",
            backend="plotly",
            class_colors=class_colors,
            save_path=plot_dir / f"target_{int(target_node_idx)}_prob_curve_data_only_order.pdf",
        )

    # Keep-only curve for data-only order
    k_keep_data_only, probs_keep_data_only = prob_curve_keep_only_by_k(
        adapter=adapter,
        data=data,
        node_idx=target_node_idx,
        ranked_edge_indices=edge_order_data_only,
        k_step=5,
        max_k=min(int(max_k), len(edge_order_data_only)),
        keep_non_ranked=False,
        drop_mode=drop_mode,
    )
    if render_all_plots and k_keep_data_only is not None:
        plot_prob_curve(
            k_values=k_keep_data_only,
            probs=probs_keep_data_only,
            title=f"Node {target_node_idx}: prob vs kept edges (data-only order)",
            x_label="Number of higher-order edges kept (k)",
            backend="plotly",
            class_colors=class_colors,
            save_path=plot_dir / f"target_{int(target_node_idx)}_prob_curve_keep_only_data_only_order.pdf",
        )
else:
    print("[warn] Data-only order empty.")


def _pad_curve(k_values, probs, max_k):
    if k_values is None or probs is None or max_k is None:
        return k_values, probs
    k = np.asarray(k_values).reshape(-1)
    p = np.asarray(probs)
    if k.size == 0:
        return k, p
    max_k = int(max_k)
    if int(k[-1]) >= max_k:
        return k, p
    k_pad = np.concatenate([k, [max_k]])
    p_pad = np.vstack([p, p[-1]])
    return k_pad, p_pad


# Combined plot: counterfactual vs margin/data-only order (overlay)
max_k_candidates = []
if k_vals is not None:
    max_k_candidates.append(int(k_vals[-1]))
if k_vals_margin is not None:
    max_k_candidates.append(int(k_vals_margin[-1]))
if k_vals_data_only is not None:
    max_k_candidates.append(int(k_vals_data_only[-1]))
max_k_common = max(max_k_candidates) if max_k_candidates else None

k_vals_pad, probs_pad = _pad_curve(k_vals, probs, max_k_common)
k_vals_margin_pad, probs_margin_pad = _pad_curve(k_vals_margin, probs_margin, max_k_common)
k_vals_data_only_pad, probs_data_only_pad = _pad_curve(k_vals_data_only, probs_data_only, max_k_common)

curves = []
if k_vals_pad is not None:
    curves.append(dict(label="cf", k_values=k_vals_pad, probs=probs_pad, line=dict(dash="solid")))
if k_vals_margin_pad is not None:
    curves.append(dict(label="margin", k_values=k_vals_margin_pad, probs=probs_margin_pad, line=dict(dash="dot")))
if k_vals_data_only_pad is not None:
    curves.append(dict(label="data_only", k_values=k_vals_data_only_pad, probs=probs_data_only_pad, line=dict(dash="dashdot")))
if render_all_plots and curves:
    plot_prob_curves_overlay(
        curves=curves,
        title=f"Node {target_node_idx}: cf vs margin/data-only order",
        x_label="Number of higher-order edges removed (k)",
        backend="plotly",
        class_colors=class_colors,
        max_k=max_k_common,
        save_path=plot_dir / f"target_{int(target_node_idx)}_prob_curve_cf_vs_margin_data_only.pdf",
    )

# Combined keep-only plot: most-important edges kept in ranked order (overlay)
max_k_keep_candidates = []
if k_keep is not None:
    max_k_keep_candidates.append(int(k_keep[-1]))
if k_keep_margin is not None:
    max_k_keep_candidates.append(int(k_keep_margin[-1]))
if k_keep_data_only is not None:
    max_k_keep_candidates.append(int(k_keep_data_only[-1]))
max_k_keep_common = max(max_k_keep_candidates) if max_k_keep_candidates else None

k_keep_pad, probs_keep_pad = _pad_curve(k_keep, probs_keep, max_k_keep_common)
k_keep_margin_pad, probs_keep_margin_pad = _pad_curve(k_keep_margin, probs_keep_margin, max_k_keep_common)
k_keep_data_only_pad, probs_keep_data_only_pad = _pad_curve(k_keep_data_only, probs_keep_data_only, max_k_keep_common)

curves_keep = []
if k_keep_pad is not None:
    curves_keep.append(dict(label="cf", k_values=k_keep_pad, probs=probs_keep_pad, line=dict(dash="solid")))
if k_keep_margin_pad is not None:
    curves_keep.append(dict(label="margin", k_values=k_keep_margin_pad, probs=probs_keep_margin_pad, line=dict(dash="dot")))
if k_keep_data_only_pad is not None:
    curves_keep.append(dict(label="data_only", k_values=k_keep_data_only_pad, probs=probs_keep_data_only_pad, line=dict(dash="dashdot")))
if render_all_plots and curves_keep:
    plot_prob_curves_overlay(
        curves=curves_keep,
        title=f"Node {target_node_idx}: keep-only cf vs margin/data-only order",
        x_label="Number of higher-order edges kept (k)",
        backend="plotly",
        class_colors=class_colors,
        max_k=max_k_keep_common,
        save_path=plot_dir / f"target_{int(target_node_idx)}_prob_curve_keep_only_cf_vs_margin_data_only.pdf",
    )

# Removed-edge view (tiny graph of just the deleted HO edges)
if render_all_plots and result.removed_edges_as_node_ids:
    plot_removed_edges_graph(
        removed_edges_as_node_ids=result.removed_edges_as_node_ids,
        target_node_idx=target_node_idx,
        save_path=plot_dir / f"target_{int(target_node_idx)}_removed_edges.pdf",
    )

# Optional: plot the full De Bruijn graph and highlight deleted edges.
g2 = getattr(assets, "g2", None) if assets is not None else None
if render_all_plots and g2 is not None and result.removed_edge_indices:
    # (This uses pathpyG's matplotlib backend; it can be slow for large graphs.)
    plot_de_bruijn_with_deleted_edges(
        g2=g2,
        data=data,
        removed_edge_indices=result.removed_edge_indices,
        target_node_idx=target_node_idx,
    )
