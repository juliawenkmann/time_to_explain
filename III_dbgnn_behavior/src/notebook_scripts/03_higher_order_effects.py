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
import torch.nn.functional as F

from data.registry import get_dataset_loader
from utils import get_device, set_seed, make_run_name
from checkpoints import load_dbgnn_checkpoint
from models.dbgnn import DBGNNAdapter
from viz.palette import DEFAULT_CLASS_COLORS, EDGE_GRAY

# ---------- Config ----------
seed = int(globals().get("seed", 0))
render_all_plots = bool(globals().get("render_all_plots", True))

dataset_name = str(globals().get("dataset_name", "temporal_clusters"))
dataset_kwargs = dict(globals().get("dataset_kwargs", {}))

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
run_dir = ROOT / "notebooks" / "runs"
run_name = str(globals().get("run_name", make_run_name(dataset_name, dataset_kwargs, model_name="dbgnn")))
checkpoint_path = globals().get("checkpoint_path", None)  # optional: set to an explicit path
PLOT_DIR = ROOT / "plots" / "03_higher_order_effects"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Load data + model ----------
device = get_device("auto")
set_seed(seed)

loader = get_dataset_loader(dataset_name)
data, assets = loader(device=device, seed=seed, **dataset_kwargs)

ckpt = load_dbgnn_checkpoint(
    run_dir=run_dir,
    run_name=run_name,
    data=data,
    device=device,
    checkpoint_path=checkpoint_path,
)
model = ckpt.model
adapter = DBGNNAdapter(model=model)

# Pull parameters as a plain state dict for the manual forward below
state = {k: v.detach().to(device) for k, v in model.state_dict().items()}

# ---------- Edges / weights ----------
if not hasattr(data, "edge_index"):
    raise AttributeError("data.edge_index is required")
if not hasattr(data, "edge_index_higher_order"):
    raise AttributeError("data.edge_index_higher_order is required")

edge_index = data.edge_index
edge_index_ho = data.edge_index_higher_order

if hasattr(data, "edge_weights") and data.edge_weights is not None:
    edge_weight = data.edge_weights.detach().float()
elif hasattr(data, "edge_weight") and data.edge_weight is not None:
    edge_weight = data.edge_weight.detach().float()
else:
    edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

if hasattr(data, "edge_weights_higher_order") and data.edge_weights_higher_order is not None:
    edge_weight_ho = data.edge_weights_higher_order.detach().float()
else:
    raise AttributeError("data.edge_weights_higher_order is required")

# Sizes
N = int(getattr(data, "num_nodes", int(data.x.size(0)) if hasattr(data, "x") else 0))
if hasattr(data, "x_h") and data.x_h is not None:
    N_ho = int(data.x_h.size(0))
elif hasattr(data, "num_ho_nodes"):
    N_ho = int(data.num_ho_nodes)
else:
    raise AttributeError("data.x_h or data.num_ho_nodes is required")

# Bipartite mapping HO->FO
if hasattr(data, "bipartite_edge_index") and data.bipartite_edge_index is not None:
    bip_edge_index = data.bipartite_edge_index
else:
    from pathpy_utils import idx_to_node_list
    if assets is None or getattr(assets, "g2", None) is None:
        raise AttributeError("assets.g2 is required to build bipartite_edge_index")
    idx_to_ho = idx_to_node_list(assets.g2)
    v_last = torch.tensor([int(p[1]) for p in idx_to_ho], dtype=torch.long, device=edge_index.device)
    bip_edge_index = torch.stack([torch.arange(N_ho, device=edge_index.device), v_last], dim=0)

# ---------- GCN helpers ----------
def add_missing_self_loops(edge_index, edge_weight, num_nodes, fill_value=1.0):
    row, col = edge_index
    mask = row == col
    has_loop = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
    has_loop[row[mask]] = True
    missing = (~has_loop).nonzero(as_tuple=False).view(-1)
    if missing.numel() == 0:
        return edge_index, edge_weight
    loops = torch.stack([missing, missing], dim=0)
    loop_w = torch.full((missing.numel(),), float(fill_value), dtype=edge_weight.dtype, device=edge_weight.device)
    return torch.cat([edge_index, loops], dim=1), torch.cat([edge_weight, loop_w], dim=0)


def gcn_norm(edge_index, edge_weight, num_nodes):
    edge_index2, edge_weight2 = add_missing_self_loops(edge_index, edge_weight, num_nodes, 1.0)
    row, col = edge_index2
    deg = torch.zeros(num_nodes, dtype=edge_weight2.dtype, device=edge_weight2.device)
    deg.scatter_add_(0, col, edge_weight2)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    norm = deg_inv_sqrt[row] * edge_weight2 * deg_inv_sqrt[col]
    return edge_index2, norm


def gcn_layer(x, edge_index, edge_weight, W, b):
    num_nodes = x.size(0)
    edge_index2, norm = gcn_norm(edge_index, edge_weight, num_nodes)
    row, col = edge_index2
    x_lin = x @ W.t()
    out = torch.zeros((num_nodes, x_lin.size(1)), dtype=x_lin.dtype, device=x_lin.device)
    out.index_add_(0, col, x_lin[row] * norm.unsqueeze(1))
    return out + b


def gcn_layer_decompose_self_vs_neigh(x, edge_index, edge_weight, W, b):
    """Pre-activation decomposition: z = z_self + z_neigh + b"""
    num_nodes = x.size(0)
    edge_index2, norm = gcn_norm(edge_index, edge_weight, num_nodes)
    row, col = edge_index2
    x_lin = x @ W.t()

    mask_self = (row == col)
    z_self = torch.zeros((num_nodes, x_lin.size(1)), dtype=x_lin.dtype, device=x_lin.device)
    z_neigh = torch.zeros((num_nodes, x_lin.size(1)), dtype=x_lin.dtype, device=x_lin.device)

    if mask_self.any():
        z_self.index_add_(0, col[mask_self], x_lin[row[mask_self]] * norm[mask_self].unsqueeze(1))
    if (~mask_self).any():
        z_neigh.index_add_(0, col[~mask_self], x_lin[row[~mask_self]] * norm[~mask_self].unsqueeze(1))

    z_total = z_self + z_neigh + b
    return z_total, z_self, z_neigh

# ---------- Bipartite operator ----------
def bipartite_operator(x_h, x_fo):
    xh_t = x_h @ state["bipartite_layer.lin1.weight"].t() + state["bipartite_layer.lin1.bias"]
    xf_t = x_fo @ state["bipartite_layer.lin2.weight"].t() + state["bipartite_layer.lin2.bias"]
    src, dst = bip_edge_index

    out_ho = torch.zeros((N, xh_t.size(1)), dtype=xh_t.dtype, device=xh_t.device)
    out_ho.index_add_(0, dst, xh_t[src])

    deg = torch.bincount(dst, minlength=N).to(out_ho.dtype).unsqueeze(1)
    out_fo = deg * xf_t

    return out_ho + out_fo, out_ho, out_fo

# ---------- Forward returning intermediate HO embedding ----------
def forward_all(g_w, h_w):
    x   = torch.eye(N, device=g_w.device)
    x_h = torch.eye(N_ho, device=h_w.device)

    # FO branch
    x = F.elu(gcn_layer(x, edge_index, g_w,
                       state["first_order_layers.0.lin.weight"], state["first_order_layers.0.bias"]))
    x = F.elu(gcn_layer(x, edge_index, g_w,
                       state["first_order_layers.1.lin.weight"], state["first_order_layers.1.bias"]))

    # HO branch
    z1_ho = gcn_layer(x_h, edge_index_ho, h_w,
                     state["higher_order_layers.0.lin.weight"], state["higher_order_layers.0.bias"])
    x_h = F.elu(z1_ho)
    z2_ho = gcn_layer(x_h, edge_index_ho, h_w,
                     state["higher_order_layers.1.lin.weight"], state["higher_order_layers.1.bias"])
    H = F.elu(z2_ho)  # final HO embedding

    # bipartite + classifier
    z_node, z_ho_part, z_fo_part = bipartite_operator(H, x)
    x_node = F.elu(z_node)
    logits = x_node @ state["lin.weight"].t() + state["lin.bias"]
    return H, x_node, logits, z_ho_part, z_fo_part


# ---------- 1) Quick sanity check: full vs self-loop-only HO ----------
H_full, X_full, L_full, _, _ = forward_all(edge_weight, edge_weight_ho)
H_self, X_self, L_self, _, _ = forward_all(edge_weight, torch.zeros_like(edge_weight_ho))

pred_full = L_full.argmax(dim=1)
pred_self = L_self.argmax(dim=1)
unchanged_pct = float((pred_full == pred_self).float().mean().item()) * 100.0
print(f"Predictions unchanged (full vs self-loop-only HO): {unchanged_pct:.2f}%")

# ---------- 3) Insertion (KEEP) explainer that matches FULL LOGITS for one node ----------
# Greedy add edges from baseline h_w=0 until KL(p_full || p_sub) < eps

def softmax_row(logits, i):
    return torch.softmax(logits[i], dim=0)


def kl(p, q, eps=1e-12):
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    return torch.sum(p * (torch.log(p) - torch.log(q)))


def insertion_keep_logits(target_node=19, eps_kl=1e-3, max_steps=300):
    # full target distribution
    _, _, Lf, _, _ = forward_all(edge_weight, edge_weight_ho)
    p_full = softmax_row(Lf, target_node).detach()

    # start from empty HO edges
    w = torch.zeros_like(edge_weight_ho)
    kept = []

    for _step in range(max_steps):
        _, _, Ls, _, _ = forward_all(edge_weight, w)
        q = softmax_row(Ls, target_node)
        cur_kl = float(kl(p_full, q).item())
        if cur_kl < eps_kl:
            return kept, cur_kl

        # one gradient step: choose edge that most decreases KL if added
        w_var = w.clone().detach().requires_grad_(True)
        _, _, Ls2, _, _ = forward_all(edge_weight, w_var)
        q2 = softmax_row(Ls2, target_node)
        obj = kl(p_full, q2)  # want to minimize
        grad = torch.autograd.grad(obj, w_var)[0]

        # If we add edge e, delta w_e = +edge_weight_ho[e]
        score = (-grad) * edge_weight_ho  # negative grad means adding helps
        score[w > 0] = -1e18              # don't add already added
        e_star = int(torch.argmax(score).item())
        if score[e_star].item() <= 0:
            # no single edge helps under this local linear view
            break

        w[e_star] = edge_weight_ho[e_star]
        kept.append(e_star)

    return kept, cur_kl


kept, final_kl = insertion_keep_logits(target_node=19, eps_kl=1e-3, max_steps=300)
print("Insertion keep-by-logits for node 19:")
print("  kept edges:", len(kept), " final KL:", final_kl)
print("  first 30 kept edge IDs:", kept[:30])

# Lookup-table view: HO hidden states under self-loop-only pass (PCA)
import matplotlib.pyplot as plt


def _flatten_labels(y):
    if y is None:
        raise AttributeError("data.y is required to color the lookup-table view by class.")
    if y.dim() > 1:
        if y.size(-1) == 1:
            y = y.view(-1)
        else:
            y = y.argmax(dim=-1)
    return y.view(-1).long()


def _pca_2d(X):
    X = X.detach().float().cpu()
    X = X - X.mean(dim=0, keepdim=True)
    if X.size(1) < 2:
        return torch.cat([X, torch.zeros(X.size(0), 2 - X.size(1))], dim=1)

    q = min(2, X.size(0), X.size(1))
    _, _, V = torch.pca_lowrank(X, q=q)
    Z = X @ V[:, :q]
    if Z.size(1) < 2:
        Z = torch.cat([Z, torch.zeros(Z.size(0), 2 - Z.size(1))], dim=1)
    return Z


with torch.no_grad():
    H_self, _, _, _, _ = forward_all(edge_weight, torch.zeros_like(edge_weight_ho))

ho_src = bip_edge_index[0].long()
fo_dst = bip_edge_index[1].long()
ho_to_fo = torch.full((N_ho,), -1, dtype=torch.long, device=fo_dst.device)
ho_to_fo[ho_src] = fo_dst
if (ho_to_fo < 0).any():
    missing = int((ho_to_fo < 0).sum().item())
    raise RuntimeError(f"Missing FO mapping for {missing} HO nodes in bipartite_edge_index.")

y_nodes = _flatten_labels(data.y.detach().to(fo_dst.device))
y_ho = y_nodes[ho_to_fo].detach().cpu().numpy()

Z = _pca_2d(H_self).numpy()
classes = np.unique(y_ho)
palette = list(DEFAULT_CLASS_COLORS)

if render_all_plots:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for i, cls in enumerate(classes):
        mask = y_ho == cls
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            s=22,
            alpha=0.82,
            color=palette[i % len(palette)],
            label=f"class {int(cls)}",
        )

    ax.set_title("Lookup table in hidden space: HO self-loop-only pass (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6, color=EDGE_GRAY)
    ax.legend(title="Class", frameon=False, ncol=min(4, len(classes)))
    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"lookup_table_pca_{dataset_name}.pdf", bbox_inches="tight")
    plt.show()


# --- Post-hoc: How much is features vs neighbors? (Ablations) ---
import pandas as pd
from utils import clone_data

use_test_mask = False  # set True to restrict to test nodes
max_nodes_eval = None  # set to an int for quick runs

if use_test_mask and hasattr(data, "test_mask") and data.test_mask is not None:
    nodes = torch.where(data.test_mask)[0].detach().cpu().tolist()
else:
    nodes = list(range(int(data.num_nodes)))

if max_nodes_eval is not None and len(nodes) > int(max_nodes_eval):
    nodes = nodes[: int(max_nodes_eval)]

def _margin_for_class(row, cls):
    cls = int(cls)
    if row.numel() <= 1:
        return float(row[cls].item())
    other = torch.cat([row[:cls], row[cls + 1 :]], dim=0)
    return float((row[cls] - other.max()).item())

def _summarize_ablation(logits_full, logits_ab, nodes_idx):
    idx = torch.as_tensor(nodes_idx, dtype=torch.long, device=logits_full.device)
    lf = logits_full[idx]
    la = logits_ab[idx]

    pred_full = lf.argmax(dim=-1)
    pred_ab = la.argmax(dim=-1)
    agree = (pred_full == pred_ab).float().mean().item() * 100.0

    p_full = torch.softmax(lf, dim=-1)
    p_ab = torch.softmax(la, dim=-1)
    p_full_orig = p_full[torch.arange(p_full.size(0)), pred_full]
    p_ab_orig = p_ab[torch.arange(p_ab.size(0)), pred_full]
    delta_p = (p_ab_orig - p_full_orig).mean().item()

    margins_full = []
    margins_ab = []
    for i in range(lf.size(0)):
        cls = int(pred_full[i].item())
        margins_full.append(_margin_for_class(lf[i], cls))
        margins_ab.append(_margin_for_class(la[i], cls))
    delta_margin = float(torch.tensor(margins_ab).mean().item() - torch.tensor(margins_full).mean().item())

    return {
        "agree_pct": agree,
        "delta_p_orig_mean": delta_p,
        "delta_margin_mean": delta_margin,
    }

# Baseline logits
with torch.no_grad():
    logits_full = adapter.predict_logits(data)

# HO ablations (temporal_clusters check): remove HO edges/features.
rows = []

# 1) Remove HO edges only.
data_no_ho_edges = clone_data(data)
if hasattr(data_no_ho_edges, "edge_weights_higher_order") and data_no_ho_edges.edge_weights_higher_order is not None:
    data_no_ho_edges.edge_weights_higher_order = torch.zeros_like(data_no_ho_edges.edge_weights_higher_order)
with torch.no_grad():
    logits_no_ho_edges = adapter.predict_logits(data_no_ho_edges)
rows.append({"ablation": "no_ho_edges (zero edge_weights_higher_order)", **_summarize_ablation(logits_full, logits_no_ho_edges, nodes)})

# 2) Remove HO features only.
data_no_ho_feat = clone_data(data)
if hasattr(data_no_ho_feat, "x_h") and data_no_ho_feat.x_h is not None:
    data_no_ho_feat.x_h = torch.zeros_like(data_no_ho_feat.x_h)
elif hasattr(data_no_ho_feat, "x_higher_order") and data_no_ho_feat.x_higher_order is not None:
    data_no_ho_feat.x_higher_order = torch.zeros_like(data_no_ho_feat.x_higher_order)
with torch.no_grad():
    logits_no_ho_feat = adapter.predict_logits(data_no_ho_feat)
rows.append({"ablation": "no_ho_features (zero x_h)", **_summarize_ablation(logits_full, logits_no_ho_feat, nodes)})

# 3) Remove both HO edges and HO features.
data_no_ho_all = clone_data(data)
if hasattr(data_no_ho_all, "edge_weights_higher_order") and data_no_ho_all.edge_weights_higher_order is not None:
    data_no_ho_all.edge_weights_higher_order = torch.zeros_like(data_no_ho_all.edge_weights_higher_order)
if hasattr(data_no_ho_all, "x_h") and data_no_ho_all.x_h is not None:
    data_no_ho_all.x_h = torch.zeros_like(data_no_ho_all.x_h)
elif hasattr(data_no_ho_all, "x_higher_order") and data_no_ho_all.x_higher_order is not None:
    data_no_ho_all.x_higher_order = torch.zeros_like(data_no_ho_all.x_higher_order)
with torch.no_grad():
    logits_no_ho_all = adapter.predict_logits(data_no_ho_all)
rows.append({"ablation": "no_ho_edges_and_features", **_summarize_ablation(logits_full, logits_no_ho_all, nodes)})

# Keep previous two global checks as well.
data_feat = clone_data(data)
for attr in ["edge_weight", "edge_weights", "edge_weights_higher_order"]:
    if hasattr(data_feat, attr) and getattr(data_feat, attr) is not None:
        setattr(data_feat, attr, torch.zeros_like(getattr(data_feat, attr)))
with torch.no_grad():
    logits_feat = adapter.predict_logits(data_feat)
rows.append({"ablation": "feature_only (zero all edges)", **_summarize_ablation(logits_full, logits_feat, nodes)})

data_struct = clone_data(data)
if hasattr(data_struct, "x") and data_struct.x is not None:
    data_struct.x = torch.ones_like(data_struct.x)
if hasattr(data_struct, "x_h") and data_struct.x_h is not None:
    data_struct.x_h = torch.ones_like(data_struct.x_h)
elif hasattr(data_struct, "x_higher_order") and data_struct.x_higher_order is not None:
    data_struct.x_higher_order = torch.ones_like(data_struct.x_higher_order)
with torch.no_grad():
    logits_struct = adapter.predict_logits(data_struct)
rows.append({"ablation": "structure_only (ones all features)", **_summarize_ablation(logits_full, logits_struct, nodes)})

posthoc_df = pd.DataFrame(rows)
posthoc_df[["agree_pct", "delta_p_orig_mean", "delta_margin_mean"]] = posthoc_df[
    ["agree_pct", "delta_p_orig_mean", "delta_margin_mean"]
].round(4)
print("\nHO ablation summary:")
print(posthoc_df.to_string(index=False))
try:
    from IPython.display import display as _display  # type: ignore
except Exception:
    _display = None
if _display is not None:
    _display(posthoc_df)
posthoc_df.to_csv(PLOT_DIR / f"ho_ablation_summary_{dataset_name}.csv", index=False)

if render_all_plots:
    fig_tbl, ax_tbl = plt.subplots(figsize=(10, 0.6 + 0.42 * len(posthoc_df)))
    ax_tbl.axis("off")
    tbl = ax_tbl.table(
        cellText=posthoc_df.values,
        colLabels=posthoc_df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.25)
    plt.tight_layout()
    fig_tbl.savefig(PLOT_DIR / f"ho_ablation_summary_{dataset_name}.pdf", bbox_inches="tight")
    plt.show()
