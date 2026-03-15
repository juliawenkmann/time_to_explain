# Optional installs (uncomment if needed)
# !pip -q install torch numpy networkx matplotlib

import sys, os, math, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from viz.palette import EDGE_GRAY, EVENT_BLUE, SNAPSHOT_ORANGE

print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("Numpy:", np.__version__)
print("NetworkX:", nx.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

CLASS_COLORS = [EVENT_BLUE, SNAPSHOT_ORANGE, EDGE_GRAY]

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
PLOT_DIR = PROJECT_ROOT / "plots" / "gnn_structure_removal_experiment"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
print("Saving plots to:", PLOT_DIR.resolve())
render_all_plots = bool(globals().get("render_all_plots", True))

def savefig_pdf(name: str):
    plt.savefig(PLOT_DIR / f"{name}.pdf", bbox_inches="tight")


# Reproducibility
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Graph (SBM) parameters
NUM_CLASSES = 3
N_PER_CLASS = 120     # total nodes = NUM_CLASSES * N_PER_CLASS
P_IN = 0.12           # within-class edge probability
P_OUT = 0.008         # between-class edge probability

# Train/val/test split (per class)
TRAIN_FRAC = 0.6
VAL_FRAC   = 0.2
TEST_FRAC  = 0.2
assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9

# Model/training
HIDDEN_DIM = 64
DROPOUT = 0.35
LR = 2e-2
WEIGHT_DECAY = 5e-4
EPOCHS = 500
PATIENCE = 40   # early stopping patience on val accuracy

print("Total nodes:", NUM_CLASSES * N_PER_CLASS)


def make_sbm_graph(num_classes: int, n_per_class: int, p_in: float, p_out: float, seed: int = 0):
    # Sizes of each block/community
    sizes = [n_per_class] * num_classes

    # SBM probability matrix
    p = [[p_in if i == j else p_out for j in range(num_classes)] for i in range(num_classes)]

    G = nx.stochastic_block_model(sizes, p, seed=seed, directed=False, selfloops=False)

    # Labels: contiguous blocks
    labels = []
    for c in range(num_classes):
        labels += [c] * n_per_class
    labels = torch.tensor(labels, dtype=torch.long)

    return G, labels

G, y = make_sbm_graph(NUM_CLASSES, N_PER_CLASS, P_IN, P_OUT, seed=SEED)
N = G.number_of_nodes()
E = G.number_of_edges()
print(f"SBM graph: N={N}, E={E}, avg degree={2*E/N:.2f}")

# One-hot node features: X = I_N
X = torch.eye(N, dtype=torch.float32)

# Quick sanity: within/between edge counts
def count_within_between_edges(G, y):
    within = 0
    between = 0
    for u, v in G.edges():
        if y[u].item() == y[v].item():
            within += 1
        else:
            between += 1
    return within, between

within, between = count_within_between_edges(G, y)
print("Edges within-class:", within)
print("Edges between-class:", between)
print("Within/between ratio:", within / max(1, between))


# Optional visualization (can be slow for big graphs)
def draw_graph(G, y, title="Graph", save_as=None):
    pos = nx.spring_layout(G, seed=SEED, k=1/math.sqrt(G.number_of_nodes()))
    y_np = y.cpu().numpy()
    node_colors = [CLASS_COLORS[int(c) % len(CLASS_COLORS)] for c in y_np]
    plt.figure(figsize=(7, 6))
    nx.draw_networkx_nodes(G, pos, node_size=35, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.6, edge_color=EDGE_GRAY)
    plt.title(title)
    plt.axis("off")
    if save_as:
        savefig_pdf(save_as)
    plt.show()

if render_all_plots:
    draw_graph(G, y, title="SBM graph (node color = class)", save_as="sbm_graph")


def stratified_split(y: torch.Tensor, train_frac=0.6, val_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    n = y.shape[0]
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)

    for c in torch.unique(y).tolist():
        idx = torch.where(y == c)[0].cpu().numpy()
        rng.shuffle(idx)
        n_c = len(idx)
        n_train = int(round(train_frac * n_c))
        n_val = int(round(val_frac * n_c))
        n_test = n_c - n_train - n_val

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train+n_val]
        test_idx = idx[n_train+n_val:]

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

    return train_mask, val_mask, test_mask

train_mask, val_mask, test_mask = stratified_split(y, TRAIN_FRAC, VAL_FRAC, seed=SEED)

print("Split sizes:", int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum()))
print("Train class counts:", torch.bincount(y[train_mask], minlength=NUM_CLASSES).tolist())
print("Val   class counts:", torch.bincount(y[val_mask], minlength=NUM_CLASSES).tolist())
print("Test  class counts:", torch.bincount(y[test_mask], minlength=NUM_CLASSES).tolist())


def graph_to_edge_index(G: nx.Graph):
    # Return directed edges (both directions) as torch.long tensor shape [2, 2E]
    edges = np.array(G.edges(), dtype=np.int64)
    if edges.size == 0:
        return torch.empty((2, 0), dtype=torch.long)
    u = edges[:, 0]
    v = edges[:, 1]
    # undirected -> add both directions
    row = np.concatenate([u, v])
    col = np.concatenate([v, u])
    edge_index = torch.tensor(np.stack([row, col], axis=0), dtype=torch.long)
    return edge_index

def build_gcn_norm_adj(edge_index: torch.Tensor, num_nodes: int, add_self_loops: bool = True):
    # edge_index: [2, E] directed
    row, col = edge_index[0], edge_index[1]

    if add_self_loops:
        self_loops = torch.arange(num_nodes, dtype=torch.long)
        row = torch.cat([row, self_loops], dim=0)
        col = torch.cat([col, self_loops], dim=0)

    # Values are initially 1
    val = torch.ones(row.shape[0], dtype=torch.float32)

    # Build sparse A_hat
    A = torch.sparse_coo_tensor(
        indices=torch.stack([row, col], dim=0),
        values=val,
        size=(num_nodes, num_nodes)
    ).coalesce()

    # Degree
    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # Normalize values: D^-1/2 * A * D^-1/2
    r, c = A.indices()
    v = A.values()
    v_norm = deg_inv_sqrt[r] * v * deg_inv_sqrt[c]

    A_norm = torch.sparse_coo_tensor(
        indices=torch.stack([r, c], dim=0),
        values=v_norm,
        size=(num_nodes, num_nodes)
    ).coalesce()
    return A_norm

edge_index = graph_to_edge_index(G)
A_norm = build_gcn_norm_adj(edge_index, N, add_self_loops=True)

print("A_norm nnz:", A_norm._nnz())


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.lin2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.dropout = dropout

    def forward(self, X, A_norm):
        # Layer 1: A_norm X W
        H = torch.sparse.mm(A_norm, X)
        H = self.lin1(H)
        H = F.relu(H)
        H = F.dropout(H, p=self.dropout, training=self.training)

        # Layer 2
        H = torch.sparse.mm(A_norm, H)
        H = self.lin2(H)
        return H

def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    pred = logits.argmax(dim=1)
    correct = (pred[mask] == y[mask]).float().mean().item()
    return correct

# Move tensors to device
X_d = X.to(device)
y_d = y.to(device)
train_mask_d = train_mask.to(device)
val_mask_d = val_mask.to(device)
test_mask_d = test_mask.to(device)
A_norm_d = A_norm.to(device)

model = GCN(in_dim=N, hidden_dim=HIDDEN_DIM, out_dim=NUM_CLASSES, dropout=DROPOUT).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

print(model)


best_val = -1.0
best_state = None
patience_left = PATIENCE

history = {"epoch": [], "loss": [], "train_acc": [], "val_acc": [], "test_acc": []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    opt.zero_grad()
    logits = model(X_d, A_norm_d)
    loss = F.cross_entropy(logits[train_mask_d], y_d[train_mask_d])
    loss.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_d, A_norm_d)
        tr_acc = accuracy(logits, y_d, train_mask_d)
        va_acc = accuracy(logits, y_d, val_mask_d)
        te_acc = accuracy(logits, y_d, test_mask_d)

    history["epoch"].append(epoch)
    history["loss"].append(loss.item())
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(va_acc)
    history["test_acc"].append(te_acc)

    # Early stopping
    if va_acc > best_val + 1e-4:
        best_val = va_acc
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        patience_left = PATIENCE
    else:
        patience_left -= 1

    if epoch % 25 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d} | loss {loss.item():.4f} | train {tr_acc:.3f} | val {va_acc:.3f} | test {te_acc:.3f}")

    if patience_left <= 0:
        print(f"Early stopping at epoch {epoch} (best val={best_val:.3f})")
        break

# Restore best model
model.load_state_dict(best_state)
model.to(device)

# Plot learning curves
if render_all_plots:
    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["loss"], color=EVENT_BLUE)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Loss curve")
    savefig_pdf("loss_curve")
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["train_acc"], label="train", color=EVENT_BLUE)
    plt.plot(history["epoch"], history["val_acc"], label="val", color=SNAPSHOT_ORANGE)
    plt.plot(history["epoch"], history["test_acc"], label="test", color=EDGE_GRAY)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy curves")
    savefig_pdf("accuracy_curves")
    plt.legend()
    plt.show()

# Final test acc on best val checkpoint
model.eval()
with torch.no_grad():
    logits = model(X_d, A_norm_d)
final_test_acc = accuracy(logits, y_d, test_mask_d)
print("Final test accuracy (original graph):", round(final_test_acc, 4))


@torch.no_grad()
def hidden_and_logits(model, X, A_norm):
    model.eval()
    H = torch.sparse.mm(A_norm, X)
    H = model.lin1(H)
    H = F.relu(H)
    H = F.dropout(H, p=model.dropout, training=False)
    logits = torch.sparse.mm(A_norm, H)
    logits = model.lin2(logits)
    return H.detach().cpu().numpy(), logits.detach().cpu().numpy()


def _project_pair(X_full, X_self, method: str):
    Xf = np.asarray(X_full, dtype=np.float32)
    Xs = np.asarray(X_self, dtype=np.float32)
    X = np.vstack([Xf, Xs])
    n = Xf.shape[0]
    method = method.lower()
    if method == "pca":
        Z = PCA(n_components=2).fit_transform(X)
    elif method == "tsne":
        Z = TSNE(n_components=2, random_state=SEED, init="pca", learning_rate=200.0, perplexity=30.0).fit_transform(X)
    else:
        raise ValueError(f"Unknown method: {method}")
    return Z[:n], Z[n:]


def _plot_full_vs_selfloop(Z_full, Z_self, y, title: str, save_name: str):
    y_np = y.detach().cpu().numpy()
    classes = np.unique(y_np)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)

    for c in classes:
        mask = y_np == c
        color = CLASS_COLORS[int(c) % len(CLASS_COLORS)]
        axes[0].scatter(Z_full[mask, 0], Z_full[mask, 1], s=14, alpha=0.8, color=color, label=f"class {int(c)}")
        axes[1].scatter(Z_self[mask, 0], Z_self[mask, 1], s=14, alpha=0.8, color=color, label=f"class {int(c)}")

    axes[0].set_title("Full graph")
    axes[1].set_title("Self-loop only")
    axes[0].set_xlabel("dim 1")
    axes[1].set_xlabel("dim 1")
    axes[0].set_ylabel("dim 2")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    fig.suptitle(title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(classes)))
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    savefig_pdf(save_name)
    plt.show()


# Embedding/logit projections: full graph vs self-loop-only graph
A_self = build_gcn_norm_adj(graph_to_edge_index(nx.empty_graph(n=N)), N, add_self_loops=True).to(device)
h_full, logits_full = hidden_and_logits(model, X_d, A_norm_d)
h_self, logits_self = hidden_and_logits(model, X_d, A_self)

Zf, Zs = _project_pair(h_full, h_self, method="pca")
if render_all_plots:
    _plot_full_vs_selfloop(Zf, Zs, y_d, title="PCA: hidden embeddings (full vs self-loop)", save_name="pca_full_vs_selfloop")

Zf, Zs = _project_pair(logits_full, logits_self, method="pca")
if render_all_plots:
    _plot_full_vs_selfloop(Zf, Zs, y_d, title="PCA: logits (full vs self-loop)", save_name="pca_logits_full_vs_selfloop")

Zf, Zs = _project_pair(h_full, h_self, method="tsne")
if render_all_plots:
    _plot_full_vs_selfloop(Zf, Zs, y_d, title="t-SNE: hidden embeddings (full vs self-loop)", save_name="tsne_full_vs_selfloop")


@torch.no_grad()
def inference_metrics(model, X, y, mask, A_norm, ref_pred=None):
    model.eval()
    logits = model(X, A_norm)
    probs = F.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)
    conf = probs.max(dim=1).values

    acc = (pred[mask] == y[mask]).float().mean().item()

    changed = None
    if ref_pred is not None:
        changed = (pred != ref_pred).float().mean().item()

    return {
        "acc": acc,
        "pred": pred.detach().cpu(),
        "conf": conf.detach().cpu(),
    } | ({"changed_frac": changed} if changed is not None else {})

def make_self_loop_graph(num_nodes: int):
    G0 = nx.Graph()
    G0.add_nodes_from(range(num_nodes))
    # no edges (we'll add self-loops in normalization)
    return G0

def remove_intra_class_edges(G: nx.Graph, y: torch.Tensor):
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v in G.edges():
        if y[u].item() != y[v].item():
            H.add_edge(u, v)
    return H

def degree_preserving_rewire(G: nx.Graph, seed: int = 0, swap_multiplier: int = 10):
    H = G.copy()
    m = H.number_of_edges()
    if m < 10:
        return H
    # Try degree-preserving swaps
    try:
        nx.double_edge_swap(H, nswap=swap_multiplier*m, max_tries=swap_multiplier*m*20, seed=seed)
    except Exception as e:
        print("Rewire warning:", e)
    return H

def eval_on_graph(G_mod: nx.Graph, name: str, ref_pred: torch.Tensor):
    edge_index_mod = graph_to_edge_index(G_mod)
    A_norm_mod = build_gcn_norm_adj(edge_index_mod, N, add_self_loops=True).to(device)
    out = inference_metrics(model, X_d, y_d, test_mask_d, A_norm_mod, ref_pred=ref_pred)
    return out, A_norm_mod

# Reference = original graph predictions
ref = inference_metrics(model, X_d, y_d, test_mask_d, A_norm_d)
ref_pred = ref["pred"]

results = {}

# 1) Original
results["original"] = {
    "acc": ref["acc"],
    "changed_frac": 0.0,
    "mean_conf": float(ref["conf"][test_mask].mean().item()),
}

# 2) No edges (self-loops only)
G_no_edges = make_self_loop_graph(N)
out_no, A_no = eval_on_graph(G_no_edges, "no_edges", ref_pred)
results["no_edges"] = {
    "acc": out_no["acc"],
    "changed_frac": out_no["changed_frac"],
    "mean_conf": float(out_no["conf"][test_mask].mean().item()),
}

# 3) Remove intra-class edges
G_inter_only = remove_intra_class_edges(G, y)
out_inter, A_inter = eval_on_graph(G_inter_only, "remove_intra", ref_pred)
results["remove_intra"] = {
    "acc": out_inter["acc"],
    "changed_frac": out_inter["changed_frac"],
    "mean_conf": float(out_inter["conf"][test_mask].mean().item()),
}

# 4) Degree-preserving rewiring
G_rewired = degree_preserving_rewire(G, seed=SEED + 1, swap_multiplier=8)
out_rw, A_rw = eval_on_graph(G_rewired, "rewired", ref_pred)
results["rewired"] = {
    "acc": out_rw["acc"],
    "changed_frac": out_rw["changed_frac"],
    "mean_conf": float(out_rw["conf"][test_mask].mean().item()),
}

results


import pandas as pd

df = pd.DataFrame(results).T
df = df[["acc", "changed_frac", "mean_conf"]]
df.index.name = "graph_variant"
df = df.sort_index()
try:
    from IPython.display import display as _display  # type: ignore
except Exception:
    _display = None
if _display is not None:
    _display(df)
else:
    print(df.to_string())

if render_all_plots:
    plt.figure(figsize=(7, 4))
    plt.bar(df.index, df["acc"], color=EVENT_BLUE)
    plt.ylabel("Test accuracy")
    plt.title("Accuracy vs graph variant")
    savefig_pdf("accuracy_vs_graph_variant")
    plt.xticks(rotation=20, ha="right")
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.bar(df.index, df["changed_frac"], color=SNAPSHOT_ORANGE)
    plt.ylabel("Fraction of nodes changed")
    plt.title("Prediction instability vs original")
    savefig_pdf("changed_frac_vs_original")
    plt.xticks(rotation=20, ha="right")
    plt.show()


def get_intra_class_edges(G: nx.Graph, y: torch.Tensor):
    intra = []
    for u, v in G.edges():
        if y[u].item() == y[v].item():
            intra.append((u, v))
    return intra

intra_edges = get_intra_class_edges(G, y)
print("Intra-class edges:", len(intra_edges))

def remove_fraction_of_intra_edges(G: nx.Graph, intra_edges, frac: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    k = int(round(frac * len(intra_edges)))
    remove_set = set(rng.choice(len(intra_edges), size=k, replace=False).tolist()) if k > 0 else set()
    H = G.copy()
    # remove selected intra edges
    for idx in remove_set:
        u, v = intra_edges[idx]
        if H.has_edge(u, v):
            H.remove_edge(u, v)
    return H

fractions = np.linspace(0.0, 1.0, 11)
accs = []
changeds = []

for f in fractions:
    Gf = remove_fraction_of_intra_edges(G, intra_edges, frac=float(f), seed=SEED+123)
    out_f, _ = eval_on_graph(Gf, name=f"remove_intra_{f:.1f}", ref_pred=ref_pred)
    accs.append(out_f["acc"])
    changeds.append(out_f["changed_frac"])

if render_all_plots:
    plt.figure(figsize=(7,4))
    plt.plot(fractions, accs, marker="o", color=EVENT_BLUE)
    plt.xlabel("Fraction of intra-class edges removed")
    plt.ylabel("Test accuracy")
    plt.title("Accuracy vs removing community structure")
    savefig_pdf("accuracy_vs_remove_intra")
    plt.ylim(0, 1.0)
    plt.grid(True, color=EDGE_GRAY, alpha=0.3)
    plt.show()

    plt.figure(figsize=(7,4))
    plt.plot(fractions, changeds, marker="o", color=SNAPSHOT_ORANGE)
    plt.xlabel("Fraction of intra-class edges removed")
    plt.ylabel("Fraction of nodes changed vs original")
    plt.title("Prediction changes vs removing community structure")
    savefig_pdf("changed_vs_remove_intra")
    plt.ylim(0, 1.0)
    plt.grid(True, color=EDGE_GRAY, alpha=0.3)
    plt.show()
