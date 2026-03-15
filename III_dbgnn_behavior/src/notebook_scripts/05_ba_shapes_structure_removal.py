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
from viz.palette import EDGE_GRAY, EVENT_BLUE, SNAPSHOT_ORANGE

print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("Numpy:", np.__version__)
print("NetworkX:", nx.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

CLASS_COLORS = [EDGE_GRAY, EVENT_BLUE, SNAPSHOT_ORANGE, EVENT_BLUE]

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
PLOT_DIR = PROJECT_ROOT / "plots" / "ba_shapes_structure_removal_experiment"
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

# BA-Shapes graph parameters (from GNNExplainer paper)
NUM_BA_NODES = 300
BA_M = 5                 # edges to attach for each new BA node
NUM_HOUSES = 80
ADD_NOISE_EDGES = True
NOISE_RATIO = 0.10       # add int(NOISE_RATIO * N) random edges after motifs

# Node features: paper says "no features" -> use a constant vector for every node.
FEAT_DIM = 10             # 1 is enough; increase (e.g., 10) if you want.

# Train/val/test split (per class) — paper uses 80/10/10 for node classification
TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10
TEST_FRAC  = 0.10
assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9

# Model/training — paper trains for 1000 epochs with lr=0.001
HIDDEN_DIM = 20
NUM_LAYERS = 3           # common choice for BA-Shapes
DROPOUT = 0.0

LR = 1e-3
WEIGHT_DECAY = 0.0
EPOCHS = 5000
PATIENCE = 200           # early stopping patience on val accuracy

print("Planned total nodes:", NUM_BA_NODES + 5 * NUM_HOUSES)


def _house_edges(nodes5):
    """Return the 6 undirected edges of the 5-node house motif.

    Node ordering convention:
    - nodes5[0], nodes5[1] are the 'middle' (top corners of the square)
    - nodes5[2], nodes5[3] are the 'bottom' corners
    - nodes5[4] is the 'roof' node ('top')
    """
    a, b, c, d, e = nodes5
    edges = [
        (a, b),  # square
        (b, c),
        (c, d),
        (d, a),
        (a, e),  # roof connections
        (b, e),
    ]
    return edges

def generate_ba_shapes(
    num_ba_nodes=300,
    ba_m=5,
    num_houses=80,
    add_noise_edges=True,
    noise_ratio=0.10,
    seed=0,
    connect_house_node=0,        # which node inside each house connects to BA (0 or 1 are "middle" nodes)
    unique_connectors=True,      # sample distinct BA nodes for attachments (PyG does this)
):
    rng = np.random.default_rng(seed)

    # 1) Base BA graph
    G = nx.barabasi_albert_graph(n=num_ba_nodes, m=ba_m, seed=seed)

    # 2) Labels: 0 for BA nodes
    labels = [0] * num_ba_nodes

    motif_edges = set()
    connector_edges = set()
    noise_edges = set()

    # Pick attachment points on the BA graph:
    if unique_connectors:
        if num_houses > num_ba_nodes:
            raise ValueError("unique_connectors=True requires num_houses <= num_ba_nodes")
        attach_nodes = rng.permutation(num_ba_nodes)[:num_houses].tolist()
    else:
        attach_nodes = rng.integers(0, num_ba_nodes, size=num_houses).tolist()

    next_node = num_ba_nodes

    for i in range(num_houses):
        house_nodes = list(range(next_node, next_node + 5))
        G.add_nodes_from(house_nodes)

        # Add motif (house) edges
        he = _house_edges(house_nodes)
        G.add_edges_from(he)
        for u, v in he:
            motif_edges.add(tuple(sorted((u, v))))

        # Node labels inside house: [1,1,2,2,3]
        labels.extend([1, 1, 2, 2, 3])

        # Connect motif to BA graph
        ba_u = attach_nodes[i]
        house_v = house_nodes[connect_house_node]
        G.add_edge(ba_u, house_v)
        connector_edges.add(tuple(sorted((ba_u, house_v))))

        next_node += 5

    # 3) Add random perturbation edges (0.1 * N)
    if add_noise_edges:
        N = G.number_of_nodes()
        target = int(noise_ratio * N)

        # Add edges uniformly at random (avoid duplicates/self-loops)
        attempts = 0
        max_attempts = target * 50 + 1000
        while len(noise_edges) < target and attempts < max_attempts:
            u = int(rng.integers(0, N))
            v = int(rng.integers(0, N))
            attempts += 1
            if u == v:
                continue
            e = tuple(sorted((u, v)))
            if G.has_edge(*e):
                continue
            G.add_edge(*e)
            noise_edges.add(e)

        if len(noise_edges) < target:
            print(f"[warn] Only added {len(noise_edges)}/{target} noise edges (hit max_attempts).")

    y = torch.tensor(labels, dtype=torch.long)
    return G, y, motif_edges, connector_edges, noise_edges

G, y, motif_edges, connector_edges, noise_edges = generate_ba_shapes(
    num_ba_nodes=NUM_BA_NODES,
    ba_m=BA_M,
    num_houses=NUM_HOUSES,
    add_noise_edges=ADD_NOISE_EDGES,
    noise_ratio=NOISE_RATIO,
    seed=SEED,
    connect_house_node=0,
    unique_connectors=True,
)

N = G.number_of_nodes()
E = G.number_of_edges()
print(f"Graph: N={N}, E={E} (undirected)")
print("Class counts:", {int(c): int((y==c).sum()) for c in torch.unique(y)})
print(f"Motif edges (house-internal): {len(motif_edges)}")
print(f"Connector edges (BA ↔ house): {len(connector_edges)}")
print(f"Noise edges: {len(noise_edges)}")


def draw_subgraph(G, nodes, y, title="subgraph", seed=0, save_as=None):
    H = G.subgraph(nodes).copy()
    pos = nx.spring_layout(H, seed=seed)
    y_np = y[list(H.nodes())].cpu().numpy()
    node_colors = [CLASS_COLORS[int(c) % len(CLASS_COLORS)] for c in y_np]

    plt.figure(figsize=(6, 5))
    nx.draw_networkx_nodes(H, pos, node_size=350, node_color=node_colors)
    nx.draw_networkx_edges(H, pos, alpha=0.8, width=2, edge_color=EDGE_GRAY)
    nx.draw_networkx_labels(H, pos, font_size=10)
    plt.title(title)
    plt.axis("off")
    if save_as:
        savefig_pdf(save_as)
    plt.show()

# Pick the first house (nodes NUM_BA_NODES..NUM_BA_NODES+4)
house0 = list(range(NUM_BA_NODES, NUM_BA_NODES + 5))

# Find which BA node connects to house0[0]
ba_neighbors = [n for n in G.neighbors(house0[0]) if n < NUM_BA_NODES]
anchor = ba_neighbors[0] if len(ba_neighbors) else None

nodes_to_plot = house0 + ([anchor] if anchor is not None else [])
if render_all_plots:
    draw_subgraph(
        G,
        nodes_to_plot,
        y,
        title="One house motif + its BA attachment",
        seed=SEED,
        save_as="house_attachment_subgraph",
    )


def stratified_split(y: torch.Tensor, train_frac=0.8, val_frac=0.1, seed=0):
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

        train_idx = idx[:n_train]
        val_idx   = idx[n_train:n_train + n_val]
        test_idx  = idx[n_train + n_val:]

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

    return train_mask, val_mask, test_mask

train_mask, val_mask, test_mask = stratified_split(y, TRAIN_FRAC, VAL_FRAC, seed=SEED)

print("Split sizes:", int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum()))
print("Check disjoint:", bool(((train_mask & val_mask) | (train_mask & test_mask) | (val_mask & test_mask)).sum() == 0))


def graph_to_edge_index(G: nx.Graph):
    # Return directed edges (both directions) as torch.long tensor shape [2, 2E]
    edges = np.array(G.edges(), dtype=np.int64)
    if edges.size == 0:
        return torch.empty((2, 0), dtype=torch.long)
    u = edges[:, 0]
    v = edges[:, 1]
    row = np.concatenate([u, v])
    col = np.concatenate([v, u])
    edge_index = torch.tensor(np.stack([row, col], axis=0), dtype=torch.long)
    return edge_index

def build_gcn_norm_adj(edge_index: torch.Tensor, num_nodes: int, add_self_loops: bool = True):
    row, col = edge_index[0], edge_index[1]

    if add_self_loops:
        self_loops = torch.arange(num_nodes, dtype=torch.long)
        row = torch.cat([row, self_loops], dim=0)
        col = torch.cat([col, self_loops], dim=0)

    val = torch.ones(row.shape[0], dtype=torch.float32)

    A = torch.sparse_coo_tensor(
        indices=torch.stack([row, col], dim=0),
        values=val,
        size=(num_nodes, num_nodes)
    ).coalesce()

    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

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


X = torch.ones((N, FEAT_DIM), dtype=torch.float32)
print("X shape:", X.shape)


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        assert num_layers >= 2, "Use num_layers >= 2"

        self.dropout = dropout
        self.num_layers = num_layers

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_dim, hidden_dim, bias=True))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
        self.lins.append(nn.Linear(hidden_dim, out_dim, bias=True))

    def forward(self, X, A_norm):
        H = X
        for i, lin in enumerate(self.lins):
            H = torch.sparse.mm(A_norm, H)
            H = lin(H)
            if i != len(self.lins) - 1:
                H = F.relu(H)
                H = F.dropout(H, p=self.dropout, training=self.training)
        return H

def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    pred = logits.argmax(dim=1)
    return (pred[mask] == y[mask]).float().mean().item()

# Move tensors to device
X_d = X.to(device)
y_d = y.to(device)
train_mask_d = train_mask.to(device)
val_mask_d = val_mask.to(device)
test_mask_d = test_mask.to(device)
A_norm_d = A_norm.to(device)

model = GCN(in_dim=FEAT_DIM, hidden_dim=HIDDEN_DIM, out_dim=int(y.max().item())+1, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

print(model)


# Quick sanity diagnostic: majority-class baseline
counts = torch.bincount(y, minlength=int(y.max())+1)
maj_class = int(torch.argmax(counts).item())
maj_acc = (y == maj_class).float().mean().item()
print("Class counts:", counts.tolist())
print("Majority class:", maj_class)
print("Majority-class baseline accuracy:", maj_acc)

# After training, if your model is stuck, check what it predicts:
# pred = logits.argmax(dim=1)
# print("Predicted class distribution:", torch.bincount(pred.cpu(), minlength=int(y.max())+1).tolist())


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
    history["loss"].append(float(loss.item()))
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(va_acc)
    history["test_acc"].append(te_acc)

    # Early stopping on val accuracy
    improved = va_acc > best_val + 1e-6
    if improved:
        best_val = va_acc
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        patience_left = PATIENCE
    else:
        patience_left -= 1

    if epoch % 50 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d} | loss={loss.item():.4f} | train={tr_acc:.3f} | val={va_acc:.3f} | test={te_acc:.3f} | patience_left={patience_left}")


# restore best model
if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
with torch.no_grad():
    logits = model(X_d, A_norm_d)
    print("Final accuracies:")
    print("  Train:", accuracy(logits, y_d, train_mask_d))
    print("  Val:  ", accuracy(logits, y_d, val_mask_d))
    print("  Test: ", accuracy(logits, y_d, test_mask_d))


if render_all_plots:
    plt.figure(figsize=(7,4))
    plt.plot(history["epoch"], history["train_acc"], label="train_acc", color=EVENT_BLUE)
    plt.plot(history["epoch"], history["val_acc"], label="val_acc", color=SNAPSHOT_ORANGE)
    plt.plot(history["epoch"], history["test_acc"], label="test_acc", color=EDGE_GRAY)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy curves")
    plt.legend()
    savefig_pdf("accuracy_curves")
    plt.show()

    plt.figure(figsize=(7,4))
    plt.plot(history["epoch"], history["loss"], color=EVENT_BLUE)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training loss")
    savefig_pdf("loss_curve")
    plt.show()


@torch.no_grad()
def get_predictions(model, X, A_norm):
    model.eval()
    logits = model(X, A_norm)
    probs = F.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)
    conf = probs.max(dim=1).values
    return logits, probs, pred, conf

def per_class_acc(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    out = {}
    for c in torch.unique(y).tolist():
        m = mask & (y == c)
        if int(m.sum()) == 0:
            out[int(c)] = float("nan")
        else:
            out[int(c)] = (pred[m] == y[m]).float().mean().item()
    return out

def remove_edges(G: nx.Graph, edges_to_remove):
    H = G.copy()
    H.remove_edges_from(list(edges_to_remove))
    return H

# Baseline predictions on original graph
A0 = A_norm_d
_, _, pred0, conf0 = get_predictions(model, X_d, A0)

def run_variant(name, G_variant: nx.Graph, ref_pred: torch.Tensor):
    edge_index_v = graph_to_edge_index(G_variant)
    A_v = build_gcn_norm_adj(edge_index_v, N, add_self_loops=True).to(device)
    _, probs, pred, conf = get_predictions(model, X_d, A_v)

    out = {}
    out["name"] = name
    out["test_acc"] = (pred[test_mask_d] == y_d[test_mask_d]).float().mean().item()
    out["flip_rate_all"] = (pred != ref_pred).float().mean().item()
    out["flip_rate_test"] = (pred[test_mask_d] != ref_pred[test_mask_d]).float().mean().item()
    out["per_class_test_acc"] = per_class_acc(pred, y_d, test_mask_d)

    # house-only / BA-only
    house = (y_d > 0)
    ba = (y_d == 0)
    out["test_acc_house_nodes"] = (pred[test_mask_d & house] == y_d[test_mask_d & house]).float().mean().item()
    out["test_acc_ba_nodes"] = (pred[test_mask_d & ba] == y_d[test_mask_d & ba]).float().mean().item()

    return out, pred.detach().cpu(), probs.detach().cpu(), conf.detach().cpu()

variants = []

# 1) Original
variants.append(("original", G, []))

# 2) Remove motif edges (house internal)
G_no_motif = remove_edges(G, motif_edges)
variants.append(("no_motif_edges", G_no_motif, motif_edges))

# 3) Remove connector edges
G_no_conn = remove_edges(G, connector_edges)
variants.append(("no_connector_edges", G_no_conn, connector_edges))

# 4) Remove both motif + connectors
G_no_motif_conn = remove_edges(G_no_motif, connector_edges)
variants.append(("no_motif_and_connectors", G_no_motif_conn, list(motif_edges) + list(connector_edges)))

# 5) Remove all edges (self loops only -> empty edge list)
G_no_edges = nx.empty_graph(n=N)  # keeps nodes, removes all edges
variants.append(("no_edges", G_no_edges, "ALL"))

results = []
preds = {}
for name, Gv, removed in variants:
    r, pred, probs, conf = run_variant(name, Gv, ref_pred=pred0)
    results.append(r)
    preds[name] = pred
    print(f"{name:24s} | test_acc={r['test_acc']:.3f} | flip_all={r['flip_rate_all']:.3f} | flip_test={r['flip_rate_test']:.3f} | house_test_acc={r['test_acc_house_nodes']:.3f}")


# Pretty-print a compact summary table
import pandas as pd

rows = []
for r in results:
    row = {
        "variant": r["name"],
        "test_acc": r["test_acc"],
        "flip_rate_all": r["flip_rate_all"],
        "flip_rate_test": r["flip_rate_test"],
        "test_acc_house": r["test_acc_house_nodes"],
        "test_acc_ba": r["test_acc_ba_nodes"],
    }
    rows.append(row)

df = pd.DataFrame(rows)
df


# Per-class accuracies as a table (columns = class id)
classes = sorted(int(c) for c in torch.unique(y).tolist())
pc_rows = []
for r in results:
    d = {"variant": r["name"]}
    for c in classes:
        d[f"class_{c}"] = r["per_class_test_acc"].get(c, float("nan"))
    pc_rows.append(d)

pd.DataFrame(pc_rows)


if render_all_plots:
    plt.figure(figsize=(7,4))
    plt.bar(df["variant"], df["test_acc"], color=EVENT_BLUE)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("test accuracy (no retrain)")
    plt.title("Accuracy after edge removals")
    plt.tight_layout()
    savefig_pdf("accuracy_after_edge_removals")
    plt.show()

    plt.figure(figsize=(7,4))
    plt.bar(df["variant"], df["flip_rate_test"], color=SNAPSHOT_ORANGE)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("fraction of test nodes that flip")
    plt.title("Prediction flips vs original")
    plt.tight_layout()
    savefig_pdf("prediction_flips_vs_original")
    plt.show()


# Pick a test node that belongs to a house (y>0)
house_test_nodes = torch.where(test_mask & (y > 0))[0]
node_idx = int(house_test_nodes[0].item()) if len(house_test_nodes) else int(torch.where(test_mask)[0][0].item())

print("Chosen node:", node_idx, "true label:", int(y[node_idx]))

# Build A_norm for original and no-motif variants (on CPU for printing)
def A_from_G(G_):
    ei = graph_to_edge_index(G_)
    return build_gcn_norm_adj(ei, N, add_self_loops=True).to(device)

A_orig = A_from_G(G)
A_no_motif = A_from_G(G_no_motif)

with torch.no_grad():
    _, probs_o, pred_o, _ = get_predictions(model, X_d, A_orig)
    _, probs_m, pred_m, _ = get_predictions(model, X_d, A_no_motif)

p_o = probs_o[node_idx].detach().cpu().numpy()
p_m = probs_m[node_idx].detach().cpu().numpy()

print("Pred (original):", int(pred_o[node_idx].item()), "probs:", np.round(p_o, 4))
print("Pred (no motif):", int(pred_m[node_idx].item()), "probs:", np.round(p_m, 4))

if render_all_plots:
    plt.figure(figsize=(7,4))
    x = np.arange(len(p_o))
    plt.bar(x - 0.2, p_o, width=0.4, label="original", color=EVENT_BLUE)
    plt.bar(x + 0.2, p_m, width=0.4, label="no motif edges", color=SNAPSHOT_ORANGE)
    plt.xticks(x, [str(i) for i in range(len(p_o))])
    plt.ylabel("probability")
    plt.title(f"Node {node_idx}: probability shift after removing motif edges")
    plt.legend()
    savefig_pdf(f"node_{node_idx}_prob_shift_no_motif")
    plt.show()
