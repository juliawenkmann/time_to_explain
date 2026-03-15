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
from train import train_model, load_or_train

from models.dbgnn import DBGNNAdapter
from models.registry import get_model_builder
from viz.palette import EDGE_GRAY, EVENT_BLUE, SNAPSHOT_ORANGE


# --- Metrics helpers (macro Precision/Recall/F1) ---
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import torch

@torch.no_grad()
def predict_labels(model, data):
    """Return hard predictions (class ids) for all nodes."""
    model.eval()
    logits = model(data)
    return logits.argmax(dim=1)

def _macro_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute macro metrics on the set of labels present in y_true."""
    labels = np.unique(y_true)
    return dict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_macro=float(
            precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        recall_macro=float(
            recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        f1_macro=float(
            f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
    )

def evaluate_macro_metrics(model, data):
    """Return (train_metrics, test_metrics) dicts with macro P/R/F1 + accuracy."""
    y = data.y.detach().cpu().numpy()
    pred = predict_labels(model, data).detach().cpu().numpy()

    train_mask = getattr(data, "train_mask", None)
    test_mask = getattr(data, "test_mask", None)
    if train_mask is None or test_mask is None:
        raise ValueError("data.train_mask / data.test_mask are required for evaluation")

    train_mask = train_mask.detach().cpu().numpy().astype(bool)
    test_mask = test_mask.detach().cpu().numpy().astype(bool)

    # ignore unlabeled nodes if encoded as -1
    labeled = (y >= 0)
    train_idx = train_mask & labeled
    test_idx = test_mask & labeled

    train_metrics = _macro_scores(y[train_idx], pred[train_idx])
    test_metrics = _macro_scores(y[test_idx], pred[test_idx])
    return train_metrics, test_metrics

# Compatibility shim for existing training loop
def evaluate_balanced_accuracy(model, data):
    train_metrics, test_metrics = evaluate_macro_metrics(model, data)
    # Balanced accuracy == macro recall for multi-class
    return train_metrics["recall_macro"], test_metrics["recall_macro"]


# --- Config ---
seed = int(globals().get("seed", 42))

# ---------------------------------------------------------------------
# Dataset selection
# ---------------------------------------------------------------------
# Choose one of: "highschool", "office", "hospital", "temporal_clusters"
dataset_key = str(globals().get("dataset_key", "hospital"))

DATASET_PROFILES = {
    "highschool": dict(
        candidates=["highschool", "sp_high_school"],
        # sp_high_school contains multiple networks: proximity/diaries/survey/facebook
        dataset_kwargs={"network": "proximity"},
        target_attr="class",  # or "gender"
    ),
    "office": dict(
        candidates=["workplace", "sp_office"],
        dataset_kwargs={},
        target_attr="department",
    ),
    "hospital": dict(
        candidates=["hospital", "sp_hospital"],
        dataset_kwargs={},
        target_attr="status",
    ),
    "temporal_clusters": dict(
        candidates=["temporal_clusters"],
        dataset_kwargs={},
        target_attr=None,  # depends on your loader; set manually if needed
    ),
}

if dataset_key not in DATASET_PROFILES:
    raise KeyError(f"Unknown dataset_key={dataset_key!r}. Available: {sorted(DATASET_PROFILES)}")

profile = DATASET_PROFILES[dataset_key]

def _resolve_dataset_name(candidates):
    last_err = None
    for name in candidates:
        try:
            _ = get_dataset_loader(name)
            return name
        except Exception as e:
            last_err = e
    raise last_err

dataset_name = str(globals().get("dataset_name", _resolve_dataset_name(profile["candidates"])))

# Optional: override the label attribute (netzschleuder datasets)
target_attr_override = globals().get("target_attr_override", profile.get("target_attr", None))
# Example overrides:
# target_attr_override = "gender"  # for highschool
# target_attr_override = None      # use loader default

# Dataset-specific kwargs (override netzschleuder defaults if needed).
dataset_kwargs_override = globals().get("dataset_kwargs", None)
if dataset_kwargs_override is None:
    dataset_kwargs = dict(profile.get("dataset_kwargs", {}))
else:
    dataset_kwargs = dict(dataset_kwargs_override)

if target_attr_override is not None:
    dataset_kwargs["target_attr"] = str(target_attr_override)

print("Selected dataset:", dataset_key, "->", dataset_name)
print("dataset_kwargs:", dataset_kwargs)

# ---------------------------------------------------------------------
# Train/test split fraction for node classification datasets
# ---------------------------------------------------------------------
num_test = float(globals().get("num_test", 0.3))

# ---------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------
epochs = int(globals().get("epochs", 8000))
lr = float(globals().get("lr", 0.001))
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
run_name = str(globals().get("run_name", make_run_name(dataset_name, dataset_kwargs, model_name="dbgnn_node2vec")))
render_all_plots = bool(globals().get("render_all_plots", True))

# Plot output (project-level)
PLOT_DIR = ROOT / "plots" / "01_train_dbgnn"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Feature learning: Node2Vec (replaces one-hot node encodings)
# ---------------------------------------------------------------------
use_node2vec_features = bool(globals().get("use_node2vec_features", True))

# Node2Vec embedding dimensionality for first-order nodes (FO graph)
n2v_dim_fo = 64
# Node2Vec embedding dimensionality for higher-order nodes (HO / De Bruijn graph)
n2v_dim_ho = 64

# Random walk params (can be tuned; keep modest for faster notebooks)
n2v_walk_length_fo = 30
n2v_num_walks_fo = 10
n2v_walk_length_ho = 30
n2v_num_walks_ho = 10

# Node2Vec bias params (p=return, q=in-out). p=q=1.0 -> DeepWalk-like.
n2v_p = 1.0
n2v_q = 1.0

# Word2Vec/Skip-gram training params
n2v_window = 10
n2v_min_count = 1
n2v_batch_words = 128
n2v_workers = 1  # set >1 for speed (may reduce determinism)
n2v_directed = False  # set True if you want directed random walks

# --- Ensure Node2Vec dims match DBGNN input dims ---
# DBGNN expects input feature dims == hidden_dims[0]
input_dim = int(hidden_dims[0])
if n2v_dim_fo != input_dim or n2v_dim_ho != input_dim:
    print(f"[warn] Adjusting Node2Vec dims to match input_dim={input_dim}")
    n2v_dim_fo = input_dim
    n2v_dim_ho = input_dim


set_seed(seed)
device = get_device()
device


# --- Load dataset ---
# Note: netzschleuder loader in src applies the pandas string-attribute patch.
loader = get_dataset_loader(dataset_name)

# Some dataset aliases may not accept all keyword args (e.g. network=...).
# Try with dataset_kwargs first; if needed, drop unsupported keys and retry.
_kwargs = dict(dataset_kwargs)
while True:
    try:
        data, assets = loader(device=device, num_test=num_test, seed=seed, **_kwargs)
        break
    except TypeError as e:
        import re
        msg = str(e)
        m = re.search(r"unexpected keyword argument '([^']+)'", msg)
        if m:
            bad = m.group(1)
            if bad in _kwargs:
                print(f"[warn] Loader did not accept kwarg '{bad}'. Removing and retrying.")
                _kwargs.pop(bad)
                continue
        raise

# Keep the actually-used kwargs around for sanity prints and reproducibility.
dataset_kwargs = _kwargs

data

# --- Drop rare class (e.g., class 2) ---
# Only apply when target is gender
if str(target_attr_override) == "gender":
    drop_class = 2
    if hasattr(data, "y") and data.y is not None:
        y_np = data.y.detach().cpu().numpy()
        keep_mask = (y_np != drop_class)
        if keep_mask.sum() < len(y_np):
            # remap labels to be contiguous starting at 0
            kept_labels = y_np[keep_mask]
            uniq = sorted(set(int(x) for x in kept_labels.tolist()))
            remap = {old: new for new, old in enumerate(uniq)}
            y_new = np.array([remap[int(x)] for x in kept_labels], dtype=int)

            # apply mask to data tensors
            data.y = torch.tensor(y_new, device=data.y.device)
            if hasattr(data, "x") and data.x is not None:
                data.x = data.x[keep_mask]

            # update masks if present
            for mask_name in ["train_mask", "val_mask", "test_mask"]:
                if hasattr(data, mask_name) and getattr(data, mask_name) is not None:
                    mask = getattr(data, mask_name).detach().cpu().numpy().astype(bool)
                    setattr(data, mask_name, torch.tensor(mask[keep_mask], device=data.y.device))

            # update num_nodes if present
            if hasattr(data, "num_nodes"):
                data.num_nodes = int(keep_mask.sum())

            # remap FO indices in edge_index and bipartite_edge_index
            old_to_new = -np.ones(len(y_np), dtype=int)
            old_to_new[keep_mask] = np.arange(int(keep_mask.sum()))

            if hasattr(data, "edge_index") and data.edge_index is not None:
                ei = data.edge_index.detach().cpu().numpy()
                keep_e = keep_mask[ei[0]] & keep_mask[ei[1]]
                ei = ei[:, keep_e]
                ei = np.vstack([old_to_new[ei[0]], old_to_new[ei[1]]])
                data.edge_index = torch.tensor(ei, device=data.y.device, dtype=torch.long)
                if hasattr(data, "edge_weights") and data.edge_weights is not None:
                    data.edge_weights = data.edge_weights[torch.tensor(keep_e, device=data.edge_weights.device)]
                if hasattr(data, "edge_weight") and data.edge_weight is not None:
                    data.edge_weight = data.edge_weight[torch.tensor(keep_e, device=data.edge_weight.device)]

            if hasattr(data, "bipartite_edge_index") and data.bipartite_edge_index is not None:
                bi = data.bipartite_edge_index.detach().cpu().numpy()
                # bi shape [2, E]: [ho_idx, fo_idx]
                keep_bi = keep_mask[bi[1]]
                bi = bi[:, keep_bi]
                bi[1] = old_to_new[bi[1]]
                data.bipartite_edge_index = torch.tensor(bi, device=data.y.device, dtype=torch.long)

            # verification check
            new_labels = data.y.detach().cpu().numpy()
            assert drop_class not in new_labels, f"Class {drop_class} still present after filtering"
            print(f"Dropped class {drop_class}. New class set: {uniq} -> remapped to 0..{len(uniq)-1}.")
        else:
            print(f"Class {drop_class} not present; no drop.")
else:
    print("Skipping class-drop: target_attr_override is not 'gender'.")

# --- Train/test split sanity: ensure test contains all classes (stratified if needed) ---
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import torch

def _mask_to_np(mask):
    return mask.detach().cpu().numpy().astype(bool)

def ensure_stratified_split(data, *, num_test: float = 0.3, seed: int = 42, force: bool = False):
    """Ensure that every class appears in the test set (if possible)."""
    y = data.y.detach().cpu().numpy()
    N = int(getattr(data, "num_nodes", len(y)))
    idx = np.arange(N)

    labeled = (y >= 0)
    idx_lab = idx[labeled]
    y_lab = y[labeled]

    if idx_lab.size == 0:
        print("[split] No labeled nodes found (y < 0 everywhere). Skipping.")
        return

    all_classes = np.unique(y_lab)

    have_masks = (
        hasattr(data, "train_mask") and data.train_mask is not None and
        hasattr(data, "test_mask") and data.test_mask is not None
    )

    if have_masks and not force:
        test_mask_np = _mask_to_np(data.test_mask)
        test_classes = np.unique(y[test_mask_np & labeled])
        if set(test_classes.tolist()) == set(all_classes.tolist()):
            print("[split] Existing split already covers all classes; keeping it.")
            return
        missing = set(all_classes.tolist()) - set(test_classes.tolist())
        print(f"[split] Test set missing classes {sorted(missing)} -> recreating split.")

    try:
        train_idx, test_idx = train_test_split(
            idx_lab,
            test_size=float(num_test),
            random_state=int(seed),
            stratify=y_lab,
        )
        stratified = True
    except Exception as e:
        print("[split] Stratified split failed (falling back to random split):", repr(e))
        train_idx, test_idx = train_test_split(
            idx_lab,
            test_size=float(num_test),
            random_state=int(seed),
            stratify=None,
        )
        stratified = False

    train_mask = np.zeros(N, dtype=bool)
    test_mask = np.zeros(N, dtype=bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    data.train_mask = torch.tensor(train_mask, device=data.y.device)
    data.test_mask = torch.tensor(test_mask, device=data.y.device)

    def _counts(mask_np):
        yy = y[mask_np & labeled]
        return Counter(yy.tolist())

    print("[split] Done. Stratified:", stratified)
    print("[split] All   counts:", _counts(np.ones(N, dtype=bool)))
    print("[split] Train counts:", _counts(train_mask))
    print("[split] Test  counts:", _counts(test_mask))

ensure_stratified_split(data, num_test=num_test, seed=seed, force=False)


# --- Node2Vec features (replace one-hot encodings) ---
#
# We learn embeddings *separately* for:
#   (1) first-order nodes on the first-order graph  (data.edge_index)
#   (2) higher-order nodes on the higher-order graph (data.edge_index_higher_order)
#
# The learned embeddings overwrite:
#   data.x   (FO features)
#   data.x_h (HO features)

if use_node2vec_features:
    # Keep a copy of the original features (often one-hot) for reference/debugging.
    try:
        data.x_onehot = data.x.detach().clone()
    except Exception:
        pass
    if hasattr(data, "x_h"):
        try:
            data.x_h_onehot = data.x_h.detach().clone()
        except Exception:
            pass

    # Dependency: `node2vec` (pip-install if missing).
    try:
        from node2vec import Node2Vec
    except Exception as e:
        import sys, subprocess

        print("Installing node2vec ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "node2vec"])
        from node2vec import Node2Vec

    import networkx as nx

    def _edge_index_to_nx(edge_index, num_nodes: int, directed: bool = False):
        """Convert a (2,E) edge_index tensor to a NetworkX graph with nodes [0..num_nodes-1]."""
        ei = edge_index.detach().cpu().numpy()
        G = nx.DiGraph() if directed else nx.Graph()
        G.add_nodes_from(range(int(num_nodes)))
        if ei.size != 0:
            G.add_edges_from(ei.T.tolist())
        return G

    def _fit_node2vec(G, dimensions: int, walk_length: int, num_walks: int, p: float, q: float, seed: int):
        """Fit node2vec and return embedding matrix (num_nodes, dimensions) in node-index order."""
        if G.number_of_nodes() == 0:
            return np.zeros((0, int(dimensions)), dtype=np.float32)
        if G.number_of_edges() == 0:
            return np.zeros((G.number_of_nodes(), int(dimensions)), dtype=np.float32)

        try:
            n2v = Node2Vec(
                G,
                dimensions=int(dimensions),
                walk_length=int(walk_length),
                num_walks=int(num_walks),
                p=float(p),
                q=float(q),
                workers=int(n2v_workers),
                seed=int(seed),
                quiet=True,
            )
            w2v = n2v.fit(
                window=int(n2v_window),
                min_count=int(n2v_min_count),
                batch_words=int(n2v_batch_words),
                seed=int(seed),
            )
    
        except Exception as e:
            print("Node2Vec failed (falling back to zeros):", repr(e))
            return np.zeros((G.number_of_nodes(), int(dimensions)), dtype=np.float32)
        emb = np.zeros((G.number_of_nodes(), int(dimensions)), dtype=np.float32)
        for n in range(G.number_of_nodes()):
            # `node2vec` stores keys as strings by default (str(node))
            key = str(n)
            if key in w2v.wv:
                emb[n] = w2v.wv[key]
            else:
                # Fallback for isolated nodes: zeros
                emb[n] = 0.0
        return emb

    # ---- First-order embeddings ----
    G_fo = _edge_index_to_nx(data.edge_index, num_nodes=data.num_nodes, directed=bool(n2v_directed))
    emb_fo = _fit_node2vec(
        G_fo,
        dimensions=n2v_dim_fo,
        walk_length=n2v_walk_length_fo,
        num_walks=n2v_num_walks_fo,
        p=n2v_p,
        q=n2v_q,
        seed=seed,
    )
    data.x = torch.tensor(emb_fo, dtype=torch.float, device=device)

    # ---- Higher-order embeddings ----
    if hasattr(data, "edge_index_higher_order") and hasattr(data, "x_h"):
        num_ho_nodes = int(data.x_h.size(0))
        G_ho = _edge_index_to_nx(data.edge_index_higher_order, num_nodes=num_ho_nodes, directed=bool(n2v_directed))
        emb_ho = _fit_node2vec(
            G_ho,
            dimensions=n2v_dim_ho,
            walk_length=n2v_walk_length_ho,
            num_walks=n2v_num_walks_ho,
            p=n2v_p,
            q=n2v_q,
            seed=seed,
        )
        data.x_h = torch.tensor(emb_ho, dtype=torch.float, device=device)

    print(" Using Node2Vec features")
    print("  data.x   (FO) shape:", tuple(data.x.shape))
    if hasattr(data, "x_h"):
        print("  data.x_h (HO) shape:", tuple(data.x_h.shape))
else:
    print("  Using original node features (no Node2Vec).")

# --- Save dataset bundle for later inspection ---
# Saved *after* feature construction so it matches whatever features you train with.
from data.cache import save_dataset_bundle

save_dir = run_dir / str(run_name) / "dataset"
saved_to = save_dataset_bundle(
    out_dir=save_dir,
    data=data,
    assets=assets,
    extra_meta=dict(dataset_name=dataset_name, dataset_kwargs=dataset_kwargs),
)
print(f"Saved dataset bundle to: {saved_to}")



# --- Node2Vec-aware model builder (match input feature dims) ---
def build_dbgnn_adapter_with_features(*, data, assets, device, hidden_dims, p_dropout):
    from pathpyG.nn.dbgnn import DBGNN
    num_classes = int(data.y.unique().numel())
    num_features = (int(data.x.size(1)), int(data.x_h.size(1)))
    model = DBGNN(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dims=list(hidden_dims),
        p_dropout=float(p_dropout),
    ).to(device)
    return DBGNNAdapter(
        model=model,
        edge_index_attr="edge_index_higher_order",
        edge_weight_attr="edge_weights_higher_order",
    )


def train_model_with_features(cfg, *, data, assets, device):
    adapter = build_dbgnn_adapter_with_features(
        data=data, assets=assets, device=device, hidden_dims=cfg.hidden_dims, p_dropout=cfg.p_dropout
    )
    model = adapter.model
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_function = torch.nn.CrossEntropyLoss()
    losses = []
    train_ba_hist = []
    test_ba_hist = []
    for epoch in range(cfg.epochs):
        model.train()
        output = model(data)
        loss = loss_function(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(float(loss.detach().cpu().item()))
        train_ba, test_ba = evaluate_balanced_accuracy(model, data)
        train_ba_hist.append(train_ba)
        test_ba_hist.append(test_ba)
        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch}, Loss: {loss.detach().item():.4f}, Train balanced accuracy: {train_ba:.4f}, Test balanced accuracy: {test_ba:.4f}"
            )
    return adapter, {"losses": losses, "train_ba": train_ba_hist, "test_ba": test_ba_hist}

# --- Train model + build adapter ---
cfg = ExperimentConfig(
    dataset_name=dataset_name,
    model_name="dbgnn",
    dataset_kwargs=dataset_kwargs,
    run_dir=str(run_dir),
    run_name=str(run_name),
    seed=seed,
    epochs=epochs,
    lr=lr,
    p_dropout=p_dropout,
    hidden_dims=hidden_dims,
    num_test=num_test,
)

if train_from_scratch:
    if enable_fo_edge_drop or enable_force_ho_epochs:
        if use_node2vec_features:
            adapter = build_dbgnn_adapter_with_features(data=data, assets=assets, device=device, hidden_dims=cfg.hidden_dims, p_dropout=cfg.p_dropout)
        else:
            model_builder = get_model_builder(cfg.model_name)
            adapter = model_builder(data=data, assets=assets, device=device, hidden_dims=cfg.hidden_dims, p_dropout=cfg.p_dropout)
        model = adapter.model

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        loss_function = torch.nn.CrossEntropyLoss()

        losses = []
        train_ba_hist = []
        test_ba_hist = []
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
        if use_node2vec_features:
            adapter, train_info = train_model_with_features(cfg, data=data, assets=assets, device=device)
        else:
            adapter, train_info = train_model(cfg, data=data, assets=assets, device=device)
else:
    if use_node2vec_features:
        # Node2Vec features change input dims; always train from scratch here
        adapter, train_info = train_model_with_features(cfg, data=data, assets=assets, device=device)
    else:
        adapter = load_or_train(cfg, data=data, assets=assets, device=device)

model = adapter.model


# --- Plot training loss and accuracy curves ---
import matplotlib.pyplot as plt

train_info = locals().get("train_info", {})
losses = train_info.get("losses", [])
train_ba = train_info.get("train_ba", None)
test_ba = train_info.get("test_ba", None)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Loss
if losses:
    axes[0].plot(range(1, len(losses) + 1), losses, label="loss", color=EVENT_BLUE)
axes[0].set_title("Training loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")

# Accuracy (balanced)
if train_ba is not None and test_ba is not None:
    axes[1].plot(range(1, len(train_ba) + 1), train_ba, label="train", color=EVENT_BLUE)
    axes[1].plot(range(1, len(test_ba) + 1), test_ba, label="test", color=SNAPSHOT_ORANGE)
    axes[1].set_title("Balanced accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Balanced accuracy")
    axes[1].legend()
else:
    # fallback to a single-point accuracy if histories missing
    try:
        train_ba_last, test_ba_last = evaluate_balanced_accuracy(model, data)
        axes[1].scatter([1], [train_ba_last], label="train", color=EVENT_BLUE)
        axes[1].scatter([1], [test_ba_last], label="test", color=SNAPSHOT_ORANGE)
        axes[1].set_title("Balanced accuracy (final)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Balanced accuracy")
        axes[1].legend()
    except Exception as e:
        axes[1].text(0.5, 0.5, f"No accuracy history: {e}", ha="center")

plt.tight_layout()
safe_run_name = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(run_name))
if render_all_plots:
    fig.savefig(PLOT_DIR / f"{safe_run_name}_training_curves.pdf", bbox_inches="tight")
    plt.show()
else:
    plt.close(fig)
# Print max test balanced accuracy if available
if test_ba is not None and len(test_ba) > 0:
    print(f"Max test balanced accuracy: {max(test_ba):.4f}")


# --- Final evaluation metrics (macro Precision / Recall / F1) ---
train_metrics, test_metrics = evaluate_macro_metrics(model, data)

def _print_metrics(name, m):
    print(
        f"{name}: "
        f"acc={m['accuracy']:.4f}  "
        f"precision_macro={m['precision_macro']:.4f}  "
        f"recall_macro={m['recall_macro']:.4f}  "
        f"f1_macro={m['f1_macro']:.4f}"
    )

_print_metrics("Train", train_metrics)
_print_metrics("Test ", test_metrics)


# ---------------------------------------------------------------------
# Sanity checks: show label names (if available) and a few predictions
# ---------------------------------------------------------------------
import numpy as np
import torch

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

def _resolve_id_to_name_map(assets):
    """Best-effort: build mapping {class_id(int) -> class_name(str)} from assets."""
    if not isinstance(assets, dict):
        return None

    # Common keys we try in order:
    candidate_keys = [
        "class_names",
        "label_names",
        "id_to_label",
        "id_to_class",
        "y_map",
        "label_map",
        "class_map",
        "classes",
    ]
    for k in candidate_keys:
        if k not in assets or assets[k] is None:
            continue
        obj = assets[k]

        # list/tuple: index is class id
        if isinstance(obj, (list, tuple)):
            return {int(i): str(obj[i]) for i in range(len(obj))}

        # dict: could be {id->name} or {name->id}
        if isinstance(obj, dict) and len(obj) > 0:
            # If keys are ints, assume {id->name}
            if all(isinstance(key, (int, np.integer)) for key in obj.keys()):
                return {int(key): str(val) for key, val in obj.items()}

            # If values are ints, assume {name->id} and reverse
            if all(isinstance(val, (int, np.integer)) for val in obj.values()):
                return {int(val): str(key) for key, val in obj.items()}

    return None

id_to_name = _resolve_id_to_name_map(assets)

def _pretty_label(cid: int) -> str:
    cid = int(cid)
    if id_to_name is None:
        return str(cid)
    return f"{cid} ({id_to_name.get(cid, 'UNKNOWN')})"

print("Dataset key:", dataset_key)
print("Dataset name:", dataset_name)
print("Target attr override:", target_attr_override)
print("Num nodes:", int(getattr(data, 'num_nodes', data.y.size(0))))
print("Num classes:", int(data.y.unique().numel()))

# Quick introspection: what did the loader attach?
try:
    print("data keys:", list(data.keys))
except Exception:
    try:
        print("data keys:", data.keys())
    except Exception as e:
        print("Could not read data keys:", e)

if isinstance(assets, dict):
    print("assets keys:", sorted(list(assets.keys())))
else:
    print("assets type:", type(assets))

if id_to_name is not None:
    print("Class id -> name mapping (from assets):")
    for cid in sorted(id_to_name):
        print(f"  {cid}: {id_to_name[cid]}")
else:
    print("No class-name mapping found in assets; showing numeric class IDs.")

# Label histograms
y_np = data.y.detach().cpu().numpy()
labeled = (y_np >= 0)

def _hist(mask: torch.Tensor):
    m = mask.detach().cpu().numpy().astype(bool) & labeled
    yy = y_np[m]
    u, c = np.unique(yy, return_counts=True)
    return [( _pretty_label(int(ui)), int(ci) ) for ui, ci in zip(u, c)]

print("\nTrain label counts:")
for name, c in _hist(data.train_mask):
    print(f"  {name}: {c}")

print("\nTest label counts:")
for name, c in _hist(data.test_mask):
    print(f"  {name}: {c}")

@torch.no_grad()
def _predict_with_probs(model, data):
    model.eval()
    logits = model(data)
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)
    conf = probs.max(dim=1).values
    return probs, pred, conf

probs, pred, conf = _predict_with_probs(model, data)

def _sample_from_mask(mask: torch.Tensor, k: int, *, rng_seed: int):
    idx = torch.where(mask)[0].detach().cpu().numpy()
    if idx.size == 0:
        return []
    rng = np.random.default_rng(int(rng_seed))
    k = int(min(k, idx.size))
    return rng.choice(idx, size=k, replace=False).tolist()

def _topk_str(i: int, k: int = 3) -> str:
    p = probs[i].detach().cpu().numpy()
    top_idx = np.argsort(-p)[:k]
    parts = []
    for j in top_idx:
        parts.append(f"{_pretty_label(int(j))}: {p[j]:.3f}")
    return ", ".join(parts)

def _make_rows(split_name: str, mask: torch.Tensor, k: int, *, rng_seed: int):
    rows = []
    for i in _sample_from_mask(mask, k, rng_seed=rng_seed):
        yt = int(data.y[i].item())
        yp = int(pred[i].item())
        rows.append(dict(
            split=split_name,
            node=int(i),
            y_true=_pretty_label(yt),
            y_pred=_pretty_label(yp),
            correct=bool(yt == yp),
            p_pred=float(conf[i].item()),
            top3=_topk_str(int(i), k=3),
        ))
    return rows

rows = []
rows += _make_rows("train", data.train_mask, k=10, rng_seed=seed)
rows += _make_rows("test",  data.test_mask,  k=10, rng_seed=seed + 1)

if _HAS_PANDAS:
    df = pd.DataFrame(rows).sort_values(["split", "correct", "p_pred"], ascending=[True, True, False])
    try:
        from IPython.display import display as _display  # type: ignore
    except Exception:
        _display = None
    if _display is not None:
        _display(df)
    else:
        print(df.to_string(index=False))
else:
    for r in rows:
        print(r)


import pathpyG as pp

g2 = getattr(assets, "g2", None)
print(type(g2), g2)

# quick plot (matplotlib backend)
if render_all_plots:
    layout = pp.layout(g2, layout="Fruchterman-Reingold", seed=1, k=0.5,
    iterations=300)
    pp.plot(g2, backend="matplotlib", layout=layout, edge_size=0.5, node_size=3,
            edge_color=EDGE_GRAY, node_color=EVENT_BLUE, show_labels=True);
