import numpy as np
import torch
import torch.nn.functional as F

from utils import clone_data

# ----------------------------
# 1) File-based inputs (optional)
# ----------------------------
STATE_PATH = "model_state.pt"

G_EDGE_INDEX = "g_edge_index.pt"
G_EDGE_WEIGHT = "g_edge_weight.pt"
G_NODE_IDS = "g_node_ids.npy"          # not strictly needed here

G2_EDGE_INDEX = "g2_edge_index.pt"
G2_EDGE_WEIGHT = "g2_edge_weight.pt"
G2_NODE_IDS = "g2_node_ids.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Globals initialized only when using file-based mode.
state = None
g_edge_index = None
g_edge_weight = None
g2_edge_index = None
g2_edge_weight0 = None
g2_node_ids = None
N = None
N_ho = None
C = None
bip_edge_index = None


def init_from_files(
    *,
    state_path: str = STATE_PATH,
    g_edge_index_path: str = G_EDGE_INDEX,
    g_edge_weight_path: str = G_EDGE_WEIGHT,
    g2_edge_index_path: str = G2_EDGE_INDEX,
    g2_edge_weight_path: str = G2_EDGE_WEIGHT,
    g2_node_ids_path: str = G2_NODE_IDS,
    num_nodes: int = 30,
    device_override: torch.device | None = None,
):
    """Initialize global tensors from saved files (legacy script mode)."""
    global state, g_edge_index, g_edge_weight, g2_edge_index, g2_edge_weight0
    global g2_node_ids, N, N_ho, C, bip_edge_index, device

    if device_override is not None:
        device = device_override

    state = torch.load(state_path, map_location="cpu")  # OrderedDict of parameters
    # Move all params to device:
    for k in list(state.keys()):
        state[k] = state[k].to(device)

    g_edge_index = torch.load(g_edge_index_path, map_location="cpu").long().to(device)   # [2, E1]
    g_edge_weight = torch.load(g_edge_weight_path, map_location="cpu").float().to(device)  # [E1]

    g2_edge_index = torch.load(g2_edge_index_path, map_location="cpu").long().to(device) # [2, E2]
    g2_edge_weight0 = torch.load(g2_edge_weight_path, map_location="cpu").float().to(device)  # [E2]
    g2_node_ids = np.load(g2_node_ids_path)  # (N_ho, 2) pairs (u,v); CPU numpy

    N = int(num_nodes)
    N_ho = int(g2_node_ids.shape[0])
    C = int(state["lin.bias"].numel())  # number of classes

    # pathpyG.utils.dbgnn.generate_bipartite_edge_index(..., mapping="last")
    v_last = torch.tensor(g2_node_ids[:, 1], dtype=torch.long, device=device)
    bip_edge_index = torch.stack([torch.arange(N_ho, device=device), v_last], dim=0)

    return True


# ----------------------------
# 2) Helper: add missing self-loops (PyG behavior)
# ----------------------------
def add_missing_self_loops(edge_index: torch.Tensor,
                           edge_weight: torch.Tensor,
                           num_nodes: int,
                           fill_value: float = 1.0):
    """
    Mimics torch_geometric.utils.add_remaining_self_loops:
    add self-loops with weight fill_value only for nodes missing a self-loop.
    """
    row, col = edge_index
    mask = row == col
    has_loop = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
    has_loop[row[mask]] = True
    missing = (~has_loop).nonzero(as_tuple=False).view(-1)
    if missing.numel() == 0:
        return edge_index, edge_weight

    loops = torch.stack([missing, missing], dim=0)
    loop_w = torch.full((missing.numel(),), float(fill_value),
                        dtype=edge_weight.dtype, device=edge_weight.device)
    edge_index2 = torch.cat([edge_index, loops], dim=1)
    edge_weight2 = torch.cat([edge_weight, loop_w], dim=0)
    return edge_index2, edge_weight2


# ----------------------------
# 3) GCNConv forward (PyG-style) in pure torch
# ----------------------------
def gcn_layer_pyg_like(x: torch.Tensor,
                       edge_index: torch.Tensor,
                       edge_weight: torch.Tensor,
                       W: torch.Tensor,
                       b: torch.Tensor,
                       add_self_loops: bool = True) -> torch.Tensor:
    """
    Implements the same normalization as torch_geometric.nn.GCNConv when edge_index is Tensor.
    See torch_geometric.nn.conv.gcn_conv.gcn_norm.  (PyG source)
    """
    num_nodes = x.size(0)
    if add_self_loops:
        edge_index2, edge_weight2 = add_missing_self_loops(edge_index, edge_weight, num_nodes, fill_value=1.0)
    else:
        edge_index2, edge_weight2 = edge_index, edge_weight

    row, col = edge_index2[0], edge_index2[1]  # source=row, target=col in PyG flow="source_to_target"
    deg = torch.zeros(num_nodes, dtype=edge_weight2.dtype, device=edge_weight2.device)
    deg.scatter_add_(0, col, edge_weight2)

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    norm = deg_inv_sqrt[row] * edge_weight2 * deg_inv_sqrt[col]

    x_lin = x @ W.t()
    out = torch.zeros((num_nodes, x_lin.size(1)), dtype=x_lin.dtype, device=x_lin.device)
    out.index_add_(0, col, x_lin[row] * norm.unsqueeze(1))
    out = out + b
    return out


# ----------------------------
# 4) BipartiteGraphOperator (as in pathpyG)
# ----------------------------
def bipartite_operator_pathpyg(x_h: torch.Tensor,
                               x_fo: torch.Tensor,
                               bip_edge_index: torch.Tensor,
                               W1: torch.Tensor, b1: torch.Tensor,
                               W2: torch.Tensor, b2: torch.Tensor,
                               n_ho: int, n_fo: int) -> torch.Tensor:
    """
    Matches pathpyG.nn.dbgnn.BipartiteGraphOperator:
      forward: x = (lin1(x_h), lin2(x_fo)); propagate(...)
      message: return x_i + x_j
    With edge_index = [ho_idx, fo_idx] ("source ho -> target fo") per generate_bipartite_edge_index(mapping="last").
    """
    xh_t = x_h @ W1.t() + b1      # (n_ho, out)
    xf_t = x_fo @ W2.t() + b2     # (n_fo, out)

    src = bip_edge_index[0]  # ho indices
    dst = bip_edge_index[1]  # fo indices

    # sum_j x_j
    out = torch.zeros((n_fo, xh_t.size(1)), dtype=xh_t.dtype, device=xh_t.device)
    out.index_add_(0, dst, xh_t[src])

    # + sum_j x_i  = deg(dst) * x_i
    deg = torch.bincount(dst, minlength=n_fo).to(out.dtype).unsqueeze(1)
    out = out + deg * xf_t
    return out


def _require_initialized():
    if state is None or g_edge_index is None or g2_edge_index is None or g2_edge_weight0 is None or g2_node_ids is None:
        raise RuntimeError("counterfactual_margin globals not initialized. Call init_from_files(...) first.")


# ----------------------------
# 6) DBGNN logits forward (pure torch)
# ----------------------------
def dbgnn_logits(edge_weights_higher_order: torch.Tensor) -> torch.Tensor:
    _require_initialized()
    # Identity features (this matches your trained weight shapes: 30 and 557 input dims)
    x = torch.eye(N, device=device)
    x_h = torch.eye(N_ho, device=device)

    # First-order convolutions with ELU after each layer (matches pathpyG docs)
    x = F.elu(gcn_layer_pyg_like(
        x, g_edge_index, g_edge_weight,
        state["first_order_layers.0.lin.weight"], state["first_order_layers.0.bias"],
        add_self_loops=True
    ))
    x = F.elu(gcn_layer_pyg_like(
        x, g_edge_index, g_edge_weight,
        state["first_order_layers.1.lin.weight"], state["first_order_layers.1.bias"],
        add_self_loops=True
    ))

    # Higher-order convolutions with ELU after each layer
    x_h = F.elu(gcn_layer_pyg_like(
        x_h, g2_edge_index, edge_weights_higher_order,
        state["higher_order_layers.0.lin.weight"], state["higher_order_layers.0.bias"],
        add_self_loops=True
    ))
    x_h = F.elu(gcn_layer_pyg_like(
        x_h, g2_edge_index, edge_weights_higher_order,
        state["higher_order_layers.1.lin.weight"], state["higher_order_layers.1.bias"],
        add_self_loops=True
    ))

    # Bipartite message passing + ELU
    x = bipartite_operator_pathpyg(
        x_h, x, bip_edge_index,
        state["bipartite_layer.lin1.weight"], state["bipartite_layer.lin1.bias"],
        state["bipartite_layer.lin2.weight"], state["bipartite_layer.lin2.bias"],
        n_ho=N_ho, n_fo=N
    )
    x = F.elu(x)

    # Linear classifier
    logits = x @ state["lin.weight"].t() + state["lin.bias"]
    return logits


# ----------------------------
# 7) Margin and greedy deletion
# ----------------------------
def margin_for_node(logits: torch.Tensor, node: int, y_ref: int) -> torch.Tensor:
    """m = logit[y_ref] - max_{c != y_ref} logit[c]. Prediction changes once m < 0."""
    row = logits[node]
    mask = torch.ones(row.numel(), dtype=torch.bool, device=row.device)
    mask[y_ref] = False
    return row[y_ref] - row[mask].max()


@torch.no_grad()
def check_any_single_edge_flip(node: int) -> bool:
    """Exact certificate for k=1 (for this node): try all single-edge deletions."""
    base_logits = dbgnn_logits(g2_edge_weight0)
    y0 = int(base_logits[node].argmax().item())
    for e in range(g2_edge_weight0.numel()):
        w = g2_edge_weight0.clone()
        w[e] = 0.0
        y = int(dbgnn_logits(w)[node].argmax().item())
        if y != y0:
            return True
    return False


def greedy_drop_until_flip(node: int,
                           max_steps: int = 2000,
                           restrict_to_positive_score: bool = True):
    """
    Gradient-based greedy estimate:
    each step deletes the higher-order edge that (under 1st-order Taylor approx)
    most decreases the classification margin of the current predicted class.
    Returns: (deleted_edge_ids, orig_pred, new_pred, final_weights)
    """
    # Baseline
    with torch.no_grad():
        base_logits = dbgnn_logits(g2_edge_weight0)
        y0 = int(base_logits[node].argmax().item())

    weights = g2_edge_weight0.clone().detach()
    deleted = []
    deleted_mask = torch.zeros_like(weights, dtype=torch.bool)

    for step in range(max_steps):
        # Make weights differentiable for this step
        w = weights.clone().detach().requires_grad_(True)

        logits = dbgnn_logits(w)
        y = int(logits[node].argmax().item())
        if y != y0:
            return deleted, y0, y, weights

        m = margin_for_node(logits, node, y0)
        grad = torch.autograd.grad(m, w, retain_graph=False, create_graph=False)[0]

        # If we drop edge e: Δw_e = -w_e, so linearized margin decrease ≈ grad_e * w_e
        score = grad * w

        # Do not pick already deleted or already-zero edges
        score = score.clone()
        score[deleted_mask] = -1e18
        score[w <= 0] = -1e18
        if restrict_to_positive_score:
            score[score <= 0] = -1e18

        e_star = int(torch.argmax(score).item())
        if score[e_star].item() <= -1e17:
            # Greedy can't find a helpful edge under linearized score
            break

        weights[e_star] = 0.0
        deleted_mask[e_star] = True
        deleted.append(e_star)

    # No flip found
    return deleted, y0, y0, weights


@torch.no_grad()
def prune_to_1_minimal(node: int, deleted_edge_ids: list[int]):
    """
    Makes the found counterfactual 1-minimal:
    remove any edge from the set that isn't actually needed.
    """
    if not deleted_edge_ids:
        return deleted_edge_ids

    # Baseline label
    y0 = int(dbgnn_logits(g2_edge_weight0)[node].argmax().item())

    # Start from fully deleted
    w = g2_edge_weight0.clone()
    w[deleted_edge_ids] = 0.0
    y_cf = int(dbgnn_logits(w)[node].argmax().item())
    if y_cf == y0:
        return deleted_edge_ids  # not a valid counterfactual set

    kept = deleted_edge_ids.copy()
    # Try adding back edges one-by-one; if still flipped, edge was unnecessary
    for e in deleted_edge_ids:
        w_try = w.clone()
        w_try[e] = g2_edge_weight0[e]
        y_try = int(dbgnn_logits(w_try)[node].argmax().item())
        if y_try != y0:
            # Edge e is not needed
            w = w_try
            kept.remove(e)

    return kept


def greedy_drop_margin_order(
    *,
    adapter,
    data,
    node_idx: int,
    max_steps: int = 200,
    restrict_to_positive_score: bool = True,
    candidate_mask: torch.Tensor | None = None,
    stop_on_flip: bool = True,
):
    """Greedy margin-based edge drop order using the live model/graph.

    Returns (deleted_edge_ids, orig_pred, new_pred).
    """

    space = adapter.explain_space()
    if space.edge_weight_attr is None or not hasattr(data, space.edge_weight_attr):
        raise ValueError("edge weights are required for margin-based greedy drops")

    edge_weight_full = getattr(data, space.edge_weight_attr)
    edge_weight_full = edge_weight_full.detach().float().view(-1)
    if edge_weight_full.numel() == 0:
        return [], None, None

    with torch.no_grad():
        logits0 = adapter.predict_logits(data)
        orig_pred = int(logits0[int(node_idx)].argmax().item())

    model = adapter.model
    was_train = model.training
    model.eval()
    reqs = [p.requires_grad for p in model.parameters()]
    try:
        for p in model.parameters():
            p.requires_grad_(False)

        data_work = clone_data(data)
        weights = edge_weight_full.clone()
        deleted_mask = torch.zeros_like(weights, dtype=torch.bool)
        deleted = []

        if candidate_mask is not None:
            if candidate_mask.dtype != torch.bool:
                candidate_mask = candidate_mask != 0
            candidate_mask = candidate_mask.to(device=weights.device).view(-1)

        for _step in range(int(max_steps)):
            w = weights.clone().detach().requires_grad_(True)
            setattr(data_work, space.edge_weight_attr, w)
            if hasattr(adapter, "reset_caches"):
                try:
                    adapter.reset_caches()  # type: ignore[attr-defined]
                except Exception:
                    pass

            logits = model(data_work)
            row = logits[int(node_idx)]
            pred = int(row.argmax().item())
            if pred != orig_pred and bool(stop_on_flip):
                return deleted, orig_pred, pred

            if row.numel() <= 1:
                margin = row[int(orig_pred)]
            else:
                other = torch.cat([row[: int(orig_pred)], row[int(orig_pred) + 1 :]], dim=0)
                margin = row[int(orig_pred)] - other.max()

            grad = torch.autograd.grad(margin, w, retain_graph=False, create_graph=False)[0]
            score = grad * w
            score = score.clone()
            score[deleted_mask] = -1e18
            score[w <= 0] = -1e18
            if candidate_mask is not None:
                score[~candidate_mask] = -1e18
            if restrict_to_positive_score:
                score[score <= 0] = -1e18

            e_star = int(torch.argmax(score).item())
            if score[e_star].item() <= -1e17:
                break

            weights[e_star] = 0.0
            deleted_mask[e_star] = True
            deleted.append(e_star)

        # No flip found within max_steps
        return deleted, orig_pred, orig_pred
    finally:
        for p, r in zip(model.parameters(), reqs):
            p.requires_grad_(r)
        if was_train:
            model.train()


if __name__ == "__main__":
    init_from_files()
    target_node = 19  # change this
    print("Any single-edge flip for this node?", check_any_single_edge_flip(target_node))

    deleted, y0, y1, _ = greedy_drop_until_flip(target_node, max_steps=3000)
    deleted_min = prune_to_1_minimal(target_node, deleted)

    print(f"Node {target_node}: {y0} -> {y1}")
    print("Greedy k =", len(deleted))
    print("1-minimal k =", len(deleted_min))
    print("Edge IDs to drop (1-minimal):", deleted_min)
