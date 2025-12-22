from typing import Union
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
# ── Set ROOT_DIR to the repository root (not the tgnnexplainer package dir) ──
from pathlib import Path
import os, sys, subprocess, importlib

# Ensure vendored tgnnexplainer (under submodules) is importable as `tgnnexplainer`
_TGNN_VENDOR = Path(__file__).resolve().parents[3] / "submodules" / "explainer" / "tgnnexplainer"
if str(_TGNN_VENDOR) not in sys.path:
    sys.path.insert(0, str(_TGNN_VENDOR))

def resolve_repo_root() -> Path:
    # 1) Respect an env var if you prefer to set it explicitly
    for key in ("PROJECT_ROOT", "REPO_ROOT", "TIME_TO_EXPLAIN_ROOT"):
        if key in os.environ:
            return Path(os.environ[key]).expanduser().resolve()
    # 2) Ask git
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return Path(out)
    except Exception:
        pass
    # 3) Walk upwards looking for common repo markers
    here = Path.cwd().resolve()
    markers = [".git", "pyproject.toml", "setup.cfg", "setup.py"]
    for p in (here, *here.parents):
        if any((p / m).exists() for m in markers):
            return p
    # 4) Fallback: current working directory
    return here

REPO_ROOT = resolve_repo_root()

# Make sure the repo is importable (in case you run notebooks from subfolders)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ROOT_DIR = REPO_ROOT

print("Using REPO_ROOT / ROOT_DIR:", REPO_ROOT)


from time_to_explain.utils.graph import NeighborFinder
from tgnnexplainer.xgraph.dataset.tg_dataset import verify_dataframe_unify

class MarginalSubgraphDataset(Dataset):
    """ Collect pair-wise graph data to calculate marginal contribution. """
    def __init__(self, data, exclude_mask, include_mask, subgraph_build_func) -> object:
        self.num_nodes = data.num_nodes
        self.X = data.x
        self.edge_index = data.edge_index
        self.device = self.X.device

        self.label = data.y
        self.exclude_mask = torch.tensor(exclude_mask).type(torch.float32).to(self.device)
        self.include_mask = torch.tensor(include_mask).type(torch.float32).to(self.device)
        self.subgraph_build_func = subgraph_build_func

    def __len__(self):
        return self.exclude_mask.shape[0]

    def __getitem__(self, idx):
        exclude_graph_X, exclude_graph_edge_index = self.subgraph_build_func(self.X, self.edge_index, self.exclude_mask[idx])
        include_graph_X, include_graph_edge_index = self.subgraph_build_func(self.X, self.edge_index, self.include_mask[idx])
        exclude_data = Data(x=exclude_graph_X, edge_index=exclude_graph_edge_index)
        include_data = Data(x=include_graph_X, edge_index=include_graph_edge_index)
        return exclude_data, include_data
# def tgat_node_reindex(u: Union[int, np.array], i: Union[int, np.array], num_users: int):
#     u = u + 1
#     i = i + 1 + num_users
#     return u, i

def construct_tgat_neighbor_finder(df):
    verify_dataframe_unify(df)

    num_nodes = df['i'].max()
    adj_list = [[] for _ in range(num_nodes + 1)]
    for i in range(len(df)):
        user, item, time, e_idx = df.u[i], df.i[i], df.ts[i], df.e_idx[i]
        adj_list[user].append((item, e_idx, time))
        adj_list[item].append((user, e_idx, time))
    neighbor_finder = NeighborFinder(adj_list, uniform=False) # default 'uniform' is False

    return neighbor_finder
