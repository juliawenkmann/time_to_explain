
# temporal_suite.py — MPS Edition (v2)
from __future__ import annotations
import dataclasses as dc
import enum
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import requests

try:
    import torch
except Exception:
    torch = None

from git import Repo

class Model(enum.Enum):
    TGN = "tgn"
    TGAT = "tgat"
    GRAPHMIXER = "graphmixer"

@dc.dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 200
    lr: float = 1e-3
    seed: int = 42
    device: str = "auto"   # 'auto' | 'mps' | 'cuda' | 'cpu'
    gpu_index: int = 0     # used when device == 'cuda'
    num_neighbors: int = 20
    heads: int = 2
    layers: int = 1
    time_dim: int = 100
    node_dim: int = 100
    drop_out: float = 0.1
    use_cached_subgraph: bool = True
    use_one_hot_nodes: bool = False
    ignore_edge_feats: bool = False

class TemporalSuite:
    def __init__(self, root: str):
        self.root = Path(os.path.expanduser(root))
        self.root.mkdir(parents=True, exist_ok=True)
        self.submodules = self.root / "submodules"
        self.resources = self.root / "resources"
        self.data_root = self.resources / "data"
        self.logs_root = self.resources / "models"
        for p in (self.submodules, self.data_root, self.logs_root):
            p.mkdir(parents=True, exist_ok=True)
        self.repos = {
            "tgn": ("https://github.com/twitter-research/tgn.git", self.submodules/"tgn"),
            "tgat": ("https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs.git", self.submodules/"tgat"),
            "graphmixer": ("https://github.com/CongWeilin/GraphMixer.git", self.submodules/"graphmixer"),
        }



    def train(self, model: Model, dataset: str, **overrides) -> Dict:
        cfg = TrainConfig(**{**dc.asdict(TrainConfig()), **overrides})
        ds = dataset.lower()
        ds_dir = self.data_root / ds
        if not ds_dir.exists():
            raise ValueError(f"Dataset '{ds}' not prepared/registered in {self.data_root}")
        device = self._resolve_device(cfg.device, cfg.gpu_index)
        env = os.environ.copy()
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = env.get("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        if model == Model.TGN:
            self._ensure_repo("tgn")
            return self._run_tgn(ds, cfg, device, env)
        if model == Model.TGAT:
            self._ensure_repo("tgat")
            return self._run_tgat(ds, cfg, device, env)
        if model == Model.GRAPHMIXER:
            self._ensure_repo("graphmixer")
            return self._run_graphmixer(ds, cfg, device, env)
        raise ValueError(model)
    
    # ---------- model runners ----------
    def _run_tgat(self, dataset: str, cfg: TrainConfig, device: str, env: Dict) -> Dict:
        repo = self._ensure_repo("tgat")
        ds_dir = self.data_root / dataset
        proc = repo / "processed"; proc.mkdir(exist_ok=True)

        csv_out = proc / f"ml_{dataset}.csv"
        edge_npy = proc / f"ml_{dataset}.npy"
        node_npy = proc / f"ml_{dataset}_node.npy"

        df = pd.read_csv(ds_dir/"events.csv")
        bip = (ds_dir/"bipartite.txt").read_text().strip() == "1"
        df2 = df.copy()
        if bip:
            max_src = df2["src"].max()
            df2["i"] = df2["dst"] + max_src + 1
            df2["u"] = df2["src"]
        else:
            df2["u"] = df2["src"]
            df2["i"] = df2["dst"]
        df2["ts"] = df2["ts"].astype(np.int64)
        df2["label"] = df2["label"].astype(int)
        df2["idx"] = np.arange(1, len(df2)+1)
        df2[["u","i","ts","label","idx"]].to_csv(csv_out, index=False)

        E = len(df2); N = int(max(df2["u"].max(), df2["i"].max()))
        ef_src = ds_dir/"edge_feat.npy"; nf_src = ds_dir/"node_feat.npy"
        if ef_src.exists():
            arr = np.load(ef_src); arr = arr[:,None] if arr.ndim==1 else arr
            Fe = arr.shape[1]; ef = np.zeros((E+1, Fe), dtype=np.float32); m = min(E+1, arr.shape[0]); ef[:m]=arr[:m]
            np.save(edge_npy, ef)
        else:
            np.save(edge_npy, np.zeros((E+1,1), dtype=np.float32))
        if nf_src.exists():
            arr = np.load(nf_src); arr = arr[:,None] if arr.ndim==1 else arr
            Fn = arr.shape[1]; nf = np.zeros((N+1, Fn), dtype=np.float32); m = min(N+1, arr.shape[0]); nf[:m]=arr[:m]
            np.save(node_npy, nf)
        else:
            np.save(node_npy, np.zeros((N+1,1), dtype=np.float32))

        # Patch for MPS
        if device == "mps":
            self._patch_device_in_files(repo, ["learn_edge.py","learn_node.py"])

        cmd = [
            sys.executable, "-u", str(repo/"learn_edge.py"),
            "-d", dataset, "--bs", str(cfg.batch_size),
            "--uniform", "--n_degree", str(cfg.num_neighbors),
            "--agg_method", "attn", "--attn_mode", "prod",
            "--n_head", str(cfg.heads),
            "--n_epoch", str(cfg.epochs),
            "--n_layer", str(cfg.layers),
            "--lr", str(cfg.lr),
            "--drop_out", str(cfg.drop_out),
            "--prefix", f"{dataset}_tgat",
        ]
        cmd += (["--gpu", str(cfg.gpu_index)] if device=="cuda" else ["--gpu","-1"])

        env2 = env.copy(); env2["PYTHONHASHSEED"] = str(cfg.seed)
        log_dir = self._make_run_dir("tgat", dataset)
        out = self._run(cmd, cwd=str(repo), env=env2, log_dir=log_dir)
        if out["returncode"] != 0:
            raise RuntimeError(f"TGAT failed; see logs at {log_dir}")
        out["log_dir"] = log_dir; out["notes"] = f"TGAT training on device={device}."
        return out

    def _run_tgn(self, dataset: str, cfg: TrainConfig, device: str, env: Dict) -> Dict:
        """
        Build TGAT-format files directly inside tgn/data so train_self_supervised.py can load:
           data/ml_{dataset}.csv, data/ml_{dataset}.npy, data/ml_{dataset}_node.npy
        """
        repo = self._ensure_repo("tgn")
        ds_dir = self.data_root / dataset
        data_dir = repo / "data"; data_dir.mkdir(exist_ok=True)

        # Build ml_* files from our events.csv (same logic as TGAT)
        csv_out = data_dir / f"ml_{dataset}.csv"
        edge_npy = data_dir / f"ml_{dataset}.npy"
        node_npy = data_dir / f"ml_{dataset}_node.npy"

        df = pd.read_csv(ds_dir/"events.csv")
        bip = (ds_dir/"bipartite.txt").read_text().strip() == "1"
        df2 = df.copy()
        if bip:
            max_src = df2["src"].max()
            df2["i"] = df2["dst"] + max_src + 1
            df2["u"] = df2["src"]
        else:
            df2["u"] = df2["src"]
            df2["i"] = df2["dst"]
        df2["ts"] = df2["ts"].astype(np.int64)
        df2["label"] = df2["label"].astype(int)
        df2["idx"] = np.arange(1, len(df2)+1)
        df2[["u","i","ts","label","idx"]].to_csv(csv_out, index=False)

        E = len(df2); N = int(max(df2["u"].max(), df2["i"].max()))
        ef_src = ds_dir/"edge_feat.npy"; nf_src = ds_dir/"node_feat.npy"
        if ef_src.exists():
            arr = np.load(ef_src); arr = arr[:,None] if arr.ndim==1 else arr
            Fe = arr.shape[1]; ef = np.zeros((E+1, Fe), dtype=np.float32); m = min(E+1, arr.shape[0]); ef[:m]=arr[:m]
            np.save(edge_npy, ef)
        else:
            np.save(edge_npy, np.zeros((E+1,1), dtype=np.float32))
        if nf_src.exists():
            arr = np.load(nf_src); arr = arr[:,None] if arr.ndim==1 else arr
            Fn = arr.shape[1]; nf = np.zeros((N+1, Fn), dtype=np.float32); m = min(N+1, arr.shape[0]); nf[:m]=arr[:m]
            np.save(node_npy, nf)
        else:
            np.save(node_npy, np.zeros((N+1,1), dtype=np.float32))

        # Patch for MPS
        if device == "mps":
            self._patch_device_in_files(repo, ["train_self_supervised.py","train_supervised.py"])

        # Train
        cmd = [
            sys.executable, "-u", str(repo/"train_self_supervised.py"),
            "-d", dataset,
            "--use_memory",
            "--n_head", str(cfg.heads),
            "--n_epoch", str(cfg.epochs),
            "--n_layer", str(cfg.layers),
            "--lr", str(cfg.lr),
            "--drop_out", str(cfg.drop_out),
            "--bs", str(cfg.batch_size),
            "--prefix", f"{dataset}_tgn",
        ]
        cmd += (["--gpu", str(cfg.gpu_index)] if device=="cuda" else ["--gpu","-1"])

        log_dir = self._make_run_dir("tgn", dataset)
        out = self._run(cmd, cwd=str(repo), env=env, log_dir=log_dir)
        if out["returncode"] != 0:
            raise RuntimeError(f"TGN failed; see logs at {log_dir}")
        out["log_dir"] = log_dir; out["notes"] = f"TGN training on device={device}."
        return out

    def _run_graphmixer(self, dataset: str, cfg: TrainConfig, device: str, env: Dict) -> Dict:
        repo = self._ensure_repo("graphmixer")
        ds_dir = self.data_root / dataset
        DATA = repo / "DATA" / dataset.upper()
        DATA.mkdir(parents=True, exist_ok=True)
        pd.read_csv(ds_dir/"events.csv")[["src","dst","ts","label"]].to_csv(DATA/"edges.csv", index=False)

        # Build extension if missing
        so_candidates = list(repo.glob("**/*.so")) + list(repo.glob("**/*.pyd"))
        if not so_candidates:
            out_build = self._run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=str(repo), env=env, log_dir=self._make_run_dir("graphmixer","build"))
            if out_build["returncode"] != 0:
                raise RuntimeError("GraphMixer build failed; see build log dir.")

        # Graph gen
        out_gen = self._run([sys.executable, str(repo/"gen_graph.py"), "--data", dataset.upper()], cwd=str(repo), env=env, log_dir=self._make_run_dir("graphmixer","gen"))
        if out_gen["returncode"] != 0:
            raise RuntimeError("GraphMixer gen_graph failed; see gen log dir.")

        # MPS -> prefer CPU for stability
        if device == "mps":
            self._patch_device_in_files(repo, ["train.py"], prefer_cpu_when_mps=True)

        cmd = [sys.executable, str(repo/"train.py"), "--data", dataset.upper(), "--num_neighbors", str(cfg.num_neighbors)]
        if cfg.use_cached_subgraph: cmd.append("--use_cached_subgraph")
        if cfg.use_one_hot_nodes:   cmd.append("--use_onehot_node_feats")
        if cfg.ignore_edge_feats:   cmd.append("--ignore_edge_feats")

        log_dir = self._make_run_dir("graphmixer", dataset)
        out = self._run(cmd, cwd=str(repo), env=env, log_dir=log_dir)
        if out["returncode"] != 0:
            raise RuntimeError(f"GraphMixer failed; see logs at {log_dir}")
        out["log_dir"] = log_dir; out["notes"] = f"GraphMixer training on device={'cpu' if device=='mps' else device}."
        return out

    # ---------- helpers ----------
    def _ensure_repo(self, key: str) -> Path:
        url, path = self.repos[key]
        if not path.exists():
            Repo.clone_from(url, path)
        return path

    def _resolve_device(self, requested: str, gpu_index: int) -> str:
        req = (requested or "auto").lower()
        if req == "cuda":
            return "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        if req == "mps":
            if torch and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if req == "cpu":
            return "cpu"
        # auto
        if torch and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        if torch and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _patch_device_in_files(self, repo: Path, files: List[str], prefer_cpu_when_mps: bool=False):
        helper = (
            "\n# --- [PATCH-MPS] device helper injected ---\n"
            "import torch\n"
            "def _patched_pick_device(args=None):\n"
            "    try:\n"
            "        has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()\n"
            "    except Exception:\n"
            "        has_mps = False\n"
            "    if has_mps and %s:\n"
            "        return torch.device('cpu')\n"
            "    if has_mps:\n"
            "        return torch.device('mps')\n"
            "    if torch.cuda.is_available():\n"
            "        gpu = getattr(args, 'gpu', 0)\n"
            "        try: gpu = int(gpu)\n"
            "        except Exception: gpu = 0\n"
            "        return torch.device(f'cuda:{gpu}')\n"
            "    return torch.device('cpu')\n"
        ) % ("True" if prefer_cpu_when_mps else "False")
        for rel in files:
            f = repo / rel
            if not f.exists(): continue
            txt = f.read_text()
            if "[PATCH-MPS]" in txt: continue
            if "import torch" not in txt:
                txt = "import torch\n" + txt
            lines = txt.splitlines()
            insert_at = 0
            for i, ln in enumerate(lines[:50]):
                if ln.strip().startswith(("import","from")):
                    insert_at = i + 1
            lines.insert(insert_at, helper)
            txt = "\n".join(lines)
            txt = re.sub(r"^\s*device\s*=\s*torch\.device\([^\n]*\)\s*$", "device = _patched_pick_device(args)", txt, flags=re.MULTILINE)
            (repo/rel).write_text(txt)

    def _run(self, cmd: List[str], cwd: Optional[str]=None, env: Optional[Dict]=None, log_dir: Optional[Path]=None) -> Dict:
        start = time.time()
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        stdout, stderr = p.communicate()
        dur = time.time() - start
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir/"stdout.txt").write_text(stdout)
            (log_dir/"stderr.txt").write_text(stderr)
            (log_dir/"cmd.json").write_text(json.dumps({"cmd": cmd, "cwd": str(cwd)}, indent=2))

        text = stdout + "\n" + stderr
        import re
        def _last_float(patterns):
            vals = []
            for ptn in patterns:
                vals += re.findall(ptn, text, flags=re.IGNORECASE)
            nums = []
            for v in vals:
                s = v if isinstance(v, str) else v[0]
                try:
                    f = float(s); nums.append(f)
                except: pass
            if not nums: return None
            in01 = [x for x in nums if 0.0 <= x <= 1.0]
            return (in01[-1] if in01 else nums[-1])

        ap  = _last_float([r"(?<!m)\bAP\b[^0-9]*([01](?:\.\d+)?)", r"Average\s*Precision[^0-9]*([01](?:\.\d+)?)"])
        auc = _last_float([r"\bAUC\b[^0-9]*([01](?:\.\d+)?)", r"ROC[-\s]*AUC[^0-9]*([01](?:\.\d+)?)"])

        notes = None
        if p.returncode != 0:
            notes = f"FAILED (code {p.returncode}). Check logs at {log_dir}."
        elif dur < 5 and ap is None and auc is None:
            notes = "Ended quickly with no metrics — inspect stdout/stderr."

        return {"returncode": p.returncode, "stdout": stdout, "stderr": stderr, "seconds": dur, "AP": ap, "AUC": auc, "notes": notes}

    def _make_run_dir(self, model: str, dataset: str) -> Path:
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = self.logs_root / f"{model}__{dataset}__{ts}"
        path.mkdir(parents=True, exist_ok=True)
        return path
