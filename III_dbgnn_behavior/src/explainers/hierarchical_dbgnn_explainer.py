#!/usr/bin/env python3
"""hierarchical_dbgnn_explainer.py

Hierarchical post-hoc explainer for a De Bruijn Graph Neural Network (DBGNN) trained
on temporal data via a 2nd-order De Bruijn (higher-order) graph.

It produces a *two-level* explanation for a target original node v:

Level 1 (HO nodes):
    Rank incoming higher-order nodes q=(u,v) (i.e., first-order edges u->v that exist at least once)
    by how much removing q's contribution reduces the model's margin for v.

Level 2 (HO edges / causal transitions):
    For each selected HO node q=(u,v), restrict to a local k-hop neighborhood in the HO graph G^(2),
    and attribute q's embedding H_q to HO edges (transitions) using:
      - Integrated Gradients (IG) on an edge mask (default), or
      - a sparse mask-optimization objective (GNNExplainer-style) (optional)

This file is self-contained (pure PyTorch + NumPy). It expects the same tensors you provided:
    model_state.pt
    g_edge_index.pt, g_edge_weight.pt
    g2_edge_index.pt, g2_edge_weight.pt, g2_node_ids.npy

Usage (CLI):
    python hierarchical_dbgnn_explainer.py --node 19 --top_m 5 --k 2 --steps 50

Or import and use the class `HierarchicalDBGNNExplainer`.

Notes:
- Your model uses one-hot identity features (FO: I_N, HO: I_Nho). This explainer is faithful to that setup.
- Level 1 masking is done *at the bipartite aggregation step* (additive sum), so it is very fast and exact
  for that notion of “removing an incoming HO node”.
- Level 2 IG attributes how HO edges shape the HO embedding of a specific HO node.

"""


from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> float:
    """Cosine similarity between 1D tensors."""
    return float((a * b).sum().item() / ((a.norm().item() + eps) * (b.norm().item() + eps)))


def _margin_from_logits_row(logits_row: torch.Tensor, y_ref: int) -> torch.Tensor:
    """Margin = logit[y_ref] - max_{c!=y_ref} logit[c]."""
    mask = torch.ones(logits_row.numel(), dtype=torch.bool, device=logits_row.device)
    mask[y_ref] = False
    return logits_row[y_ref] - logits_row[mask].max()


@dataclass
class Level1Item:
    ho_node_id: int
    u: int
    v: int
    score_margin_drop: float
    margin_before: float
    margin_after: float


@dataclass
class Level2EdgeItem:
    edge_id: int
    src_ho: int
    dst_ho: int
    src_pair: Tuple[int, int]
    dst_pair: Tuple[int, int]
    score: float


class HierarchicalDBGNNExplainer:
    """Hierarchical explainer for the specific DBGNN architecture in model_state.pt."""

    def __init__(
        self,
        state: Dict[str, torch.Tensor],
        g_edge_index: torch.Tensor,
        g_edge_weight: torch.Tensor,
        g2_edge_index: torch.Tensor,
        g2_edge_weight: torch.Tensor,
        g2_node_ids: np.ndarray,
        device: Optional[torch.device] = None,
        cache_full_forward: bool = True,
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Move weights to device
        self.state = {k: v.to(self.device) for k, v in state.items()}

        # Graph tensors
        self.g_edge_index = g_edge_index.long().to(self.device)
        self.g_edge_weight = g_edge_weight.float().to(self.device)
        self.g2_edge_index = g2_edge_index.long().to(self.device)
        self.g2_edge_weight = g2_edge_weight.float().to(self.device)

        # HO node ids: array of shape [N_ho, 2] with pairs (u,v)
        self.g2_node_ids = torch.as_tensor(g2_node_ids, dtype=torch.long, device=self.device)

        # Infer sizes from weight shapes
        self.N = int(self.state["first_order_layers.0.lin.weight"].shape[1])
        self.N_ho = int(self.g2_node_ids.shape[0])

        # Bipartite mapping ("last"): HO node (u,v) maps to original node v
        self.v_last = self.g2_node_ids[:, 1].clone()  # [N_ho]
        self.bip_src = torch.arange(self.N_ho, device=self.device, dtype=torch.long)
        self.bip_dst = self.v_last  # [N_ho]
        self.deg_bip = torch.bincount(self.bip_dst, minlength=self.N).to(torch.float32)  # [N]

        # Build adjacency lists for HO neighborhoods (for BFS)
        src = self.g2_edge_index[0].detach().cpu().numpy()
        dst = self.g2_edge_index[1].detach().cpu().numpy()
        E = int(src.shape[0])

        self.ho_in_neighbors: List[List[int]] = [[] for _ in range(self.N_ho)]
        self.ho_out_neighbors: List[List[int]] = [[] for _ in range(self.N_ho)]
        self.ho_in_edges: List[List[int]] = [[] for _ in range(self.N_ho)]
        self.ho_out_edges: List[List[int]] = [[] for _ in range(self.N_ho)]

        for e in range(E):
            a = int(src[e])
            b = int(dst[e])
            self.ho_out_neighbors[a].append(b)
            self.ho_out_edges[a].append(e)
            self.ho_in_neighbors[b].append(a)
            self.ho_in_edges[b].append(e)

        self._cache: Dict[str, torch.Tensor] = {}
        if cache_full_forward:
            self._compute_and_cache_full_forward()

    @staticmethod
    def from_files(
        model_state_path: str,
        g_edge_index_path: str,
        g_edge_weight_path: str,
        g2_edge_index_path: str,
        g2_edge_weight_path: str,
        g2_node_ids_path: str,
        device: Optional[torch.device] = None,
        cache_full_forward: bool = True,
    ) -> "HierarchicalDBGNNExplainer":
        state = torch.load(model_state_path, map_location="cpu")
        g_edge_index = torch.load(g_edge_index_path, map_location="cpu")
        g_edge_weight = torch.load(g_edge_weight_path, map_location="cpu")
        g2_edge_index = torch.load(g2_edge_index_path, map_location="cpu")
        g2_edge_weight = torch.load(g2_edge_weight_path, map_location="cpu")
        g2_node_ids = np.load(g2_node_ids_path)

        return HierarchicalDBGNNExplainer(
            state,
            g_edge_index,
            g_edge_weight,
            g2_edge_index,
            g2_edge_weight,
            g2_node_ids,
            device=device,
            cache_full_forward=cache_full_forward,
        )

    # ----------------------------
    # GCN primitives (PyG-like)
    # ----------------------------
    def _add_missing_self_loops(
        self, edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int, fill_value: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index
        mask = row == col
        has_loop = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
        has_loop[row[mask]] = True
        missing = (~has_loop).nonzero(as_tuple=False).view(-1)
        if missing.numel() == 0:
            return edge_index, edge_weight

        loops = torch.stack([missing, missing], dim=0)
        loop_w = torch.full((missing.numel(),), float(fill_value), dtype=edge_weight.dtype, device=edge_weight.device)

        edge_index2 = torch.cat([edge_index, loops], dim=1)
        edge_weight2 = torch.cat([edge_weight, loop_w], dim=0)
        return edge_index2, edge_weight2

    def _gcn_layer(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, W: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        edge_index2, edge_weight2 = self._add_missing_self_loops(edge_index, edge_weight, num_nodes, 1.0)
        row, col = edge_index2

        deg = torch.zeros(num_nodes, dtype=edge_weight2.dtype, device=edge_weight2.device)
        deg.scatter_add_(0, col, edge_weight2)

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

        norm = deg_inv_sqrt[row] * edge_weight2 * deg_inv_sqrt[col]

        x_lin = x @ W.t()
        out = torch.zeros((num_nodes, x_lin.size(1)), dtype=x_lin.dtype, device=x_lin.device)
        out.index_add_(0, col, x_lin[row] * norm.unsqueeze(1))
        return out + b

    # ----------------------------
    # Forward pieces
    # ----------------------------
    def _forward_fo_embeddings(self, g_w: Optional[torch.Tensor] = None) -> torch.Tensor:
        g_w = self.g_edge_weight if g_w is None else g_w
        x = torch.eye(self.N, device=self.device)
        x = F.elu(
            self._gcn_layer(
                x,
                self.g_edge_index,
                g_w,
                self.state["first_order_layers.0.lin.weight"],
                self.state["first_order_layers.0.bias"],
            )
        )
        x = F.elu(
            self._gcn_layer(
                x,
                self.g_edge_index,
                g_w,
                self.state["first_order_layers.1.lin.weight"],
                self.state["first_order_layers.1.bias"],
            )
        )
        return x  # [N, 32]

    def _forward_ho_embeddings(self, h_w: Optional[torch.Tensor] = None) -> torch.Tensor:
        h_w = self.g2_edge_weight if h_w is None else h_w
        x_h = torch.eye(self.N_ho, device=self.device)
        x_h = F.elu(
            self._gcn_layer(
                x_h,
                self.g2_edge_index,
                h_w,
                self.state["higher_order_layers.0.lin.weight"],
                self.state["higher_order_layers.0.bias"],
            )
        )
        x_h = F.elu(
            self._gcn_layer(
                x_h,
                self.g2_edge_index,
                h_w,
                self.state["higher_order_layers.1.lin.weight"],
                self.state["higher_order_layers.1.bias"],
            )
        )
        return x_h  # [N_ho, 32]

    def _compute_and_cache_full_forward(self) -> None:
        """Compute and store the full forward pass pieces used by Level 1."""
        with torch.no_grad():
            X_fo = self._forward_fo_embeddings(self.g_edge_weight)  # [N,32]
            H = self._forward_ho_embeddings(self.g2_edge_weight)  # [N_ho,32]

            Z_h = H @ self.state["bipartite_layer.lin1.weight"].t() + self.state["bipartite_layer.lin1.bias"]  # [N_ho,8]
            Z_fo = X_fo @ self.state["bipartite_layer.lin2.weight"].t() + self.state["bipartite_layer.lin2.bias"]  # [N,8]

            out_ho = torch.zeros((self.N, Z_h.size(1)), dtype=Z_h.dtype, device=self.device)
            out_ho.index_add_(0, self.bip_dst, Z_h[self.bip_src])

            z_pre = out_ho + self.deg_bip.unsqueeze(1) * Z_fo  # [N,8]
            X_node = F.elu(z_pre)
            logits = X_node @ self.state["lin.weight"].t() + self.state["lin.bias"]  # [N,C]

        self._cache["X_fo"] = X_fo
        self._cache["H"] = H
        self._cache["Z_h"] = Z_h
        self._cache["Z_fo"] = Z_fo
        self._cache["out_ho"] = out_ho
        self._cache["z_pre"] = z_pre
        self._cache["logits"] = logits
        self._cache["pred"] = logits.argmax(dim=1)

    # ----------------------------
    # Level 1: incoming HO nodes (u,v)
    # ----------------------------
    def explain_level1(
        self,
        v: int,
        top_m: int = 10,
        method: str = "margin_drop",
        ig_steps: int = 50,
    ) -> Tuple[List[Level1Item], int, float, int]:
        """Explain original node v using incoming HO nodes (u,v).

        Returns:
            items: ranked list (top_m) of Level1Item
            y_ref: predicted class of v on the full graph
            margin0: margin of v on the full graph
            M: number of incoming HO nodes for v
        """
        if "logits" not in self._cache:
            self._compute_and_cache_full_forward()

        logits = self._cache["logits"]
        Z_h = self._cache["Z_h"]
        Z_fo = self._cache["Z_fo"]
        z_pre = self._cache["z_pre"]

        v = int(v)
        y_ref = int(logits[v].argmax().item())
        margin0 = float(_margin_from_logits_row(logits[v], y_ref).item())

        incoming = (self.v_last == v).nonzero(as_tuple=False).view(-1)  # HO node indices mapping to v
        M = int(incoming.numel())

        if method == "margin_drop":
            # Exact occlusion at the bipartite sum: subtract Z_h[q] for one incoming q
            z_pre_v = z_pre[v].detach()
            items: List[Level1Item] = []

            for q in incoming.tolist():
                q = int(q)
                z_new = z_pre_v - Z_h[q].detach()
                logits_new = (F.elu(z_new) @ self.state["lin.weight"].t()) + self.state["lin.bias"]
                margin_new = float(_margin_from_logits_row(logits_new, y_ref).item())
                drop = margin0 - margin_new
                u = int(self.g2_node_ids[q, 0].item())
                items.append(
                    Level1Item(
                        ho_node_id=q,
                        u=u,
                        v=v,
                        score_margin_drop=float(drop),
                        margin_before=float(margin0),
                        margin_after=float(margin_new),
                    )
                )

            items.sort(key=lambda it: it.score_margin_drop, reverse=True)
            return items[:top_m], y_ref, margin0, M

        if method == "ig":
            # IG on mask variables at the bipartite stage for incoming HO nodes of v.
            incoming_idx = incoming.to(self.device)
            M = int(incoming_idx.numel())

            Z_h_in = Z_h[incoming_idx].detach()  # [M, 8]
            base = (self.deg_bip[v] * Z_fo[v]).detach()  # [8]

            # Define logits(z) for a single node
            W_lin = self.state["lin.weight"]
            b_lin = self.state["lin.bias"]

            def logits_from_z(z: torch.Tensor) -> torch.Tensor:
                return F.elu(z) @ W_lin.t() + b_lin

            # Reference class based on full (m=1)
            logits_full = logits_from_z(base + Z_h_in.sum(dim=0))
            y_ref = int(torch.argmax(logits_full).item())
            margin0 = float(_margin_from_logits_row(logits_full, y_ref).item())

            ig = torch.zeros(M, device=self.device)

            for s in range(1, ig_steps + 1):
                alpha = s / ig_steps
                m = torch.full((M,), alpha, device=self.device, requires_grad=True)
                z = base + (m.unsqueeze(1) * Z_h_in).sum(dim=0)
                logits_v = logits_from_z(z)
                scalar = _margin_from_logits_row(logits_v, y_ref)
                grad = torch.autograd.grad(scalar, m, retain_graph=False, create_graph=False)[0]
                ig += grad / ig_steps

            ig = ig.detach().cpu().numpy()

            items = []
            for j, q in enumerate(incoming_idx.tolist()):
                q = int(q)
                u = int(self.g2_node_ids[q, 0].item())
                items.append(
                    Level1Item(
                        ho_node_id=q,
                        u=u,
                        v=v,
                        score_margin_drop=float(ig[j]),
                        margin_before=float(margin0),
                        margin_after=float("nan"),
                    )
                )
            # sort by absolute IG
            items.sort(key=lambda it: abs(it.score_margin_drop), reverse=True)
            return items[:top_m], y_ref, margin0, M

        raise ValueError("method must be 'margin_drop' or 'ig'")

    # ----------------------------
    # Neighborhood on HO graph
    # ----------------------------
    def k_hop_neighborhood_ho(self, ho_node_id: int, k: int = 2, direction: str = "both") -> Tuple[Sequence[int], Sequence[int]]:
        """Return HO node indices and HO edge IDs in a k-hop neighborhood around ho_node_id.

        direction:
            'in'   : follow incoming edges only
            'out'  : follow outgoing edges only
            'both' : treat as directed but expand both ways
        """
        q = int(ho_node_id)
        visited = {q}
        frontier = {q}

        for _ in range(k):
            new_frontier = set()
            for n in frontier:
                if direction in ("in", "both"):
                    for nb in self.ho_in_neighbors[n]:
                        if nb not in visited:
                            visited.add(nb)
                            new_frontier.add(nb)
                if direction in ("out", "both"):
                    for nb in self.ho_out_neighbors[n]:
                        if nb not in visited:
                            visited.add(nb)
                            new_frontier.add(nb)
            frontier = new_frontier
            if not frontier:
                break

        nodes_k = sorted(visited)

        # Induced edge set: edges whose src and dst are both in nodes_k
        src = self.g2_edge_index[0].detach().cpu().numpy()
        dst = self.g2_edge_index[1].detach().cpu().numpy()
        node_mask = np.zeros(self.N_ho, dtype=bool)
        node_mask[nodes_k] = True
        edges_k = np.where(node_mask[src] & node_mask[dst])[0].tolist()
        return nodes_k, edges_k

    # ----------------------------
    # Level 2: HO edges (u,v)->(v,w) explaining H(u,v)
    # ----------------------------
    def explain_level2(
        self,
        ho_node_id: int,
        k: int = 2,
        direction: str = "both",
        method: str = "ig",
        steps: int = 50,
        top_e: int = 20,
        restrict_to_neighborhood: bool = True,
        # optimization hyperparams (only if method='opt')
        opt_epochs: int = 200,
        opt_lr: float = 0.1,
        l1: float = 0.05,
        entropy: float = 0.1,
        seed: int = 0,
    ) -> Dict:
        """Explain HO node embedding H_q using HO edges in a local neighborhood.

        method:
            'ig'  : Integrated Gradients on a mask for neighborhood edges
            'opt' : Optimize a sparse continuous mask (GNNExplainer-style) to reconstruct H_q

        Returns a dict with metadata + list of top edges with scores.
        """
        q = int(ho_node_id)
        nodes_k, edges_k = self.k_hop_neighborhood_ho(q, k=k, direction=direction)

        if len(edges_k) == 0:
            return {
                "ho_node_id": q,
                "pair": tuple(self.g2_node_ids[q].detach().cpu().numpy().tolist()),
                "k": k,
                "direction": direction,
                "num_nodes_neigh": len(nodes_k),
                "num_edges_neigh": 0,
                "note": "No HO edges in neighborhood.",
            }

        edges_k_t = torch.tensor(edges_k, dtype=torch.long, device=self.device)

        # Global full embedding for fidelity reporting
        with torch.no_grad():
            H_full_global = self._forward_ho_embeddings(self.g2_edge_weight)
            h_full_q_global = H_full_global[q].detach()

        # Build baseline and full weights
        if restrict_to_neighborhood:
            # Keep only neighborhood edges; everything else weight=0
            w_full = torch.zeros_like(self.g2_edge_weight)
            w_full[edges_k_t] = self.g2_edge_weight[edges_k_t]
            w_base = torch.zeros_like(self.g2_edge_weight)  # baseline: none of these neighborhood edges
        else:
            # Keep outside edges fixed at full; vary only neighborhood edges
            w_full = self.g2_edge_weight.clone()
            w_base = self.g2_edge_weight.clone()
            w_base[edges_k_t] = 0.0

        # Compute baseline and full-local embeddings for q
        with torch.no_grad():
            H_base = self._forward_ho_embeddings(w_base)
            H_full_local = self._forward_ho_embeddings(w_full)
            h_base_q = H_base[q].detach()
            h_full_q_local = H_full_local[q].detach()

        cos_local_vs_global = _cosine(h_full_q_local, h_full_q_global)
        rel_L2_local_vs_global = float((h_full_q_local - h_full_q_global).norm().item() / (h_full_q_global.norm().item() + 1e-9))

        # If neighborhood edges do not affect this embedding, return early
        delta = (h_full_q_local - h_base_q).detach()
        if float(delta.norm().item()) < 1e-9:
            return {
                "ho_node_id": q,
                "pair": tuple(self.g2_node_ids[q].detach().cpu().numpy().tolist()),
                "k": k,
                "direction": direction,
                "num_nodes_neigh": len(nodes_k),
                "num_edges_neigh": len(edges_k),
                "cosine_local_vs_global": cos_local_vs_global,
                "rel_L2_local_vs_global": rel_L2_local_vs_global,
                "note": "Neighborhood edges do not change H_q (delta ~ 0).",
            }

        # Differences on the varying edges
        delta_w_sub = (w_full[edges_k_t] - w_base[edges_k_t]).detach()

        # Helper: build w(m) = w_base + scatter(edges_k, delta_w_sub * m)
        def build_weights(m: torch.Tensor) -> torch.Tensor:
            update = torch.zeros_like(w_base).scatter(0, edges_k_t, delta_w_sub * m)
            return w_base + update

        # Build edge descriptors for reporting
        src_ho_all = self.g2_edge_index[0, edges_k_t].detach().cpu().numpy()
        dst_ho_all = self.g2_edge_index[1, edges_k_t].detach().cpu().numpy()

        def edge_item(idx_in_list: int, score: float) -> Level2EdgeItem:
            eid = int(edges_k[idx_in_list])
            s_id = int(src_ho_all[idx_in_list])
            d_id = int(dst_ho_all[idx_in_list])
            uv = tuple(self.g2_node_ids[s_id].detach().cpu().numpy().tolist())
            vw = tuple(self.g2_node_ids[d_id].detach().cpu().numpy().tolist())
            return Level2EdgeItem(
                edge_id=eid,
                src_ho=s_id,
                dst_ho=d_id,
                src_pair=uv,
                dst_pair=vw,
                score=float(score),
            )

        if method == "ig":
            # Integrated Gradients on edge-mask variables m in [0,1]^{|E_k|}
            ig = torch.zeros(len(edges_k), device=self.device)

            for s in range(1, steps + 1):
                alpha = s / steps
                m = torch.full((len(edges_k),), alpha, device=self.device, requires_grad=True)
                w_alpha = build_weights(m)
                H_alpha = self._forward_ho_embeddings(w_alpha)
                h_alpha_q = H_alpha[q]
                # scalar aligned with actual embedding change direction
                scalar = ((h_alpha_q - h_base_q) * delta).sum()
                grad = torch.autograd.grad(scalar, m, retain_graph=False, create_graph=False)[0]
                ig += grad / steps

            ig = ig.detach().cpu().numpy()
            order = np.argsort(-np.abs(ig))

            top_items = [edge_item(int(j), float(ig[int(j)])) for j in order[:top_e]]
            return {
                "ho_node_id": q,
                "pair": tuple(self.g2_node_ids[q].detach().cpu().numpy().tolist()),
                "k": k,
                "direction": direction,
                "num_nodes_neigh": len(nodes_k),
                "num_edges_neigh": len(edges_k),
                "restrict_to_neighborhood": restrict_to_neighborhood,
                "cosine_local_vs_global": cos_local_vs_global,
                "rel_L2_local_vs_global": rel_L2_local_vs_global,
                "method": "ig",
                "steps": steps,
                "top_edges": [it.__dict__ for it in top_items],
            }

        if method == "opt":
            # Sparse mask optimization to reconstruct h_full_q_local from h_base_q
            torch.manual_seed(seed)
            # unconstrained params -> sigmoid -> mask in (0,1)
            s_param = torch.zeros(len(edges_k), device=self.device, requires_grad=True)
            opt = torch.optim.Adam([s_param], lr=opt_lr)

            target = h_full_q_local.detach()
            base = h_base_q.detach()

            for _ in range(opt_epochs):
                m = torch.sigmoid(s_param)
                w_m = build_weights(m)
                H_m = self._forward_ho_embeddings(w_m)
                h_m = H_m[q]

                # reconstruction loss
                recon = (h_m - target).pow(2).mean()
                # sparsity
                spars = m.mean()
                # entropy regularizer to push toward 0/1
                ent = -(m * torch.log(m + 1e-12) + (1 - m) * torch.log(1 - m + 1e-12)).mean()

                loss = recon + l1 * spars + entropy * ent

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            with torch.no_grad():
                m = torch.sigmoid(s_param).detach().cpu().numpy()

            order = np.argsort(-m)
            top_items = [edge_item(int(j), float(m[int(j)])) for j in order[:top_e]]

            return {
                "ho_node_id": q,
                "pair": tuple(self.g2_node_ids[q].detach().cpu().numpy().tolist()),
                "k": k,
                "direction": direction,
                "num_nodes_neigh": len(nodes_k),
                "num_edges_neigh": len(edges_k),
                "restrict_to_neighborhood": restrict_to_neighborhood,
                "cosine_local_vs_global": cos_local_vs_global,
                "rel_L2_local_vs_global": rel_L2_local_vs_global,
                "method": "opt",
                "opt_epochs": opt_epochs,
                "opt_lr": opt_lr,
                "l1": l1,
                "entropy": entropy,
                "top_edges": [it.__dict__ for it in top_items],
            }

        raise ValueError("method must be 'ig' or 'opt'")


# ----------------------------
# CLI
# ----------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--node", type=int, default=19, help="Target original node v")
    p.add_argument("--top_m", type=int, default=5, help="How many incoming HO nodes to show (Level 1)")
    p.add_argument("--k", type=int, default=2, help="k-hop neighborhood for Level 2")
    p.add_argument("--direction", type=str, default="both", choices=["in", "out", "both"], help="Neighborhood expansion direction")
    p.add_argument("--steps", type=int, default=50, help="IG steps for Level 2")
    p.add_argument("--top_e", type=int, default=10, help="How many HO edges to show per HO node (Level 2)")
    p.add_argument("--lvl1_method", type=str, default="margin_drop", choices=["margin_drop", "ig"], help="Level 1 scoring")
    p.add_argument("--lvl2_method", type=str, default="ig", choices=["ig", "opt"], help="Level 2 method")
    p.add_argument("--restrict", action="store_true", help="Restrict HO graph to the k-hop neighborhood (recommended)")
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    p.add_argument("--threads", type=int, default=None, help="Optional: torch.set_num_threads(threads)")
    p.add_argument("--out_json", type=str, default=None, help="Optional path to write explanation as JSON")

    # file paths (defaults match your filenames)
    p.add_argument("--model_state", type=str, default="model_state.pt")
    p.add_argument("--g_edge_index", type=str, default="g_edge_index.pt")
    p.add_argument("--g_edge_weight", type=str, default="g_edge_weight.pt")
    p.add_argument("--g2_edge_index", type=str, default="g2_edge_index.pt")
    p.add_argument("--g2_edge_weight", type=str, default="g2_edge_weight.pt")
    p.add_argument("--g2_node_ids", type=str, default="g2_node_ids.npy")

    args = p.parse_args()

    if args.threads is not None:
        torch.set_num_threads(int(args.threads))

    device = torch.device(args.device)

    expl = HierarchicalDBGNNExplainer.from_files(
        args.model_state,
        args.g_edge_index,
        args.g_edge_weight,
        args.g2_edge_index,
        args.g2_edge_weight,
        args.g2_node_ids,
        device=device,
        cache_full_forward=True,
    )

    # Level 1
    lvl1, y_ref, margin0, M = expl.explain_level1(
        v=args.node,
        top_m=args.top_m,
        method=args.lvl1_method,
    )

    print("\n=== Level 1: incoming HO nodes (u -> v) ===")
    print(f"Target node v={args.node} | predicted class={y_ref} | margin={margin0:.4f} | #incoming HO nodes={M}")
    for it in lvl1:
        print(f"HO_node_id={it.ho_node_id:4d}  incoming edge ({it.u} -> {it.v})  score={it.score_margin_drop:+.6f}  (margin_after={it.margin_after:.4f})")

    # Level 2
    print("\n=== Level 2: HO edges explaining each selected HO node embedding ===")
    lvl2_all = []
    for it in lvl1:
        info = expl.explain_level2(
            ho_node_id=it.ho_node_id,
            k=args.k,
            direction=args.direction,
            method=args.lvl2_method,
            steps=args.steps,
            top_e=args.top_e,
            restrict_to_neighborhood=args.restrict,
        )
        lvl2_all.append(info)

        pair = info["pair"]
        print(f"\nHO node {it.ho_node_id} corresponds to pair (u,v)={pair} | neighborhood nodes={info['num_nodes_neigh']} edges={info['num_edges_neigh']}")
        if "cosine_local_vs_global" in info:
            print(f"Fidelity (local vs global embedding of this HO node): cosine={info['cosine_local_vs_global']:.4f}, rel_L2={info['rel_L2_local_vs_global']:.4f}")
        if "top_edges" in info:
            for e in info["top_edges"]:
                print(f"  edge_id={e['edge_id']:4d}  {e['src_pair']} -> {e['dst_pair']}   score={e['score']:+.6f}")
        else:
            print("  (no edges returned)")
            if "note" in info:
                print("  note:", info["note"])

    if args.out_json:
        out = {
            "node": args.node,
            "pred_class": y_ref,
            "margin": margin0,
            "level1": [it.__dict__ for it in lvl1],
            "level2": lvl2_all,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote explanation JSON to {args.out_json}")


if __name__ == "__main__":
    main()
