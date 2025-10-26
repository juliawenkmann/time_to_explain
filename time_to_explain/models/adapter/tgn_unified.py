# time_to_explain/models/adapter/tgn_unified.py
from __future__ import annotations
import torch
import numpy as np
from typing import Optional
from time_to_explain.models.adapter.connector import TGNNWrapper
from time_to_explain.data.data import ContinuousTimeDynamicGraphDataset
from time_to_explain.setup.utils import ProgressBar
from submodules.models.tgn.model.tgn import TGN
from submodules.models.tgn.utils.utils import get_neighbor_finder
from submodules.models.tgn.utils.data_processing import compute_time_statistics
from time_to_explain.models.adapter.tgn import to_data_object  # reuse your helper

class TGNWrapper(TGNNWrapper):
    """
    Unified adapter that can emulate:
      - classic TGN (memoryful)
      - TGAT (memoryless)
      - TTGN-style candidate weighting (optional)
    """
    def __init__(
        self,
        model: TGN,
        dataset: ContinuousTimeDynamicGraphDataset,
        *,
        num_hops: int = 2,
        model_name: str = "TGN",
        device: str = "cpu",
        n_neighbors: int = 20,
        batch_size: int = 32,
        checkpoint_path: Optional[str] = None,
        use_memory: bool = True,
        enable_candidates: bool = False,   # << turn on TTGN behavior
    ):
        super().__init__(model=model, dataset=dataset, num_hops=num_hops, model_name=model_name,
                         device=device, use_memory=use_memory, batch_size=batch_size)
        self.model = model
        self.n_neighbors = n_neighbors
        self.enable_candidates = enable_candidates

        # time stats for position encodings
        (self.model.mean_time_shift_src,
         self.model.std_time_shift_src,
         self.model.mean_time_shift_dst,
         self.model.std_time_shift_dst) = compute_time_statistics(
            self.dataset.source_node_ids, self.dataset.target_node_ids, self.dataset.timestamps
        )

        if checkpoint_path:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        # Ensure eval neighbor finder exists (fixes the AttributeError you saw)
        data_obj = to_data_object(self.dataset)
        self.full_ngh_finder = get_neighbor_finder(data_obj, uniform=False)

    # ---- Common API you already call elsewhere ----

    def initialize(self, event_id: int, show_progress: bool = False, memory_label: str = None):
        self.reset_model()
        return self._initialize_model(event_id, self.n_neighbors, None)

    def predict(self, event_id: int, result_as_logit: bool = False):
        s, t, ts, eid = self.extract_event_information(event_id)
        return self.compute_edge_probabilities(
            source_nodes=s, target_nodes=t, edge_timestamps=ts, edge_ids=eid,
            result_as_logit=result_as_logit, perform_memory_update=False
        )

    def post_batch_cleanup(self):
        if self.model.use_memory:
            self.model.detach_memory()

    # ---- Core scoring with optional candidate weighting (TTGN) ----
    def compute_edge_probabilities(
        self,
        source_nodes: np.ndarray,
        target_nodes: np.ndarray,
        edge_timestamps: np.ndarray,
        edge_ids: np.ndarray,
        *,
        result_as_logit: bool = False,
        perform_memory_update: bool = True,
        negative_nodes: np.ndarray | None = None,
        candidate_event_ids: np.ndarray | None = None,      # << optional TTGN
        candidate_event_weights: np.ndarray | None = None,  # << optional TTGN
        edge_idx_preserve_list=None,                         # << optional TTGN
    ):
        cand = None
        if self.enable_candidates and candidate_event_ids is not None and candidate_event_weights is not None:
            cand = {'candidate_events': candidate_event_ids, 'edge_weights': candidate_event_weights}

        return self.model.compute_edge_probabilities(
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            negative_nodes=negative_nodes,
            edge_timestamps=edge_timestamps,
            edge_ids=edge_ids,
            n_neighbors=self.n_neighbors,
            result_as_logit=result_as_logit,
            perform_memory_update=perform_memory_update,
            candidate_weights_dict=cand,
            edge_idx_preserve_list=edge_idx_preserve_list,
        )

    # Minimal stubs for memory-less configs (TGAT)
    def get_memory(self): return None if not self.model.use_memory else self.model.memory
    def detach_memory(self): 
        if self.model.use_memory: self.model.detach_memory()
    def restore_memory(self, memory_backup, event_id): 
        if self.model.use_memory: self.model.restore_memory(memory_backup, event_id)
    def reset_model(self): self.reset_latest_event_id()
