from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from pathlib import Path
import sys

_TEMGX_VENDOR = Path(__file__).resolve().parents[2] / "submodules" / "explainer" / "TemGX" / "link"
if str(_TEMGX_VENDOR) not in sys.path:
    sys.path.insert(0, str(_TEMGX_VENDOR))

_TGN_VENDOR = Path(__file__).resolve().parents[2] / "submodules" / "models" / "tgn"
if str(_TGN_VENDOR) not in sys.path:
    sys.path.insert(0, str(_TGN_VENDOR))
if "utils" in sys.modules:
    del sys.modules["utils"]

from temgxlib.data import BatchData, ContinuousTimeDynamicGraphDataset
from utils.data_processing import Data, compute_time_statistics
from utils.utils import get_neighbor_finder


class TGNNWrapper:
    """
    Wrapper exposing a common interface for temporal GNNs so CoDy can query
    predictions and roll out event histories.
    """

    node_embedding_dimension: int
    time_embedding_dimension: int

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: ContinuousTimeDynamicGraphDataset,
        num_hops: int,
        model_name: str,
        *,
        device: str | torch.device = "cpu",
        use_memory: Optional[bool] = None,
        batch_size: int = 256,
        model_event_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.num_hops = int(num_hops)
        self.name = model_name
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.latest_event_id = 0
        self.evaluation_mode = False
        self.memory_backups_map: Dict[str, Tuple[Any, int]] = {}
        self.logger = logging.getLogger("TGNNWrapper")

        self.use_memory = bool(getattr(model, "use_memory", False)) if use_memory is None else bool(use_memory)
        self._source_nodes = np.asarray(dataset.source_node_ids, dtype=np.int64)
        self._target_nodes = np.asarray(dataset.target_node_ids, dtype=np.int64)
        self._timestamps = np.asarray(dataset.timestamps, dtype=np.float32)
        self._edge_ids = np.asarray(dataset.edge_ids, dtype=np.int64)
        self._labels = (
            np.asarray(dataset.labels, dtype=np.float32)
            if hasattr(dataset, "labels") and dataset.labels is not None
            else None
        )

        if model_event_ids is None:
            model_event_ids = self._edge_ids + 1
        self._model_event_ids = np.asarray(model_event_ids, dtype=np.int64)
        self.event_id_offset = int(self._model_event_ids[0] - self._edge_ids[0]) if len(self._model_event_ids) else 0
        self._model_id_by_dataset = self._model_event_ids
        self._dataset_id_by_model = {
            int(model_id): int(dataset_id) for dataset_id, model_id in enumerate(self._model_event_ids)
        }

        self._num_nodes = int(getattr(model, "n_nodes", dataset.node_features.shape[0]))
        self.node_embedding_dimension = int(
            getattr(model, "embedding_dimension", getattr(model, "n_node_features", 0))
        )
        self.time_embedding_dimension = int(getattr(getattr(model, "time_encoder", None), "dimension", 0))

        if getattr(self.model, "num_neighbors", None) is None:
            fallback = getattr(getattr(self.model, "ngh_finder", None), "num_neighbors", 20)
            self.model.num_neighbors = fallback

        try:
            m_src, s_src, m_dst, s_dst = compute_time_statistics(
                self._source_nodes, self._target_nodes, self._timestamps
            )
            self.model.mean_time_shift_src = m_src
            self.model.std_time_shift_src = s_src
            self.model.mean_time_shift_dst = m_dst
            self.model.std_time_shift_dst = s_dst
        except Exception:
            pass

        self.base_events: list[int] = []
        self._subgraph_event_ids: Optional[np.ndarray] = None
        self.original_score: Optional[float] = None

    # ------------------------------------------------------------------ helpers #
    def _as_numpy(self, values: Sequence[int] | np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.int64)

    def _to_model_event_ids(self, dataset_event_ids: Sequence[int] | np.ndarray) -> np.ndarray:
        idx = self._as_numpy(dataset_event_ids)
        if idx.size == 0:
            return idx
        return self._model_id_by_dataset[idx]

    def _to_dataset_event_ids(self, model_event_ids: Sequence[int] | np.ndarray) -> np.ndarray:
        ids = self._as_numpy(model_event_ids)
        if ids.size == 0:
            return ids
        return np.asarray([self._dataset_id_by_model[int(i)] for i in ids], dtype=np.int64)

    def _sample_negative_nodes(self, size: int) -> np.ndarray:
        if self._num_nodes <= 1:
            return np.zeros(size, dtype=np.int64)
        return np.random.randint(1, self._num_nodes, size=size, dtype=np.int64)

    def _event_arrays(self, event_ids: Sequence[int] | np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = self._as_numpy(event_ids)
        return self._source_nodes[idx], self._target_nodes[idx], self._timestamps[idx]

    def _to_data_object(self, edges_to_drop: Optional[np.ndarray] = None) -> Data:
        if edges_to_drop is None:
            mask = np.ones(len(self._edge_ids), dtype=bool)
        else:
            mask = ~np.isin(self._edge_ids, edges_to_drop)
        labels = self._labels[mask] if self._labels is not None else np.zeros(mask.sum(), dtype=np.float32)
        return Data(
            self._source_nodes[mask],
            self._target_nodes[mask],
            self._timestamps[mask],
            self._model_event_ids[mask],
            labels,
        )

    def _compute_scores(
        self,
        source_nodes: np.ndarray,
        target_nodes: np.ndarray,
        negative_nodes: np.ndarray,
        timestamps: np.ndarray,
        edge_idxs: np.ndarray,
        *,
        result_as_logit: bool,
        perform_memory_update: bool,
        edge_idx_preserve_list: Optional[Sequence[int]] = None,
        candidate_weights_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prev_forbidden = getattr(self.model, "forbidden_memory_update", False)
        if hasattr(self.model, "forbidden_memory_update"):
            self.model.forbidden_memory_update = not perform_memory_update
        try:
            src_emb, dst_emb, neg_emb = self.model.compute_temporal_embeddings(
                source_nodes,
                target_nodes,
                negative_nodes,
                timestamps,
                edge_idxs,
                self.model.num_neighbors,
                edge_idx_preserve_list=edge_idx_preserve_list,
                candidate_weights_dict=candidate_weights_dict,
            )
            scores = self.model.affinity_score(
                torch.cat([src_emb, src_emb], dim=0),
                torch.cat([dst_emb, neg_emb], dim=0),
            ).squeeze(dim=0)
            n_samples = len(source_nodes)
            pos = scores[:n_samples]
            neg = scores[n_samples:]
            if not result_as_logit:
                pos = pos.sigmoid()
                neg = neg.sigmoid()
            return pos, neg
        finally:
            if hasattr(self.model, "forbidden_memory_update"):
                self.model.forbidden_memory_update = prev_forbidden

    # ------------------------------------------------------------------ public #
    def initialize(
        self,
        event_id: int,
        show_progress: bool = False,
        memory_label: str | None = None,
        subgraph_event_ids: Optional[Sequence[int]] = None,
    ) -> None:
        del show_progress  # progress bars are managed by callers if needed
        if event_id is None:
            return

        event_id = int(event_id)

        if subgraph_event_ids is not None:
            self._subgraph_event_ids = self._as_numpy(subgraph_event_ids)
            self.base_events = sorted(self._subgraph_event_ids.tolist())
        else:
            self._subgraph_event_ids = None
            self.base_events = []

        if event_id < 0:
            self.reset_model()
            if memory_label is not None:
                self.memory_backups_map[memory_label] = (self.get_memory(), event_id)
            return

        if memory_label in self.memory_backups_map:
            backup, backup_event_id = self.memory_backups_map[memory_label]
            if backup_event_id == event_id:
                if self.use_memory:
                    self.restore_memory(backup, event_id)
                return

        if event_id < self.latest_event_id - 1:
            self.reset_model()

        if self._subgraph_event_ids is not None:
            rollout_ids = self._subgraph_event_ids[self._subgraph_event_ids <= event_id]
            self.rollout_until_event(event_id=event_id, event_ids_to_rollout=rollout_ids)
        else:
            self.rollout_until_event(event_id=event_id)

        if memory_label is not None:
            self.memory_backups_map[memory_label] = (self.get_memory(), event_id)

    def predict(
        self,
        event_id: int,
        edge_id_preserve_list: Optional[Sequence[int]] = None,
        candidate_events: Optional[Sequence[int]] = None,
        edge_weights: Optional[Sequence[float]] = None,
        result_as_logit: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        event_id = int(event_id)
        src, dst, ts = self._event_arrays([event_id])
        edge_idxs = self._to_model_event_ids([event_id])

        if candidate_events is not None and edge_weights is not None:
            cand = self._to_model_event_ids(candidate_events)
            weights = torch.as_tensor(edge_weights, dtype=torch.float32, device=self.device).view(-1)
            candidate_weights_dict = {
                "candidate_events": torch.as_tensor(cand, dtype=torch.int64, device=self.device),
                "edge_weights": weights,
            }
            pos = self.model.get_prob(
                src, dst, ts, logit=result_as_logit, candidate_weights_dict=candidate_weights_dict
            )
            if isinstance(pos, np.ndarray):
                pos = torch.as_tensor(pos, dtype=torch.float32, device=self.device)
            self.original_score = float(pos.detach().cpu().item())
            return pos, None

        preserve = None
        if edge_id_preserve_list is not None:
            preserve = self._to_model_event_ids(edge_id_preserve_list).tolist()

        pos, neg = self._compute_scores(
            src,
            dst,
            self._sample_negative_nodes(len(src)),
            ts,
            edge_idxs,
            result_as_logit=result_as_logit,
            perform_memory_update=False,
            edge_idx_preserve_list=preserve,
        )
        if pos.numel() == 1:
            self.original_score = float(pos.detach().cpu().item())
        return pos, neg

    def rollout_until_event(
        self,
        event_id: int | None = None,
        batch_data: BatchData | None = None,
        progress_bar: Any = None,
        event_ids_to_rollout: Optional[Sequence[int]] = None,
    ) -> None:
        del progress_bar  # callers handle progress
        if not self.use_memory:
            if event_ids_to_rollout is not None and len(event_ids_to_rollout) > 0:
                self.latest_event_id = int(np.max(event_ids_to_rollout)) + 1
            elif event_id is not None:
                self.latest_event_id = int(event_id) + 1
            return

        if event_id is None and batch_data is None:
            return

        end_idx = None
        if event_id is not None:
            end_idx = int(event_id) + 1

        if batch_data is None and end_idx is not None:
            batch_data = self.dataset.get_batch_data(self.latest_event_id, end_idx)

        if batch_data is None:
            return

        number_of_batches = int(np.ceil(len(batch_data.source_node_ids) / self.batch_size))
        edge_ids_batches = None

        if event_ids_to_rollout is not None:
            event_ids = np.sort(self._as_numpy(event_ids_to_rollout))
            if len(event_ids) == 0:
                return
            boundaries = np.arange(self.latest_event_id, event_ids[-1] + self.batch_size, self.batch_size)
            edge_ids_batches = np.split(event_ids, np.searchsorted(event_ids, boundaries))
            edge_ids_batches = [array for array in edge_ids_batches if len(array) > 0]
            number_of_batches = len(edge_ids_batches)

        prev_forbidden = getattr(self.model, "forbidden_memory_update", False)
        if hasattr(self.model, "forbidden_memory_update"):
            self.model.forbidden_memory_update = False
        try:
            with torch.no_grad():
                batch_id = 0
                for batch_index in range(number_of_batches):
                    if edge_ids_batches is not None:
                        edge_idxs = edge_ids_batches[batch_index]
                        src, dst, ts = self._event_arrays(edge_idxs)
                    else:
                        batch_start = batch_id * self.batch_size
                        batch_end = min((batch_id + 1) * self.batch_size, len(batch_data.source_node_ids))
                        batch_id += 1
                        src = batch_data.source_node_ids[batch_start:batch_end]
                        dst = batch_data.target_node_ids[batch_start:batch_end]
                        ts = batch_data.timestamps[batch_start:batch_end]
                        edge_idxs = batch_data.edge_ids[batch_start:batch_end]

                    edge_idxs_model = self._to_model_event_ids(edge_idxs)
                    neg = np.zeros(len(src), dtype=np.int64)
                    self.model.compute_temporal_embeddings(
                        source_nodes=src,
                        destination_nodes=dst,
                        negative_nodes=neg,
                        edge_times=ts,
                        edge_idxs=edge_idxs_model,
                        n_neighbors=self.model.num_neighbors,
                    )
                    if self.use_memory:
                        self.model.memory.detach_memory()
        finally:
            if hasattr(self.model, "forbidden_memory_update"):
                self.model.forbidden_memory_update = prev_forbidden

        if end_idx is not None:
            self.latest_event_id = end_idx
        elif edge_ids_batches is not None:
            self.latest_event_id = int(edge_ids_batches[-1][-1]) + 1

    def compute_embeddings(
        self,
        source_nodes: Sequence[int],
        target_nodes: Sequence[int],
        edge_times: Sequence[float],
        edge_ids: Sequence[int],
        negative_nodes: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src = self._as_numpy(source_nodes)
        dst = self._as_numpy(target_nodes)
        ts = np.asarray(edge_times, dtype=np.float32)
        if negative_nodes is None:
            negative_nodes = self._sample_negative_nodes(len(src))
        neg = self._as_numpy(negative_nodes)
        edge_idxs = self._to_model_event_ids(edge_ids)
        # compute_temporal_embeddings already returns node embeddings; reuse it to get them
        prev_forbidden = getattr(self.model, "forbidden_memory_update", False)
        if hasattr(self.model, "forbidden_memory_update"):
            self.model.forbidden_memory_update = True
        try:
            src_emb, dst_emb, _ = self.model.compute_temporal_embeddings(
                src,
                dst,
                neg,
                ts,
                edge_idxs,
                self.model.num_neighbors,
            )
        finally:
            if hasattr(self.model, "forbidden_memory_update"):
                self.model.forbidden_memory_update = prev_forbidden
        return src_emb, dst_emb

    def encode_timestamps(self, timestamps: np.ndarray) -> torch.Tensor:
        ts = torch.as_tensor(timestamps, dtype=torch.float32, device=self.device)
        if ts.ndim == 1:
            ts = ts.view(-1, 1)
        return self.model.time_encoder(ts)

    def compute_edge_probabilities(
        self,
        source_nodes: Sequence[int],
        target_nodes: Sequence[int],
        edge_timestamps: Sequence[float],
        edge_ids: Sequence[int],
        negative_nodes: Optional[Sequence[int]] = None,
        result_as_logit: bool = False,
        perform_memory_update: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src = self._as_numpy(source_nodes)
        dst = self._as_numpy(target_nodes)
        ts = np.asarray(edge_timestamps, dtype=np.float32)
        if negative_nodes is None:
            negative_nodes = self._sample_negative_nodes(len(src))
        neg = self._as_numpy(negative_nodes)
        edge_idxs = self._to_model_event_ids(edge_ids)
        return self._compute_scores(
            src,
            dst,
            neg,
            ts,
            edge_idxs,
            result_as_logit=result_as_logit,
            perform_memory_update=perform_memory_update,
        )

    def compute_edge_probabilities_for_subgraph(
        self,
        event_id: int,
        edges_to_drop: np.ndarray,
        result_as_logit: bool = False,
        event_ids_to_rollout: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        event_id = int(event_id)
        edges_to_drop = self._as_numpy(edges_to_drop) if edges_to_drop is not None else np.array([], dtype=np.int64)

        original_ngh = getattr(self.model, "ngh_finder", None)
        new_finder = get_neighbor_finder(self._to_data_object(edges_to_drop), uniform=False)
        self.model.set_neighbor_finder(new_finder)
        try:
            if event_ids_to_rollout is None:
                event_ids_to_rollout = self._edge_ids[~np.isin(self._edge_ids, edges_to_drop)]
                event_ids_to_rollout = event_ids_to_rollout[event_ids_to_rollout >= self.latest_event_id]
            event_ids_to_rollout = self._as_numpy(event_ids_to_rollout)
            event_ids_to_rollout = event_ids_to_rollout[event_ids_to_rollout < event_id]

            self.rollout_until_event(event_id=event_id, event_ids_to_rollout=event_ids_to_rollout)
            src, dst, ts = self._event_arrays([event_id])
            pos, neg = self.compute_edge_probabilities(
                src,
                dst,
                ts,
                [event_id],
                result_as_logit=result_as_logit,
                perform_memory_update=False,
            )
            return pos, neg
        finally:
            if original_ngh is not None:
                self.model.set_neighbor_finder(original_ngh)

    def get_memory(self):
        if not self.use_memory or getattr(self.model, "memory", None) is None:
            return None
        return self.model.memory.backup_memory()

    def detach_memory(self):
        if self.use_memory and getattr(self.model, "memory", None) is not None:
            self.model.memory.detach_memory()

    def restore_memory(self, memory_backup, event_id):
        if not self.use_memory or getattr(self.model, "memory", None) is None:
            return
        self.model.memory.restore_memory(memory_backup)
        self.reset_latest_event_id(int(event_id) + 1)

    def reset_model(self):
        if self.use_memory and getattr(self.model, "memory", None) is not None:
            self.model.memory.detach_memory()
            self.model.memory.__init_memory__()
        self.reset_latest_event_id()
        self._subgraph_event_ids = None
        self.base_events = []
        self.original_score = None

    def get_candidate_events(self, explained_event_id: int) -> list[int]:
        if self._subgraph_event_ids is None:
            return []
        explained_event_id = int(explained_event_id)
        candidates = self._subgraph_event_ids[self._subgraph_event_ids < explained_event_id]
        return candidates.tolist()

    def set_evaluation_mode(self, activate_evaluation: bool):
        if activate_evaluation:
            self.model.eval()
            self.evaluation_mode = True
        else:
            self.model.train()
            self.evaluation_mode = False

    def post_batch_cleanup(self):
        if self.use_memory and getattr(self.model, "memory", None) is not None:
            self.model.memory.detach_memory()

    def remove_memory_backup(self, label: str):
        if label in self.memory_backups_map:
            del self.memory_backups_map[label]

    def reset_latest_event_id(self, value: int | None = None):
        self.latest_event_id = int(value) if value is not None else 0

    def extract_event_information(self, event_ids: int | np.ndarray):
        edge_mask = np.isin(self.dataset.edge_ids, event_ids)
        source_nodes = self.dataset.source_node_ids[edge_mask]
        target_nodes = self.dataset.target_node_ids[edge_mask]
        timestamps = self.dataset.timestamps[edge_mask]
        return source_nodes, target_nodes, timestamps, np.asarray(event_ids, dtype=int)
