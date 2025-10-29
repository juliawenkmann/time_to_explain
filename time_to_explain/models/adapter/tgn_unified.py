# time_to_explain/models/adapter/tgn_unified.py
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import torch

from time_to_explain.models.adapter.wrapper import TGNNWrapper
from time_to_explain.setup.constants import COL_TIMESTAMP, COL_NODE_I, COL_NODE_U
from time_to_explain.data.data import BatchData, ContinuousTimeDynamicGraphDataset
from time_to_explain.setup.utils import ProgressBar, construct_model_path
from time_to_explain.models.create_wrapper import create_wrapper

# Submodule (vendor) TGN bits
from submodules.models.tgn.TTGN.evaluation.evaluation import eval_edge_prediction
from submodules.models.tgn.TTGN.model.tgn import TGN
from submodules.models.tgn.TTGN.utils.data_processing import compute_time_statistics, Data
from submodules.models.tgn.TTGN.utils.utils import (
    EarlyStopMonitor,
    RandEdgeSampler,
    get_neighbor_finder,
    NeighborFinder,
)


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------


def _ensure_1d_negatives(neg_nodes, batch_len: int) -> np.ndarray:
    """
    Make sure `neg_nodes` is a 1-D numpy array of length == batch_len.
    If it's 2-D (e.g., [batch, K]), pick one column at random.
    """
    if torch.is_tensor(neg_nodes):
        arr = neg_nodes.detach().cpu().numpy()
    else:
        arr = np.asarray(neg_nodes)

    if arr.ndim == 1:
        if arr.shape[0] != batch_len:
            # If a scalar or wrong shape sneaks in, fall back to repeating or trimming
            if arr.size == 1:
                arr = np.repeat(arr.item(), batch_len)
            else:
                arr = arr.reshape(-1)[:batch_len]
        return arr.astype(np.int64, copy=False)

    if arr.ndim == 2:
        if arr.shape[0] != batch_len:
            # If the sampler returned a weird shape, trim rows
            arr = arr[:batch_len]
        # Pick one negative per positive (random column)
        col = np.random.randint(arr.shape[1]) if arr.shape[1] > 0 else 0
        return arr[:, col].astype(np.int64, copy=False)

    # Fallback: flatten to [batch_len, -1] and take first column
    arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[0] != batch_len:
        arr = arr[:batch_len]
    return arr[:, 0].astype(np.int64, copy=False)


def device_from_prefs(device: str = "auto", cuda: bool = False) -> torch.device:
    """
    Choose a torch.device given user prefs and availability.
    """
    # legacy flag takes precedence
    if cuda and torch.cuda.is_available():
        return torch.device("cuda")

    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "mps":
        try:
            if torch.backends.mps.is_available():
                return torch.device("mps")
        except AttributeError:
            pass
        return torch.device("cpu")
    if device == "cpu":
        return torch.device("cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    except AttributeError:
        pass
    return torch.device("cpu")


def to_data_object(dataset: ContinuousTimeDynamicGraphDataset,
                   edges_to_drop: np.ndarray | None = None) -> Data:
    """
    Convert our dataset to the submodule's Data object (for NeighborFinder).
    """
    if edges_to_drop is not None:
        mask = ~np.isin(dataset.edge_ids, edges_to_drop)
        return Data(dataset.source_node_ids[mask],
                    dataset.target_node_ids[mask],
                    dataset.timestamps[mask],
                    dataset.edge_ids[mask],
                    dataset.labels[mask])
    return Data(dataset.source_node_ids,
                dataset.target_node_ids,
                dataset.timestamps,
                dataset.edge_ids,
                dataset.labels)


def find_highest_checkpoint(
    models_root: str | Path,
    dataset: str,
    model_type: str,
    *,
    checkpoints_subdir: str = "checkpoints",
    exts: Iterable[str] = ("pth",),
    strict: bool = False,
) -> Optional[Path]:
    """
    Find the checkpoint with the largest integer suffix:
      <models_root>/<dataset>/<checkpoints_subdir>/<model_type>-<dataset>-<N>.<ext>
    e.g. CoDy/resources/models/wikipedia/checkpoints/TGAT-wikipedia-19.pth
    """
    models_root = Path(models_root)
    ckpt_dir = models_root / dataset / checkpoints_subdir
    if not ckpt_dir.is_dir():
        if strict:
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
        return None

    import re
    ext_pattern = "|".join([e.lstrip(".") for e in exts])
    pat = re.compile(rf"^{model_type}-{dataset}-(\d+)\.(?:{ext_pattern})$")
    best = None
    best_i = -1
    for p in ckpt_dir.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if m:
            i = int(m.group(1))
            if i > best_i:
                best, best_i = p, i
    if best is None and strict:
        raise FileNotFoundError(f"No checkpoints like '{model_type}-{dataset}-<N>.(pt|pth)' in {ckpt_dir}")
    return best


# --------------------------------------------------------------------------------------
# Unified core wrapper (shared by training & evaluation)
# --------------------------------------------------------------------------------------

class UnifiedTGNCore(TGNNWrapper):
    """
    One core that supports:
      - TGN (memoryful)
      - TGAT (memoryless, n_layers=2)
      - TTGN-style candidate weighting (optional on calls)

    Exposes your existing API surface:
      - compute_edge_probabilities(...)
      - compute_embeddings(...)
      - initialize(...), rollout_until_event(...)
      - get_memory/detach_memory/restore_memory/reset_model
      - attributes: dataset, batch_size, use_memory, device, model, etc.
    """

    def __init__(
        self,
        model: TGN,
        dataset: ContinuousTimeDynamicGraphDataset,
        *,
        model_name: str = "TGN",
        device: str = "cpu",
        n_neighbors: int = 20,
        batch_size: int = 32,
        checkpoint_path: str | None = None,
        use_memory: bool = True,
    ):
        super().__init__(model=model, dataset=dataset, num_hops=2,
                         model_name=model_name, device=device,
                         use_memory=use_memory, batch_size=batch_size)

        # Time statistics for the model (same as your legacy wrappers)
        (model.mean_time_shift_src,
         model.std_time_shift_src,
         model.mean_time_shift_dst,
         model.std_time_shift_dst) = compute_time_statistics(
             dataset.source_node_ids,
             dataset.target_node_ids,
             dataset.timestamps
         )

        self.model = model
        self.n_neighbors = n_neighbors
        self.device = device

        if checkpoint_path:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        # Ensure a full-graph NeighborFinder exists for eval (prevents AttributeError)
        data_obj = to_data_object(dataset)
        self.full_ngh_finder: NeighborFinder = get_neighbor_finder(data_obj, uniform=False)

        # Convenience emb dims (if used by external code)
        self.node_embedding_dimension = getattr(self.model.embedding_module, "embedding_dimension", None)
        self.time_embedding_dimension = getattr(self.model.time_encoder, "dimension", None)

        # Book-keeping used by your code
        self.memory_backups_map: dict[str, tuple[object, int]] = {}
        self.reset_model()
        self.reset_latest_event_id()

    # -----------------------------  Convenience API  -----------------------------

    def initialize(self, event_id: int, show_progress: bool = False, memory_label: str | None = None):
        """
        Initialize/restore state so that the model has processed all events up to `event_id`.
        """
        if memory_label is not None and memory_label in self.memory_backups_map:
            if show_progress:
                print(f'Restoring memory with label "{memory_label}"')
            mem_backup, backed_event_id = self.memory_backups_map[memory_label]
            if backed_event_id == event_id:
                self.restore_memory(mem_backup, event_id)
                return
            else:
                self.logger.warning(
                    "Provided event ID does not match memory label; re-processing from start."
                )
        # Rollout from the beginning up to event_id
        self.rollout_until_event(event_id, progress_bar=ProgressBar())

    def predict(self, event_id: int, result_as_logit: bool = False):
        s, t, ts, eid = self.extract_event_information(event_id)
        return self.compute_edge_probabilities(
            source_nodes=s, target_nodes=t, edge_timestamps=ts, edge_ids=eid,
            result_as_logit=result_as_logit, perform_memory_update=False
        )

    def post_batch_cleanup(self):
        if self.model.use_memory:
            self.model.detach_memory()

    # ------------------------  Rollout & batching helpers  -----------------------

    def rollout_until_event(self, event_id: int = None, batch_data: BatchData = None,
                            progress_bar: ProgressBar | None = None,
                            event_ids_to_rollout: np.ndarray | None = None) -> None:
        """
        Replay events to update internal state up to `event_id`.
        """
        assert event_id is not None or batch_data is not None
        if batch_data is None:
            event_id = int(event_id)
            # +1 because event_id is inclusive; BatchData end is exclusive
            batch_data = self.dataset.get_batch_data(self.latest_event_id, event_id + 1)
        batch_id = 0
        number_of_batches = int(math.ceil(len(batch_data.source_node_ids) / self.batch_size))
        if progress_bar is not None:
            progress_bar.reset(number_of_batches)
        with torch.no_grad():
            for _ in range(number_of_batches):
                if progress_bar is not None:
                    progress_bar.next()
                b0 = batch_id * self.batch_size
                b1 = min((batch_id + 1) * self.batch_size, len(batch_data.source_node_ids))
                self.compute_edge_probabilities(
                    source_nodes=batch_data.source_node_ids[b0:b1],
                    target_nodes=batch_data.target_node_ids[b0:b1],
                    edge_timestamps=batch_data.timestamps[b0:b1],
                    edge_ids=batch_data.edge_ids[b0:b1],
                    result_as_logit=False,
                    perform_memory_update=True
                )
                if self.model.use_memory:
                    self.model.memory.detach_memory()
                batch_id += 1

        # Keep the latest processed event
        if event_ids_to_rollout is not None and len(event_ids_to_rollout) > 0:
            self.reset_latest_event_id(event_ids_to_rollout[-1])
        elif event_id is not None:
            self.reset_latest_event_id(event_id)

    # ---------------------------  Forward computations  --------------------------

    def compute_edge_probabilities(
        self,
        *,
        source_nodes: np.ndarray,
        target_nodes: np.ndarray,
        edge_timestamps: np.ndarray,
        edge_ids: np.ndarray,
        negative_nodes: np.ndarray | None = None,
        candidate_event_ids: np.ndarray | None = None,         # TTGN-style (optional)
        candidate_event_weights: np.ndarray | None = None,     # TTGN-style (optional)
        edge_idx_preserve_list=None,                            # TTGN-style (optional)
        result_as_logit: bool = False,
        perform_memory_update: bool = True,
    ):
        candidate_weights_dict = None
        if candidate_event_ids is not None and candidate_event_weights is not None:
            candidate_weights_dict = {
                "candidate_events": candidate_event_ids,
                "edge_weights": candidate_event_weights,
            }
        if negative_nodes is None:
            negative_nodes = np.asarray(target_nodes)
        neg = _ensure_1d_negatives(negative_nodes, len(source_nodes))
        return self.model.compute_edge_probabilities(
            source_nodes, target_nodes, neg, edge_timestamps, edge_ids,
            self.n_neighbors, result_as_logit, perform_memory_update,
            candidate_weights_dict=candidate_weights_dict,
            edge_idx_preserve_list=edge_idx_preserve_list
        )

    def compute_edge_probabilities_for_subgraph(
        self,
        event_id: int,
        edges_to_drop: np.ndarray,
        *,
        result_as_logit: bool = False,
        event_ids_to_rollout: np.ndarray | None = None
    ):
        if not self.evaluation_mode:
            self.logger.info("Model not in evaluation mode. Do not use predictions for eval!")
        # Temporarily swap neighbor finder that ignores the dropped edges
        original = self.model.neighbor_finder
        self.model.set_neighbor_finder(get_neighbor_finder(
            to_data_object(self.dataset, edges_to_drop=edges_to_drop), uniform=False
        ))

        if event_ids_to_rollout is None:
            event_ids_to_rollout = self.dataset.edge_ids[~np.isin(self.dataset.edge_ids, edges_to_drop)]
            event_ids_to_rollout = event_ids_to_rollout[event_ids_to_rollout > self.latest_event_id]
        event_ids_to_rollout = event_ids_to_rollout[event_ids_to_rollout < event_id]

        self.rollout_until_event(event_id=event_id, event_ids_to_rollout=event_ids_to_rollout)

        s, t, ts, eid = self.extract_event_information(event_id)
        probs = self.compute_edge_probabilities(
            source_nodes=s, target_nodes=t, edge_timestamps=ts, edge_ids=eid,
            result_as_logit=result_as_logit, perform_memory_update=False
        )
        self.model.set_neighbor_finder(original)
        return probs

    # ---- Embedding utilities (used by TTGN paths but harmless elsewhere) ----

    def encode_timestamps(self, timestamps: np.ndarray):
        timestamps = torch.tensor(timestamps, dtype=torch.float32, device=self.device).reshape((1, -1))
        return self.model.time_encoder(timestamps)

    def compute_embeddings(
        self,
        source_nodes,
        target_nodes,
        edge_times,
        edge_ids,
        negative_nodes=None,
        candidate_event_ids: np.ndarray | None = None,
        candidate_event_weights: np.ndarray | None = None,
    ):
        candidate_weights_dict = None
        if candidate_event_ids is not None and candidate_event_weights is not None:
            candidate_weights_dict = {
                "candidate_events": candidate_event_ids,
                "edge_weights": candidate_event_weights,
            }
        return self.model.compute_temporal_embeddings(
            source_nodes, target_nodes, negative_nodes, edge_times, edge_ids,
            n_neighbors=self.n_neighbors,
            candidate_weights_dict=candidate_weights_dict
        )

    def get_current_node_embeddings(
        self,
        node_ids: np.ndarray,
        current_timestamp: int,
        candidate_event_ids: np.ndarray | None = None,
        candidate_event_weights: np.ndarray | None = None
    ):
        candidate_weights_dict = None
        if candidate_event_ids is not None and candidate_event_weights is not None:
            candidate_weights_dict = {
                "candidate_events": candidate_event_ids,
                "edge_weights": candidate_event_weights,
            }

        memory = None
        if self.model.use_memory:
            if self.model.memory_update_at_start:
                memory, _ = self.model.get_updated_memory(list(range(self.model.n_nodes)),
                                                          self.model.memory.messages)
            else:
                memory = self.model.memory.get_memory(list(range(self.model.n_nodes)))
        timestamps = np.repeat(current_timestamp, len(node_ids))
        return self.model.embedding_module.compute_embedding(
            memory=memory, source_nodes=node_ids, timestamps=timestamps,
            n_layers=self.model.n_layers, n_neighbors=self.model.n_neighbors,
            candidate_weights_dict=candidate_weights_dict
        )

    # -----------------------------  Memory management  ---------------------------

    def get_memory(self):
        if not self.use_memory:
            return None
        return self.model.memory.backup_memory()

    def detach_memory(self):
        if self.use_memory:
            self.model.memory.detach_memory()

    def restore_memory(self, memory_backup, event_id):
        if not self.use_memory:
            return
        self.reset_model()
        self.model.memory.restore_memory(memory_backup)
        self.reset_latest_event_id(event_id)

    def reset_model(self):
        self.reset_latest_event_id()
        if self.use_memory:
            self.detach_memory()
            self.model.memory.__init_memory__()


# --------------------------------------------------------------------------------------
# Dedicated training wrapper (keeps your training API/behavior)
# --------------------------------------------------------------------------------------

class TGNTrainer(UnifiedTGNCore):
    """
    Training-capable wrapper. The training loop is based on your legacy TGN wrapper.
    """

    def train_model(self, epochs: int = 50, learning_rate: float = 1e-4, early_stop_patience: int = 5,
                    checkpoint_path: str = "./saved_checkpoints/", model_path: str = "./saved_models/",
                    results_path: str = "./results/dump.pkl"):
        """
        Self-supervised training adapted from the submodule's original implementation.
        Saves checkpoints, final model, and results to the provided paths.
        """
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)

        # ---- Split & neighbor finders (same prints as before)
        (node_features, edge_features, full_data,
        train_data, val_data, test_data,
         new_node_val_data, new_node_test_data) = self.get_training_data(
            randomize_features=False,
            validation_fraction=0.15,
            test_fraction=0.15,
            new_test_nodes_fraction=0.1,
            different_new_nodes_between_val_and_test=False
        )

        train_nf = get_neighbor_finder(train_data, uniform=False)
        full_nf = get_neighbor_finder(full_data, uniform=False)
        val_nf = get_neighbor_finder(val_data, uniform=False)
        test_nf = get_neighbor_finder(test_data, uniform=False)
        nn_val_nf = get_neighbor_finder(new_node_val_data, uniform=False)
        nn_test_nf = get_neighbor_finder(new_node_test_data, uniform=False)

        # ---- Attach train finder and init optimizer/criterion
        self.model.set_neighbor_finder(train_nf)
        if hasattr(self.model, "create_optimizer"):
            self.model.create_optimizer(lr=learning_rate)
        else:
            self.model.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        if hasattr(self.model, "create_criterion"):
            self.model.create_criterion()
        else:
            # Most TGN/TGAT link-prediction setups use BCE
            self.model.criterion = torch.nn.BCELoss()

        # ---- Negative sampling
        train_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
        val_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
        nn_val_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)
        test_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
        nn_test_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=3)
        do_validation = True

        early_stopper = EarlyStopMonitor(max_round=early_stop_patience)

        num_instances = len(train_data.sources)
        num_batches = int(np.ceil(num_instances / self.batch_size))
        progress_bar = ProgressBar(num_batches)

        print(f"INFO:TGNNWrapper:num of training instances: {num_instances}")
        print(f"INFO:TGNNWrapper:num of batches per epoch: {num_batches}")

        # ---- Epoch loop
        for e in range(epochs):
            print(f"INFO:TGNNWrapper:start {e} epoch")
            progress_bar.reset(num_batches)

            self.model.set_neighbor_finder(train_nf)
            self.model.train()
            self.set_evaluation_mode(False)

            epoch_losses = []

            perm = np.random.permutation(num_instances)

            # mini-batch
            for batch_i in range(num_batches):
                progress_bar.next()
                b0 = batch_i * self.batch_size
                b1 = min((batch_i + 1) * self.batch_size, num_instances)
                batch_idx = perm[b0:b1]
                src = train_data.sources[batch_idx]
                dst = train_data.destinations[batch_idx]
                ts = train_data.timestamps[batch_idx]
                eids = train_data.edge_idxs[batch_idx]

                # Negative sampling: ensure vendor expects 1-D [batch]
                _, neg_nodes = train_sampler.sample(len(src))
                neg_nodes = _ensure_1d_negatives(neg_nodes, len(src))

                # Forward: vendor returns post-sigmoid probabilities (pos_prob, neg_prob)
                pos_prob, neg_prob = self.model.compute_edge_probabilities(
                    src, dst, neg_nodes, ts, eids, self.n_neighbors
                )

                # Convert to tensors on the right device if needed
                if not torch.is_tensor(pos_prob):
                    pos_prob = torch.as_tensor(pos_prob, dtype=torch.float32, device=self.device)
                if not torch.is_tensor(neg_prob):
                    neg_prob = torch.as_tensor(neg_prob, dtype=torch.float32, device=self.device)

                # Criterion: use BCELoss on probabilities (vendor already applied sigmoid)
                if not hasattr(self.model, "criterion") or not isinstance(self.model.criterion, torch.nn.modules.loss._Loss):
                    self.model.criterion = torch.nn.BCELoss()

                loss_pos = self.model.criterion(pos_prob, torch.ones_like(pos_prob))
                loss_neg = self.model.criterion(neg_prob, torch.zeros_like(neg_prob))
                loss = 0.5 * (loss_pos + loss_neg)
                # Backprop
                self.model.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.model.optimizer.step()

                epoch_losses.append(loss.item())

                if self.model.use_memory:
                    self.model.memory.detach_memory()

            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            print(f"INFO:TGNNWrapper:avg loss epoch {e}: {avg_loss:.4f}")

            # ---- Validation
            if do_validation:
                self.model.set_neighbor_finder(full_nf)
                self.model.eval()
                self.set_evaluation_mode(True)

                # Standard and new-node validation metrics
                val_metrics = eval_edge_prediction(
                    model=self.model, negative_edge_sampler=val_sampler, data=val_data,
                    n_neighbors=self.n_neighbors, batch_size=self.batch_size
                )
                new_val_metrics = eval_edge_prediction(
                    model=self.model, negative_edge_sampler=nn_val_sampler, data=new_node_val_data,
                    n_neighbors=self.n_neighbors, batch_size=self.batch_size
                )

                val_ap, val_auc, val_acc = self._unpack_metrics(val_metrics)
                nn_val_ap, nn_val_auc, nn_val_acc = self._unpack_metrics(new_val_metrics)

                print(f"Val:   AP {val_ap:.4f} | AUC {val_auc:.4f} | ACC {val_acc:.4f}")
                print(f"NewV:  AP {nn_val_ap:.4f} | AUC {nn_val_auc:.4f} | ACC {nn_val_acc:.4f}")

                # Early stop on validation AP
                if early_stopper.early_stop_check(val_ap):
                    print("INFO:TGNNWrapper:early stop")
                    break

            # ---- Save checkpoint each epoch (optional policy)
            ckpt_dir = Path(checkpoint_path)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_file = ckpt_dir / f"{self.name}-{self.dataset.name}-{e}.pth"
            torch.save(self.model.state_dict(), ckpt_file)

        # ---- Final evaluation (optional, mirrors your legacy code’s flow)
        self.model.set_neighbor_finder(full_nf)
        self.model.eval()
        self.set_evaluation_mode(True)

        # Final metrics on test & new-node test
        test_metrics = eval_edge_prediction(
            model=self.model, negative_edge_sampler=test_sampler, data=test_data,
            n_neighbors=self.n_neighbors, batch_size=self.batch_size
        )
        nn_test_metrics = eval_edge_prediction(
            model=self.model, negative_edge_sampler=nn_test_sampler, data=new_node_test_data,
            n_neighbors=self.n_neighbors, batch_size=self.batch_size
        )
        # Save results
        import pickle
        Path(model_path).mkdir(parents=True, exist_ok=True)
        with open(results_path, "wb") as f:
            pickle.dump({
                "test": test_metrics,
                "new_node_test": nn_test_metrics,
            }, f)

        # Save final model
        final_path = Path(model_path) / f"{self.name}-{self.dataset.name}.pth"
        torch.save(self.model.state_dict(), final_path)

    # --- Helpers ported from your wrapper ---

    @staticmethod
    def _unpack_metrics(metrics):
        # metrics is expected in (AP, AUC, ACC, ...) ordering per submodule eval
        if isinstance(metrics, (list, tuple)) and len(metrics) >= 3:
            return float(metrics[0]), float(metrics[1]), float(metrics[2])
        # fallback
        return float("nan"), float("nan"), float("nan")

    def get_training_data(self, randomize_features: bool = False,
                          validation_fraction: float = 0.15,
                          test_fraction: float = 0.15,
                          new_test_nodes_fraction: float = 0.1,
                          different_new_nodes_between_val_and_test: bool = False):
        """
        Same logic as your legacy `get_training_data` (adapted from submodule).
        """
        dataset = self.dataset
        node_features = dataset.node_features
        if randomize_features:
            node_features = np.random.rand(dataset.node_features.shape[0], node_features.shape[1])

        val_time, test_time = list(np.quantile(dataset.events[COL_TIMESTAMP],
                                               [1 - (validation_fraction + test_fraction),
                                                1 - test_fraction]))

        full_data = Data(dataset.source_node_ids, dataset.target_node_ids, dataset.timestamps, dataset.edge_ids,
                         dataset.labels)

        node_set = set(dataset.source_node_ids) | set(dataset.target_node_ids)
        unique_nodes = len(node_set)

        # Train/Val/Test masks by time
        train_mask = dataset.timestamps <= val_time
        val_mask = np.logical_and(dataset.timestamps <= test_time,
                                  dataset.timestamps > val_time)
        test_mask = dataset.timestamps > test_time

        # New node split
        test_node_set = set(dataset.source_node_ids[dataset.timestamps > val_time]) \
            .union(set(dataset.target_node_ids[dataset.timestamps > val_time]))
        new_test_node_set = set(random.sample(sorted(test_node_set),
                                              int(new_test_nodes_fraction * unique_nodes)))
        new_test_src_mask = dataset.events[COL_NODE_I].map(lambda x: x in new_test_node_set).values
        new_test_dst_mask = dataset.events[COL_NODE_U].map(lambda x: x in new_test_node_set).values
        observed_edges_mask = np.logical_and(~new_test_src_mask, ~new_test_dst_mask)

        # If new nodes should be different between val and test
        if different_new_nodes_between_val_and_test:
            val_time_mask = np.logical_and(dataset.timestamps <= val_time, observed_edges_mask)
            test_time_mask = np.logical_and(dataset.timestamps > val_time, observed_edges_mask)
        else:
            val_time_mask = np.logical_and(train_mask, observed_edges_mask)
            test_time_mask = np.logical_and(~train_mask, observed_edges_mask)

        # Build Data splits
        def mask_to_data(mask):
            return Data(dataset.source_node_ids[mask],
                        dataset.target_node_ids[mask],
                        dataset.timestamps[mask],
                        dataset.edge_ids[mask],
                        dataset.labels[mask])

        train_data = mask_to_data(train_mask & observed_edges_mask)
        val_data = mask_to_data(val_mask & observed_edges_mask)
        test_data = mask_to_data(test_mask & observed_edges_mask)

        new_node_val_data = mask_to_data(val_mask & ~observed_edges_mask)
        new_node_test_data = mask_to_data(test_mask & ~observed_edges_mask)

        print(f"The dataset has {full_data.n_interactions} interactions, involving {full_data.n_unique_nodes} different nodes")
        print(f"The training dataset has {train_data.n_interactions} interactions, involving {train_data.n_unique_nodes} different nodes")
        print(f"The validation dataset has {val_data.n_interactions} interactions, involving {val_data.n_unique_nodes} different nodes")
        print(f"The test dataset has {test_data.n_interactions} interactions, involving {test_data.n_unique_nodes} different nodes")
        print(f"The new node validation dataset has {new_node_val_data.n_interactions} interactions, involving {new_node_val_data.n_unique_nodes} different nodes")
        print(f"The new node test dataset has {new_node_test_data.n_interactions} interactions, involving {new_node_test_data.n_unique_nodes} different nodes")
        print(f"{len(new_test_node_set)} nodes were used for the inductive testing, i.e. are never seen during training")

        return (node_features, dataset.edge_features, full_data, train_data, val_data, test_data,
                new_node_val_data, new_node_test_data)


# --------------------------------------------------------------------------------------
# Read-only evaluation wrapper
# --------------------------------------------------------------------------------------

class TGNEvaluator(UnifiedTGNCore):
    """
    Evaluation/inference-only wrapper. No training method.
    """
    def train_model(self, *args, **kwargs):
        raise RuntimeError("TGNEvaluator is evaluation-only. Use TGNTrainer for training.")


# --------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------

def build_model(
    *,
    dataset: ContinuousTimeDynamicGraphDataset,
    model_type: str = "TGN",               # "TGN" | "TGAT" | "TTGN"
    device: str = "auto",
    cuda: bool = False,
    update_memory_at_start: bool = False,
    n_neighbors: int = 20,
    n_layers_tgat: int = 2,
) -> TGN:
    """
    Instantiate the submodule TGN with a configuration matching the requested variant.
    - TGN  -> use_memory=True, memory_dimension=172
    - TGAT -> use_memory=False, memory_dimension=0, n_layers=n_layers_tgat
    - TTGN -> same base model as TGN; candidate weighting handled in forward calls
    """
    dev = device_from_prefs(device, cuda)
    use_memory = (model_type.upper() != "TGAT")
    memory_dim = 172 if use_memory else 0
    n_layers = n_layers_tgat if not use_memory else 1

    model = TGN(
        neighbor_finder=get_neighbor_finder(to_data_object(dataset), uniform=False),
        node_features=dataset.node_features,
        edge_features=dataset.edge_features,
        device=dev,
        use_memory=use_memory,
        memory_update_at_start=update_memory_at_start,
        memory_dimension=memory_dim,
        embedding_module_type='graph_attention',
        message_function='identity',
        aggregator_type='last',
        memory_updater_type='gru',
        use_destination_embedding_in_message=False,
        use_source_embedding_in_message=False,
        dyrep=False,
        n_neighbors=n_neighbors,
        n_layers=n_layers,
    )
    model.to(dev)
    return model


def make_trainer(
    *,
    dataset: ContinuousTimeDynamicGraphDataset,
    model_type: str = "TGN",               # "TGN" | "TGAT" | "TTGN"
    device: str = "auto",
    cuda: bool = False,
    update_memory_at_start: bool = False,
    n_neighbors: int = 20,
    batch_size: int = 32,
    checkpoint_path: str | None = None,
) -> TGNNWrapper:
    model_key = model_type.upper()
    if model_key in {"TGN", "TGAT"}:
        wrapper = create_wrapper(
            model_type=model_type,
            dataset=dataset,
            directed=dataset.directed,
            bipartite=dataset.bipartite,
            device=device,
            cuda=cuda,
            update_memory_at_start=update_memory_at_start,
            checkpoint_path=checkpoint_path,
        )
        if hasattr(wrapper, "batch_size"):
            wrapper.batch_size = batch_size
        return wrapper

    model = build_model(dataset=dataset, model_type=model_type, device=device, cuda=cuda,
                        update_memory_at_start=update_memory_at_start, n_neighbors=n_neighbors)
    return TGNTrainer(
        model=model, dataset=dataset, model_name=model_type, device=device,
        n_neighbors=n_neighbors, batch_size=batch_size, checkpoint_path=checkpoint_path,
        use_memory=model.use_memory
    )


def make_evaluator(
    *,
    dataset: ContinuousTimeDynamicGraphDataset,
    model_type: str = "TGN",               # "TGN" | "TGAT" | "TTGN"
    device: str = "auto",
    cuda: bool = False,
    update_memory_at_start: bool = False,
    n_neighbors: int = 20,
    batch_size: int = 32,
    checkpoint_path: str | None = None,
) -> TGNEvaluator:
    model = build_model(dataset=dataset, model_type=model_type, device=device, cuda=cuda,
                        update_memory_at_start=update_memory_at_start, n_neighbors=n_neighbors)
    return TGNEvaluator(
        model=model, dataset=dataset, model_name=model_type, device=device,
        n_neighbors=n_neighbors, batch_size=batch_size, checkpoint_path=checkpoint_path,
        use_memory=model.use_memory
    )
