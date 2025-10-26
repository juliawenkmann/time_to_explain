# common_nb.py  —  parser-free utilities for dataset + wrapper construction

import glob
import logging
import os
import platform
import sys
from typing import Optional

import numpy as np
import pandas as pd
import torch

from time_to_explain.models.adapter.tgn import TGNWrapper, to_data_object
from time_to_explain.models.adapter.ttgn import TTGNWrapper
from time_to_explain.models.adapter.tgat import TGATWrapper
from time_to_explain.data.data import ContinuousTimeDynamicGraphDataset, TrainTestDatasetParameters, create_dataset


# Submodule models (the underlying architectures)
from submodules.models.tgn.model.tgn import TGN
from submodules.models.tgn.utils.utils import get_neighbor_finder
from submodules.models.tgn.model.tgn import TGN as TTGN  # legacy/alt impl used by TTGNWrapper
from submodules.models.tgn.utils.utils import get_neighbor_finder as tget_neighbor_finder
SAMPLERS = ['random', 'temporal', 'spatio-temporal', 'local-gradient']

# -----------------------
# Wrapper constructors (no argparse)
# -----------------------

def create_ttgnn_wrapper(
    *,
    model_type: str,  # "TGN" or "TGAT" (legacy alt stack)
    dataset: ContinuousTimeDynamicGraphDataset | None = None,
    dataset_dir: str | os.PathLike | None = None,
    directed: bool = False,
    bipartite: bool = False,
    device: str = "auto",
    update_memory_at_start: bool = False,  # not used by legacy TTGN when use_memory=False
    candidates_size: int = 64,
    checkpoint_path: str | None = None,
    parameters: Optional[TrainTestDatasetParameters] = None,
) -> TTGNWrapper:
    """
    Legacy/alternate implementation pathway that uses TTGN + TTGNWrapper.
    Provide either a prebuilt `dataset` or a `dataset_dir`.
    """
    print("Creating TTGNWrapper...")
    if dataset is None:
        if dataset_dir is None:
            raise ValueError("Provide either `dataset` or `dataset_dir`.")
        dataset = create_dataset(dataset_dir, directed=directed, bipartite=bipartite, parameters=parameters)

    dev_str = device.type

    if model_type == 'TGN':
        tgn = TTGN(
            neighbor_finder=tget_neighbor_finder(to_data_object(dataset), uniform=False),
            node_features=dataset.node_features,
            edge_features=dataset.edge_features,
            device=device,
            use_memory=True,
            memory_update_at_start=False,
            memory_dimension=172,
            embedding_module_type='graph_attention',
            message_function='identity',
            aggregator_type='last',
            memory_updater_type='gru',
            use_destination_embedding_in_message=False,
            use_source_embedding_in_message=False,
            dyrep=False,
            n_neighbors=20
        )
    elif model_type == 'TGAT':
        tgn = TTGN(
            neighbor_finder=tget_neighbor_finder(to_data_object(dataset), uniform=False),
            node_features=dataset.node_features,
            edge_features=dataset.edge_features,
            device=device,
            use_memory=False,
            memory_update_at_start=False,
            memory_dimension=0,
            embedding_module_type='graph_attention',
            message_function='identity',
            aggregator_type='last',
            memory_updater_type='gru',
            use_destination_embedding_in_message=False,
            use_source_embedding_in_message=False,
            dyrep=False,
            n_neighbors=20,
            n_layers=2
        )
    else:
        raise NotImplementedError(f"Unknown model_type: {model_type}")

    tgn.to(device)

    return TTGNWrapper(
        tgn, dataset, num_hops=2, model_name=model_type, device=dev_str, n_neighbors=20,
        explanation_candidates_size=candidates_size, batch_size=32, checkpoint_path=checkpoint_path,
        use_memory=tgn.use_memory
    )


def create_tgn_wrapper(
    *,
    dataset: ContinuousTimeDynamicGraphDataset | None = None,
    dataset_dir: str | os.PathLike | None = None,
    directed: bool = False,
    bipartite: bool = False,
    device: str = "auto",
    update_memory_at_start: bool = False,
    checkpoint_path: str | None = None,
    parameters: Optional[TrainTestDatasetParameters] = None,
) -> TGNWrapper:
    """
    Build a TGNWrapper (use_memory=True).
    """
    print("Creating TGNWrapper...")
    if dataset is None:
        if dataset_dir is None:
            raise ValueError("Provide either `dataset` or `dataset_dir`.")
        dataset = create_dataset(dataset_dir, directed=directed, bipartite=bipartite, parameters=parameters)

    dev_str = device.type

    tgn = TGN(
        neighbor_finder=get_neighbor_finder(to_data_object(dataset), uniform=False),
        node_features=dataset.node_features,
        edge_features=dataset.edge_features,
        device=device,
        use_memory=True,
        memory_update_at_start=update_memory_at_start,
        memory_dimension=172,
        embedding_module_type='graph_attention',
        message_function='identity',
        aggregator_type='last',
        memory_updater_type='gru',
        use_destination_embedding_in_message=False,
        use_source_embedding_in_message=False,
        dyrep=False,
        n_neighbors=20
    )
    tgn.to(device)

    return TGNWrapper(
        tgn, dataset, num_hops=2, model_name='TGN', device=dev_str, n_neighbors=20,
        batch_size=32, checkpoint_path=checkpoint_path
    )


def create_tgat_wrapper(
    *,
    dataset: ContinuousTimeDynamicGraphDataset | None = None,
    dataset_dir: str | os.PathLike | None = None,
    directed: bool = False,
    bipartite: bool = False,
    device: str = "auto",
    update_memory_at_start: bool = False,
    checkpoint_path: str | None = None,
    parameters: Optional[TrainTestDatasetParameters] = None,
) -> TGATWrapper:
    """
    Build a TGATWrapper (use_memory=False, 2 layers).
    """
    print("Creating TGATWrapper...")
    if dataset is None:
        if dataset_dir is None:
            raise ValueError("Provide either `dataset` or `dataset_dir`.")
        dataset = create_dataset(dataset_dir, directed=directed, bipartite=bipartite, parameters=parameters)

    dev_str = device.type

    tgn = TGN(
        neighbor_finder=get_neighbor_finder(to_data_object(dataset), uniform=False),
        node_features=dataset.node_features,
        edge_features=dataset.edge_features,
        device=device,
        use_memory=False,
        memory_update_at_start=update_memory_at_start,
        memory_dimension=0,
        embedding_module_type='graph_attention',
        message_function='identity',
        aggregator_type='last',
        memory_updater_type='gru',
        use_destination_embedding_in_message=False,
        use_source_embedding_in_message=False,
        dyrep=False,
        n_neighbors=20,
        n_layers=2
    )
    tgn.to(device)

    return TGATWrapper(
        tgn, dataset, num_hops=2, model_name='TGAT', device=dev_str, n_neighbors=20,
        batch_size=32, checkpoint_path=checkpoint_path
    )


def create_wrapper(
    *,
    model_type: str,  # "TGN" or "TGAT"
    dataset: ContinuousTimeDynamicGraphDataset | None = None,
    dataset_dir: str | os.PathLike | None = None,
    directed: bool = False,
    bipartite: bool = False,
    device: str = "auto",
    update_memory_at_start: bool = False,
    checkpoint_path: str | None = None,
    parameters: Optional[TrainTestDatasetParameters] = None,
):
    """
    Dispatcher: returns either TGATWrapper or TGNWrapper, based on `model_type`.
    """
    if model_type == "TGAT":
        return create_tgat_wrapper(
            dataset=dataset, dataset_dir=dataset_dir, directed=directed, bipartite=bipartite,
            device=device, update_memory_at_start=update_memory_at_start,
            checkpoint_path=checkpoint_path, parameters=parameters
        )
    elif model_type == "TGN":
        return create_tgn_wrapper(
            dataset=dataset, dataset_dir=dataset_dir, directed=directed, bipartite=bipartite,
            device=device, update_memory_at_start=update_memory_at_start,
            checkpoint_path=checkpoint_path, parameters=parameters
        )
    else:
        raise NotImplementedError(f"Unknown model_type: {model_type}")


# -----------------------
# Misc utilities (unchanged)
# -----------------------

def get_event_ids_from_file(
    event_ids_filepath: str | None,
    logger: logging.Logger,
    wrong_predictions_only: bool = False,
    tgn_wrapper: TGNWrapper | TTGNWrapper = None
):
    if os.path.exists(event_ids_filepath):
        return np.load(event_ids_filepath)
    else:
        logger.info('No event ids to explain provided. Generating new ones...')
        assert tgn_wrapper is not None, 'Cannot sample predictions if model is not provided'
        tgn_wrapper.reset_model()
        if wrong_predictions_only:
            logger.info('Generating sample consisting of wrong predictions only. This may take a while...')
            event_ids_to_explain = sample_predictions(tgn_wrapper, False)
        else:
            logger.info('Generating sample consisting of correct predictions only. This may take a while...')
            event_ids_to_explain = sample_predictions(tgn_wrapper, True)
        np.save(event_ids_filepath, event_ids_to_explain)
        return event_ids_to_explain


def sample_predictions(tgn_wrapper: TGNWrapper | TTGNWrapper, predictions_correct: bool):
    tgn_wrapper.set_evaluation_mode(True)
    max_event_id = np.max(tgn_wrapper.dataset.edge_ids)
    batch_data = tgn_wrapper.dataset.get_batch_data(0, max_event_id - 1)
    batch_id = 0
    number_of_batches = int(np.ceil(len(batch_data.source_node_ids) / tgn_wrapper.batch_size))
    all_predictions = []
    event_ids = []
    with torch.no_grad():
        for _ in range(number_of_batches):
            batch_start = batch_id * tgn_wrapper.batch_size
            batch_end = min((batch_id + 1) * tgn_wrapper.batch_size, len(batch_data.source_node_ids))
            predictions, _ = tgn_wrapper.compute_edge_probabilities(
                source_nodes=batch_data.source_node_ids[batch_start:batch_end],
                target_nodes=batch_data.target_node_ids[batch_start:batch_end],
                edge_timestamps=batch_data.timestamps[batch_start:batch_end],
                edge_ids=batch_data.edge_ids[batch_start:batch_end],
                result_as_logit=True)
            predictions = predictions.detach().cpu().numpy()
            all_predictions.append(predictions)
            event_ids.append(batch_data.edge_ids[batch_start:batch_end])
            if getattr(tgn_wrapper, "use_memory", False):
                # avoid graph growing; detach if memory module is present
                tgn_wrapper.model.memory.detach_memory()
            batch_id += 1
    all_predictions = np.concatenate(all_predictions)
    event_ids = np.concatenate(event_ids)

    results = pd.DataFrame({'edge_ids': event_ids.flatten(), 'predictions': all_predictions.flatten()})
    if predictions_correct:
        results = results[results['predictions'] > 0.2]
    else:
        results = results[results['predictions'] < -0.2]  # Wrong predictions with some margin
    filtered_results = results[
        results['edge_ids'] > int(tgn_wrapper.dataset.parameters.training_start * max_event_id)]
    filtered_results = filtered_results[
        filtered_results['edge_ids'] < int(tgn_wrapper.dataset.parameters.training_end * max_event_id)]
    sampled_results = filtered_results.sample(tgn_wrapper.dataset.parameters.train_items)
    return sampled_results.sort_values(by='edge_ids')['edge_ids'].to_numpy()
