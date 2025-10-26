import glob
import logging
import os
import platform
import sys
from argparse import Namespace, ArgumentParser

import numpy as np
import pandas as pd
import torch

from time_to_explain.models.adapter.tgn import TGNWrapper, to_data_object
from time_to_explain.models.adapter.ttgn import TTGNWrapper
from time_to_explain.models.adapter.tgat import TGATWrapper
from time_to_explain.data.data import ContinuousTimeDynamicGraphDataset, TrainTestDatasetParameters
from submodules.models.tgn.model.tgn import TGN
from submodules.models.tgn.utils.utils import get_neighbor_finder
from submodules.models.tgn.model.tgn import TGN as TTGN
from submodules.models.tgn.utils.utils import get_neighbor_finder as tget_neighbor_finder

SAMPLERS = ['random', 'temporal', 'spatio-temporal', 'local-gradient']





# -----------------------
# Arg parsing
# -----------------------

def parse_args(parser: ArgumentParser) -> Namespace:
    try:
        return parser.parse_args()
    except SystemExit:
        parser.print_help()
        sys.exit(0)


def add_dataset_arguments(parser: ArgumentParser):
    parser.add_argument('-d', '--dataset', required=True, type=str, help='Path to the dataset folder')
    parser.add_argument('--directed', action='store_true', help='Provide if the graph is directed')
    parser.add_argument('--bipartite', action='store_true', help='Provide if the graph is bipartite')


def add_wrapper_model_arguments(parser: ArgumentParser):
    parser.add_argument('-m', '--model', default=None, type=str,
                        help='Path to the model checkpoint to use')

    # Legacy flag retained for compatibility; will gracefully fall back if CUDA isn't available
    parser.add_argument('--cuda', action='store_true', help='Use cuda for GPU utilization (falls back if unavailable)')

    # New, explicit device selection (recommended)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Compute device preference (auto -> cuda/mps/cpu fallback)')

    parser.add_argument('--update_memory_at_start', action='store_true',
                        help='Provide if the memory should be updated at start')
    parser.add_argument('--type', default='TGN', required=True, choices=['TGN', 'TGAT'])
    parser.add_argument('--candidates_size', type=int, default=64,
                        help='Number of candidates from which the samples are selected')


def add_model_training_arguments(parser: ArgumentParser):
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the directory where the model checkpoints, final model and results are saved to.')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of epochs to train the model for.')


# -----------------------
# Data helpers
# -----------------------

def column_to_int_array(df, column_name):
    df[column_name] = (df[column_name].str.rstrip(']').str.lstrip('[')
                       .replace('\n', '').str.split().apply(lambda x: np.array([int(item) for item in x])))


def column_to_float_array(df, column_name):
    df[column_name] = (df[column_name].str.rstrip(']').str.lstrip('[')
                       .replace('\n', '').str.split().apply(lambda x: np.array([float(item) for item in x])))


def create_dataset_from_args(args: Namespace, parameters: TrainTestDatasetParameters | None = None) -> (
        ContinuousTimeDynamicGraphDataset):
    if parameters is None:
        parameters = TrainTestDatasetParameters(0.2, 0.6, 0.8, 1000, 500, 500)

    # Get dataset
    dataset_folder = args.dataset

    events = glob.glob(os.path.join(dataset_folder, '*_data.csv'))
    edge_features = glob.glob(os.path.join(dataset_folder, '*_edge_features.npy'))
    node_features = glob.glob(os.path.join(dataset_folder, '*_node_features.npy'))

    name = edge_features[0][:-18]
    assert len(events) == len(edge_features) == len(node_features) == 1
    assert name == edge_features[0][:-18] == events[0][:-9]

    if platform.system() == 'Windows':
        name = name.split('\\')[-1]
    else:
        name = name.split('/')[-1]
    all_event_data = pd.read_csv(events[0])
    edge_features = np.load(edge_features[0])
    node_features = np.load(node_features[0])

    return ContinuousTimeDynamicGraphDataset(all_event_data, edge_features, node_features, name,
                                             directed=args.directed, bipartite=args.bipartite,
                                             parameters=parameters)


# -----------------------
# Wrapper constructors
# -----------------------

def create_ttgnn_wrapper_from_args(args: Namespace, dataset: ContinuousTimeDynamicGraphDataset | None = None):
    if dataset is None:
        dataset = create_dataset_from_args(args)

    dev_str = dev.type  # for wrappers that expect a string

    if args.type == 'TGN':
        tgn = TTGN(
            neighbor_finder=tget_neighbor_finder(to_data_object(dataset), uniform=False),
            node_features=dataset.node_features,
            edge_features=dataset.edge_features,
            device=dev,
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
    elif args.type == 'TGAT':
        tgn = TTGN(
            neighbor_finder=tget_neighbor_finder(to_data_object(dataset), uniform=False),
            node_features=dataset.node_features,
            edge_features=dataset.edge_features,
            device=dev,
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
        raise NotImplementedError

    tgn.to(dev)

    return TTGNWrapper(tgn, dataset, num_hops=2, model_name=args.type, device=dev_str, n_neighbors=20,
                       explanation_candidates_size=args.candidates_size, batch_size=32, checkpoint_path=args.model,
                       use_memory=tgn.use_memory)


def create_tgnn_wrapper_from_args(args: Namespace, dataset: ContinuousTimeDynamicGraphDataset | None = None):
    if dataset is None:
        dataset = create_dataset_from_args(args)

    if args.type == 'TGAT':
        return create_tgat_wrapper_from_args(args, dataset)
    elif args.type == 'TGN':
        return create_tgn_wrapper_from_args(args, dataset)
    else:
        raise NotImplementedError


def create_tgn_wrapper_from_args(args: Namespace, dataset: ContinuousTimeDynamicGraphDataset | None = None):
    if dataset is None:
        dataset = create_dataset_from_args(args)

    dev = device_from_args(args)
    dev_str = dev.type

    tgn = TGN(
        neighbor_finder=get_neighbor_finder(to_data_object(dataset), uniform=False),
        node_features=dataset.node_features,
        edge_features=dataset.edge_features,
        device=dev,
        use_memory=True,
        memory_update_at_start=args.update_memory_at_start,
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

    tgn.to(dev)

    return TGNWrapper(tgn, dataset, num_hops=2, model_name='TGN', device=dev_str, n_neighbors=20,
                      batch_size=32, checkpoint_path=args.model)


def create_tgat_wrapper_from_args(args: Namespace, dataset: ContinuousTimeDynamicGraphDataset | None = None):
    if dataset is None:
        dataset = create_dataset_from_args(args)

    dev = device_from_args(args)
    dev_str = dev.type

    tgn = TGN(
        neighbor_finder=get_neighbor_finder(to_data_object(dataset), uniform=False),
        node_features=dataset.node_features,
        edge_features=dataset.edge_features,
        device=dev,
        use_memory=False,
        memory_update_at_start=args.update_memory_at_start,
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

    tgn.to(dev)

    return TGATWrapper(tgn, dataset, num_hops=2, model_name='TGAT', device=dev_str, n_neighbors=20,
                       batch_size=32, checkpoint_path=args.model)


# -----------------------
# Misc utilities
# -----------------------

def get_event_ids_from_file(event_ids_filepath: str | None, logger: logging.Logger,
                            wrong_predictions_only: bool = False, tgn_wrapper: TGNWrapper | TTGNWrapper = None):
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
        results = results[results['predictions'] < - 0.2]  # Wrong predictions with some margin
    filtered_results = results[
        results['edge_ids'] > int(tgn_wrapper.dataset.parameters.training_start * max_event_id)]
    filtered_results = filtered_results[
        filtered_results['edge_ids'] < int(tgn_wrapper.dataset.parameters.training_end * max_event_id)]
    sampled_results = filtered_results.sample(tgn_wrapper.dataset.parameters.train_items)
    return sampled_results.sort_values(by='edge_ids')['edge_ids'].to_numpy()
