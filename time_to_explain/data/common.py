import glob
import os
import platform
import sys
from argparse import Namespace, ArgumentParser

import numpy as np
import pandas as pd
from .data import ContinuousTimeDynamicGraphDataset, TrainTestDatasetParameters

SAMPLERS = ['random', 'temporal', 'spatio-temporal', 'local-gradient']


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
    parser.add_argument('--cuda', action='store_true', help='Use cuda for GPU utilization')
    parser.add_argument('--update_memory_at_start', action='store_true',
                        help='Provide if the memory should be updated at start')
    parser.add_argument('--type', default='TGN', required=True, choices=['TGN', 'TGAT'])
    parser.add_argument('--candidates_size', type=int, default=64,
                        help='Number of candidates from which the samples are selected')


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

