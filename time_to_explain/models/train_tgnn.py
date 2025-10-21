import argparse
import os


import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from time_to_explain.models.common import (add_dataset_arguments, create_tgn_wrapper_from_args, create_tgnn_wrapper_from_args, add_wrapper_model_arguments,
                    add_model_training_arguments, parse_args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('T-GNN Training')
    add_dataset_arguments(parser)
    add_wrapper_model_arguments(parser)
    add_model_training_arguments(parser)

    args = parse_args(parser)

    tgnn_wrapper = create_tgnn_wrapper_from_args(args)

    model_path = args.model_path
    checkpoints_path = os.path.join(model_path, 'checkpoints/')
    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)

    tgnn_wrapper.train_model(args.epochs, checkpoint_path=checkpoints_path, model_path=model_path,
                            results_path=model_path + '/results.pkl')
