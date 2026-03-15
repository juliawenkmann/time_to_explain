import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from argparse import ArgumentParser
from temgxlib.common import parse_args, add_dataset_arguments, add_wrapper_model_arguments
from temgxlib.runners.temgx_runner import TemGXRunner

def build_parser():
    p = ArgumentParser("TemGX")
    add_dataset_arguments(p)
    add_wrapper_model_arguments(p)
    p.add_argument('--max_explain', type=int, default=50)
    p.add_argument('--candidate_cap', type=int, default=200)
    p.add_argument('--sparsity', type=int, default=5)
    p.add_argument('--l_hops', type=int, default=2)
    p.add_argument('--time_window', type=int, default=None)
    p.add_argument('--icm_alpha', type=float, default=1.0)
    p.add_argument('--icm_beta', type=float, default=0.1)
    p.add_argument('--icm_lambda', type=float, default=0.01)
    p.add_argument('--icm_gamma', type=float, default=0.5)
    p.add_argument('--trd_scale', type=float, default=1.0)
    p.add_argument('--explained_ids', type=str, default=None)
    p.add_argument('--verbose', action='store_true')
    return p

def main():
    args = parse_args(build_parser())
    TemGXRunner().run(args)

if __name__ == "__main__":
    main()