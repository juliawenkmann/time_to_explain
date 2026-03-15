import logging, os, numpy as np
from temgxlib.models.factory import build_dataset, build_wrapper
from temgxlib.explain import metrics
from temgxlib.common import get_event_ids_from_file

class TemGXRunner:
    def __init__(self, logger=None):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logger or logging.getLogger("TemGX")
    def run(self, args):
        self.logger.info(f"Arguments: {vars(args)}")
        dataset = build_dataset(args)
        wrapper = build_wrapper(args, dataset)
        ids = get_event_ids_from_file(args.explained_ids, self.logger, False, wrapper)
        results = self._explain_and_eval(args, wrapper, ids)
        self.logger.info(f"Results: {results}")
    def _explain_and_eval(self, args, wrapper, explained_ids):
        import sys, os, subprocess
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # -> Link/
        entry = os.path.join(root, 'scripts', 'TemGx.py')

        py_paths = [
            root,
            os.path.join(root, 'scripts'),
            os.path.join(root, 'submodules', 'tgn'),
            os.path.join(root, 'submodules', 'ttgn'),
        ]
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join(py_paths + [env.get('PYTHONPATH', '')])

        argv = [
            sys.executable, entry,
            '--dataset', args.dataset,
            '--type', args.type,
            '--model', args.model,
            '--candidates_size', str(args.candidates_size),
            '--max_explain', str(args.max_explain),
            '--candidate_cap', str(args.candidate_cap),
            '--sparsity', str(args.sparsity),
            '--l_hops', str(args.l_hops),
            '--icm_alpha', str(args.icm_alpha),
            '--icm_beta', str(args.icm_beta),
            '--icm_lambda', str(args.icm_lambda),
            '--icm_gamma', str(args.icm_gamma),
            '--trd_scale', str(args.trd_scale),
        ]
        if args.cuda:
            argv.append('--cuda')
        if args.update_memory_at_start:
            argv.append('--update_memory_at_start')
        if args.time_window is not None:
            argv += ['--time_window', str(args.time_window)]
        if args.explained_ids:
            argv += ['--explained_ids', args.explained_ids]
        if args.verbose:
            argv.append('--verbose')

        subprocess.run(argv, check=True, env=env)
        return {"fidelity_minus": None, "sparsity": None, "aufsc": None}