from ctypes import Union
from fileinput import filename
from typing import List
import numpy as np
from pandas import DataFrame
from tqdm import tqdm
from pathlib import Path
import sys

# Ensure vendored tgnnexplainer (under submodules) is importable as `tgnnexplainer`
_TGNN_VENDOR = Path(__file__).resolve().parents[3] / "submodules" / "explainer" / "tgnnexplainer"
if str(_TGNN_VENDOR) not in sys.path:
    sys.path.insert(0, str(_TGNN_VENDOR))

from tgnnexplainer.xgraph.method.attn_explainer_tg import AttnExplainerTG
from tgnnexplainer.xgraph.method.subgraphx_tg import BaseExplainerTG, SubgraphXTG
from tgnnexplainer.xgraph.evaluation.metrics_tg_utils import fidelity_inv_tg, sparsity_tg


class BaseEvaluator():
    def __init__(self, model_name: str, explainer_name: str, dataset_name: str, 
                explainer: BaseExplainerTG = None,
                results_dir=None,
                threshold_num=25,
    ) -> None:
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.explainer_name = explainer_name

        self.explainer = explainer

        self.results_dir = results_dir
        self.suffix = None
        self.threshold_num = threshold_num


    @staticmethod
    def _save_path(results_dir,
                   model_name,
                   dataset_name,
                   explainer_name,
                   event_idxs,
                   suffix=None,
                   threshold_num=25):
        if isinstance(event_idxs, int):
            event_idxs = [event_idxs, ]
        
        if suffix is not None:
            filename = Path(results_dir)/f'{model_name}_{dataset_name}_{explainer_name}_{event_idxs[0]}_to_{event_idxs[-1]}_eval_{suffix}_th{threshold_num}.csv'
        else:
            filename = Path(results_dir)/f'{model_name}_{dataset_name}_{explainer_name}_{event_idxs[0]}_to_{event_idxs[-1]}_eval_th{threshold_num}.csv'
        return filename

    def _save_value_results(self, event_idxs, value_results, suffix=None):
        """save to a csv for plotting"""
        filename = self._save_path(results_dir=self.results_dir, model_name=self.model_name, dataset_name=self.dataset_name, explainer_name=self.explainer_name, event_idxs=event_idxs, suffix=suffix, threshold_num=self.threshold_num)

        df = DataFrame(value_results)
        df.to_csv(filename, index=False)
        
        print(f'evaluation value results saved at {str(filename)}')

    def _evaluate_one(self, single_results, event_idx):
        raise NotImplementedError
        
    
    def evaluate(self, explainer_results, event_idxs):
        event_idxs_results = []
        sparsity_results = []
        fid_inv_results = []
        fid_inv_best_results = []

        print('\nevaluating...')
        for i, (single_results, event_idx) in enumerate(zip(explainer_results, event_idxs)):
            print(f'\nevaluate {i}th: {event_idx}')
            self.explainer._initialize(event_idx)

            sparsity_list, fid_inv_list, fid_inv_best_list =  self._evaluate_one(single_results, event_idx)

            event_idxs_results.extend([event_idx]*len(sparsity_list))
            sparsity_results.extend(sparsity_list)
            fid_inv_results.extend(fid_inv_list)
            fid_inv_best_results.extend(fid_inv_best_list)
        
        results = {
            'event_idx': event_idxs_results,
            'sparsity': sparsity_results,
            'fid_inv': fid_inv_results,
            'fid_inv_best': fid_inv_best_results,
            
        }

        self._save_value_results(event_idxs, results, self.suffix)
        return results



class EvaluatorAttenTG(BaseEvaluator):
    def __init__(self, model_name: str, explainer_name: str, dataset_name: str,
                explainer: AttnExplainerTG,
                results_dir=None,
                threshold_num=25,
        ) -> None:
        super(EvaluatorAttenTG, self).__init__(model_name=model_name,
                                              explainer_name=explainer_name,
                                              dataset_name=dataset_name,
                                              results_dir=results_dir,
                                              explainer = explainer,
                                              threshold_num=threshold_num
                                              )
    
    def _evaluate_one(self, single_results, event_idx):
        candidates, candidate_weights = single_results

        candidate_events = self.explainer.candidate_events
        candidate_num = len(candidate_events)
        assert len(candidates) == candidate_num

        fid_inv_list = []
        sparsity_list = np.arange(0, 1.05, 0.05)
        for spar in sparsity_list:
            num = int(spar * candidate_num)
            important_events = candidates[:num+1]
            b_i_events = self.explainer.base_events + important_events
            important_pred = self.explainer.tgnn_reward_wraper._compute_gnn_score(b_i_events, event_idx)
            ori_pred = self.explainer.tgnn_reward_wraper.original_scores
            fid_inv = fidelity_inv_tg(ori_pred, important_pred)
            fid_inv_list.append(fid_inv)
            
        fid_inv_best = array_best(fid_inv_list)
        sparsity = np.array(sparsity_list)
        

        return sparsity, fid_inv_list, fid_inv_best


def array_best(values):
    if len(values) == 0:
        return values
    best_values = [values[0], ]
    best = values[0]
    for i in range(1, len(values)):
        if best < values[i]:
            best = values[i]
        best_values.append(best)
    return np.array(best_values)


class EvaluatorMCTSTG(BaseEvaluator):
    def __init__(self, 
        model_name: str, explainer_name: str, dataset_name: str, 
        explainer: SubgraphXTG,
        results_dir = None,
        threshold_num=25
        ) -> None:
        super(EvaluatorMCTSTG, self).__init__(model_name=model_name,
                                              explainer_name=explainer_name,
                                              dataset_name=dataset_name,
                                              results_dir=results_dir,
                                              threshold_num=threshold_num
                                              )
        self.explainer = explainer
        self.suffix = self.explainer.suffix
    
    def _evaluate_one(self, single_results, event_idx):
        
        tree_nodes, _ = single_results
        sparsity_list = []
        fid_inv_list = []
        
        candidate_events = self.explainer.candidate_events
        candidate_num = len(candidate_events)
        for node in tqdm(tree_nodes, total=len(tree_nodes)):
            spar = sparsity_tg(node, candidate_num)
            assert np.isclose(spar, node.Sparsity)
            
            fid_inv = node.P
            
            fid_inv_list.append(fid_inv)
            sparsity_list.append(spar)
        
        sparsity_list = np.array(sparsity_list)
        fid_inv_list = np.array(fid_inv_list)
        
        sort_idx = np.argsort(sparsity_list) # ascending of sparsity
        sparsity_list = sparsity_list[sort_idx]
        fid_inv_list = fid_inv_list[sort_idx]
        fid_inv_best = array_best(fid_inv_list)

        sparsity_thresholds = np.arange(0, 1.05, 0.05)
        indices = []
        for sparsity in sparsity_thresholds:
            indices.append( np.where(sparsity_list <= sparsity)[0].max() )
        
        fid_inv_list = fid_inv_list[indices]
        fid_inv_best = fid_inv_best[indices]

        return sparsity_thresholds, fid_inv_list, fid_inv_best
