import os, sys, argparse, json, glob, math
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import time
from collections import defaultdict
from scipy.special import expit
from temgxlib.implementations.tgn import TGNWrapper, to_data_object
from temgxlib.implementations.ttgn import TTGNWrapper
from temgxlib.implementations.tgat import TGATWrapper
from temgxlib.data import ContinuousTimeDynamicGraphDataset, TrainTestDatasetParameters

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common import (
    add_dataset_arguments,
    add_wrapper_model_arguments,
    parse_args,
    create_dataset_from_args,
    create_tgnn_wrapper_from_args,
)
from temgxlib.data import TrainTestDatasetParameters

def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('temgx_icm_debug.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def safe_float_conversion(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return float("nan")
        return float(arr.reshape(-1)[0])
    except Exception:
        try:
            return float(x)
        except Exception:
            return float("nan")

def detect_column_mapping(df: pd.DataFrame, logger) -> Dict[str, str]:
    logger.debug(f"Available columns: {list(df.columns)}")
    logger.debug(f"DataFrame shape: {df.shape}")
    logger.debug(f"Sample data:\n{df.head()}")

    def find_column(candidates, required=True):
        for c in df.columns:
            c_lower = c.lower()
            for pattern in candidates:
                if pattern.lower() in c_lower or c_lower == pattern.lower():
                    return c
        if required:
            logger.error(f"Could not find column for patterns: {candidates}")
        return None

    mapping = {}
    if 'idx' in df.columns and 'user_id' in df.columns and 'item_id' in df.columns and 'timestamp' in df.columns:
        mapping['e_id'] = 'idx'
        mapping['u'] = 'user_id'
        mapping['i'] = 'item_id'
        mapping['ts'] = 'timestamp'
    else:
        mapping['e_id'] = find_column(['idx', 'e_id', 'edge_id', 'event_id'], required=False)
        mapping['u'] = find_column(['user_id', 'u', 'src', 'source', 'from', 'node_1'])
        mapping['i'] = find_column(['item_id', 'i', 'dst', 'dest', 'to', 'node_2'])
        mapping['ts'] = find_column(['timestamp', 'ts', 't', 'time', 'event_time'])

    logger.info(f"Column mapping: {mapping}")
    return mapping

def extract_events_from_dataset(dataset, dataset_path, logger):
    def try_load_dataframe():
        for attr in ['data', 'events', 'df', 'edge_df', 'edges', 'all_events']:
            if hasattr(dataset, attr):
                obj = getattr(dataset, attr)
                if isinstance(obj, pd.DataFrame) and len(obj) > 0:
                    logger.info(f"Found data in dataset.{attr}")
                    return obj.copy()
        arrays = {}
        for name_group, attr_names in [
            ('src', ['sources', 'source_nodes', 'u', 'src']),
            ('dst', ['destinations', 'destination_nodes', 'i', 'dst']),
            ('ts', ['timestamps', 'time', 'ts', 'times']),
            ('eid', ['edge_idxs', 'e_id', 'edge_id', 'event_id', 'idx'])
        ]:
            for attr in attr_names:
                if hasattr(dataset, attr):
                    arrays[name_group] = getattr(dataset, attr)
                    break
        if 'src' in arrays and 'dst' in arrays and 'ts' in arrays:
            logger.info("Constructing DataFrame from dataset arrays")
            eids = arrays.get('eid', np.arange(len(arrays['ts'])))
            return pd.DataFrame({'u': arrays['src'], 'i': arrays['dst'], 'ts': arrays['ts'], 'e_id': eids})
        raise RuntimeError(f"Cannot load data from {dataset_path}")

    df = try_load_dataframe()
    for col in list(df.columns):
        if col.lower() in ['unnamed: 0', 'index', 'level_0']:
            df = df.drop(columns=[col])

    mapping = detect_column_mapping(df, logger)
    required = ['u', 'i', 'ts']
    missing = [k for k in required if mapping.get(k) is None]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    result_data = {}
    for std_name, col_name in mapping.items():
        if col_name is not None:
            result_data[std_name] = df[col_name].values

    if 'e_id' not in result_data:
        result_data['e_id'] = np.arange(len(df))

    events_df = pd.DataFrame(result_data)
    events_df['e_id'] = events_df['e_id'].astype(np.int64)
    events_df['u'] = events_df['u'].astype(np.int64)
    events_df['i'] = events_df['i'].astype(np.int64)
    events_df['ts'] = events_df['ts'].astype(np.float64)

    logger.info(f"Extracted {len(events_df)} events")
    logger.info(f"Columns: {list(events_df.columns)}")
    logger.info(f"Data types: {events_df.dtypes}")
    logger.info(f"Time range: {events_df['ts'].min():.1f} - {events_df['ts'].max():.1f}")
    return events_df

def detect_time_scale(events_df: pd.DataFrame, logger) -> Tuple[str, float]:
    timestamps = events_df['ts'].values
    time_diffs = np.diff(np.sort(timestamps))
    median_diff = np.median(time_diffs) if len(time_diffs) > 0 else 0.0
    q25_diff = np.percentile(time_diffs, 25) if len(time_diffs) > 0 else 0.0
    q75_diff = np.percentile(time_diffs, 75) if len(time_diffs) > 0 else 0.0
    logger.info(f"Time difference stats - Q25: {q25_diff:.1f}, Median: {median_diff:.1f}, Q75: {q75_diff:.1f}")

    if median_diff < 10:
        scale = "steps"
        conversion_factor = 1.0
    elif median_diff < 3600:
        scale = "seconds"
        conversion_factor = 3600.0
    elif median_diff < 86400:
        scale = "hours"
        conversion_factor = 1.0
    else:
        scale = "days"
        conversion_factor = 24.0
    logger.info(f"Detected time scale: {scale}")
    return scale, conversion_factor

class TemporalGraphIndex:
    def __init__(self, events_df: pd.DataFrame, wrapper=None, logger=None):
        self.events_df = events_df.sort_values('ts').reset_index(drop=True)
        self.wrapper = wrapper
        self.logger = logger or logging.getLogger(__name__)
        self.edge_info = {}
        self.node_to_edges = defaultdict(list)
        self.temporal_neighbors = {}
        self._build_index()

    def _build_index(self):
        for idx, row in self.events_df.iterrows():
            eid = int(row["e_id"])
            u = int(row["u"])
            i = int(row["i"])
            ts = float(row["ts"])
            self.edge_info[eid] = {'u': u, 'i': i, 'ts': ts, 'idx': idx}
            self.node_to_edges[u].append((eid, ts))
            self.node_to_edges[i].append((eid, ts))
        for node in self.node_to_edges:
            self.node_to_edges[node].sort(key=lambda x: x[1])

    def _get_node_embedding(self, node_id: int, timestamp: float = None) -> Optional[np.ndarray]:
        if self.wrapper is None:
            return None
        try:
            if hasattr(self.wrapper, 'get_raw_node_embedding'):
                return self.wrapper.get_raw_node_embedding(node_id)
            elif hasattr(self.wrapper, 'get_node_embedding'):
                return self.wrapper.get_node_embedding(node_id)
            elif hasattr(self.wrapper, 'node_embeddings'):
                embeddings = self.wrapper.node_embeddings
                if hasattr(embeddings, '__getitem__'):
                    return embeddings[node_id]
        except Exception as e:
            self.logger.debug(f"Failed to get embedding for node {node_id}: {e}")
        return None

    def get_all_past_edges(self, target_eid: int) -> List[int]:
        if target_eid not in self.edge_info:
            return []
        target_ts = self.edge_info[target_eid]['ts']
        past_edges = []
        for eid, info in self.edge_info.items():
            if info['ts'] < target_ts:
                past_edges.append(eid)
        return sorted(past_edges)

    def get_l_hop_temporal_neighbors(self, target_eid: int, l_hops: int, time_window: float, time_scale: str = "hours") -> List[int]:
        if target_eid not in self.edge_info:
            return []
        target_info = self.edge_info[target_eid]
        target_ts = target_info['ts']
        target_nodes = [target_info['u'], target_info['i']]

        def is_valid_time(ts):
            if ts >= target_ts:
                return False
            return ts >= target_ts - time_window

        visited_edges = set()
        current_nodes = set(target_nodes)
        all_neighbor_edges = set()

        for _ in range(l_hops):
            next_nodes = set()
            hop_edges = set()
            for node in current_nodes:
                for eid, ts in self.node_to_edges[node]:
                    if eid == target_eid or eid in visited_edges:
                        continue
                    if not is_valid_time(ts):
                        continue
                    hop_edges.add(eid)
                    visited_edges.add(eid)
                    edge_info = self.edge_info[eid]
                    other_node = edge_info['i'] if edge_info['u'] == node else edge_info['u']
                    next_nodes.add(other_node)
            all_neighbor_edges.update(hop_edges)
            current_nodes = next_nodes
            if not current_nodes:
                break
        return list(all_neighbor_edges)

class ICMInfluenceCalculator:
    def __init__(self, graph_index: TemporalGraphIndex, alpha: float = 1.0, beta: float = 0.1, lambda_decay: float = 0.01, gamma: float = 0.5, trd_scale: float = 1.0, time_scale: str = "hours"):
        self.graph_index = graph_index
        self.alpha = alpha
        self.beta = beta
        self.base_lambda_decay = lambda_decay
        self.lambda_decay = self.get_adaptive_lambda(time_scale, lambda_decay)
        self.gamma = gamma
        self.trd_scale = trd_scale
        self.time_scale = time_scale

    def get_adaptive_lambda(self, time_scale: str, base_lambda: float = 0.01) -> float:
        if time_scale == "seconds":
            return base_lambda / 3600
        elif time_scale == "hours":
            return base_lambda
        else:
            return base_lambda * 24

    def compute_temporal_resistance_distance(self, vs_eid: int, target_eid: int) -> float:
        if vs_eid not in self.graph_index.edge_info or target_eid not in self.graph_index.edge_info:
            return 2.0
            
        vs_info = self.graph_index.edge_info[vs_eid]
        target_info = self.graph_index.edge_info[target_eid]
        
        vs_embedding = self.graph_index._get_node_embedding(vs_info['u'])
        target_embedding = self.graph_index._get_node_embedding(target_info['u'])
        
        if vs_embedding is not None and target_embedding is not None:
            embedding_distance = np.linalg.norm(vs_embedding - target_embedding)
            return float(embedding_distance)
        else:
            vs_nodes = {vs_info['u'], vs_info['i']}
            target_nodes = {target_info['u'], target_info['i']}
            if vs_nodes & target_nodes:
                return 0.3
            else:
                return 1.5

    def compute_enhanced_similarity(self, vs_eid: int, target_eid: int) -> float:
        vs_info = self.graph_index.edge_info[vs_eid]
        target_info = self.graph_index.edge_info[target_eid]
        
        vs_embedding = self.graph_index._get_node_embedding(vs_info['u'])
        target_embedding = self.graph_index._get_node_embedding(target_info['u'])
        
        if vs_embedding is not None and target_embedding is not None:
            cos_sim = np.dot(vs_embedding, target_embedding) / (
                np.linalg.norm(vs_embedding) * np.linalg.norm(target_embedding) + 1e-8
            )
            return (cos_sim + 1) / 2
        else:
            vs_nodes = {vs_info['u'], vs_info['i']}
            target_nodes = {target_info['u'], target_info['i']}
            if vs_nodes & target_nodes:
                return 0.8
            else:
                return 0.3

    def trd_to_connectivity(self, trd_value: float) -> float:
        return np.exp(-trd_value / self.trd_scale)

    def compute_path_activation_probability(self, vs_eid: int, target_eid: int) -> float:
        if vs_eid not in self.graph_index.edge_info or target_eid not in self.graph_index.edge_info:
            return 0.0
        
        vs_info = self.graph_index.edge_info[vs_eid]
        target_info = self.graph_index.edge_info[target_eid]
        vs_nodes = {vs_info['u'], vs_info['i']}
        target_nodes = {target_info['u'], target_info['i']}
        
        if vs_nodes & target_nodes:
            path_length = 1
        else:
            path_length = 2
        
        similarity = self.compute_enhanced_similarity(vs_eid, target_eid)
        trd_value = self.compute_temporal_resistance_distance(vs_eid, target_eid)
        connectivity_strength = self.trd_to_connectivity(trd_value)
        
        base_logit = self.alpha * similarity - self.beta * path_length
        connectivity_adjusted_logit = base_logit + self.gamma * np.log(connectivity_strength + 1e-8)
        
        return expit(connectivity_adjusted_logit)

    def compute_single_edge_influence(self, vs_eid: int, target_eid: int, target_ts: float) -> float:
        if vs_eid not in self.graph_index.edge_info:
            return 0.0
        vs_ts = self.graph_index.edge_info[vs_eid]['ts']
        if vs_ts >= target_ts:
            return 0.0
        path_prob = self.compute_path_activation_probability(vs_eid, target_eid)
        time_diff = target_ts - vs_ts
        time_decay = np.exp(-self.lambda_decay * time_diff)
        return path_prob * time_decay

    def normalize_icm_scores(self, candidates: List[int], target_eid: int, target_ts: float) -> List[Tuple[int, float]]:
        scores = []
        for candidate in candidates:
            score = self.compute_single_edge_influence(candidate, target_eid, target_ts)
            scores.append((candidate, score))
        
        if len(scores) <= 1:
            return scores
        
        score_values = np.array([s[1] for s in scores])
        if np.std(score_values) < 1e-10:
            ranks = np.ones(len(scores))
        else:
            ranks = len(scores) - np.argsort(np.argsort(score_values))
            ranks = ranks / len(scores)
        
        return [(scores[i][0], ranks[i]) for i in range(len(scores))]

    def compute_edge_set_influence(self, vs_eids: List[int], target_eid: int) -> float:
        if target_eid not in self.graph_index.edge_info:
            return 0.0
        target_ts = self.graph_index.edge_info[target_eid]['ts']
        independent_probs = []
        for vs_eid in vs_eids:
            prob = self.compute_single_edge_influence(vs_eid, target_eid, target_ts)
            independent_probs.append(prob)
        total_prob = 1.0
        for prob in independent_probs:
            total_prob *= (1.0 - prob)
        return 1.0 - total_prob

    def compute_temporal_influence(self, vs_eids: List[int], target_eid: int, delta: float) -> float:
        return self.compute_edge_set_influence(vs_eids, target_eid)

def compute_original_prediction_with_fallback(wrapper, s: int, d: int, t: float, eid: int, logger) -> float:
    try:
        p_full = wrapper.compute_edge_probabilities(
            np.array([s], dtype=np.int64),
            np.array([d], dtype=np.int64),
            np.array([t], dtype=np.float64),
            np.array([eid], dtype=np.int64),
            result_as_logit=False
        )
        if isinstance(p_full, (list, tuple)):
            p_full = p_full[0]
        p_full = safe_float_conversion(p_full)
        if not np.isnan(p_full):
            return p_full
        
        logger.debug(f"Original prediction NaN, trying subgraph API with all past edges")
        all_past_ids = wrapper.dataset.events[wrapper.dataset.events['timestamp'] < t]['idx'].values
        edges_to_drop = np.array([], dtype=np.int64)
        p_full = wrapper.compute_edge_probabilities_for_subgraph(
            int(eid),
            edges_to_drop=edges_to_drop,
            result_as_logit=False,
            event_ids_to_rollout=all_past_ids
        )
        if isinstance(p_full, (list, tuple)):
            p_full = p_full[0]
        p_full = safe_float_conversion(p_full)
        return p_full
    except Exception as e:
        logger.error(f"All prediction attempts failed for edge {eid}: {e}")
        return float('nan')

def compute_prediction_with_subgraph(wrapper, graph_index, s: int, d: int, t: float, eid: int,
                                     allowed_ids: np.ndarray, logger) -> Tuple[float, str]:
    try:
        all_past_edges = graph_index.get_all_past_edges(eid)
        allowed_ids = np.unique(allowed_ids.astype(np.int64))
        edges_to_drop = np.setdiff1d(all_past_edges, allowed_ids, assume_unique=False).astype(np.int64)
        
        logger.debug(f"Computing CF for edge {eid}: past_edges={len(all_past_edges)}, allowed={len(allowed_ids)}, dropping={len(edges_to_drop)}")
        
        p = wrapper.compute_edge_probabilities_for_subgraph(
            int(eid),
            edges_to_drop=edges_to_drop,
            result_as_logit=False,
            event_ids_to_rollout=allowed_ids
        )
        if isinstance(p, (list, tuple)):
            p = p[0]
        p = safe_float_conversion(p)
        return p, "subgraph_edges_to_drop"
    except Exception as e:
        logger.debug(f"Subgraph computation failed: {e}")
        return float('nan'), "failed"

def verify_counterfactual_property(wrapper, graph_index, target_eid: int, candidate_nodes: List[int], logger) -> bool:
    try:
        target_info = graph_index.edge_info[target_eid]
        s, d, t = target_info['u'], target_info['i'], target_info['ts']

        p_full = compute_original_prediction_with_fallback(wrapper, s, d, t, target_eid, logger)
        if np.isnan(p_full):
            return False
        

        p_counterfactual, _ = compute_prediction_with_subgraph(
            wrapper, graph_index, s, d, t, target_eid, 
            np.array(candidate_nodes, dtype=np.int64), logger
        )
        
        if np.isnan(p_counterfactual):
            return False

        return abs(p_full - p_counterfactual) > 0.01
        
    except Exception as e:
        logger.debug(f"Verification failed for target {target_eid}: {e}")
        return False

def select_explanatory_nodes_greedy(wrapper, graph_index: TemporalGraphIndex,
                                   icm_calculator: ICMInfluenceCalculator,
                                   target_eid: int, k: int, candidate_pool: List[int],
                                   logger) -> List[int]:
    if len(candidate_pool) <= k:
        return candidate_pool
    
    target_info = graph_index.edge_info[target_eid]
    t = target_info['ts']
    
    icm_scores = icm_calculator.normalize_icm_scores(candidate_pool, target_eid, t)
    icm_scores.sort(key=lambda x: x[1], reverse=True)

    pre_selected_size = min(len(candidate_pool), max(k * 3, 20))
    pre_selected = [c for c, _ in icm_scores[:pre_selected_size]]
    
    selected_nodes = []
    remaining_candidates = pre_selected.copy()
    
    for _ in range(min(k, len(pre_selected))):
        if not remaining_candidates:
            break
            
        best_candidate = None
        best_score = -1
        
        for candidate in remaining_candidates:
            test_set = selected_nodes + [candidate]
            influence_score = icm_calculator.compute_edge_set_influence(test_set, target_eid)
            
            if influence_score > best_score:
                best_score = influence_score
                best_candidate = candidate
        
        if best_candidate is not None:
            selected_nodes.append(best_candidate)
            remaining_candidates.remove(best_candidate)
    
    return selected_nodes

def genInstanceX(wrapper, graph_index: TemporalGraphIndex,
                 icm_calculator: ICMInfluenceCalculator,
                 target_eid: int, k: int, candidate_pool: List[int],
                 delta: float, time_scale: str, logger) -> Tuple[List[int], float, Dict[str, Any]]:

    if target_eid not in graph_index.edge_info:
        return [], float('nan'), {"error": "target edge not found"}

    target_info = graph_index.edge_info[target_eid]
    s, d, t = target_info['u'], target_info['i'], target_info['ts']

    p_full = compute_original_prediction_with_fallback(wrapper, s, d, t, target_eid, logger)
    if np.isnan(p_full):
        logger.warning(f"Skipping edge {target_eid}: original prediction is NaN after all attempts")
        return [], float('nan'), {"error": "original prediction NaN"}


    valid_candidates = []
    for eid in candidate_pool:
        if eid in graph_index.edge_info:
            eid_ts = graph_index.edge_info[eid]['ts']
            if eid_ts < t:  
                valid_candidates.append(eid)
    
    if not valid_candidates:
        logger.warning(f"No valid candidates for target edge {target_eid}")
        return [], float('nan'), {"error": "no valid candidates", "original_prediction": p_full}

    logger.debug(f"genInstanceX: Target {target_eid}, Valid candidates: {len(valid_candidates)}")

    stats = {
        "icm_evaluations": 0, 
        "prediction_calls": 0, 
        "api_types": [], 
        "original_prediction": p_full
    }

    if len(valid_candidates) > 100:
        icm_scores = icm_calculator.normalize_icm_scores(valid_candidates, target_eid, t)
        icm_scores.sort(key=lambda x: x[1], reverse=True)
        valid_candidates = [c for c, _ in icm_scores[:100]]
    
    selected_edges = select_explanatory_nodes_greedy(
        wrapper, graph_index, icm_calculator, target_eid,
        k, valid_candidates, logger
    )

    if selected_edges:
        is_counterfactual = verify_counterfactual_property(
            wrapper, graph_index, target_eid, selected_edges, logger
        )
        
        if is_counterfactual:
            logger.debug(f"genInstanceX: Selected {len(selected_edges)} edges with verified counterfactual property")
            for i, edge in enumerate(selected_edges):
                logger.debug(f"  Selected edge {edge} (rank {i+1})")
        else:
            logger.debug(f"genInstanceX: Selected edges do not satisfy counterfactual property")

    if selected_edges:
        try:
            p_counterfactual, api_type = compute_prediction_with_subgraph(
                wrapper, graph_index, s, d, t, target_eid, 
                np.array(selected_edges, dtype=np.int64), logger
            )
            stats["prediction_calls"] += 1
            stats["api_types"].append(api_type)
            stats["counterfactual_valid"] = True
        except Exception as e:
            logger.debug(f"Failed to compute counterfactual prediction: {e}")
            p_counterfactual = float('nan')
            stats["counterfactual_valid"] = False
    else:
        p_counterfactual = float('nan')
        stats["counterfactual_valid"] = False

    final_icm_score = icm_calculator.compute_temporal_influence(selected_edges, target_eid, delta)
    stats["final_icm_score"] = final_icm_score
    stats["explanation_size"] = len(selected_edges)
    
    logger.debug(f"genInstanceX completed: edges={len(selected_edges)}, icm_score={final_icm_score:.4f}")
    
    return selected_edges, p_counterfactual, stats

def determine_time_window(events_df: pd.DataFrame, time_scale: str, logger) -> float:
    timestamps = events_df['ts'].values
    time_diffs = np.diff(np.sort(timestamps))
    median_diff = np.median(time_diffs) if len(time_diffs) > 0 else 0.0
    
    if time_scale == "seconds":
        window = min(86400, max(3600, median_diff * 100))
    else:
        window = 1000
    
    logger.info(f"Determined time window: {window} {time_scale}")
    return window

def load_dataset_and_events(args, logger):
    try:
        dataset = create_dataset_from_args(args, TrainTestDatasetParameters(0.2, 0.6, 0.8, 1000, 500, 500))
        wrapper = create_tgnn_wrapper_from_args(args, dataset)
        wrapper.set_evaluation_mode(True)
        logger.info("Dataset and wrapper created successfully")
    except Exception as e:
        logger.error(f"Failed to create dataset/wrapper: {e}")
        raise
    events_df = extract_events_from_dataset(dataset, args.dataset, logger).sort_values("ts").reset_index(drop=True)
    logger.info(f"Loaded {len(events_df)} events")
    return dataset, wrapper, events_df

def create_argument_parser():
    ap = argparse.ArgumentParser("TemGX: Temporal Graph eXplainer with ICM and Enhanced TRD")
    ap.add_argument("--dataset", type=str, required=True, help="Dataset path")
    ap.add_argument("--directed", action="store_true", help="Directed graph")
    ap.add_argument("--bipartite", action="store_true", help="Bipartite graph")
    ap.add_argument("--model", type=str, help="Model path")
    ap.add_argument("--cuda", action="store_true", help="Use CUDA")
    ap.add_argument("--type", type=str, required=True, help="Model type (e.g., TGAT)")
    ap.add_argument("--candidates_size", type=int, default=64, help="Candidates size")
    ap.add_argument("--update_memory_at_start", action="store_true")
    ap.add_argument("--max_explain", type=int, default=50, help="Max events to explain")
    ap.add_argument("--candidate_cap", type=int, default=200, help="Max candidate pool size")
    ap.add_argument("--sparsity", type=int, default=5, help="Explanation size (k)")
    ap.add_argument("--l_hops", type=int, default=2, help="L-hop neighbor limit")
    ap.add_argument("--time_window", type=float, default=None, help="Time window")
    ap.add_argument("--icm_alpha", type=float, default=1.0, help="ICM similarity weight")
    ap.add_argument("--icm_beta", type=float, default=0.1, help="ICM path length penalty")
    ap.add_argument("--icm_lambda", type=float, default=0.01, help="ICM time decay")
    ap.add_argument("--icm_gamma", type=float, default=0.5, help="TRD connectivity weight")
    ap.add_argument("--trd_scale", type=float, default=1.0, help="TRD scaling factor")
    ap.add_argument("--explained_ids", type=str, default=None, help="Specific event IDs to explain")
    ap.add_argument("--verbose", action="store_true")
    return ap

def main():
    ap = create_argument_parser()
    args = ap.parse_args()
    logger = setup_logging(args.verbose)
    logger.info("Starting TemGX: Temporal Graph eXplainer with Enhanced ICM and TRD")
    logger.info(f"Arguments: {vars(args)}")

    try:
        dataset, wrapper, events_df = load_dataset_and_events(args, logger)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    wrapper_type = type(wrapper).__name__
    logger.info(f"Using wrapper: {wrapper_type}")

    time_scale, conversion_factor = detect_time_scale(events_df, logger)
    logger.info("Building temporal graph index...")
    start_time = time.time()
    graph_index = TemporalGraphIndex(events_df, wrapper, logger)
    index_time = time.time() - start_time
    logger.info(f"Graph index built in {index_time:.2f}s")

    icm_calculator = ICMInfluenceCalculator(
        graph_index, 
        alpha=args.icm_alpha, 
        beta=args.icm_beta, 
        lambda_decay=args.icm_lambda,
        gamma=args.icm_gamma,
        trd_scale=args.trd_scale,
        time_scale=time_scale
    )
   

    if args.time_window is None:
        time_window = determine_time_window(events_df, time_scale, logger)
    else:
        time_window = float(args.time_window)
    logger.info(f"Using time window: {time_window}")

    if args.explained_ids and os.path.exists(args.explained_ids):
        if args.explained_ids.endswith(".npy"):
            target_eids = np.load(args.explained_ids)
        else:
            target_eids = np.loadtxt(args.explained_ids, dtype=np.int64)
    else:
        all_eids = events_df["e_id"].values
        start_idx = len(all_eids) // 2
        candidate_eids = all_eids[start_idx:]
        target_eids = np.random.choice(candidate_eids, size=min(args.max_explain, len(candidate_eids)), replace=False)

    logger.info(f"Will explain {len(target_eids)} events using genInstanceX algorithm")

    results = []
    total_start_time = time.time()

    for i, target_eid in enumerate(tqdm(target_eids, desc="Running genInstanceX")):
        if target_eid not in graph_index.edge_info:
            logger.warning(f"Target edge {target_eid} not found")
            continue

        candidate_pool = graph_index.get_l_hop_temporal_neighbors(target_eid, args.l_hops, time_window, time_scale)

        if len(candidate_pool) > args.candidate_cap:
            candidate_pool = np.random.choice(candidate_pool, args.candidate_cap, replace=False).tolist()

        if not candidate_pool:
            logger.warning(f"No candidates found for target edge {target_eid}")
            continue

        try:
            explanation_eids, p_counterfactual, stats = genInstanceX(
                wrapper, graph_index, icm_calculator, target_eid,
                args.sparsity, candidate_pool, time_window, time_scale, logger
            )
        except Exception as e:
            logger.error(f"genInstanceX failed for edge {target_eid}: {e}")
            continue

        if "error" in stats and stats["error"] == "original prediction NaN":
            continue

        p_full = stats.get("original_prediction", float("nan"))
        p_full = safe_float_conversion(p_full)
        p_counterfactual = safe_float_conversion(p_counterfactual)

        if np.isnan(p_full):
            continue

        if not np.isnan(p_counterfactual) and stats.get("counterfactual_valid", False):
            fidelity_minus = max(0.0, p_full - p_counterfactual)
            fidelity_plus = max(0.0, p_counterfactual - p_full)
            is_counterfactual = fidelity_minus > 0.01
            prediction_change = abs(p_full - p_counterfactual)
        else:
            fidelity_minus = np.nan
            fidelity_plus = np.nan
            is_counterfactual = False
            prediction_change = np.nan

        sparsity = len(explanation_eids) / len(candidate_pool) if candidate_pool else 0.0

        result = {
            "explained_event_id": int(target_eid),
            "original_prediction": p_full,
            "counterfactual_prediction": p_counterfactual,
            "fidelity_minus": fidelity_minus,
            "fidelity_plus": fidelity_plus,
            "explanation_event_ids": json.dumps([int(x) for x in explanation_eids]),
            "explanation_size": len(explanation_eids),
            "candidate_size": len(candidate_pool),
            "sparsity": sparsity,
            "is_counterfactual": is_counterfactual,
            "prediction_change": prediction_change,
            "icm_score": stats.get("final_icm_score", np.nan),
            "counterfactual_valid": stats.get("counterfactual_valid", False),
            "l_hops": args.l_hops,
            "time_window": time_window,
            "time_scale": time_scale,
            "algorithm": "temgx_genInstanceX"
        }
        results.append(result)

        fidelity_str = f"{fidelity_minus:.4f}" if not np.isnan(fidelity_minus) else "nan"
        cf_str = f"{p_counterfactual:.4f}" if not np.isnan(p_counterfactual) else "nan"
        logger.info(f"Edge {int(target_eid)} ({i+1}/{len(target_eids)}): pred={p_full:.4f}→{cf_str}, fidelity-={fidelity_str}, size={len(explanation_eids)}")

    total_time = time.time() - total_start_time
    logger.info(f"Total processing time: {total_time:.2f}s")

    if results:
        dataset_name = getattr(dataset, "name", None) or os.path.basename(args.dataset.rstrip('/'))
        out_dir = os.path.join("resources", "results", dataset_name, "temgx")
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f"results_{args.type}_{dataset_name}_temgx_genInstanceX.csv")
        df = pd.DataFrame(results)
        df.to_csv(out_csv, index=False)
        logger.info(f"Saved results to: {out_csv}")

        valid_results = df[(~df["original_prediction"].isna()) & (~df["counterfactual_prediction"].isna())]
        valid_counterfactuals = valid_results[valid_results["counterfactual_valid"] == True]

        print("="*60)
        print("TEMGX RESULTS USING genInstanceX ALGORITHM:")
        print(f"Total explanations: {len(results)}")
        print(f"Valid counterfactual predictions: {len(valid_counterfactuals)}")
        if len(valid_counterfactuals) > 0:
            print(f"True counterfactual explanations: {int(valid_counterfactuals['is_counterfactual'].sum())}")
            if not valid_counterfactuals['fidelity_minus'].isna().all():
                print(f"Median fidelity-: {float(valid_counterfactuals['fidelity_minus'].median()):.4f}")
            if not valid_counterfactuals['prediction_change'].isna().all():
                print(f"Mean prediction change: {float(valid_counterfactuals['prediction_change'].mean()):.4f}")
        print(f"Time window: {float(time_window)} {time_scale}")
        print(f"Adaptive lambda: {icm_calculator.lambda_decay:.6f}")
        print(f"Algorithm: genInstanceX with ICM and TRD")
        print("="*60)
    else:
        logger.error("No valid results generated")

if __name__ == "__main__":
    main()
    