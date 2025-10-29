import numpy as np
import itertools
import numpy as np
from time_to_explain.data.data import ContinuousTimeDynamicGraphDataset
from time_to_explain.setup.constants import COL_ID
from submodules.models.tgn.TTGN.utils.utils import NeighborFinder
from time_to_explain.data.data import BatchData, ContinuousTimeDynamicGraphDataset
from submodules.models.tgn.TTGN.utils.data_processing import Data


def to_data_object(dataset: ContinuousTimeDynamicGraphDataset, edges_to_drop: np.ndarray = None) -> Data:
    """
    Convert the dataset to a data object that can be used as input for a neighborhood finder
    @param dataset: Dataset to convert to the data object
    @param edges_to_drop: Edges that should be excluded from the data
    @return: Data object of the dataset
    """
    if edges_to_drop is not None:
        edge_mask = ~np.isin(dataset.edge_ids, edges_to_drop)
        return Data(dataset.source_node_ids[edge_mask], dataset.target_node_ids[edge_mask],
                    dataset.timestamps[edge_mask], dataset.edge_ids[edge_mask], dataset.labels[edge_mask])
    return Data(dataset.source_node_ids, dataset.target_node_ids, dataset.timestamps, dataset.edge_ids, dataset.labels)

def _unpack_metrics(metrics):
    """
    Accept (ap, auc, acc), (ap, auc), or a dict with keys like
    'ap'/'average_precision', 'auc'/'roc_auc', 'acc'/'accuracy'.
    Returns: (ap, auc, acc_or_None)
    """
    if isinstance(metrics, dict):
        ap = metrics.get('ap', metrics.get('average_precision'))
        auc = metrics.get('auc', metrics.get('roc_auc'))
        acc = metrics.get('acc', metrics.get('accuracy'))
        return ap, auc, acc

    if isinstance(metrics, (list, tuple)):
        if len(metrics) >= 3:
            return metrics[0], metrics[1], metrics[2]
        if len(metrics) == 2:
            return metrics[0], metrics[1], None

    # Unexpected shape
    raise ValueError(f"Unsupported metrics format from eval_edge_prediction: {type(metrics)} -> {metrics}")


def find_candidate_events(dataset: ContinuousTimeDynamicGraphDataset, neighborhood_finder: NeighborFinder,
                          target_event_idx: int, num_hops: int, candidates_size: int, subgraph_event_ids):
    target_mask = np.isin(dataset.events[COL_ID], target_event_idx)
    target_nodes = dataset.target_node_ids[target_mask][0]
    source_nodes = dataset.source_node_ids[target_mask][0]
    timestamps = dataset.timestamps[target_mask][0]

    accu_edge_idx = []
    accu_node = [[target_nodes, source_nodes, ]]
    accu_ts = [[timestamps, timestamps, ]]

    for i in range(num_hops):
        last_nodes = accu_node[-1]
        last_ts = accu_ts[-1]

        out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = neighborhood_finder.get_temporal_neighbor(
            last_nodes,
            last_ts,
            n_neighbors=candidates_size,
            edge_idx_preserve_list=subgraph_event_ids,  # NOTE: not needed?
        )

        out_ngh_node_batch = out_ngh_node_batch.flatten()
        out_ngh_eidx_batch = out_ngh_eidx_batch.flatten()
        out_ngh_t_batch = out_ngh_t_batch.flatten()

        mask = out_ngh_node_batch != 0
        out_ngh_node_batch = out_ngh_node_batch[mask]
        out_ngh_eidx_batch = out_ngh_eidx_batch[mask]
        out_ngh_t_batch = out_ngh_t_batch[mask]

        out_ngh_node_batch = out_ngh_node_batch.tolist()
        out_ngh_t_batch = out_ngh_t_batch.tolist()
        out_ngh_eidx_batch = out_ngh_eidx_batch.tolist()

        accu_node.append(out_ngh_node_batch)
        accu_ts.append(out_ngh_t_batch)
        accu_edge_idx.append(out_ngh_eidx_batch)

    unique_e_idx = np.array(list(itertools.chain.from_iterable(accu_edge_idx)))
    unique_e_idx = unique_e_idx[unique_e_idx != 0]  # NOTE: 0 are padded e_idxs
    # unique_e_idx = unique_e_idx - 1 # NOTE: -1, because ngh_finder stored +1 e_idxs
    unique_e_idx = np.unique(unique_e_idx).tolist()

    candidate_events = unique_e_idx
    if len(candidate_events) > candidates_size:
        candidate_events = candidate_events[-candidates_size:]
        candidate_events = sorted(candidate_events)

    return candidate_events, unique_e_idx
