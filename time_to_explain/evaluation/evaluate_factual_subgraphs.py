import argparse
import ast
import os.path
from pathlib import Path

import numpy as np
import pandas as pd

from cody.data import TrainTestDatasetParameters
from cody.utils import ProgressBar
from common import (add_dataset_arguments, add_wrapper_model_arguments, create_dataset_from_args,
                    create_tgnn_wrapper_from_args, parse_args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Explainer Evaluation')
    add_dataset_arguments(parser)
    add_wrapper_model_arguments(parser)
    parser.add_argument('--results', required=True, type=str,
                        help='Path to the file containing the results of the factual explainer evaluation')

    args = parse_args(parser)

    dataset = create_dataset_from_args(args, TrainTestDatasetParameters(0.2, 0.6, 0.8, 500,
                                                                        500, 500))

    tgn_wrapper = create_tgnn_wrapper_from_args(args, dataset)
    tgn_wrapper.set_evaluation_mode(True)

    def resolve_results_path(results_path: str) -> str:
        path = Path(results_path)
        if path.is_dir():
            parquet_files = sorted(path.glob('*.parquet'))
            csv_files = sorted(path.glob('*.csv'))
            candidates = parquet_files or csv_files
            if not candidates:
                raise FileNotFoundError(f'No .parquet or .csv files found in {path}')
            if len(candidates) > 1:
                names = '\n  '.join(p.name for p in candidates)
                raise FileExistsError(
                    f'Multiple result files found in {path}. Provide an explicit file:\n  {names}'
                )
            return str(candidates[0])
        if not path.exists():
            if path.parent.exists():
                available = sorted(path.parent.glob('*.parquet')) + sorted(path.parent.glob('*.csv'))
                if available:
                    names = '\n  '.join(p.name for p in available)
                    raise FileNotFoundError(
                        f'Failed to locate {path}. Available results in {path.parent}:\n  {names}'
                    )
            raise FileNotFoundError('Failed to locate the file containing the results')
        return str(path)

    results_path = resolve_results_path(args.results)
    results_file = Path(results_path)

    if results_file.suffix == '.parquet':
        results_df = pd.read_parquet(results_file)
    elif results_file.suffix == '.csv':
        results_df = pd.read_csv(results_file)
        # drop the common index column written by pandas
        if len(results_df.columns) > 0 and str(results_df.columns[0]).startswith('Unnamed'):
            results_df = results_df.iloc[:, 1:]
    else:
        raise ValueError(f'Unsupported results extension: {results_file.suffix}')

    out_base = results_file.with_suffix('')
    out_csv = out_base.with_suffix('.csv')
    out_parquet = out_base.with_suffix('.parquet')

    def _parse_results_cell(cell):
        """Parse the per-row 'results' cell.

        - Parquet typically stores this as a list/np.ndarray of dicts.
        - CSV stores a string representation.
        """
        if cell is None:
            return []
        if isinstance(cell, (list, tuple)):
            return list(cell)
        if hasattr(cell, 'tolist'):
            try:
                return cell.tolist()
            except Exception:
                pass
        if isinstance(cell, str):
            s = cell.strip()
            if s in {'', 'nan', 'NaN', 'None'}:
                return []
            # CSV stores python-literal-like strings; ast is robust for this.
            try:
                return ast.literal_eval(s)
            except Exception:
                # Fallback: try JSON after replacing single quotes.
                try:
                    import json

                    return json.loads(s.replace("'", '"'))
                except Exception as exc:
                    raise ValueError(f"Could not parse 'results' cell: {s[:200]}") from exc
        # Last resort
        return []

    explanation_events = []
    explanation_sizes = []
    scores = []
    progress_bar = ProgressBar(len(results_df), 'Adding counterfacual information')
    for index, row in results_df.iterrows():
        progress_bar.next()
        results = _parse_results_cell(row.get('results'))
        if not results:
            # keep alignment; write empty explanation
            explanation_events.append([])
            explanation_sizes.append(0)
            scores.append(np.nan)
            continue

        best_prediction = results[0]['prediction']
        best_result = results[0]
        for result in results:
            if result['prediction'] > best_prediction:
                best_prediction = result['prediction']
                best_result = result
        explanation_events.append(best_result['event_ids_in_explanation'])
        explanation_sizes.append(len(best_result['event_ids_in_explanation']))

        score, _ = tgn_wrapper.compute_edge_probabilities_for_subgraph(row['explained_event_id'],
                                                                       edges_to_drop=np.array(
                                                                           best_result['event_ids_in_explanation']),
                                                                       result_as_logit=True)
        tgn_wrapper.reset_model()
        scores.append(score.detach().cpu().item())

    progress_bar.close()
    results_df['explanation_event_ids'] = explanation_events
    results_df['explanation_size'] = explanation_sizes
    results_df['sparsity'] = results_df['explanation_size'] / results_df['candidate_size']
    results_df['counterfactual_prediction'] = scores

    results_df.to_csv(out_csv, index=False)
    print(f'Results saved to {out_csv}')
    try:
        results_df.to_parquet(out_parquet)
        print(f'Results saved to {out_parquet}')
    except ImportError:
        print('Could not save to parquet format. Install pyarrow if you want to export to parquet format')
