import pandas as pd
import numpy as np
import json
from pgmpy.models import BayesianNetwork, DynamicBayesianNetwork as DBN
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from sklearn.preprocessing import KBinsDiscretizer

np.random.seed(42)
json_file_path = 'replaced with your instacen level results!'

with open(json_file_path, 'r') as f:
    data_json = json.load(f)

union_Vs = data_json.get('union_Vs', [])
union_edges = data_json.get('union_edges', [])
time_slices = data_json.get('time_slices', [])

num_time_slices = len(time_slices)

all_nodes = set(union_Vs)
if num_time_slices > 0:
    Vt = time_slices[0].get('Vt')
    if Vt is not None:
        all_nodes.add(Vt)
    else:
        raise KeyError("Missing key 'Vt' in the first time slice.")
else:
    raise ValueError("No 'time_slices' found in JSON data.")

all_nodes = list(all_nodes)
print(f"All nodes: {all_nodes}")
print(f"Total time slices: {num_time_slices}")

data_snapshots = []
for t in range(num_time_slices):
    time_slice = time_slices[t]
    Vs = time_slice.get('Vs', [])
    Vt_current = time_slice.get('Vt')
    nodes_in_time_slice = set(all_nodes)
    node_labels = time_slice.get('node_labels', {})
    node_features = time_slice.get('node_features', {})
    shapley_values = time_slice.get('shapley_values', {})
    edges = time_slice.get('edges', [])
    df = pd.DataFrame({
        'node': list(nodes_in_time_slice),
        'label': [int(node_labels.get(str(node), 0)) for node in nodes_in_time_slice],
        'feature': [float(node_features.get(str(node), 0.0)) for node in nodes_in_time_slice],
        'snapshot': t
    })
    data_snapshots.append(df)

data = pd.concat(data_snapshots, ignore_index=True)
print("\nCombined Data:")
print(data.head())

flattened_data = data.pivot(index="snapshot", columns="node", values="label")
flattened_data.columns = [f"node_{int(col)}" for col in flattened_data.columns]
flattened_data = flattened_data.reset_index(drop=True)

print("\nFlattened Data (Labels):")
print(flattened_data.head())

flattened_data.fillna(0, inplace=True)
flattened_data = flattened_data.astype(int)

node_columns = flattened_data.columns.tolist()
Vt_column = f"node_{Vt}"
subnode_columns = [col for col in node_columns if col != Vt_column]
fixed_edges = [(col, Vt_column) for col in subnode_columns]
all_possible_edges = [(u, v) for u in node_columns for v in node_columns if u != v]
blacklist = [edge for edge in all_possible_edges if edge not in fixed_edges]

hc = HillClimbSearch(flattened_data)
model = hc.estimate(
    scoring_method=BicScore(flattened_data),
    fixed_edges=fixed_edges,
    blacklist=blacklist,
    max_iter=100
)

print("\nLearned BN structure (Edges):")
print(model.edges())

inter_snapshot_data = []
for t in range(num_time_slices - 1):
    snapshot_t = flattened_data.iloc[t]
    snapshot_t1 = flattened_data.iloc[t + 1]
    inter_data = {}
    for col in node_columns:
        inter_data[f"{col}_t0"] = snapshot_t[col]
        inter_data[f"{col}_t1"] = snapshot_t1[col]
    inter_snapshot_data.append(inter_data)

inter_snapshot_df = pd.DataFrame(inter_snapshot_data)
inter_snapshot_df.fillna(0, inplace=True)
inter_snapshot_df = inter_snapshot_df.astype(int)

print("\nInter-snapshot Data:")
print(inter_snapshot_df.head())

fixed_edges_inter = [(f"{col}_t0", f"{Vt_column}_t1") for col in node_columns]

hc_inter = HillClimbSearch(inter_snapshot_df)
inter_model = hc_inter.estimate(
    scoring_method=BicScore(inter_snapshot_df),
    fixed_edges=fixed_edges_inter,
    blacklist=[],
    max_iter=100
)

print("\nLearned Inter-snapshot structure (Edges):")
print(inter_model.edges())

def filter_inter_snapshot_edges(edges):
    filtered_edges = []
    for u, v in edges:
        if u.endswith('_t0') and v == f"{Vt_column}_t1":
            filtered_edges.append((u, v))
    return filtered_edges

filtered_inter_edges = filter_inter_snapshot_edges(inter_model.edges())
print("\nFiltered Inter-snapshot Edges:")
print(filtered_inter_edges)

dbn = DBN()
for t in range(num_time_slices):
    for u, v in model.edges():
        dbn.add_edge(f"{u}_t{t}", f"{v}_t{t}")
for u, v in filtered_inter_edges:
    dbn.add_edge(u, v)

print("\nDBN Edges:")
print(dbn.edges())

train_data = []
for t in range(num_time_slices - 1):
    snapshot_t = flattened_data.iloc[t]
    snapshot_t1 = flattened_data.iloc[t + 1]
    data_dict = {}
    for col in node_columns:
        data_dict[f"{col}_t0"] = snapshot_t[col]
        data_dict[f"{col}_t1"] = snapshot_t1[col]
    train_data.append(data_dict)

train_df = pd.DataFrame(train_data)
print("\nTraining Data for DBN (before perturbation):")
print(train_df.head())

discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')

feature_means = data.copy()
feature_means['mean_feature'] = feature_means['feature']
feature_flat = feature_means.pivot(index="snapshot", columns="node", values="mean_feature")
feature_flat.columns = [f"node_{int(col)}" for col in feature_flat.columns]
feature_flat = feature_flat.reset_index(drop=True)

print("\nFeature Means (flattened):")
print(feature_flat.head())

all_delta_values = []
for t in range(1, num_time_slices):
    current_means = feature_flat.iloc[t]
    prev_means = feature_flat.iloc[t - 1]
    delta = current_means - prev_means
    for col in subnode_columns:
        delta_value = delta[col]
        all_delta_values.append([delta_value])

all_delta_values = np.array(all_delta_values)
all_delta_values = all_delta_values[~np.isnan(all_delta_values)].reshape(-1, 1)
discretizer.fit(all_delta_values)
print("Discretizer fitted on feature differences.")

def reassign_labels(train_df, feature_flat, discretizer, num_time_slices, node_columns):
    perturbed_train_df = train_df.copy()
    for t in range(num_time_slices - 1):
        current_means = feature_flat.iloc[t + 1]
        prev_means = feature_flat.iloc[t]
        delta = current_means - prev_means
        for col in node_columns:
            delta_value = delta[col]
            if np.isnan(delta_value):
                new_label = 0
            else:
                new_label = discretizer.transform([[delta_value]])[0][0]
            perturbed_train_df.at[t, f"{col}_t1"] = int(new_label)
    return perturbed_train_df

perturbed_train_df = reassign_labels(train_df, feature_flat, discretizer, num_time_slices, node_columns)

print("\nTraining Data for DBN (after perturbation):")
print(perturbed_train_df.head())
print(f"Perturbed DBN Data Shape: {perturbed_train_df.shape}")

perturbed_train_df = perturbed_train_df.astype(int)

num_augment = 20
augmented_data = []
for _ in range(num_augment):
    augmented_sample = perturbed_train_df.sample(n=1, replace=True).copy()
    noise = np.random.randint(-1, 2, size=augmented_sample.shape)
    augmented_sample += noise
    augmented_sample = augmented_sample.clip(lower=0)
    augmented_data.append(augmented_sample)

augmented_train_df = pd.concat(augmented_data, ignore_index=True)
final_train_df = pd.concat([perturbed_train_df, augmented_train_df], ignore_index=True)

print("\nFinal Training Data for DBN (after augmentation):")
print(final_train_df.head())
print(f"Final DBN Training Data Shape: {final_train_df.shape}")

try:
    dbn.fit(final_train_df, estimator=MaximumLikelihoodEstimator)
    print("\nDBN fitting completed successfully.")
except ValueError as e:
    print(f"Error during DBN fitting: {e}")

try:
    print("\nConditional Probability Distributions:")
    for cpd in dbn.get_cpds():
        print(cpd)
except Exception as e:
    print(f"Error while retrieving CPDs: {e}")

node_to_query = f"{Vt_column}_t1"
if node_to_query in dbn.nodes():
    try:
        cpd = dbn.get_cpds(node_to_query)
        print(f"\nCPD for node {node_to_query}:")
        print(cpd)
    except KeyError:
        print(f"\nCPD for node {node_to_query} not found.")
else:
    print(f"\nNode {node_to_query} not found in the DBN.")