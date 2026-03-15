# TemGX

**TemGX (Temporal Graph eXplainer)** is a training‑free explanation framework for **Temporal Graph Neural Networks (TGNNs)**.  
It produces **instance‑level counterfactual explanations** for a given, pretrained TGNN by discovering compact temporal subgraphs whose removal changes the model’s prediction.

---

## Highlights

- **Training‑free**: Runs on a fixed, pretrained TGNN. No explainer retraining is required.
- **Instance‑level counterfactuals**: Generates edge‑level temporal explanations around a target event.
- **Temporal influence scoring**: Combines ICM‑style propagation influence, Temporal Resistance Distance (TRD), and time decay.
- **Greedy selection**: Efficient candidate selection over an L‑hop temporal neighborhood (details in the paper).

---

## Repository Layout

```
TemGX/
├─ Link/
│  ├─ scripts/
│  │  ├─ temgx.py           # main entry for explanations (TemGX)
│  │  ├─ temgx_cli.py     
│  │  ├─ train_tgn.py       # TGNN training (TGN)
│  │  └─ train_tgat.py      # TGNN training (TGAT)
│  ├─ temgxlib/             # core library (ICM/TRD)
│  ├─ resources/
│  │  ├─ datasets/
│  │  │  └─ processed/      # processed datasets (e.g., ucim, wiki, ...)
│  │  └─ models/            # pretrained checkpoints (e.g., TGN-ucim.pth)
│  └─ submodules/
│     ├─ tgn/               # TGN implementation 
│     └─ ttgn/              # TGATimplementation
└─ README.md
```

> **Note**: `submodules/tgn` and `submodules/ttgn` must be on `PYTHONPATH` to import the model code.

---

## Environment

- Python ≥ 3.9
- PyTorch ≥ 1.12
- Other packages: `numpy`, `pandas`, `scipy`, `tqdm`,`networkx`

Install:
```bash
pip install -r requirements.txt
```

Set `PYTHONPATH` (adjust to your repo root):
```bash
export PYTHONPATH="$(pwd)/Link:$(pwd)/Link/submodules/tgn:$(pwd)/Link/submodules/ttgn:$PYTHONPATH"
```

---

## Datasets
###  Prepare Datasets
The first step to running the experiments is the preparation of the datasets.  
The datasets can be downloaded from the following sites:

- **Wikipedia**: [https://snap.stanford.edu/jodie/wikipedia.csv](https://snap.stanford.edu/jodie/wikipedia.csv)  
- **UCI-Messages**: [http://opsahl.co.uk/tnet/datasets/OCnodeslinks.txt](http://opsahl.co.uk/tnet/datasets/OCnodeslinks.txt)  
- **METR-LA**: [https://github.com/liyaguang/DCRNN/tree/master/data](https://github.com/liyaguang/DCRNN/tree/master/data)  
- **PEMS-BAY**: [https://github.com/liyaguang/DCRNN/tree/master/data](https://github.com/liyaguang/DCRNN/tree/master/data)

The raw dataset files should have the same format as the Wikipedia dataset, that is:

- **First column**: Source node ids  
- **Second column**: Target node ids  
- **Third column**: UNIX timestamp  
- **Fourth column**: State label (not necessary for link prediction task)  
- **Fifth column and onwards**: Comma-separated list of edge features  

The UCI-Messages dataset does not have this form by default.  
To make the conversion easier, the `format_uci_data.py` script is provided.  
First download the dataset file, for example for UCI-Messages:

```bash
curl http://opsahl.co.uk/tnet/datasets/OCnodeslinks.txt > UCI-Messages.txt
```
TemGX expects **processed temporal interaction data** under:
```
Link/resources/datasets/processed/<dataset_name>/
  ├─ <name>_data.csv
  ├─ <name>_edge_features.npy
  └─ <name>_node_features.npy
```

`*_data.csv` should contain at least: an **edge/event id**, **source node**, **destination node**, and **timestamp** columns.  
  The loader will auto-detect common headers (e.g., `idx/e_id/event_id`, `user_id/src`, `item_id/dst`, `timestamp/ts`).
  Feature arrays should match the number of unique edges/nodes.

For quick start, you can place already‑processed datasets into the `processed/` folder as shown above.



## Training TGNN Models

TemGX uses pretrained TGNN checkpoints. You can train **TGN** or **TGAT** using the provided scripts.

### Train TGN
```bash
python Link/scripts/train_tgn.py \
  --dataset Link/resources/datasets/processed/ucim \
  --save    Link/resources/models/ucim/TGN-ucim.pth \
  --epochs  50 \
  --cuda
```

### Train TGAT
```bash
python Link/scripts/train_tgat.py \
  --dataset Link/resources/datasets/processed/ucim \
  --save    Link/resources/models/ucim/TGAT-ucim.pth \
  --epochs  50 \
  --cuda
```

Common options (see `-h` for full list):
- `--dataset` : path to processed dataset folder
- `--save`    : output checkpoint file
- `--epochs`  : training epochs
- `--cuda`    : enable GPU

---

## Generating Explanations (TemGX)

Example: explain 50 target edges on **UCIM** with a pretrained **TGN**:

```bash
export PYTHONPATH="$(pwd)/Link:$(pwd)/Link/submodules/tgn:$(pwd)/Link/submodules/ttgn:$PYTHONPATH"

python Link/scripts/temgx.py \
  --dataset Link/resources/datasets/processed/ucim \
  --type TGN \
  --model Link/resources/models/ucim/TGN-ucim.pth \
  --cuda \
  --max_explain 50 \
  --sparsity 5 \
  --l_hops 2
```

You can also provide explicit target event IDs:
```bash
python Link/scripts/temgx.py \
  --dataset Link/resources/datasets/processed/ucim \
  --type TGN \
  --model Link/resources/models/ucim/TGN-ucim.pth \
  --cuda \
  --explained_ids Link/resources/datasets/processed/ucim/explained_ids.npy
```

### Key arguments
- `--dataset`         : processed dataset path
- `--type`            : TGNN type (`TGN` or `TGAT`)
- `--model`           : checkpoint path
- `--cuda`            : enable GPU
- `--max_explain`     : number of events to explain
- `--sparsity`        : explanation size `k` (max #edges in explanation)
- `--l_hops`          : L‑hop temporal neighborhood
- `--candidate_cap`   : cap of candidate pool size
- `--time_window`     : fixed window (if omitted, TemGX detects an adaptive window)
- `--icm_/lambda/gamma`, `--trd_scale` : scoring hyperparameters

Outputs are written to:
```
resources/results/<dataset>/temgx/results_<TYPE>_<dataset>_temgx_genInstanceX.csv
```

The console will also print a short summary (counts, median Fidelity‑, time window, etc.).

---

## Metrics

Let `p_full` be the original prediction probability on the full graph at time `t`, and  
`p_cf` the prediction after removing the explanation subgraph (`Gε`) around the target.

**Fidelity** (higher is better):
\[
\text{Fidelity} = \max(0,\; p_{\text{full}} - p_{\text{cf}})
\]

**Sparsity** (higher means more concise):
\[
\text{Sparsity} = \frac{|E_\epsilon|}{|E^L(V_T)|}
\]
where \(E_\epsilon\) are selected explanatory temporal edges and \(E^L(V_T)\) are temporal edges within L‑hop of target nodes.

(If enabled in the evaluation pipeline, we also report a summary curve and its area, **AUFSC**.)



## License

This code is released for research purposes. See the `LICENSE` file for details.
