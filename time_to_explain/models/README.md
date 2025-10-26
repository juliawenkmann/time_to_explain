# Train the TGNN Models

Next, a TGNN model is trained for each dataset. To do the training run the following command from the 
[/scripts](./scripts) directory:

```shell
bash train_tgnn_model.bash MODEL-TYPE DATASET-NAME --bipartite

```
e.g.:

```shell
bash time_to_explain/models/train_tgnn_model.bash TGN wikipedia --bipartite
```

Replace ``MODEL-TYPE`` with the type of the model you want to evaluate, e.g., 'TGAT' or 'TGN'.

Replace ``DATASET-NAME`` with the name of the dataset on which you want to train the TGN model, e.g., 'uci', 
'wikipedia', etc.

Only provide the ``--bipartite`` flag if the underlying dataset is a bipartite graph (Wikipedia/UCI-Forums), else
omit the ``--bipartite`` flag from the command.

## Disclaimer: This was taken from https://github.com/daniel-gomm/CoDy. 