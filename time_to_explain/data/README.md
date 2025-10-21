# Prepare Datasets
The first step to running the experiments is the preparation of the datasets. The datasets can be downloaded from the 
following sites:

- [Wikipedia](http://snap.stanford.edu/jodie/#datasets)
- [UCI-Messages/UCI-Forums](https://toreopsahl.com/datasets/)

For Wikipedia do: 

```shell
curl -O http://snap.stanford.edu/jodie/wikipedia.csv
```

The raw dataset files should have the same format as the Wikipedia dataset; that is: 

- First column: Source node ids
- Second column: Target node ids
- Third column: UNIX timestamp
- Fourth column: State label (not necessary for link prediction task)
- Fifth column and onwards: Comma seperated list of edge features

The UCI datasets do not have this form by default. To make the conversion easier, the 
[format_uci_data.py](./format_uci_data.py) is provided. First download the dataset to a file from the website 
mentioned above, for example for the UCI-Messages dataset by running:

```shell
curl http://opsahl.co.uk/tnet/datasets/OCnodeslinks.txt > UCI-Messages.txt
```

Then use the script to convert the downloaded file to an appropriate .csv file by running:

```shell
python format_uci_data.py --input UCI-Messages.txt --output ../resources/datasets/raw/uci_forums.csv
```
Adjust the ``--input`` and the ``--output`` parameters accordingly.

Place the correctly formatted datasets as .csv files in the [/resources/datasets/raw](./resources/datasets/raw) directory.

To prepare the raw datasets for the usage run the following command from the [/data](./data) directory:

Wikipedia/UCI-Forums:
```shell 
bash preprocess_data.bash DATASET-NAME --bipartite
```

Replace ``DATASET-NAME`` with the name of the dataset that should be processed, e.g., 'wikipedia' or 'uci_forums'. The name 
corresponds to the file name without extension, i.e., if the wikipedia dataset is named 'wikipedia.csv', pass 'wikipedia'
as the ``DATASET-NAME``.

UCI-Messages:

```shell 
bash preprocess_data.bash DATASET-NAME
```

For the UCI-Messages dataset the ``--bipartite`` flag is omitted, as the dataset describes a unipartite graph.


## Disclaimer: This is heavily based on [CoDy](https://github.com/daniel-gomm/CoDy). 
