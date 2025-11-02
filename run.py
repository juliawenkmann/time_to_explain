from time_to_explain.models.models import TemporalSuite, Model

suite = TemporalSuite(root="~/temporal_workspace")

# Built-in datasets
suite.prepare_dataset("wikipedia")     # downloads from JODIE
suite.prepare_dataset("uci_messages")  # downloads SNAP CollegeMsg

# For UCI forums (Opsahl), put OCnodeslinks.txt under ~/temporal_workspace/data/uci_forums/
# then:
suite.prepare_dataset("uci_forums")

# Train all three models
suite.train(Model.TGN, "wikipedia", gpu=0, epochs=5)
#suite.train(Model.TGAT, "uci_messages", gpu=0, epochs=5, num_neighbors=20)
#suite.train(Model.GRAPHMIXER, "uci_forums", gpu=0, num_neighbors=30, use_one_hot_nodes=True)

# Train on your custom dataset
#suite.register_custom_csv(
#    name="my_data",
#    csv_path="data/my_data.csv",   # columns: src,dst,ts,label,(edge_id optional)
#    bipartite=False
#)
#suite.train(Model.TGN, "my_data", gpu=0)
