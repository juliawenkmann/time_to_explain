
# suite_cli.py
import argparse
from time_to_explain.models.models import TemporalSuite, Model

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--dataset", required=True)   # wikipedia | uci_messages | uci_forums | custom_name
p.add_argument("--model", required=True, choices=[m.value for m in Model])
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--batch_size", type=int, default=200)
p.add_argument("--gpu", type=int, default=-1)
args = p.parse_args()

suite = TemporalSuite(args.root)
if args.dataset in {"wikipedia","uci_messages","uci_forums"}:
    suite.prepare_dataset(args.dataset)

res = suite.train(model=Model(args.model), dataset=args.dataset,
                  epochs=args.epochs, batch_size=args.batch_size, gpu=args.gpu)
print("Log dir:", res["log_dir"])
print("AP:", res.get("AP"), "AUC:", res.get("AUC"))
