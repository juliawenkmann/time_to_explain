
# eval/protocol.py
def run_experiment(cfg):
    ds = REG["datasets"][cfg.dataset](cfg)
    train, val, test = ds.splits(cfg.splits, seed=cfg.seed)

    model = REG["models"][cfg.model](cfg)
    model.fit(ds, train)                   # single training per dataset/seed
    save_ckpt(model)

    results = []
    for exp_name in cfg.explainers:
        explainer = REG["explainers"][exp_name](model, **cfg.explainer_overrides.get(exp_name, {}))
        for instance in iter_instances(ds, test, task=cfg.task):
            ex = timed_budgeted_explain(explainer, instance, cfg.budget)
            for mname in cfg.metrics:
                metric = REG["metrics"][mname](cfg)
                score = metric(ex, model, instance)
                results.append(row(exp_name, instance.id, mname, score))

    table = aggregate_and_test(results)    # paired tests + CIs + Holm–Bonferroni
    save_jsonl(results); save_table(table)
    return table
