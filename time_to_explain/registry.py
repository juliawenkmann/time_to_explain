# gxai/registry.py
REG = dict(models={}, explainers={}, metrics={}, datasets={})

def register(kind, name):
    def deco(cls):
        REG[kind][name] = cls
        return cls
    return deco
