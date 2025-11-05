# Importing here auto-registers the recipes so available_datasets() works immediately
from .erdos_temporal import ErdosTemporal
try:
    from .hawkes_exp import HawkesExp
except Exception:
    # It's fine if tick isn't installed; users can still import erdos_temporal.
    pass