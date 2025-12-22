# Importing here auto-registers the recipes so available_datasets() works immediately
from .erdos_temporal import ErdosTemporal
from .nicolaus_temporal import NicolausTemporal
from .stick_figure import StickFigure, StickyHips
from .triadic_closure import TriadicClosure
try:
    from .hawkes_exp import HawkesExp
except Exception:
    # It's fine if tick isn't installed; users can still import erdos_temporal.
    pass
