# metrics/base.py

class Metric(ABC):
    name: str
    directional: str  # 'higher-better' or 'lower-better'
    def __call__(self, explanation, model, batch, target_idx=None) -> float: ...
