import torch
from argparse import Namespace
from typing import Any


def _mps_available() -> bool:
    try:
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()
    except Exception:
        return False


def pick_device(prefer: str | None = None) -> torch.device:
    """
    Prefer a specific device if available, otherwise fall back gracefully:
    CUDA -> MPS -> CPU.
    prefer may be one of: None/'auto', 'cuda', 'mps', 'cpu'.
    """
    prefer = (prefer or "auto").lower()

    def _cuda_ok() -> bool:
        return torch.cuda.is_available()

    if prefer in ("auto", ""):
        if _cuda_ok():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if prefer in ("cuda", "gpu"):
        if _cuda_ok():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if prefer in ("mps", "metal"):
        if _mps_available():
            return torch.device("mps")
        if _cuda_ok():
            return torch.device("cuda")
        return torch.device("cpu")

    # 'cpu' or anything else defaults to cpu
    return torch.device("cpu")


def resolve_device(prefer: str | torch.device | None = None, *, cuda: bool = False) -> torch.device:
    """
    Normalize string/torch.device preferences into a torch.device.
    """
    if isinstance(prefer, torch.device):
        return prefer

    if cuda and torch.cuda.is_available():
        return torch.device("cuda")

    return pick_device(prefer)


def device_from_args(args: Namespace | Any) -> torch.device:
    """
    Backward-compatible helper shared by CLI + notebooks.
    - If --device provided, use it with graceful fallback.
    - Else if --cuda, try cuda else fallback.
    - Else auto-detect.
    """
    device_attr = getattr(args, "device", None)
    prefer = device_attr if device_attr not in (None, "", "auto") else None
    cuda_flag = bool(getattr(args, "cuda", False))
    return resolve_device(prefer, cuda=cuda_flag)

