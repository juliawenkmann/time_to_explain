import torch
import platform
import sys
from argparse import Namespace, ArgumentParser

def _mps_available() -> bool:
    # MPS is available on Apple Silicon with macOS 12.3+ and PyTorch built with MPS
    try:
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()
    except Exception:
        return False


def pick_device(prefer: str | None = None) -> torch.device:
    """
    Prefer a specific device if available, otherwise fall back gracefully:
    CUDA -> MPS -> CPU.
    prefer may be one of: None/'auto', 'cuda', 'mps', 'cpu'
    """
    prefer = (prefer or "auto").lower()

    def _cuda_ok():
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
        # graceful fallback
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if prefer in ("mps", "metal"):
        if _mps_available():
            return torch.device("mps")
        # graceful fallback
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # 'cpu' or anything else defaults to cpu
    return torch.device("cpu")


def device_from_args(args: Namespace) -> torch.device:
    """
    Backward-compatible:
    - If --device provided, use it with graceful fallback.
    - Else if --cuda, try cuda else fallback.
    - Else auto-detect.
    """
    # New flag (added below in add_wrapper_model_arguments)
    if hasattr(args, "device") and args.device is not None:
        return pick_device(args.device)

    # Legacy behavior with safe fallback
    if getattr(args, "cuda", False):
        return pick_device("cuda")

    device =  pick_device("auto")
    print(device)
    return device