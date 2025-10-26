from __future__ import annotations
import subprocess, sys
from time_to_explain.tools.install_third_party import main as install_tp

def run(*args: str) -> None:
    print("$", " ".join(args)); subprocess.check_call([sys.executable, *args])

def main() -> None:
    # Install torch/pyg placeholders (users can edit for CUDA)
    try:
        run("-m", "pip", "install", "--upgrade", "pip")
        run("-m", "pip", "install", "torch>=2.2", "torch-geometric>=2.4")
    except Exception as e:
        print("Torch/pyg install skipped/failed:", e)
    # Install third-party modules (editable)
    install_tp()
    # Install this package with dev extras
    run("-m", "pip", "install", "-e", ".[dev]")

if __name__ == "__main__":
    main()
