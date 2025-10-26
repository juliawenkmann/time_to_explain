# time_to_explain/tools/install_third_party.py
from __future__ import annotations
import subprocess, sys, os
from pathlib import Path

def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)

def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]  # .../time_to_explain
    third_party = repo_root / "third_party"
    if not third_party.exists():
        print("No third_party/ directory found; nothing to install.")
        return

    # 1) Ensure submodules are present
    try:
        run(["git", "submodule", "update", "--init", "--recursive"], cwd=repo_root)
    except Exception as e:
        print("Warning: couldn't update submodules:", e)

    # 2) Find all subprojects that have a pyproject.toml (you added these in each fork)
    projects = []
    for p in third_party.rglob("pyproject.toml"):
        # only take immediate project roots (ignore nested test projects)
        projects.append(p.parent)

    # (Optional) stable order
    projects = sorted(set(projects), key=lambda p: str(p))

    if not projects:
        print("No third-party projects with pyproject.toml found.")
        return

    print("Installing third-party modules (editable):")
    for proj in projects:
        # Make sure packages are packages (helpful if upstream lacks __init__.py)
        for pkg_dir in ["model", "utils", "src"]:
            d = proj / pkg_dir
            if d.is_dir():
                (d / "__init__.py").touch(exist_ok=True)
        # pip install -e <project>
        run([sys.executable, "-m", "pip", "install", "-e", str(proj)])

    print("\nAll third-party modules installed. ✅")

if __name__ == "__main__":
    main()
