from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

REPOS = {
    "vrsbench": "https://github.com/lx709/VRSBench.git",
    "rsvqa": "https://github.com/syvlo/RSVQA.git",
    "cdvqa": "https://github.com/YZHJessica/CDVQA.git",
}


def main() -> None:
    p = argparse.ArgumentParser(description="Clone the public benchmark code/annotation repositories used by SatQueryX.")
    p.add_argument("--root", default="data/benchmarks")
    p.add_argument("--dataset", choices=["all", *REPOS.keys()], default="all")
    args = p.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    names = list(REPOS) if args.dataset == "all" else [args.dataset]
    for name in names:
        dest = root / name
        if dest.exists():
            print(f"exists: {dest}")
            continue
        print(f"cloning {name} -> {dest}")
        subprocess.run(["git", "clone", "--depth", "1", REPOS[name], str(dest)], check=True)
    print("Public benchmark repositories prepared. Dataset image archives may require separate downloads/licensing steps.")


if __name__ == "__main__":
    main()
