from __future__ import annotations

"""Prepare the metadata/loader side of BigEarthNet.txt for SatQueryX training.

This script intentionally does not download the multi-tens-of-GB Sentinel-1/Sentinel-2
image archive. The official BigEarthNet.txt instructions require BigEarthNet v2.0 imagery
to be converted to an LMDB with rico-hdl before paired image/text training.
"""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/bigearthnet_txt")
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    parquet = hf_hub_download(repo_id=REPO, filename="BigEarthNet.txt.parquet", local_dir=str(root))
    loader = hf_hub_download(repo_id=REPO, filename="ben_txt_datamodule.py", local_dir=str(root))
    print(f"Metadata: {parquet}")
    print(f"Official loader: {loader}")
    print()
    print("Next: download the official BigEarthNet v2.0 Sentinel-1 and Sentinel-2 image data")
    print("and convert it with rico-hdl, for example:")
    print("  rico-hdl bigearthnet --bigearthnet-s1-dir <S1_ROOT_DIR> --bigearthnet-s2-dir <S2_ROOT_DIR> --target-dir <LMDB_DIR>")
    print()
    print("Then set:")
    print(f"  BIGEARTHNET_METADATA={Path(parquet).resolve()}")
    print(f"  BIGEARTHNET_LOADER={Path(loader).resolve()}")
    print("  BIGEARTHNET_IMAGE_LMDB=<absolute path to Encoded-BigEarthNet>")


if __name__ == "__main__":
    main()
