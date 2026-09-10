from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    role: str
    source: str
    tasks: tuple[str, ...]


DATASETS = {
    "bigearthnet_txt": DatasetSpec(
        "BigEarthNet.txt",
        "domain adaptation",
        "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt",
        ("captioning", "vqa", "grounding", "optical_sar"),
    ),
    "vrsbench": DatasetSpec(
        "VRSBench",
        "single-image evaluation",
        "https://github.com/lx709/VRSBench",
        ("captioning", "grounding", "vqa"),
    ),
    "rsvqa": DatasetSpec(
        "RSVQA",
        "single-image VQA evaluation",
        "https://github.com/syvlo/RSVQA",
        ("vqa",),
    ),
    "cdvqa": DatasetSpec(
        "CDVQA",
        "bi-temporal change VQA",
        "https://github.com/YZHJessica/CDVQA",
        ("change_vqa",),
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    key = name.lower().replace("-", "_")
    if key not in DATASETS:
        raise KeyError(f"Unknown dataset: {name}")
    return DATASETS[key]


def load_bigearthnet_metadata(split: str = "train", limit: int | None = None):
    """Load BigEarthNet.txt metadata from the official HF dataset.

    Image bytes are deliberately not downloaded here. The official dataset uses
    an image LMDB alongside its parquet annotations; the training pipeline joins
    those resources using the patch identifiers. This keeps the repository small
    and makes data acquisition explicit.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the `datasets` package to load BigEarthNet.txt metadata.") from exc

    ds = load_dataset("BIFOLD-BigEarthNetv2-0/BigEarthNet.txt", split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def dataset_status() -> list[dict[str, Any]]:
    status: list[dict[str, Any]] = []
    for key, spec in DATASETS.items():
        env_name = {
            "bigearthnet_txt": "BIGEARTHNET_METADATA",
            "vrsbench": "VRSBENCH_ROOT",
            "rsvqa": "RSVQA_ROOT",
            "cdvqa": "CDVQA_ROOT",
        }[key]
        value = os.getenv(env_name, "").strip()
        status.append({
            "key": key,
            "name": spec.name,
            "role": spec.role,
            "tasks": list(spec.tasks),
            "configured": bool(value),
            "location": value or "not configured",
        })
    return status
