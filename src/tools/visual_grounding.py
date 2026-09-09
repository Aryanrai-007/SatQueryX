from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tempfile

import numpy as np
from PIL import Image


@dataclass
class Detection:
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]


def run_yolo(image: Image.Image, model_path: str | None = None, confidence: float = 0.25) -> list[Detection]:
    """Run a real Ultralytics detector. No fallback detections are generated."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed. Install requirements.txt to enable visual grounding.") from exc

    model_ref = model_path or os.getenv("YOLO_MODEL_PATH", "yolo11n.pt")
    if not model_ref:
        raise RuntimeError("YOLO_MODEL_PATH is empty. Provide a real detector checkpoint.")

    model = YOLO(model_ref)
    results = model.predict(source=np.asarray(image), conf=confidence, verbose=False)
    detections: list[Detection] = []
    for result in results:
        names = result.names
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls.item())
            detections.append(
                Detection(
                    label=str(names[cls_id]),
                    confidence=float(box.conf.item()),
                    xyxy=tuple(float(v) for v in box.xyxy[0].tolist()),
                )
            )
    return detections
