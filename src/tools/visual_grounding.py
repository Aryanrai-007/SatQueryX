from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
from PIL import Image, ImageDraw


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
            detections.append(Detection(str(names[cls_id]), float(box.conf.item()), tuple(float(v) for v in box.xyxy[0].tolist())))
    return detections


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    """Draw only detector-produced boxes; never invents labels or boxes."""
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    width = max(2, out.width // 500)
    for d in detections:
        box = tuple(int(round(v)) for v in d.xyxy)
        draw.rectangle(box, outline=(255, 255, 255), width=width)
        label = f"{d.label} {d.confidence:.0%}"
        y = max(0, box[1] - 18)
        bbox = draw.textbbox((box[0], y), label)
        draw.rectangle(bbox, fill=(0, 0, 0))
        draw.text((box[0], y), label, fill=(255, 255, 255))
    return out
