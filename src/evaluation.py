from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any


def normalize_answer(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9.%+-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    p = normalize_answer(prediction).split()
    r = normalize_answer(reference).split()
    if not p or not r:
        return float(p == r)
    pc, rc = Counter(p), Counter(r)
    overlap = sum((pc & rc).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def mean_iou(pred_bbox: tuple[float, float, float, float], ref_bbox: tuple[float, float, float, float]) -> float:
    px1, py1, px2, py2 = pred_bbox
    rx1, ry1, rx2, ry2 = ref_bbox
    ix1, iy1, ix2, iy2 = max(px1, rx1), max(py1, ry1), min(px2, rx2), min(py2, ry2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    pa = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    ra = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = pa + ra - inter
    return inter / union if union else 0.0


def evaluate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate generic VQA/change-VQA records with exact match and token F1.

    Each record should contain `prediction` and `reference`; optional `task` is
    retained in the summary so the same evaluator can process RSVQA/CDVQA/BigEarthNet
    benchmark exports without assuming a dataset-specific schema.
    """
    if not records:
        return {"count": 0, "exact_match": None, "token_f1": None, "by_task": {}}
    em = [exact_match(r.get("prediction", ""), r.get("reference", "")) for r in records]
    f1 = [token_f1(r.get("prediction", ""), r.get("reference", "")) for r in records]
    by_task: dict[str, list[float]] = {}
    for r, score in zip(records, f1):
        by_task.setdefault(str(r.get("task", "unknown")), []).append(score)
    return {
        "count": len(records),
        "exact_match": sum(em) / len(em),
        "token_f1": sum(f1) / len(f1),
        "by_task": {k: sum(v) / len(v) for k, v in by_task.items()},
    }


def load_prediction_jsonl(path: str | Path) -> list[dict[str, Any]]:
    import json
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows
