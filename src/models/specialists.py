from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.models.remote_vlm import RemoteSensingVLM, VLMConfig


@dataclass
class SpecialistResult:
    task: str
    answer: str
    model: str
    evidence: list[str]
    limitations: list[str]
    regions: list[dict[str, Any]] | None = None


def _vlm(env_name: str = "REMOTE_VLM_MODEL_ID", max_tokens: int = 256) -> RemoteSensingVLM:
    model_id = os.getenv(env_name, "").strip()
    if not model_id and env_name != "REMOTE_VLM_MODEL_ID":
        model_id = os.getenv("REMOTE_VLM_MODEL_ID", "").strip()
    if not model_id:
        raise RuntimeError(f"{env_name} is not configured. Provide the required domain-adapted checkpoint.")
    return RemoteSensingVLM(VLMConfig(model_id=model_id, max_new_tokens=max_tokens))


def run_single_vqa(image: Image.Image, query: str) -> SpecialistResult:
    model = _vlm()
    answer = model.generate(image, "remote sensing visual question answering. Answer only from the image and question. State uncertainty when unsupported.\nQuestion: " + query + "\nAnswer:")
    return SpecialistResult("single_vqa", answer, model.config.model_id, ["domain-adapted RS-VLM output"], [])


def run_captioning(image: Image.Image) -> SpecialistResult:
    model = _vlm()
    answer = model.generate(image, "remote sensing scene captioning. Describe land cover, major visible objects, spatial relationships and environmental context. Do not invent coordinates or sensor metadata.\nCaption:")
    return SpecialistResult("captioning", answer, model.config.model_id, ["domain-adapted RS-VLM caption"], [])


def run_grounding(image: Image.Image, query: str) -> SpecialistResult:
    model = _vlm("RS_GROUNDING_MODEL_ID", 192)
    prompt = ('remote sensing referring-expression grounding. Return JSON only in this schema: '
              '{"regions":[{"label":"...","bbox":[x1,y1,x2,y2]}],"answer":"..."}. '
              'Coordinates are normalized 0..1000 relative to the image. Ground only the regions referred to by the query.\n'
              f'Query: {query}\nJSON:')
    raw = model.generate(image, prompt)
    import json
    try:
        parsed = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
    except Exception as exc:
        raise RuntimeError(f"Grounding checkpoint did not return valid JSON regions: {raw[:300]}") from exc
    regions = parsed.get("regions") or []
    return SpecialistResult("text_grounding", str(parsed.get("answer", "Grounded region(s) returned.")), model.config.model_id, ["remote-sensing grounding checkpoint output"], [], regions)


def _pair_canvas(left: Image.Image, right: Image.Image, left_label: str, right_label: str) -> Image.Image:
    left, right = left.convert("RGB"), right.convert("RGB")
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), "black")
    canvas.paste(left, (0, 0)); canvas.paste(right, (left.width, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, min(220, left.width), 30), fill="black")
    draw.rectangle((left.width, 0, min(left.width + 220, canvas.width), 30), fill="black")
    draw.text((8, 8), left_label, fill="white")
    draw.text((left.width + 8, 8), right_label, fill="white")
    return canvas


def run_change_vqa(before: Image.Image, after: Image.Image, query: str) -> SpecialistResult:
    env = "RS_CHANGE_VQA_MODEL_ID" if os.getenv("RS_CHANGE_VQA_MODEL_ID") else "REMOTE_VLM_MODEL_ID"
    model = _vlm(env)
    canvas = _pair_canvas(before, after, "T1 / BEFORE", "T2 / AFTER")
    answer = model.generate(canvas, "remote sensing change visual question answering. Left is T1/before and right is T2/after. Describe only changes supported by both observations.\nQuestion: " + query + "\nAnswer:")
    limitation = [] if env == "RS_CHANGE_VQA_MODEL_ID" else ["Native paired-input change checkpoint is not configured; the adapted RS-VLM is receiving an explicit T1/T2 labelled representation."]
    return SpecialistResult("change_vqa", answer, model.config.model_id, ["domain-adapted RS-VLM temporal reasoning"], limitation)


def run_optical_sar(image_optical: Image.Image, image_sar: Image.Image, query: str) -> SpecialistResult:
    env = "RS_MULTIMODAL_MODEL_ID" if os.getenv("RS_MULTIMODAL_MODEL_ID") else "REMOTE_VLM_MODEL_ID"
    model = _vlm(env)
    canvas = _pair_canvas(image_optical, image_sar, "OPTICAL", "SAR")
    answer = model.generate(canvas, "remote sensing cross-modal reasoning. Left is OPTICAL and right is SAR. Use complementary spectral/contextual and radar structural evidence. Do not invent precise class areas unless computed evidence supplies them.\nQuestion: " + query + "\nAnswer:")
    limitation = [] if env == "RS_MULTIMODAL_MODEL_ID" else ["Native dual-encoder optical/SAR checkpoint is not configured; the adapted RS-VLM is receiving an explicit labelled joint representation."]
    return SpecialistResult("optical_sar", answer, model.config.model_id, ["domain-adapted RS-VLM cross-modal reasoning"], limitation)


def image_from_array(array: np.ndarray) -> Image.Image:
    arr = np.asarray(array)
    if arr.ndim == 2:
        lo, hi = np.nanpercentile(arr, [2, 98])
        scaled = np.zeros_like(arr, dtype=np.float32) if hi <= lo else np.clip((arr - lo) / (hi - lo), 0, 1)
        return Image.fromarray((scaled * 255).astype(np.uint8)).convert("RGB")
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB")
