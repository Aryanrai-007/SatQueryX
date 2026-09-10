from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO
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


def _vlm() -> RemoteSensingVLM:
    model_id = os.getenv("REMOTE_VLM_MODEL_ID", "").strip()
    if not model_id:
        raise RuntimeError(
            "REMOTE_VLM_MODEL_ID is not configured. Configure a domain-adapted SatQueryX checkpoint "
            "before using the specialist VLM workflows. A generic VLM is not presented as the SIH solution."
        )
    return RemoteSensingVLM(VLMConfig(model_id=model_id, max_new_tokens=int(os.getenv("REMOTE_VLM_MAX_NEW_TOKENS", "256"))))


def run_single_vqa(image: Image.Image, query: str) -> SpecialistResult:
    prompt = (
        "remote sensing visual question answering. Answer only from the image and question. "
        "Use remote-sensing terminology and state uncertainty when the image does not support a claim.\n"
        f"Question: {query}\nAnswer:"
    )
    model = _vlm()
    answer = model.generate(image, prompt)
    return SpecialistResult("single_vqa", answer, model.config.model_id, ["domain-adapted RS-VLM output"], [])


def run_captioning(image: Image.Image) -> SpecialistResult:
    prompt = (
        "remote sensing scene captioning. Describe land cover, major visible objects, spatial relationships, "
        "and relevant environmental context. Do not invent geographic coordinates or sensor metadata.\nCaption:"
    )
    model = _vlm()
    answer = model.generate(image, prompt)
    return SpecialistResult("captioning", answer, model.config.model_id, ["domain-adapted RS-VLM caption"], [])


def run_change_vqa(before: Image.Image, after: Image.Image, query: str) -> SpecialistResult:
    # The paired montage is an explicit two-observation representation. A future
    # checkpoint can consume native two-image tensors; until then the same adapted
    # RS-VLM sees both observations with unambiguous T1/T2 labels.
    canvas = Image.new("RGB", (before.width + after.width, max(before.height, after.height)), "black")
    canvas.paste(before.convert("RGB"), (0, 0))
    canvas.paste(after.convert("RGB"), (before.width, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, min(180, before.width), 28), fill="black")
    draw.rectangle((before.width, 0, min(before.width + 180, canvas.width), 28), fill="black")
    draw.text((8, 7), "T1 / BEFORE", fill="white")
    draw.text((before.width + 8, 7), "T2 / AFTER", fill="white")
    prompt = (
        "remote sensing change visual question answering. The image is a paired temporal observation: "
        "left=T1/before and right=T2/after. Describe only changes supported by both observations. "
        "Distinguish changed location/type from unchanged context.\n"
        f"Question: {query}\nAnswer:"
    )
    model = _vlm()
    answer = model.generate(canvas, prompt)
    return SpecialistResult("change_vqa", answer, model.config.model_id, ["domain-adapted RS-VLM temporal reasoning"], ["Current inference uses a labelled T1/T2 montage; native two-image encoder support requires a checkpoint trained for paired inputs."])


def run_optical_sar(image_optical: Image.Image, image_sar: Image.Image, query: str) -> SpecialistResult:
    canvas = Image.new("RGB", (image_optical.width + image_sar.width, max(image_optical.height, image_sar.height)), "black")
    canvas.paste(image_optical.convert("RGB"), (0, 0))
    canvas.paste(image_sar.convert("RGB"), (image_optical.width, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, min(190, image_optical.width), 28), fill="black")
    draw.rectangle((image_optical.width, 0, min(image_optical.width + 190, canvas.width), 28), fill="black")
    draw.text((8, 7), "OPTICAL", fill="white")
    draw.text((image_optical.width + 8, 7), "SAR", fill="white")
    prompt = (
        "remote sensing cross-modal reasoning. The paired image contains OPTICAL on the left and SAR on the right. "
        "Use complementary spectral/contextual and radar structural evidence. Explicitly say when one modality "
        "is insufficient. Do not infer precise class areas unless they are supplied by computed evidence.\n"
        f"Question: {query}\nAnswer:"
    )
    model = _vlm()
    answer = model.generate(canvas, prompt)
    return SpecialistResult("optical_sar", answer, model.config.model_id, ["domain-adapted RS-VLM cross-modal reasoning"], ["Current inference represents the pair as a labelled joint canvas; a native dual-encoder checkpoint can replace this adapter without changing the workflow contract."])


def image_from_array(array: np.ndarray) -> Image.Image:
    arr = np.asarray(array)
    if arr.ndim == 2:
        lo, hi = np.nanpercentile(arr, [2, 98])
        scaled = np.zeros_like(arr, dtype=np.float32) if hi <= lo else np.clip((arr - lo) / (hi - lo), 0, 1)
        return Image.fromarray((scaled * 255).astype(np.uint8)).convert("RGB")
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB")
