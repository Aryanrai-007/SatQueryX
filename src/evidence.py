from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from .models.gemini import GeminiClient
from .models.remote_vlm import RemoteSensingVLM


@dataclass
class EvidenceBundle:
    facts: list[str]
    confidence: float | None
    sources: list[str]


def build_evidence(results: dict[str, Any]) -> EvidenceBundle:
    facts: list[str] = []
    sources: list[str] = []
    confidences: list[float] = []

    if "ndvi" in results:
        nd = results["ndvi"]
        facts.append(f"NDVI mean={nd['mean']:.4f}, median={nd['median']:.4f}, vegetated_fraction={nd['vegetated_fraction']:.2%} using the configured red/NIR bands.")
        sources.append("NDVI raster computation")
    if "change_detection" in results:
        ch = results["change_detection"]
        facts.append(f"Change mask covers {ch['changed_fraction']:.2%} of valid pixels; mean absolute normalized change={ch['mean_absolute_change']:.4f}; SSIM={ch['ssim']:.4f}.")
        sources.append("Aligned raster change analysis")
    if "detections" in results:
        dets = results["detections"]
        counts: dict[str, int] = {}
        for d in dets:
            counts[d.label] = counts.get(d.label, 0) + 1
            confidences.append(d.confidence)
        facts.append("Visual grounding detections: " + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) if counts else "none returned by the detector."))
        sources.append("Configured YOLO checkpoint")
    if "fusion" in results:
        fu = results["fusion"]
        facts.append(f"Optical/SAR fused standardized evidence computed from aligned inputs; optical/SAR correlation={fu['correlation']:.4f}.")
        sources.append("Optical/SAR statistical fusion")
    if "sar" in results:
        s = results["sar"]
        facts.append(f"SAR statistics: mean={s['mean']:.4f}, std={s['std']:.4f}, min={s['min']:.4f}, max={s['max']:.4f}.")
        sources.append("SAR raster statistics")

    confidence = float(sum(confidences) / len(confidences)) if confidences else None
    return EvidenceBundle(facts=facts, confidence=confidence, sources=sources)


def synthesize_answer(query: str, evidence: EvidenceBundle, image: Image.Image | None = None) -> tuple[str, str]:
    """Return answer + provider. If no provider is configured, fail rather than fabricate."""
    gemini = GeminiClient()
    confidence_text = f"{evidence.confidence:.2f}" if evidence.confidence is not None else "not estimated"
    prompt = (
        "You are SatQueryX, a remote-sensing analysis assistant. Answer ONLY from the supplied evidence. "
        "Do not invent objects, locations, dates, sensor properties, or certainty. Clearly state limitations.\n\n"
        f"User query: {query}\n\n"
        "Computed evidence:\n- " + "\n- ".join(evidence.facts) +
        f"\n\nDetector confidence (only when available): {confidence_text}\nSources: {', '.join(evidence.sources)}"
    )
    if gemini.configured:
        return gemini.generate(prompt, [image] if image else []), "Gemini"

    if image is not None:
        try:
            vlm = RemoteSensingVLM()
            return vlm.generate(image, prompt), "Configured remote-sensing VLM"
        except RuntimeError:
            pass
    raise RuntimeError("No language model is configured. Set GEMINI_API_KEY or REMOTE_VLM_MODEL_ID; SatQueryX will not fabricate an answer.")
