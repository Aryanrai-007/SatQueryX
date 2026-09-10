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
    """Aggregate already-computed analysis results into grounded evidence.

    Site intelligence is deliberately NOT recomputed here. The application executes it
    once before this function and stores the result under ``results['site_summary']``.
    This prevents duplicate remote requests and keeps the execution trace truthful.
    """
    facts: list[str] = []
    sources: list[str] = []
    confidences: list[float] = []

    aoi = results.get("aoi")
    if aoi:
        location = aoi.get("location") or "Unavailable"
        scene = aoi.get("scene_id") or "Unavailable"
        acquired = aoi.get("datetime") or "Unavailable"
        cloud = aoi.get("cloud_cover")
        cloud_text = f"{cloud}%" if cloud is not None else "Unavailable"
        bbox = aoi.get("bbox")
        bbox_text = ", ".join(f"{float(v):.6f}" for v in bbox) if bbox else "Unavailable"
        platform = aoi.get("platform") or "Sentinel-2"
        instruments = ",".join(aoi.get("instruments", [])) or "MSI"
        tile = aoi.get("mgrs_tile") or "Unavailable"
        bands = ",".join(aoi.get("bands", [])) or "Unavailable"
        facts.append(
            f"AOI location={location}; WGS84 bbox={bbox_text}; scene={scene}; acquisition={acquired}; "
            f"platform={platform}; instrument={instruments}; MGRS tile={tile}; cloud cover={cloud_text}; "
            f"bands={bands}; CRS={aoi.get('crs', 'Unavailable')}."
        )
        sources.append("Earth Search Sentinel-2 metadata + OpenStreetMap/Nominatim AOI context")

    # Consume the single site-intelligence result produced by app.py.
    site = results.get("site_summary")
    if site:
        area = site.get("area_ha")
        if area is not None:
            facts.append(
                f"AOI footprint area={float(area):.3f} hectares; area basis={site.get('area_basis', 'unknown')}."
            )
            sources.append("WGS84 geodesic AOI area calculation" if site.get("area_basis") == "exact drawn polygon" else "WGS84 AOI footprint calculation")

        elevation = site.get("elevation")
        if elevation:
            facts.append(
                f"{elevation['source']}: mean elevation={elevation['mean_m']:.1f} m, "
                f"median={elevation['median_m']:.1f} m, minimum={elevation['min_m']:.1f} m, "
                f"maximum={elevation['max_m']:.1f} m, relief={elevation['relief_m']:.1f} m, "
                f"valid samples={elevation.get('samples', 0)}."
            )
            sources.append(elevation["source"])

        worldcover = site.get("worldcover")
        if worldcover and worldcover.get("rows"):
            rows = worldcover["rows"]
            facts.append(
                "ESA WorldCover 2021 v200 land-cover composition: "
                + ", ".join(
                    f"{row['label']}={row['fraction']:.1%} ({row['area_ha']:.3f} ha)"
                    for row in rows
                )
                + "."
            )
            for cls, label in ((30, "grassland"), (50, "built-up"), (80, "permanent-water")):
                row = next((r for r in rows if r["class"] == cls), None)
                if row:
                    facts.append(
                        f"Mapped {label} area={row['area_ha']:.3f} hectares "
                        f"({row['fraction']:.1%} of WorldCover-mapped pixels)."
                    )
            sources.append("ESA WorldCover 2021 v200 (10 m)")

        buildings = site.get("buildings")
        if buildings is not None:
            facts.append(
                f"OpenStreetMap mapped buildings={int(buildings.get('count', 0))}; "
                f"scope={buildings.get('scope', 'AOI')}; source={buildings.get('source', 'OpenStreetMap / Overpass API')}."
            )
            sources.append(buildings.get("source", "OpenStreetMap / Overpass API"))

        waterways = site.get("waterways")
        if waterways is not None:
            if waterways.get("names"):
                description = ", ".join(waterways["names"])
            elif waterways.get("count"):
                description = f"{waterways['count']} mapped feature(s), unnamed"
            else:
                description = "no mapped features returned"
            types = ", ".join(waterways.get("types", [])) or "none"
            facts.append(
                f"OpenStreetMap mapped water features={waterways.get('count', 0)}; "
                f"types={types}; names={description}; "
                f"context radius={waterways.get('context_radius_degrees', 'AOI only')} degrees."
            )
            sources.append(waterways.get("source", "OpenStreetMap / Overpass API"))

        ndvi_area = site.get("ndvi_vegetated_area_ha")
        if ndvi_area is not None:
            facts.append(
                f"NDVI-derived vegetated area={float(ndvi_area):.3f} hectares above the configured NDVI threshold; "
                "this is vegetation extent and is not equivalent to grassland classification."
            )
            sources.append("Sentinel-2 NDVI raster computation")

        for limitation in site.get("limitations", []):
            facts.append("Site-intelligence limitation: " + str(limitation))

    if "ndvi" in results:
        nd = results["ndvi"]
        facts.append(
            f"NDVI mean={nd['mean']:.4f}, median={nd['median']:.4f}, "
            f"vegetated_fraction={nd['vegetated_fraction']:.2%} using configured red/NIR bands."
        )
        sources.append("NDVI raster computation")

    if "change_detection" in results:
        ch = results["change_detection"]
        facts.append(
            f"Change mask covers {ch['changed_fraction']:.2%} of valid pixels; "
            f"mean absolute normalized change={ch['mean_absolute_change']:.4f}; SSIM={ch['ssim']:.4f}."
        )
        sources.append("Aligned raster change analysis")

    if "detections" in results:
        dets = results["detections"]
        counts: dict[str, int] = {}
        for detection in dets:
            counts[detection.label] = counts.get(detection.label, 0) + 1
            confidences.append(float(detection.confidence))
        facts.append(
            "Visual grounding detections: "
            + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) if counts else "none returned by the detector.")
        )
        sources.append("Configured YOLO checkpoint")

    if results.get("detector_limitation"):
        facts.append("Visual grounding limitation: " + str(results["detector_limitation"]))

    if "fusion" in results:
        fu = results["fusion"]
        facts.append(
            f"Optical/SAR fused standardized evidence computed from aligned inputs; "
            f"optical/SAR correlation={fu['correlation']:.4f}."
        )
        sources.append("Optical/SAR statistical fusion")

    if "sar" in results:
        s = results["sar"]
        facts.append(
            f"SAR statistics: mean={s['mean']:.4f}, std={s['std']:.4f}, "
            f"min={s['min']:.4f}, max={s['max']:.4f}."
        )
        sources.append("SAR raster statistics")

    confidence = float(sum(confidences) / len(confidences)) if confidences else None
    return EvidenceBundle(facts=facts, confidence=confidence, sources=sources)


def synthesize_answer(
    query: str,
    evidence: EvidenceBundle,
    image: Image.Image | None = None,
) -> tuple[str, str]:
    gemini = GeminiClient()
    confidence_text = f"{evidence.confidence:.2f}" if evidence.confidence is not None else "not estimated"
    prompt = (
        "You are SatQueryX, a remote-sensing analysis assistant. Answer ONLY from the supplied evidence. "
        "Do not invent objects, locations, dates, sensor properties, areas, or certainty. Clearly state limitations. "
        "Use exact computed values when present. Distinguish mapped datasets such as ESA WorldCover and OSM "
        "from direct Sentinel-2 measurements. If the user asks for a quantity that is unavailable, say it is unavailable.\n\n"
        f"User query: {query}\n\n"
        "Computed evidence:\n- " + "\n- ".join(evidence.facts) +
        f"\n\nDetector confidence (only when available): {confidence_text}\nSources: {', '.join(dict.fromkeys(evidence.sources))}"
    )
    if gemini.configured:
        return gemini.generate(prompt, [image] if image else []), "Gemini"
    if image is not None:
        try:
            vlm = RemoteSensingVLM()
            return vlm.generate(image, prompt), "Configured remote-sensing VLM"
        except RuntimeError:
            pass
    raise RuntimeError(
        "No language model is configured. Set GEMINI_API_KEY or REMOTE_VLM_MODEL_ID; "
        "SatQueryX will not fabricate an answer."
    )
