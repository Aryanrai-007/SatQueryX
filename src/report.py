from __future__ import annotations

from io import BytesIO
from datetime import datetime, timezone
from html import escape
from typing import Any

import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _safe_text(value: Any) -> str:
    return escape(str(value))


def _array_png(array: np.ndarray, mode: str = "gray") -> bytes | None:
    arr = np.asarray(array)
    if arr.size == 0:
        return None
    if mode == "mask":
        img_arr = (np.nan_to_num(arr, nan=0.0) > 0).astype(np.uint8) * 255
        image = PILImage.fromarray(img_arr, mode="L")
    elif arr.ndim == 2:
        finite = np.isfinite(arr)
        if not finite.any():
            return None
        values = arr.astype(np.float32)
        lo = float(np.nanpercentile(values, 2))
        hi = float(np.nanpercentile(values, 98))
        normalized = np.zeros_like(values, dtype=np.float32) if hi <= lo else np.clip((values - lo) / (hi - lo), 0, 1)
        image = PILImage.fromarray((normalized * 255).astype(np.uint8), mode="L")
    else:
        values = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        if values.max() <= 1.0:
            values *= 255.0
        image = PILImage.fromarray(np.clip(values, 0, 255).astype(np.uint8))
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _report_image(png: bytes, max_width_mm: float = 170.0, max_height_mm: float = 100.0) -> RLImage:
    image = PILImage.open(BytesIO(png))
    width_px, height_px = image.size
    scale = min((max_width_mm * mm) / width_px, (max_height_mm * mm) / height_px)
    return RLImage(BytesIO(png), width=width_px * scale, height=height_px * scale)


def _metadata_table(rows: list[list[str]]) -> Table:
    return Table(rows, colWidths=[55 * mm, 105 * mm], repeatRows=1, style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))


def _site_summary_story(aoi: dict[str, Any], results: dict[str, Any], styles: Any) -> list[Any]:
    site = results.get("site_summary")
    if not site:
        return []

    location = aoi.get("location") or "Selected AOI"
    area = site.get("bbox_area_ha")
    story: list[Any] = [
        Spacer(1, 8),
        Paragraph("Location & environmental summary", styles["Heading2"]),
        Paragraph(
            _safe_text(
                f"The selected area is {location}. Its selected WGS84 bounding-box footprint is approximately "
                f"{area:.2f} hectares. The summary combines Sentinel-2 evidence with independent "
                f"Copernicus DEM, ESA WorldCover and OpenStreetMap context; values are reported only when the "
                f"corresponding source was successfully retrieved."
            ),
            styles["BodyText"],
        ),
    ]

    e = site.get("elevation")
    if e:
        story.extend([
            Spacer(1, 5),
            Paragraph("Terrain / elevation", styles["Heading3"]),
            _metadata_table([
                ["Property", "Value"],
                ["Elevation source", _safe_text(e.get("source", "Unavailable"))],
                ["Elevation model", _safe_text(e.get("model", "Unavailable"))],
                ["Mean elevation", f"{e['mean_m']:.1f} m"],
                ["Median elevation", f"{e['median_m']:.1f} m"],
                ["Minimum / maximum", f"{e['min_m']:.1f} m / {e['max_m']:.1f} m"],
                ["Relief across sampled AOI", f"{e['relief_m']:.1f} m"],
                ["Valid elevation samples / cells", str(e.get("samples", "Unavailable"))],
            ]),
        ])
        if "DSM" in str(e.get("model", "")).upper():
            story.append(Paragraph("Copernicus DEM is a digital surface model: buildings, infrastructure and vegetation can contribute to the measured surface height.", styles["BodyText"]))
        else:
            story.append(Paragraph("Elevation is reported from the named elevation model; it should not be interpreted as building height or bare-earth elevation unless the source explicitly provides that product.", styles["BodyText"]))

    wc = site.get("worldcover")
    if wc and wc.get("rows"):
        rows = [["Land-cover class", "Share", "Estimated area"]]
        for item in wc["rows"]:
            rows.append([_safe_text(item["label"]), f"{item['fraction']:.1%}", f"{item['area_ha']:.2f} ha"])
        story.extend([
            Spacer(1, 6),
            Paragraph("Land cover", styles["Heading3"]),
            Paragraph("ESA WorldCover 2021 v200 classification at 10 m resolution. Areas are proportional estimates for the AOI footprint from the mapped pixels; this is a land-cover product, not a pixel-perfect cadastral survey:", styles["BodyText"]),
            Table(rows, colWidths=[75 * mm, 35 * mm, 50 * mm], repeatRows=1, style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ])),
        ])
    elif results.get("ndvi"):
        nd = results["ndvi"]
        area_ha = site.get("bbox_area_ha", 0.0) * nd.get("vegetated_fraction", 0.0)
        story.extend([
            Spacer(1, 6),
            Paragraph("Vegetation", styles["Heading3"]),
            Paragraph(_safe_text(f"NDVI-based vegetated fraction (NDVI > 0.3): {nd['vegetated_fraction']:.1%}, corresponding to approximately {area_ha:.2f} hectares of the selected footprint. This is vegetation extent, not a grass-only classification."), styles["BodyText"]),
        ])

    water = site.get("waterways")
    if water:
        story.extend([Spacer(1, 6), Paragraph("Rivers, waterways & surface-water context", styles["Heading3"])])
        if water.get("names"):
            story.append(Paragraph(_safe_text("Named mapped water features returned by OpenStreetMap in the queried context area: " + ", ".join(water["names"]) + "."), styles["BodyText"]))
        elif water.get("count"):
            radius = water.get("context_radius_degrees")
            context = f"within approximately {radius}° of the AOI bbox" if radius else "in the queried AOI/context bbox"
            story.append(Paragraph(_safe_text(f"OpenStreetMap returned {water['count']} mapped water/waterway feature(s) {context}, but no names were returned."), styles["BodyText"]))
        else:
            story.append(Paragraph("OpenStreetMap returned no mapped river, stream, canal, drain or natural-water feature in the queried AOI/context area.", styles["BodyText"]))
        wc_water = next((r for r in (wc or {}).get("rows", []) if r["class"] == 80), None)
        if wc_water:
            story.append(Paragraph(_safe_text(f"ESA WorldCover permanent-water class covers approximately {wc_water['area_ha']:.2f} hectares ({wc_water['fraction']:.1%}) of the mapped AOI pixels."), styles["BodyText"]))

    if site.get("ndvi_vegetated_area_ha") is not None:
        story.append(Paragraph(_safe_text(f"Sentinel-2 NDVI vegetation estimate: approximately {site['ndvi_vegetated_area_ha']:.2f} hectares above the configured NDVI threshold."), styles["BodyText"]))

    if site.get("limitations"):
        story.append(Paragraph("Site-intelligence limitations: " + _safe_text("; ".join(site["limitations"])), styles["BodyText"]))
    return story


def build_pdf(query: str, plan: Any, evidence: Any, results: dict[str, Any], preview_png: bytes | None = None) -> bytes:
    out = BytesIO()
    doc = SimpleDocTemplate(out, pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm)
    styles = getSampleStyleSheet()
    story: list[Any] = [
        Paragraph("SatQueryX — Remote Sensing Analysis Report", styles["Title"]),
        Paragraph(datetime.now(timezone.utc).strftime("Generated %Y-%m-%d %H:%M UTC"), styles["Normal"]),
        Spacer(1, 8),
        Paragraph("Query", styles["Heading2"]),
        Paragraph(_safe_text(query), styles["BodyText"]),
    ]

    aoi = results.get("aoi")
    if aoi and results.get("site_summary"):
        story.extend(_site_summary_story(aoi, results, styles))

    story.extend([
        Spacer(1, 6),
        Paragraph("Execution plan", styles["Heading2"]),
        Paragraph("Intents: " + _safe_text(", ".join(plan.intents)), styles["BodyText"]),
        Paragraph("Tools: " + _safe_text(", ".join(plan.tools)), styles["BodyText"]),
    ])

    steps = results.get("execution_steps", [])
    if steps:
        story.extend([Spacer(1, 6), Paragraph("Execution trace", styles["Heading2"])])
        trace_rows = [["Status", "Step", "Detail"]]
        for step in steps:
            trace_rows.append(["SUCCESS" if step.status == "success" else "ERROR", _safe_text(step.name), _safe_text(step.detail)])
        story.append(Table(trace_rows, colWidths=[25 * mm, 45 * mm, 100 * mm], repeatRows=1, style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ])))

    story.extend([Spacer(1, 8), Paragraph("Evidence", styles["Heading2"])])
    if evidence.facts:
        story.extend(Paragraph("• " + _safe_text(fact), styles["BodyText"]) for fact in evidence.facts)
    else:
        story.append(Paragraph("No computed evidence was produced by the selected tools.", styles["BodyText"]))
    confidence_text = f"{evidence.confidence:.2f}" if evidence.confidence is not None else "Not estimated"
    story.append(Paragraph(f"Aggregate confidence: {_safe_text(confidence_text)}", styles["BodyText"]))
    sources = ", ".join(evidence.sources) if evidence.sources else "None"
    story.append(Paragraph("Sources: " + _safe_text(sources), styles["BodyText"]))

    if aoi:
        bbox = aoi.get("bbox")
        bbox_text = ", ".join(f"{float(v):.6f}" for v in bbox) if bbox else "Unavailable"
        story.extend([
            Spacer(1, 8),
            Paragraph("AOI & acquisition", styles["Heading2"]),
            _metadata_table([
                ["Property", "Value"],
                ["Location", _safe_text(aoi.get("location") or "Unavailable")],
                ["Scene ID", _safe_text(aoi.get("scene_id", "Unavailable"))],
                ["Acquisition time", _safe_text(aoi.get("datetime", "Unavailable"))],
                ["Cloud cover", _safe_text(aoi.get("cloud_cover", "Unavailable"))],
                ["Source", _safe_text(aoi.get("source", "Unavailable"))],
                ["AOI bbox (WGS84)", _safe_text(bbox_text)],
                ["Bands", _safe_text(", ".join(aoi.get("bands", [])))],
                ["Scene CRS", _safe_text(aoi.get("crs", "Unavailable"))],
            ]),
        ])

    info = results.get("primary_info")
    if info is not None:
        story.extend([
            Spacer(1, 8),
            Paragraph("Primary raster metadata", styles["Heading2"]),
            _metadata_table([
                ["Property", "Value"],
                ["File", _safe_text(info.name)],
                ["Dimensions", f"{info.width} × {info.height} pixels"],
                ["Bands", str(info.count)],
                ["Data type", _safe_text(info.dtype)],
                ["CRS", _safe_text(info.crs or "Not available")],
                ["Resolution", f"{info.resolution_x:g} × {info.resolution_y:g}"],
            ]),
        ])

    if results.get("secondary_info"):
        info2 = results["secondary_info"]
        story.extend([
            Spacer(1, 6),
            Paragraph("Secondary raster metadata", styles["Heading3"]),
            _metadata_table([
                ["Property", "Value"],
                ["File", _safe_text(info2.name)],
                ["Dimensions", f"{info2.width} × {info2.height} pixels"],
                ["Bands", str(info2.count)],
                ["CRS", _safe_text(info2.crs or "Not available")],
                ["Resolution", f"{info2.resolution_x:g} × {info2.resolution_y:g}"],
            ]),
        ])

    if results.get("answer"):
        story.extend([
            Spacer(1, 8),
            Paragraph("SatQueryX answer", styles["Heading2"]),
            Paragraph(_safe_text(results["answer"]).replace("\n", "<br/>"), styles["BodyText"]),
            Paragraph("Language provider: " + _safe_text(results.get("provider", "unknown")), styles["BodyText"]),
        ])

    if preview_png:
        story.extend([Spacer(1, 8), Paragraph("Input preview", styles["Heading2"]), _report_image(preview_png)])

    rows = [["Metric", "Value"]]
    if "change_detection" in results:
        ch = results["change_detection"]
        rows += [["Changed fraction", f"{ch['changed_fraction']:.2%}"], ["Mean absolute change", f"{ch['mean_absolute_change']:.4f}"], ["SSIM", f"{ch['ssim']:.4f}"]]
    if "ndvi" in results:
        nd = results["ndvi"]
        rows += [["NDVI mean", f"{nd['mean']:.4f}"], ["NDVI median", f"{nd['median']:.4f}"], ["Vegetated fraction (NDVI > 0.3)", f"{nd['vegetated_fraction']:.2%}"]]
    if "detections" in results:
        rows.append(["Detections", str(len(results["detections"]))])
    if "sar" in results:
        s = results["sar"]
        rows += [["SAR mean", f"{s['mean']:.4f}"], ["SAR std", f"{s['std']:.4f}"]]
    if "fusion" in results:
        rows.append(["Optical/SAR correlation", f"{results['fusion']['correlation']:.4f}"])
    if len(rows) > 1:
        story.extend([Spacer(1, 8), Paragraph("Computed metrics", styles["Heading2"]), Table(rows, colWidths=[90 * mm, 70 * mm], repeatRows=1, style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))])

    if "ndvi" in results:
        ndvi_png = _array_png(results["ndvi"]["array"])
        if ndvi_png:
            story.extend([Spacer(1, 8), Paragraph("NDVI result", styles["Heading2"]), _report_image(ndvi_png, max_height_mm=85)])
    if "change_detection" in results:
        heat_png = _array_png(results["change_detection"]["heatmap"])
        mask_png = _array_png(results["change_detection"]["mask"], mode="mask")
        if heat_png:
            story.extend([Spacer(1, 8), Paragraph("Change heatmap", styles["Heading2"]), _report_image(heat_png, max_height_mm=85)])
        if mask_png:
            story.extend([Spacer(1, 6), Paragraph("Change threshold mask", styles["Heading2"]), _report_image(mask_png, max_height_mm=85)])

    doc.build(story)
    return out.getvalue()
