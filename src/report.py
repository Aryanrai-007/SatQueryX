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
        Spacer(1, 6),
        Paragraph("Execution plan", styles["Heading2"]),
        Paragraph("Intents: " + _safe_text(", ".join(plan.intents)), styles["BodyText"]),
        Paragraph("Tools: " + _safe_text(", ".join(plan.tools)), styles["BodyText"]),
    ]

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

    aoi = results.get("aoi")
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
