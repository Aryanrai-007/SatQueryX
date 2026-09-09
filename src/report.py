from __future__ import annotations

from io import BytesIO
from datetime import datetime, timezone
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib import colors


def build_pdf(query: str, plan: Any, evidence: Any, results: dict[str, Any], preview_png: bytes | None = None) -> bytes:
    out = BytesIO()
    doc = SimpleDocTemplate(out, pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("SatQueryX — Remote Sensing Analysis Report", styles["Title"]),
        Paragraph(datetime.now(timezone.utc).strftime("Generated %Y-%m-%d %H:%M UTC"), styles["Normal"]),
        Spacer(1, 8),
        Paragraph("Query", styles["Heading2"]),
        Paragraph(query.replace("&", "&amp;"), styles["BodyText"]),
        Spacer(1, 6),
        Paragraph("Execution plan", styles["Heading2"]),
        Paragraph("Intents: " + ", ".join(plan.intents), styles["BodyText"]),
        Paragraph("Tools: " + ", ".join(plan.tools), styles["BodyText"]),
        Spacer(1, 6),
        Paragraph("Evidence", styles["Heading2"]),
    ]
    story.extend(Paragraph(f"• {f}", styles["BodyText"]) for f in evidence.facts)

    # Confidence is optional: for example, there is no detector confidence when
    # no object detector was executed. Never invent a confidence value.
    confidence_text = (
        f"{evidence.confidence:.2f}"
        if evidence.confidence is not None
        else "Not estimated"
    )
    story.append(Paragraph(f"Aggregate confidence: {confidence_text}", styles["BodyText"]))
    story.append(Paragraph("Sources: " + ", ".join(evidence.sources), styles["BodyText"]))

    if preview_png:
        story.extend([
            Spacer(1, 8),
            Paragraph("Input preview", styles["Heading2"]),
            RLImage(BytesIO(preview_png), width=170 * mm, height=105 * mm),
        ])

    rows = [["Result", "Value"]]
    if "change_detection" in results:
        ch = results["change_detection"]
        rows += [
            ["Changed fraction", f"{ch['changed_fraction']:.2%}"],
            ["Mean absolute change", f"{ch['mean_absolute_change']:.4f}"],
            ["SSIM", f"{ch['ssim']:.4f}"],
        ]
    if "ndvi" in results:
        nd = results["ndvi"]
        rows += [
            ["NDVI mean", f"{nd['mean']:.4f}"],
            ["Vegetated fraction", f"{nd['vegetated_fraction']:.2%}"],
        ]
    if "detections" in results:
        rows += [["Detections", str(len(results["detections"]))]]
    if "sar" in results:
        s = results["sar"]
        rows += [
            ["SAR mean", f"{s['mean']:.4f}"],
            ["SAR std", f"{s['std']:.4f}"],
        ]
    if len(rows) > 1:
        story.extend([
            Spacer(1, 8),
            Paragraph("Metrics", styles["Heading2"]),
            Table(
                rows,
                colWidths=[70 * mm, 90 * mm],
                style=TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]),
            ),
        ])
    doc.build(story)
    return out.getvalue()
