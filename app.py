from __future__ import annotations

from io import BytesIO
import os

import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from src.agent import AgenticOrchestrator, ExecutionStep
from src.evidence import build_evidence, synthesize_answer
from src.geospatial import compute_ndvi, raster_info, raster_or_image, read_preview, reproject_to_reference, rgb_preview, sar_to_db, validate_geospatial
from src.report import build_pdf
from src.tools.change_detection import detect_change
from src.tools.optical_sar_fusion import fuse_optical_sar
from src.tools.visual_grounding import draw_detections, run_yolo

load_dotenv()
st.set_page_config(page_title="SatQueryX", page_icon="🛰️", layout="wide")
st.markdown("""<style>.block-container {max-width: 1450px; padding-top: 1.5rem;}</style>""", unsafe_allow_html=True)

st.title("🛰️ SatQueryX")
st.caption("Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries")
st.info("No login is required. SatQueryX performs real analysis only; unavailable models or services are reported instead of replaced with dummy outputs.")

with st.sidebar:
    st.header("Analysis inputs")
    primary = st.file_uploader("Primary satellite image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="primary")
    secondary = st.file_uploader("Second image (change/fusion)", type=["tif", "tiff", "png", "jpg", "jpeg"], key="secondary")
    st.divider()
    st.subheader("Spectral settings")
    red_band = st.number_input("Red band (1-based)", min_value=1, value=3, step=1)
    nir_band = st.number_input("NIR band (1-based)", min_value=1, value=4, step=1)
    sar_db = st.checkbox("Convert SAR linear power to dB", value=False)
    st.divider()
    st.subheader("Detection")
    yolo_conf = st.slider("YOLO confidence", 0.05, 0.95, float(os.getenv("YOLO_CONFIDENCE", "0.25")), 0.05)
    st.caption("Uses the real checkpoint configured by YOLO_MODEL_PATH.")

st.subheader("Ask SatQueryX")
query = st.text_area("Natural-language query", placeholder="e.g. Detect vehicles and buildings, calculate NDVI, or compare these two images for change.", height=90)
run = st.button("Run analysis", type="primary", use_container_width=True, disabled=not bool(primary and query.strip()))

if primary:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Primary input")
        st.write(f"**{primary.name}** · {primary.size / 1024:.1f} KB")
    with col2:
        if secondary:
            st.markdown("### Secondary input")
            st.write(f"**{secondary.name}** · {secondary.size / 1024:.1f} KB")

if run:
    orchestrator = AgenticOrchestrator()
    plan = orchestrator.plan(query, has_second_image=bool(secondary))
    errors = orchestrator.validate_required_inputs(plan, bool(primary), bool(secondary))
    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    steps: list[ExecutionStep] = []
    results: dict = {}
    preview_image: Image.Image | None = None
    preview_png: bytes | None = None
    primary_mem = primary_ds = secondary_mem = secondary_ds = None

    try:
        with st.status("Running SatQueryX analysis…", expanded=True) as status:
            steps.append(ExecutionStep("intent_classification", "success", ", ".join(plan.intents)))
            st.write("Intent classified: " + ", ".join(plan.intents))
            steps.append(ExecutionStep("tool_selection", "success", ", ".join(plan.tools)))
            st.write("Tools selected: " + ", ".join(plan.tools))

            kind, obj = raster_or_image(primary.getvalue(), primary.name)
            if kind == "raster":
                primary_mem, primary_ds = obj
                validation = validate_geospatial(primary_ds)
                if validation:
                    raise ValueError("Primary GeoTIFF validation failed: " + " ".join(validation))
                info = raster_info(primary_ds, primary.name)
                arr, _ = read_preview(primary_ds)
                preview_image = Image.fromarray((rgb_preview(primary_ds) * 255).astype(np.uint8))
                results["primary_info"] = info
                steps.append(ExecutionStep("geospatial_validation", "success", f"CRS={info.crs}; {info.width}×{info.height}; {info.count} bands; resolution={info.resolution_x:g}×{info.resolution_y:g}"))
            else:
                rgb = obj
                preview_image = Image.fromarray(rgb)
                arr = np.moveaxis(rgb, -1, 0).astype(np.float32)
                steps.append(ExecutionStep("image_ingestion", "success", f"{rgb.shape[1]}×{rgb.shape[0]} RGB image"))

            pbuf = BytesIO()
            preview_image.save(pbuf, format="PNG")
            preview_png = pbuf.getvalue()

            if "ndvi" in plan.tools:
                if primary_ds is None:
                    raise ValueError("NDVI requires a GeoTIFF with explicit red and NIR spectral bands.")
                ndvi = compute_ndvi(primary_ds, int(red_band), int(nir_band))
                valid = np.isfinite(ndvi)
                results["ndvi"] = {"array": ndvi, "mean": float(np.nanmean(ndvi)), "median": float(np.nanmedian(ndvi)), "vegetated_fraction": float(np.mean(ndvi[valid] > 0.3)) if valid.any() else 0.0}
                steps.append(ExecutionStep("ndvi", "success", f"red={red_band}, nir={nir_band}; mean={results['ndvi']['mean']:.4f}; vegetated_fraction={results['ndvi']['vegetated_fraction']:.2%}"))

            if "sar_statistics" in plan.tools:
                sar_arr = arr[0]
                if sar_db:
                    sar_arr = sar_to_db(sar_arr)
                results["sar"] = {"mean": float(np.nanmean(sar_arr)), "std": float(np.nanstd(sar_arr)), "min": float(np.nanmin(sar_arr)), "max": float(np.nanmax(sar_arr))}
                steps.append(ExecutionStep("sar_statistics", "success", "Computed from actual raster pixels"))

            if secondary:
                skind, sobj = raster_or_image(secondary.getvalue(), secondary.name)
                if skind == "raster":
                    secondary_mem, secondary_ds = sobj
                    validation = validate_geospatial(secondary_ds)
                    if validation:
                        raise ValueError("Secondary GeoTIFF validation failed: " + " ".join(validation))
                    secondary_info = raster_info(secondary_ds, secondary.name)
                    results["secondary_info"] = secondary_info
                    steps.append(ExecutionStep("secondary_geospatial_validation", "success", f"CRS={secondary_info.crs}; {secondary_info.width}×{secondary_info.height}; {secondary_info.count} bands"))
                else:
                    secondary_ds = None
                    steps.append(ExecutionStep("secondary_image_ingestion", "success", f"{sobj.shape[1]}×{sobj.shape[0]} RGB image"))

                if "change_detection" in plan.tools:
                    if primary_ds is None or secondary_ds is None:
                        raise ValueError("Change detection requires two georeferenced GeoTIFF inputs so the second image can be aligned to the first grid.")
                    before = primary_ds.read(1).astype(np.float32)
                    after = reproject_to_reference(secondary_ds, primary_ds, 1)
                    ch = detect_change(before, after)
                    results["change_detection"] = {"difference": ch.difference, "heatmap": ch.normalized_heatmap, "mask": ch.threshold_mask, "changed_fraction": ch.changed_fraction, "mean_absolute_change": ch.mean_absolute_change, "ssim": ch.ssim}
                    steps.append(ExecutionStep("change_detection", "success", f"changed_fraction={ch.changed_fraction:.2%}; SSIM={ch.ssim:.4f}; mean_change={ch.mean_absolute_change:.4f}"))

                if "optical_sar_fusion" in plan.tools:
                    if primary_ds is None or secondary_ds is None:
                        raise ValueError("Optical/SAR fusion requires two georeferenced GeoTIFF inputs.")
                    optical = primary_ds.read(1).astype(np.float32)
                    sar = reproject_to_reference(secondary_ds, primary_ds, 1)
                    fu = fuse_optical_sar(optical, sar)
                    results["fusion"] = {"fused": fu.fused, "correlation": fu.correlation, "optical_mean": fu.optical_mean, "sar_mean": fu.sar_mean, "optical_std": fu.optical_std, "sar_std": fu.sar_std}
                    steps.append(ExecutionStep("optical_sar_fusion", "success", f"correlation={fu.correlation:.4f}"))

            if "visual_grounding" in plan.tools:
                detections = run_yolo(preview_image, confidence=yolo_conf)
                results["detections"] = detections
                steps.append(ExecutionStep("visual_grounding", "success", f"{len(detections)} detections"))

            evidence = build_evidence(results)
            confidence_detail = f"{evidence.confidence:.2f}" if evidence.confidence is not None else "not estimated"
            steps.append(ExecutionStep("evidence_synthesis", "success", f"{len(evidence.facts)} evidence statements; detector confidence={confidence_detail}"))
            answer, provider = synthesize_answer(query, evidence, preview_image)
            results["answer"] = answer
            results["provider"] = provider
            steps.append(ExecutionStep("language_synthesis", "success", provider))
            # Preserve the exact execution trace for the downloadable report.
            results["execution_steps"] = list(steps)
            status.update(label="Analysis complete", state="complete", expanded=False)

        st.session_state["satquery_result"] = (query, plan, steps, results, evidence, preview_png)
    except Exception as exc:
        steps.append(ExecutionStep("analysis", "error", str(exc)))
        results["execution_steps"] = list(steps)
        st.session_state["satquery_result"] = (query, plan, steps, results, None, preview_png)
        st.error(str(exc))
    finally:
        if primary_mem:
            primary_mem.close()
        if secondary_mem:
            secondary_mem.close()

if "satquery_result" in st.session_state:
    query0, plan0, steps0, results0, evidence0, preview0 = st.session_state["satquery_result"]
    st.divider()
    left, right = st.columns([2.1, 1])
    with left:
        st.header("Results & evidence")
        if results0.get("answer"):
            st.markdown("### SatQueryX answer")
            st.write(results0["answer"])
            st.caption(f"Language provider: {results0.get('provider', 'unknown')}")
        if evidence0:
            st.markdown("### Computed evidence")
            for fact in evidence0.facts:
                st.write("• " + fact)
            st.metric("Detector confidence", f"{evidence0.confidence:.0%}" if evidence0.confidence is not None else "Not estimated")
        if "ndvi" in results0:
            st.markdown("### NDVI")
            st.image(results0["ndvi"]["array"], clamp=True, caption="Computed NDVI raster")
        if "change_detection" in results0:
            st.markdown("### Change heatmap")
            st.image(results0["change_detection"]["heatmap"], clamp=True, caption="Computed normalized change intensity")
            st.image(results0["change_detection"]["mask"], clamp=True, caption="Thresholded changed-pixel mask")
        if "detections" in results0:
            st.markdown("### Visual grounding")
            if preview0:
                base_image = Image.open(BytesIO(preview0)).convert("RGB")
                st.image(draw_detections(base_image, results0["detections"]), caption="Detector-produced bounding boxes")
            if results0["detections"]:
                for d in results0["detections"]:
                    st.write(f"**{d.label}** · {d.confidence:.1%} · box={tuple(round(x, 1) for x in d.xyxy)}")
            else:
                st.write("The configured detector returned no detections.")
    with right:
        st.header("Execution trace")
        for s in steps0:
            icon = "✅" if s.status == "success" else "❌"
            st.markdown(f"{icon} **{s.name}** — {s.detail}")
        if evidence0 and preview0:
            pdf = build_pdf(query0, plan0, evidence0, results0, preview0)
            st.download_button("Download PDF report", data=pdf, file_name="satqueryx_report.pdf", mime="application/pdf", use_container_width=True)

st.divider()
st.caption("SatQueryX • no authentication • real-data pipeline • external model/API credentials are never committed to the repository")
