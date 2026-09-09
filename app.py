from __future__ import annotations

from io import BytesIO
import os

import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from src.agent import AgenticOrchestrator, ExecutionStep
from src.aoi import (
    bbox_center,
    build_aoi_map,
    default_dates,
    geometry_bbox,
    geometry_from_drawing,
    reverse_geocode,
    search_sentinel2,
    fetch_sentinel2_snippet,
    select_items,
)
from src.evidence import build_evidence, synthesize_answer
from src.geospatial import compute_ndvi, raster_info, raster_or_image, read_preview, reproject_to_reference, rgb_preview, sar_to_db, validate_geospatial
from src.report import build_pdf
from src.tools.change_detection import detect_change
from src.tools.optical_sar_fusion import fuse_optical_sar
from src.tools.visual_grounding import draw_detections, run_yolo

load_dotenv()
st.set_page_config(page_title="SatQueryX", page_icon="🛰️", layout="wide")
st.markdown("""<style>.block-container {max-width: 1500px; padding-top: 1.2rem;}</style>""", unsafe_allow_html=True)

st.title("🛰️ SatQueryX")
st.caption("Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries")
st.info("No login is required. SatQueryX performs real analysis only; unavailable models or services are reported instead of replaced with dummy outputs.")

if "aoi_center" not in st.session_state:
    st.session_state["aoi_center"] = (28.6139, 77.2090)
if "aoi_geometry" not in st.session_state:
    st.session_state["aoi_geometry"] = None
if "aoi_location" not in st.session_state:
    st.session_state["aoi_location"] = None
if "aoi_primary_bytes" not in st.session_state:
    st.session_state["aoi_primary_bytes"] = None
if "aoi_secondary_bytes" not in st.session_state:
    st.session_state["aoi_secondary_bytes"] = None
if "aoi_primary_meta" not in st.session_state:
    st.session_state["aoi_primary_meta"] = None
if "aoi_secondary_meta" not in st.session_state:
    st.session_state["aoi_secondary_meta"] = None

st.header("🗺️ Select an Area of Interest")
st.caption("Pan/zoom the OpenStreetMap base map, then use the Leaflet rectangle or polygon tool to draw the exact area you want SatQueryX to analyze.")
map_data = build_aoi_map(st.session_state["aoi_center"], zoom=11)
drawing = geometry_from_drawing(map_data.get("last_active_drawing") if map_data else None)
if drawing:
    st.session_state["aoi_geometry"] = drawing
    bbox = geometry_bbox(drawing)
    center = bbox_center(bbox)
    st.session_state["aoi_center"] = center
    try:
        st.session_state["aoi_location"] = reverse_geocode(center[0], center[1])
    except Exception:
        st.session_state["aoi_location"] = "Reverse geocoding unavailable"

if st.session_state["aoi_geometry"]:
    aoi_bbox = geometry_bbox(st.session_state["aoi_geometry"])
    c1, c2, c3 = st.columns(3)
    with c1:
        lat, lon = bbox_center(aoi_bbox)
        st.metric("Selected centre", f"{lat:.5f}, {lon:.5f}")
    with c2:
        st.metric("Latitude span", f"{aoi_bbox[3] - aoi_bbox[1]:.5f}°")
    with c3:
        st.metric("Longitude span", f"{aoi_bbox[2] - aoi_bbox[0]:.5f}°")
    st.caption(f"📍 **Location:** {st.session_state.get('aoi_location') or 'Resolving…'}")

with st.expander("🛰️ Automatic satellite data fetch", expanded=True):
    d1, d2, d3 = st.columns(3)
    start_default, end_default = default_dates(365)
    with d1:
        cloud_limit = st.slider("Maximum cloud cover", 1, 80, 15, 1)
    with d2:
        start_date = st.date_input("Search from", start_default)
    with d3:
        end_date = st.date_input("Search to", end_default)
    mode = st.radio("Acquisition mode", ["Single latest Sentinel-2 scene", "Two dates for change detection"], horizontal=True)
    fetch_button = st.button("🔎 Find & fetch imagery for selected area", type="primary", use_container_width=True, disabled=st.session_state["aoi_geometry"] is None)
    if fetch_button:
        try:
            bbox = geometry_bbox(st.session_state["aoi_geometry"])
            with st.spinner("Searching Earth Search and clipping Sentinel-2 imagery to your AOI…"):
                features = search_sentinel2(bbox, start_date, end_date, max_cloud=float(cloud_limit), limit=12)
                selected = select_items(features, count=2, min_gap_days=14) if mode.startswith("Two") else select_items(features, count=1)
                if not selected:
                    raise ValueError("No Sentinel-2 L2A scene matching the AOI, date range and cloud limit was found.")
                primary_bytes, primary_meta = fetch_sentinel2_snippet(selected[0], bbox)
                secondary_bytes = None
                secondary_meta = None
                if mode.startswith("Two"):
                    if len(selected) < 2:
                        raise ValueError("Only one suitable Sentinel-2 date was found. Widen the date range or increase the cloud limit.")
                    secondary_bytes, secondary_meta = fetch_sentinel2_snippet(selected[1], bbox)
            st.session_state["aoi_primary_bytes"] = primary_bytes
            st.session_state["aoi_primary_meta"] = primary_meta
            st.session_state["aoi_secondary_bytes"] = secondary_bytes
            st.session_state["aoi_secondary_meta"] = secondary_meta
            st.success("Satellite snippet fetched successfully. The selected AOI is now ready for SatQueryX analysis.")
        except Exception as exc:
            st.error(f"AOI satellite fetch failed: {exc}")

if st.session_state.get("aoi_primary_meta"):
    meta = st.session_state["aoi_primary_meta"]
    st.success(f"Ready: {meta['scene_id']} · {meta.get('datetime', 'date unavailable')} · cloud {meta.get('cloud_cover', 'n/a')}% · bands B02/B03/B04/B08")
    if st.session_state.get("aoi_secondary_meta"):
        sm = st.session_state["aoi_secondary_meta"]
        st.info(f"Comparison scene: {sm['scene_id']} · {sm.get('datetime', 'date unavailable')} · cloud {sm.get('cloud_cover', 'n/a')}%")

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
query = st.text_area("Natural-language query", placeholder="e.g. Analyze this area, calculate vegetation health, detect objects, or compare the two selected dates for change.", height=90)
has_primary = bool(primary or st.session_state.get("aoi_primary_bytes"))
run = st.button("Run analysis", type="primary", use_container_width=True, disabled=not bool(has_primary and query.strip()))

if primary:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Primary input")
        st.write(f"**{primary.name}** · {primary.size / 1024:.1f} KB")
    with col2:
        if secondary:
            st.markdown("### Secondary input")
            st.write(f"**{secondary.name}** · {secondary.size / 1024:.1f} KB")
elif st.session_state.get("aoi_primary_meta"):
    st.markdown("### AOI-derived primary input")
    st.write(f"**Sentinel-2 AOI snippet** · {len(st.session_state['aoi_primary_bytes']) / 1024:.1f} KB")

if run:
    orchestrator = AgenticOrchestrator()
    second_available = bool(secondary or st.session_state.get("aoi_secondary_bytes"))
    plan = orchestrator.plan(query, has_second_image=second_available)
    errors = orchestrator.validate_required_inputs(plan, has_primary, second_available)
    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    steps: list[ExecutionStep] = []
    results: dict = {}
    preview_image: Image.Image | None = None
    preview_png: bytes | None = None
    primary_mem = primary_ds = secondary_mem = secondary_ds = None
    primary_bytes = primary.getvalue() if primary else st.session_state["aoi_primary_bytes"]
    primary_name = primary.name if primary else "sentinel2_aoi.tif"
    secondary_bytes = secondary.getvalue() if secondary else st.session_state.get("aoi_secondary_bytes")
    secondary_name = secondary.name if secondary else "sentinel2_aoi_secondary.tif"

    try:
        with st.status("Running SatQueryX analysis…", expanded=True) as status:
            steps.append(ExecutionStep("intent_classification", "success", ", ".join(plan.intents)))
            st.write("Intent classified: " + ", ".join(plan.intents))
            steps.append(ExecutionStep("tool_selection", "success", ", ".join(plan.tools)))
            st.write("Tools selected: " + ", ".join(plan.tools))
            kind, obj = raster_or_image(primary_bytes, primary_name)
            if kind == "raster":
                primary_mem, primary_ds = obj
                validation = validate_geospatial(primary_ds)
                if validation:
                    raise ValueError("Primary GeoTIFF validation failed: " + " ".join(validation))
                info = raster_info(primary_ds, primary_name)
                arr, _ = read_preview(primary_ds)
                preview_image = Image.fromarray((rgb_preview(primary_ds) * 255).astype(np.uint8))
                results["primary_info"] = info
                steps.append(ExecutionStep("geospatial_validation", "success", f"CRS={info.crs}; {info.width}×{info.height}; {info.count} bands; resolution={info.resolution_x:g}×{info.resolution_y:g}"))
            else:
                rgb = obj
                preview_image = Image.fromarray(rgb)
                arr = np.moveaxis(rgb, -1, 0).astype(np.float32)
                steps.append(ExecutionStep("image_ingestion", "success", f"{rgb.shape[1]}×{rgb.shape[0]} RGB image"))

            if st.session_state.get("aoi_primary_meta") and not primary:
                results["aoi"] = {**st.session_state["aoi_primary_meta"], "location": st.session_state.get("aoi_location")}
                steps.append(ExecutionStep("aoi_context", "success", f"Location={results['aoi']['location']}; scene={results['aoi']['scene_id']}"))

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

            if secondary_bytes:
                skind, sobj = raster_or_image(secondary_bytes, secondary_name)
                if skind == "raster":
                    secondary_mem, secondary_ds = sobj
                    validation = validate_geospatial(secondary_ds)
                    if validation:
                        raise ValueError("Secondary GeoTIFF validation failed: " + " ".join(validation))
                    secondary_info = raster_info(secondary_ds, secondary_name)
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
            results["execution_steps"] = list(steps)
            steps.append(ExecutionStep("language_synthesis", "success", provider))
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
        if results0.get("aoi"):
            st.markdown("### AOI & acquisition")
            a = results0["aoi"]
            st.write(f"**Location:** {a.get('location') or 'Unavailable'}")
            st.write(f"**Scene:** {a.get('scene_id', 'Unavailable')} · **Acquired:** {a.get('datetime', 'Unavailable')} · **Cloud:** {a.get('cloud_cover', 'Unavailable')}%")
            st.write(f"**Bands:** {', '.join(a.get('bands', []))} · **CRS:** {a.get('crs', 'Unavailable')}")
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
st.caption("SatQueryX • no authentication • real-data pipeline • OpenStreetMap + Leaflet AOI selection • external model/API credentials are never committed to the repository")
