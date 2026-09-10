from __future__ import annotations

from io import BytesIO
import os

import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from src.agent import AgenticOrchestrator, ExecutionStep
from src.aoi import (
    bbox_center, build_aoi_map, default_dates, geometry_bbox, geometry_from_drawing,
    reverse_geocode, search_sentinel2, fetch_sentinel2_snippet, select_items,
)
from src.evidence import build_evidence, synthesize_answer
from src.geospatial import (
    compute_ndvi, raster_info, raster_or_image, reproject_to_reference,
    rgb_preview, sar_to_db, validate_geospatial,
)
from src.report import build_pdf
from src.site_analysis import build_site_summary
from src.tools.change_detection import detect_change
from src.tools.optical_sar_fusion import fuse_optical_sar
from src.tools.visual_grounding import draw_detections, run_yolo

load_dotenv()
st.set_page_config(page_title="SatQueryX", page_icon="🛰️", layout="wide")
st.markdown("<style>.block-container {max-width: 1500px; padding-top: 1.2rem;}</style>", unsafe_allow_html=True)

st.title("🛰️ SatQueryX")
st.caption("Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries")
st.info("No login is required. SatQueryX performs real analysis only; unavailable models or services are reported instead of replaced with dummy outputs.")

for key, default in {
    "aoi_center": (28.6139, 77.2090), "aoi_geometry": None, "aoi_location": None,
    "aoi_primary_bytes": None, "aoi_secondary_bytes": None,
    "aoi_primary_meta": None, "aoi_secondary_meta": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.header("🗺️ Select an Area of Interest")
st.caption("Pan/zoom the map, switch basemaps, then draw the exact polygon or rectangle you want SatQueryX to analyze.")
map_data = build_aoi_map(st.session_state["aoi_center"], zoom=11)
drawing = geometry_from_drawing(map_data.get("last_active_drawing") if map_data else None)
if drawing:
    st.session_state["aoi_geometry"] = drawing
    bbox = geometry_bbox(drawing)
    st.session_state["aoi_center"] = bbox_center(bbox)
    try:
        st.session_state["aoi_location"] = reverse_geocode(*bbox_center(bbox))
    except Exception:
        st.session_state["aoi_location"] = "Reverse geocoding unavailable"

if st.session_state["aoi_geometry"]:
    bbox = geometry_bbox(st.session_state["aoi_geometry"])
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Selected centre", f"{bbox_center(bbox)[0]:.5f}, {bbox_center(bbox)[1]:.5f}")
    with c2: st.metric("Latitude span", f"{bbox[3]-bbox[1]:.5f}°")
    with c3: st.metric("Longitude span", f"{bbox[2]-bbox[0]:.5f}°")
    st.caption(f"📍 **Location:** {st.session_state.get('aoi_location') or 'Resolving…'}")

with st.expander("🛰️ Automatic satellite data fetch", expanded=True):
    d1, d2, d3 = st.columns(3)
    start_default, end_default = default_dates()
    with d1: cloud_limit = st.slider("Maximum cloud cover", 1, 80, 15, 1)
    with d2: start_date = st.date_input("Search from", start_default)
    with d3: end_date = st.date_input("Search to", end_default)
    mode = st.radio("Acquisition mode", ["Single latest Sentinel-2 scene", "Two dates for change detection"], horizontal=True)
    fetch_button = st.button("🔎 Find & fetch imagery for selected area", type="primary", width="stretch", disabled=st.session_state["aoi_geometry"] is None)
    if fetch_button:
        try:
            bbox = geometry_bbox(st.session_state["aoi_geometry"])
            with st.spinner("Searching Earth Search and clipping Sentinel-2 imagery to your AOI…"):
                features = search_sentinel2(bbox, start_date, end_date, max_cloud=float(cloud_limit))
                selected = select_items(features, count=2, min_gap_days=14) if mode.startswith("Two") else select_items(features, count=1)
                if not selected: raise ValueError("No Sentinel-2 L2A scene matching the AOI, date range and cloud limit was found.")
                primary_bytes, primary_meta = fetch_sentinel2_snippet(selected[0], bbox)
                secondary_bytes = secondary_meta = None
                if mode.startswith("Two"):
                    if len(selected) < 2: raise ValueError("Only one suitable Sentinel-2 date was found. Widen the date range or increase the cloud limit.")
                    secondary_bytes, secondary_meta = fetch_sentinel2_snippet(selected[1], bbox)
            st.session_state.update(aoi_primary_bytes=primary_bytes, aoi_primary_meta=primary_meta, aoi_secondary_bytes=secondary_bytes, aoi_secondary_meta=secondary_meta)
            st.success("Satellite snippet fetched successfully. The selected AOI is ready for SatQueryX analysis.")
        except Exception as exc:
            st.error(f"AOI satellite fetch failed: {exc}")

if st.session_state.get("aoi_primary_meta"):
    meta = st.session_state["aoi_primary_meta"]
    st.success(f"Ready: {meta['scene_id']} · {meta.get('datetime','date unavailable')} · cloud {meta.get('cloud_cover','n/a')}% · bands B02/B03/B04/B08")
    if st.session_state.get("aoi_secondary_meta"):
        sm = st.session_state["aoi_secondary_meta"]
        st.info(f"Comparison scene: {sm['scene_id']} · {sm.get('datetime','date unavailable')} · cloud {sm.get('cloud_cover','n/a')}%")

with st.sidebar:
    st.header("Analysis inputs")
    primary = st.file_uploader("Primary satellite image", type=["tif","tiff","png","jpg","jpeg"], key="primary")
    secondary = st.file_uploader("Second image (change/fusion)", type=["tif","tiff","png","jpg","jpeg"], key="secondary")
    st.caption("GeoTIFF is recommended for spectral analysis. PNG/JPG supports visual analysis only.")
    st.divider(); st.subheader("Spectral settings")
    red_band = st.number_input("Red band (1-based)", min_value=1, value=3, step=1)
    nir_band = st.number_input("NIR band (1-based)", min_value=1, value=4, step=1)
    sar_db = st.checkbox("Convert SAR linear power to dB", value=False)
    st.divider(); st.subheader("Detection")
    yolo_conf = st.slider("YOLO confidence", 0.05, 0.95, float(os.getenv("YOLO_CONFIDENCE", "0.25")), 0.05)
    st.caption("Image detection requires an installed Ultralytics checkpoint. No fake detections are generated.")

st.subheader("Ask SatQueryX")
query = st.text_area("Natural-language query", placeholder="e.g. number of buildings, analyze vegetation, detect objects, or compare the two dates for change.", height=90)
has_primary = bool(primary or st.session_state.get("aoi_primary_bytes"))
run = st.button("Run analysis", type="primary", width="stretch", disabled=not bool(has_primary and query.strip()))

if primary:
    st.write(f"**Primary input:** {primary.name} · {primary.size/1024:.1f} KB")
    if secondary: st.write(f"**Secondary input:** {secondary.name} · {secondary.size/1024:.1f} KB")
elif st.session_state.get("aoi_primary_meta"):
    st.write(f"**AOI-derived primary input:** Sentinel-2 AOI snippet · {len(st.session_state['aoi_primary_bytes'])/1024:.1f} KB")

if run:
    orchestrator = AgenticOrchestrator()
    second_available = bool(secondary or st.session_state.get("aoi_secondary_bytes"))
    plan = orchestrator.plan(query, has_second_image=second_available)
    errors = orchestrator.validate_required_inputs(plan, has_primary, second_available)
    if errors:
        for e in errors: st.error(e)
        st.stop()

    steps: list[ExecutionStep] = []
    results: dict = {}
    preview_image = None
    preview_png = None
    primary_mem = primary_ds = secondary_mem = secondary_ds = None
    evidence = None
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
                preview_image = Image.fromarray((rgb_preview(primary_ds) * 255).astype(np.uint8))
                arr = primary_ds.read().astype(np.float32)
                results["primary_info"] = info
                steps.append(ExecutionStep("geospatial_validation", "success", f"CRS={info.crs}; {info.width}×{info.height}; {info.count} bands; resolution={info.resolution_x:g}×{info.resolution_y:g}"))
            else:
                rgb = obj
                preview_image = Image.fromarray(rgb)
                arr = np.moveaxis(rgb, -1, 0).astype(np.float32)
                steps.append(ExecutionStep("image_ingestion", "success", f"{rgb.shape[1]}×{rgb.shape[0]} RGB image"))

            if st.session_state.get("aoi_primary_meta") and not primary:
                results["aoi"] = {
                    **st.session_state["aoi_primary_meta"],
                    "location": st.session_state.get("aoi_location"),
                    "bbox": geometry_bbox(st.session_state["aoi_geometry"]),
                    "geometry": st.session_state["aoi_geometry"],
                }
                steps.append(ExecutionStep("aoi_context", "success", f"Location={results['aoi']['location']}; scene={results['aoi']['scene_id']}"))

            if "ndvi" in plan.tools:
                if primary_ds is None:
                    raise ValueError("NDVI requires a GeoTIFF with explicit red and NIR spectral bands.")
                ndvi = compute_ndvi(primary_ds, int(red_band), int(nir_band))
                valid = np.isfinite(ndvi)
                results["ndvi"] = {
                    "array": ndvi,
                    "mean": float(np.nanmean(ndvi)),
                    "median": float(np.nanmedian(ndvi)),
                    "vegetated_fraction": float(np.mean(ndvi[valid] > .3)) if valid.any() else 0.0,
                }
                steps.append(ExecutionStep("ndvi", "success", f"red={red_band}, nir={nir_band}; mean={results['ndvi']['mean']:.4f}; vegetated_fraction={results['ndvi']['vegetated_fraction']:.2%}"))

            if "sar_statistics" in plan.tools:
                sar_arr = arr[0]
                sar_arr = sar_to_db(sar_arr) if sar_db else sar_arr
                results["sar"] = {
                    "mean": float(np.nanmean(sar_arr)), "std": float(np.nanstd(sar_arr)),
                    "min": float(np.nanmin(sar_arr)), "max": float(np.nanmax(sar_arr)),
                }
                steps.append(ExecutionStep("sar_statistics", "success", "Computed from actual raster pixels"))

            if secondary_bytes:
                skind, sobj = raster_or_image(secondary_bytes, secondary_name)
                if skind == "raster":
                    secondary_mem, secondary_ds = sobj
                    validation = validate_geospatial(secondary_ds)
                    if validation:
                        raise ValueError("Secondary GeoTIFF validation failed: " + " ".join(validation))
                    results["secondary_info"] = raster_info(secondary_ds, secondary_name)
                    steps.append(ExecutionStep("secondary_geospatial_validation", "success", "Validated georeferenced secondary raster"))
                if "change_detection" in plan.tools:
                    if primary_ds is None or secondary_ds is None:
                        raise ValueError("Change detection requires two georeferenced GeoTIFF inputs.")
                    before = primary_ds.read(1).astype(np.float32)
                    after = reproject_to_reference(secondary_ds, primary_ds, 1)
                    ch = detect_change(before, after)
                    results["change_detection"] = {
                        "difference": ch.difference, "heatmap": ch.normalized_heatmap,
                        "mask": ch.threshold_mask, "changed_fraction": ch.changed_fraction,
                        "mean_absolute_change": ch.mean_absolute_change, "ssim": ch.ssim,
                    }
                    steps.append(ExecutionStep("change_detection", "success", f"changed_fraction={ch.changed_fraction:.2%}; SSIM={ch.ssim:.4f}"))
                if "optical_sar_fusion" in plan.tools:
                    if primary_ds is None or secondary_ds is None:
                        raise ValueError("Optical/SAR fusion requires two georeferenced GeoTIFF inputs.")
                    optical = primary_ds.read(1).astype(np.float32)
                    sar = reproject_to_reference(secondary_ds, primary_ds, 1)
                    fu = fuse_optical_sar(optical, sar)
                    results["fusion"] = {
                        "fused": fu.fused, "correlation": fu.correlation,
                        "optical_mean": fu.optical_mean, "sar_mean": fu.sar_mean,
                        "optical_std": fu.optical_std, "sar_std": fu.sar_std,
                    }

            # Run site intelligence exactly once. build_evidence consumes this result and never re-fetches it.
            if results.get("aoi", {}).get("bbox"):
                with st.spinner("Computing bounded site intelligence…"):
                    site = build_site_summary(
                        results["aoi"]["bbox"],
                        results.get("ndvi"),
                        results["aoi"].get("geometry"),
                    )
                results["site_summary"] = site
                building_count = (site.get("buildings") or {}).get("count")
                detail = f"area={site['area_ha']:.3f} ha; buildings={building_count if building_count is not None else 'unavailable'}"
                steps.append(ExecutionStep("site_intelligence", "success", detail))

            if "visual_grounding" in plan.tools:
                try:
                    detections = run_yolo(preview_image, confidence=yolo_conf)
                    results["detections"] = detections
                    steps.append(ExecutionStep("visual_grounding", "success", f"{len(detections)} detections"))
                except Exception as exc:
                    results["detector_limitation"] = str(exc)
                    steps.append(ExecutionStep("visual_grounding", "warning", str(exc)))
                    st.warning("Image detector unavailable. SatQueryX will continue with available geospatial/site evidence instead of fabricating detections.")

            evidence = build_evidence(results)
            confidence_detail = f"{evidence.confidence:.2f}" if evidence.confidence is not None else "not estimated"
            steps.append(ExecutionStep("evidence_synthesis", "success", f"{len(evidence.facts)} evidence statements; detector confidence={confidence_detail}"))
            answer, provider = synthesize_answer(query, evidence, preview_image)
            results["answer"] = answer
            results["provider"] = provider
            steps.append(ExecutionStep("language_synthesis", "success", provider))
            results["execution_steps"] = list(steps)
            status.update(label="Analysis complete", state="complete", expanded=False)

        pbuf = BytesIO()
        preview_image.save(pbuf, format="PNG")
        preview_png = pbuf.getvalue()
        st.session_state["satquery_result"] = (query, plan, steps, results, evidence, preview_png)
    except Exception as exc:
        steps.append(ExecutionStep("analysis", "error", str(exc)))
        results["execution_steps"] = list(steps)
        st.session_state["satquery_result"] = (query, plan, steps, results, evidence, preview_png)
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
        if preview0:
            st.markdown("### Input imagery")
            st.image(preview0, caption="AOI / primary true-colour preview", width="stretch")
        if results0.get("answer"):
            st.markdown("### SatQueryX answer")
            st.write(results0["answer"])
            st.caption(f"Language provider: {results0.get('provider', 'unknown')}")
        if results0.get("aoi"):
            a = results0["aoi"]
            st.markdown("### AOI & acquisition")
            st.write(f"**Location:** {a.get('location') or 'Unavailable'}")
            st.write(f"**Scene:** {a.get('scene_id', 'Unavailable')} · **Acquired:** {a.get('datetime', 'Unavailable')} · **Cloud:** {a.get('cloud_cover', 'Unavailable')}%")
            st.write(f"**Bands:** {', '.join(a.get('bands', []))} · **CRS:** {a.get('crs', 'Unavailable')}")
        site = results0.get("site_summary")
        if site:
            st.markdown("### Site intelligence")
            st.metric("AOI area", f"{site['area_ha']:.3f} ha")
            st.caption(f"Area basis: {site.get('area_basis', 'unknown')}")
            elev = site.get("elevation")
            if elev:
                st.write(f"**Elevation:** mean {elev['mean_m']:.1f} m · median {elev['median_m']:.1f} m · range {elev['min_m']:.1f}–{elev['max_m']:.1f} m · relief {elev['relief_m']:.1f} m ({elev['source']})")
            wc = site.get("worldcover")
            if wc:
                st.markdown("**Land cover (ESA WorldCover 2021 v200)**")
                for row in wc["rows"]:
                    st.write(f"• {row['label']}: {row['area_ha']:.3f} ha ({row['fraction']:.1%})")
            b = site.get("buildings")
            if b:
                st.write(f"**Mapped buildings:** {b['count']} ({b['source']})")
            w = site.get("waterways")
            if w:
                st.write(f"**Mapped water features:** {w['count']} · types: {', '.join(w['types']) or 'none'} · names: {', '.join(w['names']) or 'unnamed/unavailable'}")
            for limitation in site.get("limitations", []):
                st.warning(limitation)
        if results0.get("detector_limitation"):
            st.warning("Image detector limitation: " + results0["detector_limitation"])
        if evidence0:
            st.markdown("### Computed evidence")
            for fact in evidence0.facts:
                st.write("• " + fact)
            st.metric("Detector confidence", f"{evidence0.confidence:.0%}" if evidence0.confidence is not None else "Not estimated")
        if results0.get("ndvi"):
            n = results0["ndvi"]
            st.markdown("### NDVI")
            st.write(f"Mean: {n['mean']:.4f} · Median: {n['median']:.4f} · Vegetated fraction (NDVI > 0.3): {n['vegetated_fraction']:.1%}")
        if results0.get("detections"):
            st.markdown("### Image detections")
            st.image(draw_detections(Image.open(BytesIO(preview0)), results0["detections"]), width="stretch")
        if results0.get("change_detection"):
            ch = results0["change_detection"]
            st.markdown("### Change detection")
            st.write(f"Changed fraction: {ch['changed_fraction']:.2%} · SSIM: {ch['ssim']:.4f} · mean absolute change: {ch['mean_absolute_change']:.4f}")
        try:
            # build_pdf expects the actual plan, evidence bundle, result dictionary, then preview PNG.
            if evidence0 is not None:
                pdf = build_pdf(query0, plan0, evidence0, results0, preview0)
                st.download_button("📄 Download PDF report", pdf, "satqueryx_report.pdf", "application/pdf", width="stretch")
        except Exception as exc:
            st.warning(f"PDF report unavailable: {exc}")
    with right:
        st.header("Execution trace")
        for step in steps0:
            st.write(f"**{step.name}** · {step.status} — {step.detail}")
