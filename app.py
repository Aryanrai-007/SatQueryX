from __future__ import annotations

from io import BytesIO
import os

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from dotenv import load_dotenv

from src.agent import AgenticOrchestrator, ExecutionStep
from src.aoi import (
    bbox_center, build_aoi_map, default_dates, geometry_bbox, geometry_from_drawing,
    reverse_geocode, search_sentinel2, fetch_sentinel2_snippet, select_items,
)
from src.datasets import dataset_status
from src.evidence import build_evidence, synthesize_answer
from src.geospatial import compute_ndvi, raster_info, raster_or_image, reproject_to_reference, rgb_preview, sar_to_db, validate_geospatial
from src.models.specialists import run_change_vqa, run_captioning, run_grounding, run_optical_sar, run_single_vqa
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
st.info("No login. No fabricated detections, benchmark scores, sensor metadata or confidence values. Missing specialist checkpoints are reported explicitly.")

for key, default in {
    "aoi_center": (28.6139, 77.2090), "aoi_geometry": None, "aoi_location": None,
    "aoi_primary_bytes": None, "aoi_secondary_bytes": None,
    "aoi_primary_meta": None, "aoi_secondary_meta": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.header("🗺️ Select an Area of Interest")
st.caption("Draw the exact polygon/rectangle to fetch real Sentinel-2 data. Uploaded GeoTIFFs remain supported for benchmark and paired-sensor workflows.")
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

with st.expander("🛰️ Automatic Sentinel-2 AOI fetch", expanded=True):
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
            with st.spinner("Searching Earth Search and clipping Sentinel-2 imagery to the AOI…"):
                features = search_sentinel2(bbox, start_date, end_date, max_cloud=float(cloud_limit))
                selected = select_items(features, count=2, min_gap_days=14) if mode.startswith("Two") else select_items(features, count=1)
                if not selected:
                    raise ValueError("No Sentinel-2 L2A scene matching the AOI, date range and cloud limit was found.")
                primary_bytes, primary_meta = fetch_sentinel2_snippet(selected[0], bbox)
                secondary_bytes = secondary_meta = None
                if mode.startswith("Two"):
                    if len(selected) < 2:
                        raise ValueError("Only one suitable Sentinel-2 date was found. Widen the date range or increase the cloud limit.")
                    secondary_bytes, secondary_meta = fetch_sentinel2_snippet(selected[1], bbox)
            st.session_state.update(aoi_primary_bytes=primary_bytes, aoi_primary_meta=primary_meta, aoi_secondary_bytes=secondary_bytes, aoi_secondary_meta=secondary_meta)
            st.success("Satellite snippet fetched successfully.")
        except Exception as exc:
            st.error(f"AOI satellite fetch failed: {exc}")

if st.session_state.get("aoi_primary_meta"):
    meta = st.session_state["aoi_primary_meta"]
    st.success(f"Ready: {meta['scene_id']} · {meta.get('datetime','date unavailable')} · cloud {meta.get('cloud_cover','n/a')}% · B02/B03/B04/B08")
    if st.session_state.get("aoi_secondary_meta"):
        sm = st.session_state["aoi_secondary_meta"]
        st.info(f"Comparison scene: {sm['scene_id']} · {sm.get('datetime','date unavailable')} · cloud {sm.get('cloud_cover','n/a')}%")

with st.sidebar:
    st.header("Analysis inputs")
    primary = st.file_uploader("Primary image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="primary")
    secondary = st.file_uploader("Second image (temporal or optical/SAR)", type=["tif", "tiff", "png", "jpg", "jpeg"], key="secondary")
    st.caption("GeoTIFF/TIFF is required for geospatial/spectral workflows. PNG/JPG are accepted for benchmark visual inputs.")
    st.divider()
    st.subheader("Spectral settings")
    red_band = st.number_input("Red band (1-based)", min_value=1, value=3, step=1)
    nir_band = st.number_input("NIR band (1-based)", min_value=1, value=4, step=1)
    sar_db = st.checkbox("Convert SAR linear power to dB", value=False)
    st.divider()
    st.subheader("Detection")
    yolo_conf = st.slider("YOLO confidence", 0.05, 0.95, float(os.getenv("YOLO_CONFIDENCE", "0.25")), 0.05)
    st.caption("YOLO is auxiliary. SIH text-guided grounding uses RS_GROUNDING_MODEL_ID.")
    st.divider()
    st.subheader("SIH dataset/model readiness")
    for item in dataset_status():
        icon = "✅" if item["configured"] else "⚪"
        st.write(f"{icon} **{item['name']}** — {item['role']}")

st.subheader("Ask SatQueryX")
query = st.text_area("Natural-language query", placeholder="Describe the land-cover… / Highlight the water body… / What changed… / Use optical and SAR together…", height=90)
has_primary = bool(primary or st.session_state.get("aoi_primary_bytes"))
has_second = bool(secondary or st.session_state.get("aoi_secondary_bytes"))
run = st.button("Run analysis", type="primary", width="stretch", disabled=not bool(has_primary and query.strip()))

if primary:
    st.write(f"**Primary input:** {primary.name} · {primary.size/1024:.1f} KB")
    if secondary: st.write(f"**Secondary input:** {secondary.name} · {secondary.size/1024:.1f} KB")
elif st.session_state.get("aoi_primary_meta"):
    st.write(f"**AOI-derived primary input:** Sentinel-2 AOI snippet · {len(st.session_state['aoi_primary_bytes'])/1024:.1f} KB")


def _modality(name: str, ds=None, meta=None) -> str:
    text = f"{name} {meta or ''}".lower()
    if any(x in text for x in ("sar", "sentinel-1", "risat", "radar", "vv", "vh")):
        return "sar"
    if ds is not None and any((d or "").upper() in {"VV", "VH"} for d in getattr(ds, "descriptions", ())):
        return "sar"
    return "optical"


def _grounded_preview(image: Image.Image, regions: list[dict]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for r in regions:
        box = r.get("bbox", [])
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(v) / 1000.0 for v in box]
        px = (max(0, min(w, int(x1*w))), max(0, min(h, int(y1*h))), max(0, min(w, int(x2*w))), max(0, min(h, int(y2*h))))
        draw.rectangle(px, outline="red", width=3)
        if r.get("label"):
            draw.text((px[0] + 3, px[1] + 3), str(r["label"]), fill="white")
    return out

if run:
    orchestrator = AgenticOrchestrator()
    steps: list[ExecutionStep] = []
    results: dict = {}
    evidence = None
    preview_image = None
    preview_png = None
    primary_mem = primary_ds = secondary_mem = secondary_ds = None
    primary_bytes = primary.getvalue() if primary else st.session_state["aoi_primary_bytes"]
    primary_name = primary.name if primary else "sentinel2_aoi.tif"
    secondary_bytes = secondary.getvalue() if secondary else st.session_state.get("aoi_secondary_bytes")
    secondary_name = secondary.name if secondary else "sentinel2_aoi_secondary.tif"

    try:
        with st.status("Running SatQueryX analysis…", expanded=True) as status:
            # Load/validate primary input first so modality-aware planning is possible.
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
                primary_modality = _modality(primary_name, primary_ds)
                steps.append(ExecutionStep("geospatial_validation", "success", f"CRS={info.crs}; {info.width}×{info.height}; {info.count} bands; resolution={info.resolution_x:g}×{info.resolution_y:g}; modality={primary_modality}"))
            else:
                rgb = obj
                preview_image = Image.fromarray(rgb)
                arr = np.moveaxis(rgb, -1, 0).astype(np.float32)
                primary_modality = _modality(primary_name)
                steps.append(ExecutionStep("image_ingestion", "success", f"{rgb.shape[1]}×{rgb.shape[0]} RGB image; modality={primary_modality}"))

            secondary_modality = None
            if secondary_bytes:
                skind, sobj = raster_or_image(secondary_bytes, secondary_name)
                if skind == "raster":
                    secondary_mem, secondary_ds = sobj
                    validation = validate_geospatial(secondary_ds)
                    if validation:
                        raise ValueError("Secondary GeoTIFF validation failed: " + " ".join(validation))
                    results["secondary_info"] = raster_info(secondary_ds, secondary_name)
                    secondary_modality = _modality(secondary_name, secondary_ds)
                    secondary_preview = Image.fromarray((rgb_preview(secondary_ds) * 255).astype(np.uint8))
                    steps.append(ExecutionStep("secondary_geospatial_validation", "success", f"Validated secondary GeoTIFF; modality={secondary_modality}"))
                else:
                    secondary_preview = Image.fromarray(sobj)
                    secondary_modality = _modality(secondary_name)
                    steps.append(ExecutionStep("secondary_image_ingestion", "success", f"Secondary RGB input; modality={secondary_modality}"))

            plan = orchestrator.plan(query, has_second_image=has_second, modalities=(primary_modality, secondary_modality) if secondary_modality else (primary_modality,))
            errors = orchestrator.validate_required_inputs(plan, has_primary, has_second, (primary_modality, secondary_modality) if secondary_modality else (primary_modality,))
            if errors:
                for error in errors:
                    st.error(error)
                raise ValueError("Input compatibility check failed.")
            steps.append(ExecutionStep("intent_classification", "success", ", ".join(plan.intents)))
            steps.append(ExecutionStep("workflow_selection", "success", f"{plan.workflow}; datasets={', '.join(plan.datasets)}"))
            steps.append(ExecutionStep("tool_selection", "success", ", ".join(plan.tools)))
            st.write(f"**Workflow:** {plan.workflow}")
            st.write(f"**Datasets:** {', '.join(plan.datasets)}")

            if st.session_state.get("aoi_primary_meta") and not primary:
                results["aoi"] = {
                    **st.session_state["aoi_primary_meta"],
                    "location": st.session_state.get("aoi_location"),
                    "bbox": geometry_bbox(st.session_state["aoi_geometry"]),
                    "geometry": st.session_state["aoi_geometry"],
                }
                steps.append(ExecutionStep("aoi_context", "success", f"Location={results['aoi']['location']}; scene={results['aoi']['scene_id']}"))

            # Deterministic specialist workflows.
            if plan.workflow == "single_vqa":
                try:
                    spec = run_single_vqa(preview_image, query)
                    results["specialist"] = spec
                    steps.append(ExecutionStep("rs_vqa", "success", f"model={spec.model}"))
                except Exception as exc:
                    results["specialist_limitation"] = str(exc)
                    steps.append(ExecutionStep("rs_vqa", "warning", str(exc)))

            if plan.workflow == "single_captioning":
                try:
                    spec = run_captioning(preview_image)
                    results["specialist"] = spec
                    steps.append(ExecutionStep("rs_captioning", "success", f"model={spec.model}"))
                except Exception as exc:
                    results["specialist_limitation"] = str(exc)
                    steps.append(ExecutionStep("rs_captioning", "warning", str(exc)))

            if plan.workflow == "text_grounding":
                try:
                    spec = run_grounding(preview_image, query)
                    results["specialist"] = spec
                    steps.append(ExecutionStep("rs_grounding", "success", f"model={spec.model}; regions={len(spec.regions or [])}"))
                except Exception as exc:
                    results["specialist_limitation"] = str(exc)
                    steps.append(ExecutionStep("rs_grounding", "warning", str(exc)))

            if plan.workflow == "bi_temporal_change":
                if primary_ds is None or secondary_ds is None:
                    raise ValueError("Bi-temporal change analysis requires two georeferenced GeoTIFFs.")
                before = primary_ds.read(1).astype(np.float32)
                after = reproject_to_reference(secondary_ds, primary_ds, 1)
                ch = detect_change(before, after)
                results["change_detection"] = {"difference": ch.difference, "heatmap": ch.normalized_heatmap, "mask": ch.threshold_mask, "changed_fraction": ch.changed_fraction, "mean_absolute_change": ch.mean_absolute_change, "ssim": ch.ssim}
                steps.append(ExecutionStep("change_detection", "success", f"changed_fraction={ch.changed_fraction:.2%}; SSIM={ch.ssim:.4f}"))
                try:
                    spec = run_change_vqa(preview_image, secondary_preview, query)
                    results["specialist"] = spec
                    steps.append(ExecutionStep("change_vqa", "success", f"model={spec.model}"))
                except Exception as exc:
                    results["specialist_limitation"] = str(exc)
                    steps.append(ExecutionStep("change_vqa", "warning", str(exc)))

            if plan.workflow == "optical_sar":
                if not secondary_bytes:
                    raise ValueError("Optical/SAR workflow requires both images.")
                try:
                    spec = run_optical_sar(preview_image, secondary_preview, query)
                    results["specialist"] = spec
                    steps.append(ExecutionStep("multimodal_rs_vlm", "success", f"model={spec.model}"))
                except Exception as exc:
                    results["specialist_limitation"] = str(exc)
                    steps.append(ExecutionStep("multimodal_rs_vlm", "warning", str(exc)))
                if primary_ds is not None and secondary_ds is not None:
                    optical = primary_ds.read(1).astype(np.float32)
                    sar = reproject_to_reference(secondary_ds, primary_ds, 1)
                    fu = fuse_optical_sar(optical, sar)
                    results["fusion"] = {"fused": fu.fused, "correlation": fu.correlation, "optical_mean": fu.optical_mean, "sar_mean": fu.sar_mean, "optical_std": fu.optical_std, "sar_std": fu.sar_std}
                    steps.append(ExecutionStep("optical_sar_fusion", "success", f"correlation={fu.correlation:.4f}"))

            if "ndvi" in plan.tools:
                if primary_ds is None:
                    raise ValueError("NDVI requires a georeferenced GeoTIFF with spectral bands.")
                ndvi = compute_ndvi(primary_ds, int(red_band), int(nir_band))
                valid = np.isfinite(ndvi)
                results["ndvi"] = {"array": ndvi, "mean": float(np.nanmean(ndvi)), "median": float(np.nanmedian(ndvi)), "vegetated_fraction": float(np.mean(ndvi[valid] > .3)) if valid.any() else 0.0}
                steps.append(ExecutionStep("ndvi", "success", f"mean={results['ndvi']['mean']:.4f}; vegetated_fraction={results['ndvi']['vegetated_fraction']:.2%}"))

            if "sar_statistics" in plan.tools:
                sar_arr = arr[0]
                sar_arr = sar_to_db(sar_arr) if sar_db else sar_arr
                results["sar"] = {"mean": float(np.nanmean(sar_arr)), "std": float(np.nanstd(sar_arr)), "min": float(np.nanmin(sar_arr)), "max": float(np.nanmax(sar_arr))}
                steps.append(ExecutionStep("sar_statistics", "success", "Computed from actual raster pixels"))

            if "visual_grounding" in plan.tools:
                try:
                    detections = run_yolo(preview_image, confidence=yolo_conf)
                    results["detections"] = detections
                    steps.append(ExecutionStep("visual_grounding_auxiliary", "success", f"{len(detections)} detections"))
                except Exception as exc:
                    results["detector_limitation"] = str(exc)
                    steps.append(ExecutionStep("visual_grounding_auxiliary", "warning", str(exc)))

            if results.get("aoi", {}).get("bbox"):
                with st.spinner("Computing bounded site intelligence once…"):
                    results["site_summary"] = build_site_summary(results["aoi"]["bbox"], results.get("ndvi"), results["aoi"].get("geometry"))
                site = results["site_summary"]
                steps.append(ExecutionStep("site_intelligence", "success", f"area={site['area_ha']:.3f} ha; buildings={(site.get('buildings') or {}).get('count', 'unavailable')}"))

            evidence = build_evidence(results)
            if results.get("specialist"):
                spec = results["specialist"]
                evidence.facts.insert(0, f"Specialist workflow answer ({spec.task}, model={spec.model}): {spec.answer}")
                evidence.sources.insert(0, spec.model)
                for limitation in spec.limitations:
                    evidence.facts.append("Specialist limitation: " + limitation)
            if results.get("specialist_limitation"):
                evidence.facts.append("Specialist model limitation: " + results["specialist_limitation"])
            steps.append(ExecutionStep("evidence_synthesis", "success", f"{len(evidence.facts)} evidence statements; detector confidence={'not estimated' if evidence.confidence is None else f'{evidence.confidence:.2f}'}"))

            # Gemini is synthesis only; the RS specialist remains the task model.
            try:
                answer, provider = synthesize_answer(query, evidence, preview_image)
            except Exception as exc:
                if results.get("specialist"):
                    answer, provider = results["specialist"].answer, results["specialist"].model
                    results["language_synthesis_limitation"] = str(exc)
                else:
                    raise
            results["answer"] = answer
            results["provider"] = provider
            steps.append(ExecutionStep("language_synthesis", "success", provider))
            results["execution_steps"] = list(steps)
            status.update(label="Analysis complete", state="complete", expanded=False)

        pbuf = BytesIO(); preview_image.save(pbuf, format="PNG"); preview_png = pbuf.getvalue()
        st.session_state["satquery_result"] = (query, plan, steps, results, evidence, preview_png)
    except Exception as exc:
        steps.append(ExecutionStep("analysis", "error", str(exc)))
        results["execution_steps"] = list(steps)
        st.session_state["satquery_result"] = (query, plan if 'plan' in locals() else None, steps, results, evidence, preview_png)
        st.error(str(exc))
    finally:
        if primary_mem: primary_mem.close()
        if secondary_mem: secondary_mem.close()

if "satquery_result" in st.session_state:
    query0, plan0, steps0, results0, evidence0, preview0 = st.session_state["satquery_result"]
    st.divider()
    left, right = st.columns([2.1, 1])
    with left:
        st.header("Results & evidence")
        if preview0:
            st.markdown("### Input imagery")
            st.image(preview0, caption="Primary input / true-colour preview", width="stretch")
        if results0.get("specialist"):
            spec = results0["specialist"]
            st.markdown(f"### Specialist result — {spec.task}")
            st.write(spec.answer)
            st.caption(f"Model: {spec.model}")
            if spec.regions:
                st.image(_grounded_preview(Image.open(BytesIO(preview0)), spec.regions), caption="Text-guided grounded regions", width="stretch")
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
            if site.get("elevation"):
                e = site["elevation"]; st.write(f"**Elevation:** mean {e['mean_m']:.1f} m · median {e['median_m']:.1f} m · range {e['min_m']:.1f}–{e['max_m']:.1f} m · relief {e['relief_m']:.1f} m ({e['source']})")
            if site.get("worldcover"):
                st.markdown("**Land cover — ESA WorldCover 2021 v200**")
                for row in site["worldcover"]["rows"]:
                    st.write(f"• {row['label']}: {row['area_ha']:.3f} ha ({row['fraction']:.1%})")
            if site.get("buildings") is not None:
                b = site["buildings"]; st.write(f"**Mapped buildings:** {b['count']} ({b['source']})")
            if site.get("waterways") is not None:
                w = site["waterways"]; st.write(f"**Mapped water features:** {w['count']} · types: {', '.join(w['types']) or 'none'} · names: {', '.join(w['names']) or 'unnamed/unavailable'}")
            for limitation in site.get("limitations", []): st.warning(limitation)
        if results0.get("specialist_limitation"): st.warning("Specialist limitation: " + results0["specialist_limitation"])
        if results0.get("detector_limitation"): st.warning("Auxiliary detector limitation: " + results0["detector_limitation"])
        if evidence0:
            st.markdown("### Computed evidence")
            for fact in evidence0.facts: st.write("• " + fact)
            st.metric("Detector confidence", f"{evidence0.confidence:.0%}" if evidence0.confidence is not None else "Not estimated")
        if results0.get("ndvi"):
            n = results0["ndvi"]; st.markdown("### NDVI"); st.write(f"Mean: {n['mean']:.4f} · Median: {n['median']:.4f} · Vegetated fraction (NDVI > 0.3): {n['vegetated_fraction']:.1%}")
        if results0.get("detections"):
            st.markdown("### Auxiliary image detections"); st.image(draw_detections(Image.open(BytesIO(preview0)), results0["detections"]), width="stretch")
        if results0.get("change_detection"):
            ch = results0["change_detection"]; st.markdown("### Change map"); st.write(f"Changed fraction: {ch['changed_fraction']:.2%} · SSIM: {ch['ssim']:.4f} · mean absolute change: {ch['mean_absolute_change']:.4f}"); st.image(ch["heatmap"], caption="Normalized change heatmap", width="stretch")
        if results0.get("fusion"):
            fu = results0["fusion"]; st.markdown("### Optical/SAR fusion"); st.write(f"Optical/SAR correlation: {fu['correlation']:.4f}; optical mean={fu['optical_mean']:.4f}; SAR mean={fu['sar_mean']:.4f}")
        try:
            if evidence0 is not None and plan0 is not None:
                pdf = build_pdf(query0, plan0, evidence0, results0, preview0)
                st.download_button("📄 Download PDF report", pdf, "satqueryx_report.pdf", "application/pdf", width="stretch")
        except Exception as exc:
            st.warning(f"PDF report unavailable: {exc}")
    with right:
        st.header("Execution trace")
        for step in steps0: st.write(f"**{step.name}** · {step.status} — {step.detail}")
