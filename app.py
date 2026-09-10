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
st.set_page_config(page_title="SatQueryX", page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { --sx-cyan:#22d3ee; --sx-blue:#38bdf8; --sx-bg:#071018; --sx-panel:#0d1822; --sx-border:rgba(148,163,184,.18); }
.block-container { max-width: 1520px; padding: 1.1rem 2rem 3rem; }
[data-testid="stAppViewContainer"] { background: radial-gradient(circle at 12% 0%, rgba(34,211,238,.08), transparent 28%), #071018; }
[data-testid="stSidebar"] { background:#08131d; border-right:1px solid var(--sx-border); }
[data-testid="stSidebar"] > div:first-child { padding-top:1rem; }
.sx-hero { border:1px solid rgba(34,211,238,.20); border-radius:22px; padding:24px 28px; margin-bottom:18px; background:linear-gradient(135deg,rgba(14,42,56,.92),rgba(8,18,28,.96)); box-shadow:0 16px 50px rgba(0,0,0,.24); }
.sx-kicker { color:var(--sx-cyan); font-size:.72rem; font-weight:800; letter-spacing:.18em; text-transform:uppercase; }
.sx-title { font-size:2.35rem; font-weight:800; letter-spacing:-.04em; margin:.15rem 0 .25rem; color:#f2fbff; }
.sx-sub { color:#9fb4c2; font-size:.95rem; max-width:900px; }
.sx-chip { display:inline-block; border:1px solid rgba(34,211,238,.24); background:rgba(34,211,238,.07); color:#b9f5ff; border-radius:999px; padding:5px 10px; margin:4px 5px 0 0; font-size:.72rem; }
.sx-section { font-size:1.1rem; font-weight:750; margin:18px 0 8px; color:#e6f4f7; }
.sx-card { border:1px solid var(--sx-border); background:rgba(13,24,34,.72); border-radius:16px; padding:16px 18px; }
.sx-muted { color:#8ea3b1; font-size:.82rem; }
.sx-answer { border:1px solid rgba(34,211,238,.25); border-left:4px solid var(--sx-cyan); background:linear-gradient(90deg,rgba(34,211,238,.09),rgba(13,24,34,.72)); border-radius:15px; padding:18px 20px; }
.sx-trace { border-left:2px solid rgba(34,211,238,.35); padding-left:12px; margin:8px 0; }
div[data-testid="stMetric"] { background:rgba(13,24,34,.68); border:1px solid var(--sx-border); border-radius:14px; padding:10px 12px; }
button[kind="primary"] { border-radius:12px; font-weight:750; }
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] { border-radius:10px; padding:7px 14px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sx-hero">
  <div class="sx-kicker">REMOTE SENSING INTELLIGENCE CONSOLE</div>
  <div class="sx-title">🛰️ SatQueryX</div>
  <div class="sx-sub">Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries.</div>
  <div style="margin-top:10px">
    <span class="sx-chip">Agentic orchestration</span><span class="sx-chip">Optical + SAR</span><span class="sx-chip">Bi-temporal change</span><span class="sx-chip">Text-guided evidence</span><span class="sx-chip">Auditable trace</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.info("No login. No fabricated detections, benchmark scores, sensor metadata or confidence values. Missing specialist checkpoints are reported explicitly.")

for key, default in {
    "aoi_center": (28.6139, 77.2090), "aoi_geometry": None, "aoi_location": None,
    "aoi_primary_bytes": None, "aoi_secondary_bytes": None,
    "aoi_primary_meta": None, "aoi_secondary_meta": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.markdown("### 🛰️ Mission inputs")
    primary = st.file_uploader("Primary image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="primary")
    secondary = st.file_uploader("Second image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="secondary")
    st.caption("Use GeoTIFF/TIFF for geospatial, spectral, temporal and paired-sensor workflows. PNG/JPG are accepted for benchmark visual inputs.")
    st.divider()
    st.markdown("**Spectral / SAR**")
    red_band = st.number_input("Red band (1-based)", min_value=1, value=3, step=1)
    nir_band = st.number_input("NIR band (1-based)", min_value=1, value=4, step=1)
    sar_db = st.checkbox("Convert SAR linear power to dB", value=False)
    st.divider()
    st.markdown("**Auxiliary detection**")
    yolo_conf = st.slider("YOLO confidence", 0.05, 0.95, float(os.getenv("YOLO_CONFIDENCE", "0.25")), 0.05)
    st.caption("YOLO remains auxiliary. SIH text-guided grounding uses RS_GROUNDING_MODEL_ID.")
    st.divider()
    st.markdown("**SIH readiness**")
    for item in dataset_status():
        icon = "🟢" if item["configured"] else "⚪"
        st.write(f"{icon} **{item['name']}**")
        st.caption(item["role"])

st.markdown('<div class="sx-section">🗺️ Area of Interest</div>', unsafe_allow_html=True)
st.caption("Draw an AOI to fetch real Sentinel-2 imagery, or upload benchmark GeoTIFFs in the mission panel.")
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
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Centre", f"{bbox_center(bbox)[0]:.4f}, {bbox_center(bbox)[1]:.4f}")
    with c2: st.metric("Lat span", f"{bbox[3]-bbox[1]:.4f}°")
    with c3: st.metric("Lon span", f"{bbox[2]-bbox[0]:.4f}°")
    with c4: st.metric("Location", st.session_state.get("aoi_location") or "Resolving…")

with st.expander("🛰️ Automatic Sentinel-2 acquisition", expanded=not bool(st.session_state.get("aoi_primary_meta"))):
    d1, d2, d3 = st.columns(3)
    start_default, end_default = default_dates()
    with d1: cloud_limit = st.slider("Maximum cloud cover", 1, 80, 15, 1)
    with d2: start_date = st.date_input("Search from", start_default)
    with d3: end_date = st.date_input("Search to", end_default)
    mode = st.radio("Acquisition mode", ["Single latest Sentinel-2 scene", "Two dates for change detection"], horizontal=True)
    fetch_button = st.button("🔎 Find & fetch imagery", type="primary", width="stretch", disabled=st.session_state["aoi_geometry"] is None)
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
            st.success("Satellite imagery fetched successfully.")
        except Exception as exc:
            st.error(f"AOI satellite fetch failed: {exc}")

if st.session_state.get("aoi_primary_meta"):
    meta = st.session_state["aoi_primary_meta"]
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Primary scene", meta["scene_id"])
    with c2: st.metric("Acquisition", meta.get("datetime", "unavailable"))
    with c3: st.metric("Cloud cover", f"{meta.get('cloud_cover','n/a')}%")
    if st.session_state.get("aoi_secondary_meta"):
        sm = st.session_state["aoi_secondary_meta"]
        st.caption(f"Comparison scene: **{sm['scene_id']}** · {sm.get('datetime','date unavailable')} · cloud {sm.get('cloud_cover','n/a')}%")

st.markdown('<div class="sx-section">💬 Query console</div>', unsafe_allow_html=True)
q1, q2 = st.columns([5, 1.15])
with q1:
    query = st.text_area("Natural-language request", placeholder="Ask about land cover, water, objects, change, or complementary optical/SAR information…", height=88, label_visibility="collapsed")
with q2:
    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
    has_primary = bool(primary or st.session_state.get("aoi_primary_bytes"))
    has_second = bool(secondary or st.session_state.get("aoi_secondary_bytes"))
    run = st.button("▶ Run\nanalysis", type="primary", width="stretch", disabled=not bool(has_primary and query.strip()))
if primary:
    st.caption(f"Primary: **{primary.name}** · {primary.size/1024:.1f} KB" + (f" · Secondary: **{secondary.name}** · {secondary.size/1024:.1f} KB" if secondary else ""))
elif st.session_state.get("aoi_primary_meta"):
    st.caption(f"AOI-derived primary: **Sentinel-2 {st.session_state['aoi_primary_meta']['scene_id']}**")


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
                for error in errors: st.error(error)
                raise ValueError("Input compatibility check failed.")
            steps.append(ExecutionStep("intent_classification", "success", ", ".join(plan.intents)))
            steps.append(ExecutionStep("workflow_selection", "success", f"{plan.workflow}; datasets={', '.join(plan.datasets)}"))
            steps.append(ExecutionStep("tool_selection", "success", ", ".join(plan.tools)))

            if st.session_state.get("aoi_primary_meta") and not primary:
                results["aoi"] = {**st.session_state["aoi_primary_meta"], "location": st.session_state.get("aoi_location"), "bbox": geometry_bbox(st.session_state["aoi_geometry"]), "geometry": st.session_state["aoi_geometry"]}
                steps.append(ExecutionStep("aoi_context", "success", f"Location={results['aoi']['location']}; scene={results['aoi']['scene_id']}"))

            if plan.workflow == "single_vqa":
                try:
                    spec = run_single_vqa(preview_image, query); results["specialist"] = spec; steps.append(ExecutionStep("rs_vqa", "success", f"model={spec.model}"))
                except Exception as exc:
                    results["specialist_limitation"] = str(exc); steps.append(ExecutionStep("rs_vqa", "warning", str(exc)))

            if plan.workflow == "single_captioning":
                try:
                    spec = run_captioning(preview_image); results["specialist"] = spec; steps.append(ExecutionStep("rs_captioning", "success", f"model={spec.model}"))
                except Exception as exc:
                    results["specialist_limitation"] = str(exc); steps.append(ExecutionStep("rs_captioning", "warning", str(exc)))

            if plan.workflow == "text_grounding":
                try:
                    spec = run_grounding(preview_image, query); results["specialist"] = spec; steps.append(ExecutionStep("rs_grounding", "success", f"model={spec.model}; regions={len(spec.regions or [])}"))
                except Exception as exc:
                    results["specialist_limitation"] = str(exc); steps.append(ExecutionStep("rs_grounding", "warning", str(exc)))

            if plan.workflow == "bi_temporal_change":
                if primary_ds is None or secondary_ds is None: raise ValueError("Bi-temporal change analysis requires two georeferenced GeoTIFFs.")
                before = primary_ds.read(1).astype(np.float32); after = reproject_to_reference(secondary_ds, primary_ds, 1); ch = detect_change(before, after)
                results["change_detection"] = {"difference": ch.difference, "heatmap": ch.normalized_heatmap, "mask": ch.threshold_mask, "changed_fraction": ch.changed_fraction, "mean_absolute_change": ch.mean_absolute_change, "ssim": ch.ssim}
                steps.append(ExecutionStep("change_detection", "success", f"changed_fraction={ch.changed_fraction:.2%}; SSIM={ch.ssim:.4f}"))
                try:
                    spec = run_change_vqa(preview_image, secondary_preview, query); results["specialist"] = spec; steps.append(ExecutionStep("change_vqa", "success", f"model={spec.model}"))
                except Exception as exc:
                    results["specialist_limitation"] = str(exc); steps.append(ExecutionStep("change_vqa", "warning", str(exc)))

            if plan.workflow == "optical_sar":
                if not secondary_bytes: raise ValueError("Optical/SAR workflow requires both images.")
                try:
                    spec = run_optical_sar(preview_image, secondary_preview, query); results["specialist"] = spec; steps.append(ExecutionStep("multimodal_rs_vlm", "success", f"model={spec.model}"))
                except Exception as exc:
                    results["specialist_limitation"] = str(exc); steps.append(ExecutionStep("multimodal_rs_vlm", "warning", str(exc)))
                if primary_ds is not None and secondary_ds is not None:
                    fu = fuse_optical_sar(primary_ds.read(1).astype(np.float32), reproject_to_reference(secondary_ds, primary_ds, 1))
                    results["fusion"] = {"fused": fu.fused, "correlation": fu.correlation, "optical_mean": fu.optical_mean, "sar_mean": fu.sar_mean, "optical_std": fu.optical_std, "sar_std": fu.sar_std}
                    steps.append(ExecutionStep("optical_sar_fusion", "success", f"correlation={fu.correlation:.4f}"))

            if "ndvi" in plan.tools:
                if primary_ds is None: raise ValueError("NDVI requires a georeferenced GeoTIFF with spectral bands.")
                ndvi = compute_ndvi(primary_ds, int(red_band), int(nir_band)); valid = np.isfinite(ndvi)
                results["ndvi"] = {"array": ndvi, "mean": float(np.nanmean(ndvi)), "median": float(np.nanmedian(ndvi)), "vegetated_fraction": float(np.mean(ndvi[valid] > .3)) if valid.any() else 0.0}
                steps.append(ExecutionStep("ndvi", "success", f"mean={results['ndvi']['mean']:.4f}; vegetated_fraction={results['ndvi']['vegetated_fraction']:.2%}"))

            if "sar_statistics" in plan.tools:
                sar_arr = sar_to_db(arr[0]) if sar_db else arr[0]
                results["sar"] = {"mean": float(np.nanmean(sar_arr)), "std": float(np.nanstd(sar_arr)), "min": float(np.nanmin(sar_arr)), "max": float(np.nanmax(sar_arr))}
                steps.append(ExecutionStep("sar_statistics", "success", "Computed from actual raster pixels"))

            if "visual_grounding" in plan.tools:
                try:
                    detections = run_yolo(preview_image, confidence=yolo_conf); results["detections"] = detections; steps.append(ExecutionStep("visual_grounding_auxiliary", "success", f"{len(detections)} detections"))
                except Exception as exc:
                    results["detector_limitation"] = str(exc); steps.append(ExecutionStep("visual_grounding_auxiliary", "warning", str(exc)))

            if results.get("aoi", {}).get("bbox"):
                with st.spinner("Computing bounded site intelligence…"):
                    results["site_summary"] = build_site_summary(results["aoi"]["bbox"], results.get("ndvi"), results["aoi"].get("geometry"))
                site = results["site_summary"]
                steps.append(ExecutionStep("site_intelligence", "success", f"area={site['area_ha']:.3f} ha; buildings={(site.get('buildings') or {}).get('count', 'unavailable')}"))

            evidence = build_evidence(results)
            if results.get("specialist"):
                spec = results["specialist"]
                evidence.facts.insert(0, f"Specialist workflow answer ({spec.task}, model={spec.model}): {spec.answer}")
                evidence.sources.insert(0, spec.model)
                for limitation in spec.limitations: evidence.facts.append("Specialist limitation: " + limitation)
            if results.get("specialist_limitation"): evidence.facts.append("Specialist model limitation: " + results["specialist_limitation"])
            steps.append(ExecutionStep("evidence_synthesis", "success", f"{len(evidence.facts)} evidence statements; confidence={'not estimated' if evidence.confidence is None else f'{evidence.confidence:.2f}'}"))

            try:
                answer, provider = synthesize_answer(query, evidence, preview_image)
            except Exception as exc:
                if results.get("specialist"):
                    answer, provider = results["specialist"].answer, results["specialist"].model; results["language_synthesis_limitation"] = str(exc)
                else: raise
            results["answer"] = answer; results["provider"] = provider
            steps.append(ExecutionStep("language_synthesis", "success", provider)); results["execution_steps"] = list(steps)
            status.update(label="Analysis complete", state="complete", expanded=False)

        pbuf = BytesIO(); preview_image.save(pbuf, format="PNG"); preview_png = pbuf.getvalue()
        st.session_state["satquery_result"] = (query, plan, steps, results, evidence, preview_png)
    except Exception as exc:
        steps.append(ExecutionStep("analysis", "error", str(exc))); results["execution_steps"] = list(steps)
        st.session_state["satquery_result"] = (query, plan if 'plan' in locals() else None, steps, results, evidence, preview_png); st.error(str(exc))
    finally:
        if primary_mem: primary_mem.close()
        if secondary_mem: secondary_mem.close()

if "satquery_result" in st.session_state:
    query0, plan0, steps0, results0, evidence0, preview0 = st.session_state["satquery_result"]
    st.markdown('<div class="sx-section">📡 Analysis output</div>', unsafe_allow_html=True)
    if plan0:
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Workflow", plan0.workflow.replace("_", " ").title())
        with m2: st.metric("Intents", len(plan0.intents))
        with m3: st.metric("Tools", len(plan0.tools))
        with m4: st.metric("Trace steps", len(steps0))

    tabs = st.tabs(["🛰️ Evidence", "🧠 Agent trace", "📊 Metrics", "📄 Report"])
    with tabs[0]:
        left, right = st.columns([1.55, 1])
        with left:
            if preview0:
                st.image(preview0, caption="Primary input / true-colour preview", width="stretch")
            if results0.get("specialist"):
                spec = results0["specialist"]
                st.markdown(f"#### Specialist result · `{spec.task}`")
                st.markdown(f'<div class="sx-card"><b>{spec.answer}</b><br><span class="sx-muted">Model: {spec.model}</span></div>', unsafe_allow_html=True)
                if spec.regions and preview0:
                    st.image(_grounded_preview(Image.open(BytesIO(preview0)), spec.regions), caption="Text-guided grounded regions", width="stretch")
            if results0.get("answer"):
                st.markdown("#### SatQueryX answer")
                st.markdown(f'<div class="sx-answer">{results0["answer"]}</div>', unsafe_allow_html=True)
                st.caption(f"Language synthesis provider: {results0.get('provider', 'unknown')}")
            if results0.get("change_detection"):
                ch = results0["change_detection"]
                st.markdown("#### Change evidence")
                a,b,c = st.columns(3)
                a.metric("Changed area", f"{ch['changed_fraction']:.2%}"); b.metric("SSIM", f"{ch['ssim']:.4f}"); c.metric("Mean |Δ|", f"{ch['mean_absolute_change']:.4f}")
                st.image(ch["heatmap"], caption="Normalized change heatmap", width="stretch")
            if results0.get("fusion"):
                fu = results0["fusion"]
                st.markdown("#### Optical / SAR complementarity")
                a,b,c = st.columns(3); a.metric("Correlation", f"{fu['correlation']:.4f}"); b.metric("Optical mean", f"{fu['optical_mean']:.4f}"); c.metric("SAR mean", f"{fu['sar_mean']:.4f}")
            if results0.get("detections") and preview0:
                st.markdown("#### Auxiliary detections")
                st.image(draw_detections(Image.open(BytesIO(preview0)), results0["detections"]), width="stretch")
        with right:
            if results0.get("aoi"):
                a = results0["aoi"]
                st.markdown("#### Acquisition context")
                st.markdown(f'<div class="sx-card"><b>{a.get("location") or "Location unavailable"}</b><br>Scene: {a.get("scene_id", "Unavailable")}<br>Acquired: {a.get("datetime", "Unavailable")}<br>Cloud: {a.get("cloud_cover", "Unavailable")}%<br>CRS: {a.get("crs", "Unavailable")}</div>', unsafe_allow_html=True)
            if evidence0:
                st.markdown("#### Evidence ledger")
                for fact in evidence0.facts: st.write("• " + fact)
                st.metric("Confidence", f"{evidence0.confidence:.0%}" if evidence0.confidence is not None else "Not estimated")
            if results0.get("specialist_limitation"): st.warning("Specialist limitation: " + results0["specialist_limitation"])
            if results0.get("detector_limitation"): st.warning("Auxiliary detector limitation: " + results0["detector_limitation"])

    with tabs[1]:
        st.caption("Auditable execution trace: intent → workflow → tools/models → evidence → synthesis.")
        for i, step in enumerate(steps0, 1):
            icon = "🟢" if step.status == "success" else ("🟡" if step.status == "warning" else "🔴")
            st.markdown(f'<div class="sx-trace"><b>{i:02d} · {icon} {step.name}</b><br><span class="sx-muted">{step.detail}</span></div>', unsafe_allow_html=True)

    with tabs[2]:
        if results0.get("ndvi"):
            n = results0["ndvi"]; a,b,c = st.columns(3); a.metric("NDVI mean", f"{n['mean']:.4f}"); b.metric("NDVI median", f"{n['median']:.4f}"); c.metric("Vegetated fraction", f"{n['vegetated_fraction']:.1%}")
        if results0.get("sar"):
            s = results0["sar"]; a,b,c,d = st.columns(4); a.metric("SAR mean", f"{s['mean']:.4f}"); b.metric("SAR std", f"{s['std']:.4f}"); c.metric("SAR min", f"{s['min']:.4f}"); d.metric("SAR max", f"{s['max']:.4f}")
        site = results0.get("site_summary")
        if site:
            st.markdown("#### Site intelligence")
            st.metric("AOI area", f"{site['area_ha']:.3f} ha")
            st.caption(f"Area basis: {site.get('area_basis', 'unknown')}")
            if site.get("elevation"):
                e=site["elevation"]; st.write(f"Elevation: mean {e['mean_m']:.1f} m · median {e['median_m']:.1f} m · range {e['min_m']:.1f}–{e['max_m']:.1f} m · relief {e['relief_m']:.1f} m ({e['source']})")
            if site.get("worldcover"):
                for row in site["worldcover"]["rows"]: st.write(f"• {row['label']}: {row['area_ha']:.3f} ha ({row['fraction']:.1%})")
            if site.get("buildings") is not None:
                st.write(f"Mapped buildings: {site['buildings']['count']} ({site['buildings']['source']})")
            if site.get("waterways") is not None:
                w=site["waterways"]; st.write(f"Mapped water features: {w['count']} · types: {', '.join(w['types']) or 'none'}")
            for limitation in site.get("limitations", []): st.warning(limitation)

    with tabs[3]:
        try:
            if evidence0 is not None and plan0 is not None:
                pdf = build_pdf(query0, plan0, evidence0, results0, preview0)
                st.download_button("📄 Download SatQueryX PDF report", pdf, "satqueryx_report.pdf", "application/pdf", width="stretch")
                st.caption("Report contains the query, selected workflow, evidence and computed outputs available from this run.")
        except Exception as exc:
            st.warning(f"PDF report unavailable: {exc}")

else:
    st.markdown('<div class="sx-card"><b>Ready for analysis.</b><br><span class="sx-muted">Upload an image or fetch a Sentinel-2 AOI, then ask SatQueryX a natural-language question.</span></div>', unsafe_allow_html=True)
