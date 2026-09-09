# SatQueryX

**Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries**

SatQueryX is a no-login Streamlit application for real remote-sensing analysis. It accepts optical/RGB imagery, SAR imagery, and GeoTIFF rasters, validates geospatial metadata, derives NDVI where spectral bands permit it, performs raster change detection, optionally runs YOLO-based visual grounding, optionally loads a Hugging Face remote-sensing VLM, and uses Google Gemini for evidence-grounded natural-language synthesis when configured.

## Design principles

- **No authentication:** there is deliberately no login, signup, auth middleware, or session authentication layer.
- **No fabricated results:** unavailable models, missing API keys, unsupported raster bands, or invalid geospatial inputs produce explicit errors/warnings instead of placeholder answers.
- **Optional integrations are explicit:** Gemini and the Hugging Face VLM are adapters. The application can run deterministic geospatial tools without them, but VLM synthesis requires a configured provider.
- **Traceability:** every analysis records the selected tools, validation steps, warnings, and execution results.

## Architecture

`Streamlit UI -> Geospatial ingestion/preprocessing -> Agentic orchestrator -> analysis tools -> optional RS-VLM -> Evidence synthesizer -> results/report`

Implemented modules:

- `src/geospatial.py` — GeoTIFF/imagery inspection, CRS and resolution validation, optical/SAR preparation, NDVI, tiling.
- `src/tools/visual_grounding.py` — real Ultralytics YOLO inference with configurable weights.
- `src/tools/change_detection.py` — real aligned raster difference/SSIM-style change metrics and heatmaps.
- `src/tools/optical_sar_fusion.py` — aligned feature/statistical fusion for optical + SAR evidence.
- `src/models/gemini.py` — Google Gemini API adapter.
- `src/models/remote_vlm.py` — Hugging Face Transformers adapter for configurable PaliGemma/remote-sensing VLM checkpoints.
- `src/agent.py` — intent classification, input validation, tool selection, execution trace and safety gates.
- `src/evidence.py` — evidence aggregation and grounded Gemini/VLM response generation.
- `src/report.py` — PDF report generation from actual analysis outputs.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/macOS
streamlit run app.py
```

### Required/optional external services

The core raster operations do not require an API key.

For natural-language VLM synthesis:

1. Create a Gemini API key and set `GEMINI_API_KEY` in `.env`.
2. Set `GEMINI_MODEL` to a model available to your account.

For the domain-adapted VLM:

- Set `REMOTE_VLM_MODEL_ID` to the Hugging Face repository/checkpoint you will provide.
- If the checkpoint is gated, also provide `HF_TOKEN`.
- The application will report a clear configuration error rather than silently substituting another model.

For visual grounding:

- Set `YOLO_MODEL_PATH` to a real YOLO checkpoint, or use a supported Ultralytics model name such as `yolo11n.pt` if your environment can download it.
- No detection is claimed when the model cannot be loaded.

## Data requirements

### Optical

GeoTIFFs with valid georeferencing are preferred. NDVI requires red and near-infrared bands. You can provide their 1-based band indexes in the UI.

### SAR

GeoTIFF SAR products are accepted. Calibration is data-product dependent; SatQueryX supports explicit linear-to-dB conversion and validates numeric ranges. Product-specific calibration constants should be supplied through metadata or preprocessing before analysis when required by the sensor/product.

### Change detection

Provide two temporally separated, spatially compatible rasters. SatQueryX reprojects the second raster to the first raster's grid before calculating the change metric.

## Safety / truthfulness

SatQueryX does not generate synthetic detections, fake confidence scores, invented satellite metadata, or placeholder API responses. If an external dependency is missing, the UI exposes the exact reason and the setup required to enable that capability.

## Project status

This repository is the initial implementation scaffold and working application foundation. Model weights and API credentials are intentionally not committed to Git.
