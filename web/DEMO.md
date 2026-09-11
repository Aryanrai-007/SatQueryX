# SatQueryX — SIH Internal Demo Runbook

## Before the room

1. Start the web console.
2. Put either an OpenRouter or Gemini key in `web/.env.local`.
3. Keep one known-good satellite JPG/PNG ready.
4. Keep a second image ready for the change or optical/SAR scene.
5. Open the console at a laptop-friendly width.

## 5-minute flow

### 0:00 — Problem

"Satellite imagery contains rich spatial and spectral information, but a user normally has to understand multiple specialist workflows. SatQueryX turns that into a natural-language interface with an agentic controller in the middle."

### 0:30 — Single image VQA

Upload the primary image.

Query:

> What major land-cover characteristics are visible in this scene?

Run analysis. Keep **Agent trace** visible while the steps animate, then switch to **Evidence**.

Say:

> "The important part is not just that a model answers. The controller validates the input, interprets intent, selects a specialist workflow, assembles evidence and records the execution trace."

### 1:40 — Bi-temporal change

Upload a second image.

Query:

> Compare these two observations and describe the major visible changes.

The prototype computes a deterministic RGB difference statistic when both images can be decoded in the browser. The production path uses the existing Python GeoTIFF/SSIM/change pipeline.

Show:
- changed fraction
- mean difference
- model explanation
- trace

### 2:45 — Optical + SAR

Mark one input `OPTICAL` and the second `SAR`.

Query:

> What complementary information does the SAR observation provide?

Say:

> "Optical and SAR are complementary evidence sources. Optical emphasizes spectral response, while SAR contributes radar backscatter and structural information and is less dependent on visible-light conditions."

### 3:35 — AOI / map

Click the AOI map to move the center.

Say:

> "The production acquisition path starts here: AOI, STAC catalog search, cloud filtering, COG or GeoTIFF retrieval, then agent orchestration."

Do not claim that the web prototype is already performing full STAC acquisition; the existing Python application contains that path.

### 4:15 — Architecture

Show the right-side architecture panel:

`Natural language → Agent controller → Specialist tools/models → Evidence → Language synthesis → Audit trace`

Say:

> "The general multimodal API is deliberately only the language layer in this prototype. After selection, we replace it with the remote-sensing VLM trained on the prescribed open training data, dedicated grounding/change models, benchmark evaluation, and finally validation on the hidden ISRO/SAC data."

### 4:45 — Close

> "So the internal prototype proves the interaction and orchestration contract. The external-round build focuses the same contract on research-grade remote-sensing specialists, quantitative benchmark evaluation and the hidden test set."

## Judge vocabulary

- **Agentic orchestration:** the controller chooses the workflow rather than hard-coding one model for every request.
- **Multimodal VLM:** a model that reasons over visual inputs and natural-language instructions.
- **Cross-modal fusion:** combining complementary evidence from different sensor modalities.
- **Bi-temporal change detection:** comparing observations of the same area across dates.
- **Visual grounding:** linking a language request to a spatial image region.
- **Evidence-grounded generation:** generating language from explicit visual/measurement evidence rather than unsupported claims.
- **Auditable trace:** recording selected task, model/provider, tools, parameters, intermediate evidence and final output.
- **COG:** Cloud Optimized GeoTIFF, useful for efficient geospatial raster access.
- **STAC:** SpatioTemporal Asset Catalog, used to discover geospatial scenes by space and time.
- **SAR:** Synthetic Aperture Radar, a radar sensing modality that provides complementary structural/backscatter information.

## Never claim tomorrow

- Do not claim the general OpenRouter/Gemini model is a fine-tuned remote-sensing specialist.
- Do not claim benchmark scores that have not been measured on the prescribed test subset.
- Do not claim hidden ISRO/SAC validation has happened.
- Do not present prototype evidence coverage as calibrated model confidence.
- Do not claim browser RGB comparison is equivalent to production GeoTIFF change detection.
