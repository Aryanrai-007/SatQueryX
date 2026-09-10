# SatQueryX

**Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries**

SatQueryX is a no-login Streamlit application implementing the SIH2026 problem scope: single-image remote-sensing VQA plus captioning/grounding, bi-temporal change understanding, co-registered optical/SAR reasoning, and an observable agentic controller. The system reports missing models/services rather than substituting dummy outputs.

## SIH2026 requirement coverage

| Requirement | SatQueryX implementation |
|---|---|
| Single-image VQA | `rs_vqa` specialist + BigEarthNet.txt/RSVQA evaluation |
| Additional single-image task | captioning and text-guided grounding |
| Remote-sensing adaptation | PaliGemma QLoRA training entrypoint using BigEarthNet.txt |
| Bi-temporal change | aligned raster change map + `change_vqa` specialist |
| Optical/SAR pair | optical/SAR fusion + multimodal RS-VLM workflow |
| Agentic orchestration | `src/agent.py` workflow registry/routing |
| Input validation | CRS, resolution, modality, image-count and pair compatibility checks |
| Evidence/confidence/trace | evidence bundle, model provenance, confidence only from real detector scores, execution trace |
| Downloadable report | PDF containing evidence, provenance and metrics |
| Public benchmark evaluation | BigEarthNet.txt bench, VRSBench, RSVQA and CDVQA prediction/evaluation paths |
| Hidden ISRO/SAC readiness | sensor-agnostic GeoTIFF pair contract for Cartosat-2S/RISAT |

The problem statement requires single-image VQA plus captioning or grounding, change understanding from bi-temporal inputs, optical/SAR joint analysis, remote-sensing adaptation, and automatic model/tool orchestration. It also states that generic LLM/VLM use without remote-sensing adaptation is insufficient.

## Dataset roles

### 1. BigEarthNet.txt — adaptation foundation

Official HF dataset: `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt`  
Paper: https://arxiv.org/abs/2603.29630

BigEarthNet.txt provides co-registered Sentinel-1 SAR and Sentinel-2 multispectral imagery with image-text annotations for captioning, VQA and referring/bounding-box tasks. SatQueryX uses it for remote-sensing VLM adaptation and as a multisensor benchmark contract.

The official dataset loader exposes RGB, S2 multispectral and S1+S2 band combinations. Image LMDB and metadata parquet are intentionally kept outside Git.

### 2. VRSBench — single-image captioning/grounding/VQA

Repository: https://github.com/lx709/VRSBench

Used for single-image captioning, visual grounding and VQA evaluation. Its grounding annotations are the target contract for `RS_GROUNDING_MODEL_ID`.

### 3. RSVQA — quantitative single-image VQA

Repository: https://github.com/syvlo/RSVQA

Used to evaluate questions involving object presence, counting, spatial relationships and other remote-sensing VQA tasks.

### 4. CDVQA — bi-temporal change VQA

Repository: https://github.com/YZHJessica/CDVQA

Used for change-question evaluation and temporal reasoning. SatQueryX combines its specialist answer with an independently computed aligned change map when georeferenced rasters are available.

### 5. ISRO/SAC — hidden final evaluation

The problem statement specifies pre-georeferenced/co-registered Cartosat-2S optical and RISAT SAR pairs with task-specific answers, labels, bounding boxes or masks. Those files are not committed and are accepted through the same modality/pair validation layer.

## Architecture

```text
Natural-language query
        |
        v
Agentic orchestrator
(intent + image count + modality + compatibility)
        |
   +----+---------+------------------+
   |              |                  |
   v              v                  v
Single image   Bi-temporal       Optical + SAR
   |              |                  |
RS VQA       Change map +       Fusion + multimodal
Caption      Change VQA         RS-VLM
Grounding        |                  |
   +--------------+------------------+
                  |
                  v
          Evidence synthesizer
                  |
          Answer + visual proof
                  |
          Trace + confidence
                  |
               PDF report
```

## Model adaptation

`training/train_rs_vlm.py` performs real QLoRA adaptation of `google/paligemma-3b-pt-448` using the official BigEarthNet.txt image/text loader. It starts with the RGB Sentinel-2 subset because the base PaliGemma image processor expects a three-channel image. The paired S1/S2 dataset remains the canonical data contract for the optical/SAR workflow and native multimodal checkpoint.

The adapted checkpoint should be uploaded to Hugging Face and configured with:

```bash
REMOTE_VLM_MODEL_ID=<your-adapted-checkpoint>
HF_TOKEN=<read-token-if-gated>
```

For text-guided grounding, configure a checkpoint trained on BigEarthNet.txt/VRSBench grounding annotations:

```bash
RS_GROUNDING_MODEL_ID=<your-grounding-checkpoint>
```

Do not use the generic PaliGemma checkpoint as the final SIH remote-sensing model.

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

For adaptation/evaluation:

```bash
pip install -r requirements-training.txt
```

Copy the official BigEarthNet.txt `ben_txt_datamodule.py` into the training environment and configure:

```bash
BIGEARTHNET_IMAGE_LMDB=/path/to/Encoded-BigEarthNet
BIGEARTHNET_METADATA=/path/to/BigEarthNet.txt.parquet
BIGEARTHNET_LOADER=/path/to/ben_txt_datamodule.py
```

Then:

```bash
python training/train_rs_vlm.py --limit 5000 --output artifacts/satqueryx-rsvlm
```

For the final adaptation run, remove `--limit` and use a suitable GPU environment.

## Benchmark evaluation

Generate JSONL prediction records containing `task`, `prediction` and `reference`, then run:

```bash
python scripts/evaluate_predictions.py predictions.jsonl
```

The evaluator reports exact match and token F1 by task. Dataset-specific official metrics should also be run where the prescribed benchmark code requires them; SatQueryX does not invent benchmark scores.

## Input workflows

### Single image

- GeoTIFF/TIFF for geospatial/spectral analysis.
- PNG/JPEG for approved benchmark visual inputs.
- VQA, captioning and text-guided grounding.

### Bi-temporal

Two spatially corresponding GeoTIFFs are validated, the second is reprojected to the first raster grid, and a real difference/SSIM change map is produced. The adapted RS-VLM receives an explicitly labelled T1/T2 representation for change VQA. A native paired checkpoint can replace that adapter without changing the workflow contract.

### Optical + SAR

Two inputs are checked for modality. Optical and SAR statistics/fusion are computed from actual pixels, and the multimodal RS-VLM receives a labelled optical/SAR joint representation. A native dual-encoder checkpoint can replace this adapter.

### AOI explorer

The Leaflet/OpenStreetMap interface lets users draw an AOI, reverse-geocode it, query Earth Search, fetch Sentinel-2 L2A B02/B03/B04/B08 COGs and clip them to a local georeferenced GeoTIFF. Two dates can be selected for change analysis.

## Truthfulness and provenance

SatQueryX never generates fake detections, fake confidence, invented acquisition metadata, placeholder benchmark scores or synthetic satellite observations. Gemini is used as a language/evidence synthesizer; the SIH task capability is routed through the configured remote-sensing specialist checkpoint.

Model weights, dataset binaries and API credentials are intentionally excluded from Git.
