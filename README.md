# SatQueryX

**Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries**

SatQueryX is a no-login Streamlit application implementing the SIH2026 problem scope: single-image remote-sensing VQA plus captioning/grounding, bi-temporal change understanding, co-registered optical/SAR reasoning, and an observable agentic controller. The system reports missing models/services rather than substituting dummy outputs.

## SIH2026 requirement coverage

| Requirement | SatQueryX implementation |
|---|---|
| Single-image VQA | `rs_vqa` specialist + BigEarthNet.txt/RSVQA evaluation |
| Additional single-image task | captioning and text-guided grounding |
| Remote-sensing adaptation | executable PaliGemma QLoRA training on real BigEarthNet.txt image/text pairs |
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

`training/train_rs_vlm.py` performs real QLoRA adaptation of `google/paligemma2-3b-pt-224` using the official BigEarthNet.txt image/text loader. It starts with the RGB Sentinel-2 subset because the base PaliGemma image processor expects a three-channel image. The paired S1/S2 dataset remains the canonical data contract for the optical/SAR workflow and native multimodal checkpoint.

### Training setup

See `training/README.md` for the complete data preparation and training sequence. Run `scripts/check_training_env.py` before training.
