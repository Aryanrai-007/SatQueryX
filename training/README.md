# SatQueryX model adaptation

The SIH problem requires at least one remote-sensing visual/vision-language component to be fine-tuned or adapted. SatQueryX uses **BigEarthNet.txt** as the primary adaptation source and keeps model weights outside Git.

## 1. Prepare BigEarthNet.txt

Use the official dataset and its supplied `ben_txt_datamodule.py` loader. The dataset exposes co-registered Sentinel-1/Sentinel-2 imagery plus `input`/`output` annotations for captioning, VQA and referring/bounding-box tasks.

Required environment variables:

```bash
export BIGEARTHNET_IMAGE_LMDB=/data/Encoded-BigEarthNet
export BIGEARTHNET_METADATA=/data/BigEarthNet.txt.parquet
```

The repository does not commit the image LMDB or parquet because the image store is large.

## 2. Adapt PaliGemma

```bash
pip install -r requirements-training.txt
python training/train_rs_vlm.py --model google/paligemma-3b-pt-448 --split train --limit 5000 --output artifacts/satqueryx-rsvlm
```

For a real submission checkpoint, remove `--limit` and train on a GPU with validation/benchmark evaluation. The resulting adapter/checkpoint must be uploaded to a private or public Hugging Face repository and configured with `REMOTE_VLM_MODEL_ID`.

## 3. Evaluation datasets

- VRSBench: captioning, grounding and VQA.
- RSVQA: single-image VQA.
- CDVQA: bi-temporal change VQA.
- BigEarthNet.txt `bench` split: multisensor VQA/captioning/grounding benchmark.

Prediction files should be JSONL records with at least:

```json
{"task":"vqa","prediction":"yes","reference":"yes"}
```

Evaluate them with:

```bash
python scripts/evaluate_predictions.py predictions.jsonl
```

## 4. Optical/SAR adaptation

BigEarthNet.txt provides a `S1S2-10m20m` band configuration. SatQueryX uses this as the canonical paired-modality data contract. The application can already route a co-registered optical/SAR pair to the multimodal workflow; a native dual-encoder checkpoint can be substituted through `RS_MULTIMODAL_MODEL_ID` without changing the agent contract.

## 5. Hidden ISRO/SAC evaluation

Do not hard-code the hidden data. The input validator accepts georeferenced optical and SAR GeoTIFFs and the workflow registry is sensor-agnostic, so Cartosat-2S/RISAT pairs can enter the same cross-modal workflow.
