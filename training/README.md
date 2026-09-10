# SatQueryX remote-sensing VLM training

This directory contains the real BigEarthNet.txt adaptation path. No synthetic images or fabricated metrics are used.

## 1. Install training dependencies

Use a CUDA-enabled Linux environment with enough GPU memory for PaliGemma 2 3B QLoRA:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-training.txt
```

## 2. Prepare BigEarthNet.txt metadata

```bash
python scripts/bootstrap_bigearthnet.py
```

The helper downloads the official `BigEarthNet.txt.parquet` and `ben_txt_datamodule.py`. The actual Sentinel-1/Sentinel-2 imagery is intentionally not stored in Git or downloaded by the helper.

BigEarthNet.txt's official loading path uses BigEarthNet v2.0 imagery converted to an LMDB with `rico-hdl`.

```bash
rico-hdl bigearthnet \
  --bigearthnet-s1-dir <S1_ROOT_DIR> \
  --bigearthnet-s2-dir <S2_ROOT_DIR> \
  --target-dir <LMDB_DIR>
```

Set these in `.env` or export them in the shell:

```text
HF_TOKEN=<only if your HF environment requires authentication>
BIGEARTHNET_METADATA=/absolute/path/to/BigEarthNet.txt.parquet
BIGEARTHNET_LOADER=/absolute/path/to/ben_txt_datamodule.py
BIGEARTHNET_IMAGE_LMDB=/absolute/path/to/Encoded-BigEarthNet
```

## 3. Run a real smoke-training job first

The smoke run still uses real BigEarthNet images and annotations; it only limits the number of examples.

```bash
python training/train_rs_vlm.py \
  --train-limit 128 \
  --val-limit 32 \
  --epochs 1 \
  --output artifacts/satqueryx-rsvlm-smoke
```

Check that the adapter is saved and that validation loss is finite. This is a pipeline test, not a benchmark result.

## 4. Run the actual adaptation

```bash
python training/train_rs_vlm.py \
  --train-limit 0 \
  --val-limit 1000 \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 8 \
  --output artifacts/satqueryx-rsvlm
```

The training script uses real Sentinel-2 B04/B03/B02 imagery and real BigEarthNet.txt binary VQA, multiple-choice VQA and captioning annotations. The model is PaliGemma 2 3B with a QLoRA adapter.

## 5. Connect the adapter to SatQueryX

For a local adapter directory:

```text
REMOTE_VLM_MODEL_ID=/absolute/path/to/artifacts/satqueryx-rsvlm
REMOTE_VLM_IS_PEFT=1
RS_VLM_BASE_MODEL_ID=google/paligemma2-3b-pt-224
```

For a Hugging Face-hosted adapter, use its repository ID and keep `HF_TOKEN` available when required.

## Important

The first adaptation is RGB Sentinel-2 because PaliGemma's native vision input is three-channel. BigEarthNet.txt remains the project's source for optical/SAR paired data, grounding annotations and benchmark evaluation. Native dual-image optical/SAR fusion and dedicated grounding/change adapters are separate training stages and must not be claimed as trained until their checkpoints exist.
