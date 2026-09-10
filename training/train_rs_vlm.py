from __future__ import annotations

"""QLoRA entrypoint for adapting PaliGemma on BigEarthNet.txt.

This script intentionally requires real BigEarthNet.txt image storage and does
not manufacture images. The official dataset exposes `input`, `output`, `type`,
`category`, and Sentinel-1/Sentinel-2 patch identifiers. A local image resolver
must map each row to a real paired image before training.
"""

import argparse
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.getenv("BASE_VLM_MODEL_ID", "google/paligemma-3b-pt-448"))
    p.add_argument("--dataset", default="BIFOLD-BigEarthNetv2-0/BigEarthNet.txt")
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", default="artifacts/satqueryx-rsvlm")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    args = p.parse_args()

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, TrainingArguments
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install the training extras first: "
            "pip install -r requirements-training.txt"
        ) from exc

    lmdb_path = os.getenv("BIGEARTHNET_IMAGE_LMDB", "").strip()
    if not lmdb_path or not Path(lmdb_path).exists():
        raise SystemExit(
            "BIGEARTHNET_IMAGE_LMDB must point to the downloaded BigEarthNet image store. "
            "The metadata parquet alone is not sufficient for image-text fine-tuning."
        )

    ds = load_dataset(args.dataset, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    # Import the official image reader only after verifying that the image store exists.
    try:
        from ben_txt_datamodule import BENTxTDataset  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Copy/import the official BigEarthNet.txt `ben_txt_datamodule.py` into the training environment "
            "or expose it on PYTHONPATH. This avoids silently guessing the binary image encoding."
        ) from exc

    processor = AutoProcessor.from_pretrained(args.model)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # The official BigEarthNet.txt loader handles S1/S2 band normalization and
    # spatial alignment. We use its datamodule as the source of truth rather than
    # inventing a new binary image decoder here.
    dm = BENTxTDataset  # retained as an explicit dependency check/documentation
    _ = dm

    raise SystemExit(
        "Environment validated and PaliGemma+QLoRA initialized. Wire the official "
        "BENTxTDataModule batch collation into Trainer for the exact image schema; "
        "no fake training loop is provided. See training/README.md."
    )


if __name__ == "__main__":
    main()
