from __future__ import annotations

"""Train a remote-sensing-adapted PaliGemma checkpoint on BigEarthNet.txt.

The official BigEarthNet.txt loader is used for the real image/text pairs. The
training path starts with RGB Sentinel-2 because PaliGemma consumes 3-channel
images natively. The same dataset is also used by the project for S1/S2 paired
workflow development and benchmark evaluation.
"""

import argparse
import importlib.util
import os
from pathlib import Path


def _load_official_loader(path: str):
    spec = importlib.util.spec_from_file_location("ben_txt_datamodule", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load BigEarthNet.txt loader from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BENTxTDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("BASE_VLM_MODEL_ID", "google/paligemma-3b-pt-448"))
    parser.add_argument("--lmdb", default=os.getenv("BIGEARTHNET_IMAGE_LMDB", ""))
    parser.add_argument("--metadata", default=os.getenv("BIGEARTHNET_METADATA", ""))
    parser.add_argument("--loader", default=os.getenv("BIGEARTHNET_LOADER", "ben_txt_datamodule.py"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="artifacts/satqueryx-rsvlm")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()

    try:
        import torch
        from PIL import Image
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit("Install requirements-training.txt before model adaptation.") from exc

    if not args.lmdb or not Path(args.lmdb).exists():
        raise SystemExit("BIGEARTHNET_IMAGE_LMDB must point to the official BigEarthNet image LMDB.")
    if not args.metadata or not Path(args.metadata).exists():
        raise SystemExit("BIGEARTHNET_METADATA must point to BigEarthNet.txt.parquet.")
    if not Path(args.loader).exists():
        raise SystemExit("BIGEARTHNET_LOADER must point to the official ben_txt_datamodule.py file.")

    BENTxTDataset = _load_official_loader(args.loader)
    raw = BENTxTDataset(
        lmdb_file=args.lmdb,
        metadata_file=args.metadata,
        bands=("B04", "B03", "B02"),
        img_size=448,
        splits=(args.split,),
    )
    if args.limit:
        raw = torch.utils.data.Subset(raw, range(min(args.limit, len(raw))))

    processor = AutoProcessor.from_pretrained(args.model)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    class DatasetAdapter(torch.utils.data.Dataset):
        def __len__(self):
            return len(raw)

        def __getitem__(self, idx):
            sample = raw[idx]
            arr = sample["image_input"]
            if hasattr(arr, "detach"):
                arr = arr.detach().cpu().numpy()
            arr = arr.astype("float32")
            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = arr.transpose(1, 2, 0)
            arr = arr - arr.min()
            mx = arr.max()
            if mx > 0:
                arr = arr / mx
            image = Image.fromarray((arr * 255).clip(0, 255).astype("uint8"), mode="RGB")
            prompt = str(sample["text_input"])
            reference = str(sample["reference_output"])
            return {"image": image, "prompt": prompt, "reference": reference}

    adapted = DatasetAdapter()

    def collate(batch):
        prompts = [f"remote sensing: {x['prompt']}" for x in batch]
        images = [x["image"] for x in batch]
        answers = [x["reference"] for x in batch]
        full = [f"{p}\n{x}" for p, x in zip(prompts, answers)]
        inputs = processor(text=full, images=images, return_tensors="pt", padding=True)
        labels = inputs["input_ids"].clone()
        pad = processor.tokenizer.pad_token_id
        if pad is not None:
            labels[labels == pad] = -100
        inputs["labels"] = labels
        return inputs

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=adapted, data_collator=collate)
    trainer.train()
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    print(f"Saved adapted remote-sensing checkpoint to {args.output}")


if __name__ == "__main__":
    main()
