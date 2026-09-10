from __future__ import annotations

"""Fine-tune PaliGemma on real BigEarthNet.txt remote-sensing annotations.

The official BigEarthNet.txt PyTorch dataset is used for the image/text pairs.
The first adapter is Sentinel-2 RGB because PaliGemma natively consumes three
image channels. The trained PEFT adapter is then loadable by SatQueryX.

Prerequisites:
  1. Install requirements-training.txt on a CUDA machine.
  2. Download BigEarthNet.txt.parquet and the official ben_txt_datamodule.py.
  3. Convert BigEarthNet v2.0 S1/S2 imagery to the official LMDB format.
  4. Set BIGEARTHNET_IMAGE_LMDB, BIGEARTHNET_METADATA and BIGEARTHNET_LOADER.
"""

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Any


def load_official_dataset(loader_path: str):
    spec = importlib.util.spec_from_file_location("ben_txt_datamodule", loader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import BigEarthNet.txt loader: {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BENTxTDataset


def tensor_to_rgb(image_tensor: Any):
    """Convert real B04/B03/B02 values to an RGB PIL image without inventing data."""
    import numpy as np
    from PIL import Image

    arr = image_tensor.detach().cpu().numpy() if hasattr(image_tensor, "detach") else np.asarray(image_tensor)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W] Sentinel-2 RGB tensor, received {arr.shape}")
    arr = arr.astype("float32")
    channels = []
    for channel in arr:
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            raise ValueError("Sentinel-2 channel contains no finite pixels")
        lo, hi = np.percentile(finite, [2, 98])
        if hi <= lo:
            scaled = np.zeros_like(channel, dtype="float32")
        else:
            scaled = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
        channels.append(scaled)
    rgb = np.stack(channels, axis=-1)
    return Image.fromarray((rgb * 255.0).astype("uint8"), mode="RGB")


class WrappedDataset:
    def __init__(self, raw, torch_module):
        self.raw = raw
        self.torch = torch_module

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index):
        sample = self.raw[index]
        return {
            "image": tensor_to_rgb(sample["image_input"]),
            "prompt": str(sample["text_input"]),
            "answer": str(sample["reference_output"]),
        }


def close_lmdb_dataset(dataset) -> None:
    """Close an official BigEarthNet LMDB environment before opening the same path again.

    py-lmdb 2.x rejects opening the same environment path twice in one process.
    Trainer's training dataset remains referenced after training, so explicit
    validation must close that environment before the validation dataset opens it.
    This helper unwraps torch.utils.data.Subset and WrappedDataset instances and
    closes the official loader's lazily-created image-reader environment.
    """
    current = dataset
    while hasattr(current, "raw"):
        current = current.raw
    if hasattr(current, "dataset") and not hasattr(current, "image_reader"):
        current = current.dataset
    image_reader = getattr(current, "image_reader", None)
    env = getattr(image_reader, "env", None)
    if env is not None:
        env.close()
        image_reader.env = None


def evaluate_loss(model, dataset, collate_fn, torch_module, batch_size: int) -> float:
    """Run explicit multimodal validation outside Trainer's generic eval loop.

    Transformers 5.x Trainer evaluation can re-enter custom LMDB-backed
    multimodal datasets through an incompatible path. The model/processor
    themselves are fully compatible, so validation is performed explicitly
    with the same collator used for training.
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model.eval()
    total_loss = 0.0
    batches = 0

    with torch_module.no_grad():
        for batch in loader:
            batch = {
                key: value.to("cuda") if torch_module.is_tensor(value) else value
                for key, value in batch.items()
            }
            outputs = model(**batch)
            loss = outputs.loss
            if not torch_module.isfinite(loss):
                raise RuntimeError(f"Validation produced a non-finite loss: {loss.item()}")
            total_loss += float(loss.detach().cpu())
            batches += 1

    model.train()
    return total_loss / max(batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="SatQueryX BigEarthNet.txt QLoRA training")
    parser.add_argument("--model", default=os.getenv("BASE_VLM_MODEL_ID", "google/paligemma2-3b-pt-224"))
    parser.add_argument("--lmdb", default=os.getenv("BIGEARTHNET_IMAGE_LMDB", ""))
    parser.add_argument("--metadata", default=os.getenv("BIGEARTHNET_METADATA", ""))
    parser.add_argument("--loader", default=os.getenv("BIGEARTHNET_LOADER", ""))
    parser.add_argument("--output", default="artifacts/satqueryx-rsvlm")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-limit", type=int, default=0, help="0 = full training split")
    parser.add_argument("--val-limit", type=int, default=1000)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512, help="Must retain PaliGemma's 256 image tokens at 224px")
    args = parser.parse_args()

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            PaliGemmaForConditionalGeneration,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit("Install requirements-training.txt before training.") from exc

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for the default QLoRA training path. No CPU fallback is claimed.")
    for name, value in (("BIGEARTHNET_IMAGE_LMDB", args.lmdb), ("BIGEARTHNET_METADATA", args.metadata), ("BIGEARTHNET_LOADER", args.loader)):
        if not value or not Path(value).exists():
            raise SystemExit(f"{name} must point to an existing BigEarthNet asset: {value!r}")

    BENTxTDataset = load_official_dataset(args.loader)
    common = dict(
        lmdb_file=args.lmdb,
        metadata_file=args.metadata,
        bands=("B04", "B03", "B02"),
        img_size=args.image_size,
        upsample_mode="bilinear",
        types=("binary", "mcq", "captioning"),
    )
    train_raw = BENTxTDataset(**common, splits=("train",))
    val_raw = BENTxTDataset(**common, splits=("validation",))
    if args.train_limit:
        train_raw = torch.utils.data.Subset(train_raw, range(min(args.train_limit, len(train_raw))))
    if args.val_limit:
        val_raw = torch.utils.data.Subset(val_raw, range(min(args.val_limit, len(val_raw))))

    train_ds = WrappedDataset(train_raw, torch)
    val_ds = WrappedDataset(val_raw, torch)
    processor = AutoProcessor.from_pretrained(args.model, token=os.getenv("HF_TOKEN") or None)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        token=os.getenv("HF_TOKEN") or None,
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    # Transformers 5.x nests the PaliGemma vision backbone under `model`.
    # Keep the vision tower frozen for memory efficiency; LoRA adapts the
    # language-side remote-sensing reasoning while the multimodal projector
    # remains part of the base model.
    for parameter in model.model.vision_tower.parameters():
        parameter.requires_grad = False

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def collate(batch):
        prompts = [f"remote sensing: {item['prompt']}" for item in batch]
        images = [item["image"] for item in batch]
        answers = [item["answer"] for item in batch]
        encoded = processor(
            text=prompts,
            images=images,
            suffix=answers,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=args.max_length,
        )
        return encoded

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=0.03,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        dataloader_pin_memory=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collate,
    )
    print(f"Training samples: {len(train_ds)} | validation samples: {len(val_ds)}")
    trainer.train()

    # The training dataset has lazily opened the shared LMDB environment.
    # py-lmdb 2.x intentionally rejects a second open of the same path in the
    # same process, so close the training environment before validation opens it.
    close_lmdb_dataset(train_ds)

    print("Running explicit validation...")
    val_loss = evaluate_loss(
        model=model,
        dataset=val_ds,
        collate_fn=collate,
        torch_module=torch,
        batch_size=args.batch_size,
    )
    print(f"Validation loss: {val_loss:.6f}")

    trainer.save_model(str(output))
    processor.save_pretrained(str(output))
    print(f"Saved SatQueryX PEFT adapter to {output.resolve()}")


if __name__ == "__main__":
    main()
