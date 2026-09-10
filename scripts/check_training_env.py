from __future__ import annotations

import os
from pathlib import Path


def check_path(name: str) -> bool:
    value = os.getenv(name, "").strip()
    ok = bool(value) and Path(value).exists()
    print(f"{name}: {'OK' if ok else 'MISSING'}" + (f" -> {value}" if value else ""))
    return ok


def main() -> None:
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA: {torch.version.cuda}")
            print(f"BF16: {torch.cuda.is_bf16_supported()}")
    except Exception as exc:
        print(f"PyTorch check failed: {exc}")

    try:
        import transformers
        import peft
        import bitsandbytes
        print(f"Transformers: {transformers.__version__}")
        print(f"PEFT: {peft.__version__}")
        print(f"bitsandbytes: {bitsandbytes.__version__}")
    except Exception as exc:
        print(f"Training dependency check failed: {exc}")

    required = all(check_path(name) for name in (
        "BIGEARTHNET_IMAGE_LMDB",
        "BIGEARTHNET_METADATA",
        "BIGEARTHNET_LOADER",
    ))
    if not os.getenv("HF_TOKEN"):
        print("HF_TOKEN: not set in this shell. Public dataset metadata may still work, but gated PaliGemma weights require an authenticated Hugging Face session.")
    raise SystemExit(0 if required else 1)


if __name__ == "__main__":
    main()
