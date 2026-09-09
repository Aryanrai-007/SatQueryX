from __future__ import annotations

import os
from dataclasses import dataclass

from PIL import Image


@dataclass
class VLMConfig:
    model_id: str
    max_new_tokens: int = 128


class RemoteSensingVLM:
    """Configurable Hugging Face vision-language model adapter.

    The checkpoint is intentionally supplied by the project owner rather than
    silently downloading an unrelated model. This makes the architecture ready
    for the domain-adapted PaliGemma/QLoRA checkpoint.
    """

    def __init__(self, config: VLMConfig | None = None):
        model_id = (config.model_id if config else None) or os.getenv("REMOTE_VLM_MODEL_ID", "").strip()
        if not model_id:
            raise RuntimeError(
                "REMOTE_VLM_MODEL_ID is not configured. Provide the exact Hugging Face checkpoint "
                "for the domain-adapted remote-sensing VLM."
            )
        self.config = config or VLMConfig(
            model_id=model_id,
            max_new_tokens=int(os.getenv("REMOTE_VLM_MAX_NEW_TOKENS", "128")),
        )
        self.processor = None
        self.model = None

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Transformers/PyTorch is required for the remote-sensing VLM.") from exc

        token = os.getenv("HF_TOKEN") or None
        self.processor = AutoProcessor.from_pretrained(self.config.model_id, token=token)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.config.model_id,
            token=token,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.model.eval()

    def generate(self, image: Image.Image, prompt: str) -> str:
        if self.model is None or self.processor is None:
            self.load()
        import torch

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
        decoded = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        return decoded.strip()
