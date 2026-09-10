from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class VLMConfig:
    model_id: str
    max_new_tokens: int = 128


class RemoteSensingVLM:
    """Configurable PaliGemma adapter with optional PEFT/QLoRA support.

    ``REMOTE_VLM_MODEL_ID`` may point either to a full Hugging Face model or to
    a PEFT adapter directory/repository. For an adapter, ``RS_VLM_BASE_MODEL_ID``
    identifies the frozen PaliGemma base checkpoint.
    """

    def __init__(self, config: VLMConfig | None = None):
        model_id = (config.model_id if config else None) or os.getenv("REMOTE_VLM_MODEL_ID", "").strip()
        if not model_id:
            raise RuntimeError(
                "REMOTE_VLM_MODEL_ID is not configured. Provide the domain-adapted SatQueryX checkpoint or PEFT adapter."
            )
        self.config = config or VLMConfig(
            model_id=model_id,
            max_new_tokens=int(os.getenv("REMOTE_VLM_MAX_NEW_TOKENS", "256")),
        )
        self.processor = None
        self.model = None

    @staticmethod
    def _is_peft_adapter(model_id: str) -> bool:
        if Path(model_id).is_dir():
            return (Path(model_id) / "adapter_config.json").exists()
        return os.getenv("REMOTE_VLM_IS_PEFT", "0").lower() in {"1", "true", "yes"}

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, BitsAndBytesConfig, PaliGemmaForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Transformers/PyTorch is required for the remote-sensing VLM.") from exc

        token = os.getenv("HF_TOKEN") or None
        adapter = self._is_peft_adapter(self.config.model_id)
        base_id = os.getenv("RS_VLM_BASE_MODEL_ID", "google/paligemma2-3b-pt-448").strip()
        processor_id = base_id if adapter else self.config.model_id
        self.processor = AutoProcessor.from_pretrained(processor_id, token=token)

        if adapter:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("PEFT is required to load the trained SatQueryX adapter.") from exc
            load_kwargs = {
                "token": token,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "device_map": "auto",
            }
            base = PaliGemmaForConditionalGeneration.from_pretrained(base_id, **load_kwargs)
            self.model = PeftModel.from_pretrained(base, self.config.model_id, token=token)
        else:
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

        inputs = self.processor(text=prompt, images=image.convert("RGB"), return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens, do_sample=False)
        input_len = inputs["input_ids"].shape[-1]
        generated = output[0][input_len:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()
