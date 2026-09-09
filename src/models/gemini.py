from __future__ import annotations

import os
from typing import Iterable

from PIL import Image


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self._client = None
        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError as exc:
                raise RuntimeError("google-genai is not installed. Install requirements.txt.") from exc

    @property
    def configured(self) -> bool:
        return self._client is not None

    def generate(self, prompt: str, images: Iterable[Image.Image] = ()) -> str:
        if not self.configured:
            raise RuntimeError("GEMINI_API_KEY is not configured. Add it to .env to enable LLM synthesis.")
        contents = [prompt]
        contents.extend(list(images))
        response = self._client.models.generate_content(model=self.model_name, contents=contents)
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini returned no text response.")
        return text.strip()
