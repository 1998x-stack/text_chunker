"""Hugging Face tokenizer, loaded only when explicitly selected."""
from __future__ import annotations


class HFTokenizer:
    def __init__(self, model_name: str) -> None:
        if not model_name:
            raise ValueError("HF model name must not be empty")
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Install optional transformer extras to use a HF tokenizer"
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.name = f"hf:{model_name}"

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))
