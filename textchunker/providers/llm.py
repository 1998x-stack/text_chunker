from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ..exceptions import ProviderError


class BaseLLMProvider:
    """Base class for LLM providers."""

    def propose_spans(
        self, text: str, system_prompt: str, max_chars: int
    ) -> List[Tuple[int, int, Optional[str]]]:
        raise NotImplementedError


class HFProvider(BaseLLMProvider):
    """HuggingFace transformers text generation provider."""

    def __init__(self, model: str) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # type: ignore

        logger.info("Loading HF model: {}", model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(model)
        self.pipe = pipeline("text-generation", model=self._model, tokenizer=self.tokenizer)

    def propose_spans(
        self, text: str, system_prompt: str, max_chars: int
    ) -> List[Tuple[int, int, Optional[str]]]:
        prompt = system_prompt.strip() + "\n\n" + text[:max_chars]
        try:
            out = self.pipe(prompt, max_new_tokens=512, do_sample=False)[0]["generated_text"]
        except Exception as e:
            logger.warning("HF inference failed: {}", e)
            return []

        match = re.search(r'\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]', out, re.DOTALL)
        if not match:
            logger.warning("No JSON array found in HF response")
            return []

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from HF response")
            return []

        spans: List[Tuple[int, int, Optional[str]]] = []
        for item in data:
            spans.append((int(item.get("start", 0)), int(item.get("end", 0)), item.get("title")))
        return spans
