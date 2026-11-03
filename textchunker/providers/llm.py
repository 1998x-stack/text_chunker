"""Protocol-compatible local and optional LLM inference backends."""
from __future__ import annotations

from typing import List, Optional, Tuple

from loguru import logger

from .response_parser import parse_span_response


class BaseLLMProvider:
    def propose_spans(
        self, text: str, system_prompt: str, max_chars: int,
    ) -> List[Tuple[int, int, Optional[str]]]:
        raise NotImplementedError


class HFProvider(BaseLLMProvider):
    """Optional local Hugging Face model; generation disabled until selected."""

    def __init__(self, model: str) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        logger.info("Loading HF model: {}", model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(model)
        self.pipe = pipeline(
            "text-generation", model=self._model, tokenizer=self.tokenizer
        )

    def propose_spans(
        self, text: str, system_prompt: str, max_chars: int,
    ) -> List[Tuple[int, int, Optional[str]]]:
        if len(text) > max_chars:
            raise ValueError("Window exceeds provider max_chars")
        prompt = system_prompt.strip() + "\n\n" + text
        try:
            result = self.pipe(prompt, max_new_tokens=512, do_sample=False)
            content = result[0]["generated_text"]
        except Exception as exc:
            logger.warning("HF model failure: {}", type(exc).__name__)
            return []
        return parse_span_response(content)
