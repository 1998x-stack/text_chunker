"""DashScope/OpenAI-compatible provider with bounded response parsing."""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

from loguru import logger

from .llm import BaseLLMProvider
from .response_parser import parse_span_response


class DashScopeProvider(BaseLLMProvider):
    def __init__(
        self, model: str = "qwen-max", api_key: str = "",
        api_base_url: str = "", timeout_seconds: float = 30.0,
    ) -> None:
        from openai import OpenAI

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.model = model
        key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        base = api_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if not key:
            logger.warning("DASHSCOPE_API_KEY missing; provider calls may fail")
        # Retrying is owned by LLMChunker, so SDK retry defaults are disabled.
        self.client = OpenAI(
            api_key=key or "missing", base_url=base, timeout=timeout_seconds,
            max_retries=0,
        )

    def propose_spans(
        self, text: str, system_prompt: str, max_chars: int
    ) -> List[Tuple[int, int, Optional[str]]]:
        if len(text) > max_chars:
            raise ValueError("Window exceeds provider max_chars")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                "Return a JSON list with title, start and end character offsets "
                "for consecutive spans of this entire text:\n\n" + text
            )},
        ]
        # Do not log user text, secrets or provider response contents.
        try:
            result = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0,
            )
            content = result.choices[0].message.content or "[]"
        except Exception as exc:
            logger.warning("LLM provider failure: {}", type(exc).__name__)
            return []
        return self._parse_spans(content)

    @staticmethod
    def _parse_spans(content: str) -> List[Tuple[int, int, Optional[str]]]:
        rows = parse_span_response(content)
        if not rows:
            logger.warning("LLM response contained no valid span list")
        return rows
