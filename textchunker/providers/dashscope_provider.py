from __future__ import annotations

import json
import os
import re
from typing import List, Optional, Tuple

from loguru import logger

from ..exceptions import ProviderError
from .llm import BaseLLMProvider


class DashScopeProvider(BaseLLMProvider):
    """DashScope LLM provider using OpenAI-compatible SDK."""

    def __init__(self, model: str = "qwen-max", api_key: str = "", api_base_url: str = "") -> None:
        from openai import OpenAI

        self.model = model
        key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        base = api_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"

        if not key:
            logger.warning("DASHSCOPE_API_KEY not set — DashScope provider will fail on API calls")

        self.client = OpenAI(api_key=key, base_url=base)

    def propose_spans(
        self, text: str, system_prompt: str, max_chars: int
    ) -> List[Tuple[int, int, Optional[str]]]:
        prompt = text[:max_chars]
        msg = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"请基于如下文本生成 JSON 切分：\n\n{prompt}\n\n"
                f'返回格式示例：[{{"title": "节名或null", "start": 0, "end": 100}}, ...]'
            )},
        ]
        try:
            logger.debug("DashScope API call: model={}, prompt_len={}", self.model, len(prompt))
            resp = self.client.chat.completions.create(
                model=self.model, messages=msg, temperature=0,
            )
            content = resp.choices[0].message.content or "[]"
        except Exception as e:
            logger.warning("DashScope API call failed: {}", e)
            return []

        return self._parse_spans(content)

    @staticmethod
    def _parse_spans(content: str) -> List[Tuple[int, int, Optional[str]]]:
        """Parse JSON spans from LLM response with robust extraction."""
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return [(int(d.get("start", 0)), int(d.get("end", 0)), d.get("title")) for d in data]
        except (json.JSONDecodeError, TypeError):
            pass

        match = re.search(r'\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]', content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return [(int(d.get("start", 0)), int(d.get("end", 0)), d.get("title")) for d in data]
            except (json.JSONDecodeError, TypeError):
                pass

        logger.warning("Failed to parse spans from LLM response: {}", content[:200])
        return []
