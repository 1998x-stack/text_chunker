from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional, Tuple


class BaseLLMProvider:
    """LLM 提供者接口。"""
    def propose_spans(self, text: str, system_prompt: str, max_chars: int) -> List[Tuple[int, int, Optional[str]]]:
        raise NotImplementedError


class OpenAIProvider(BaseLLMProvider):
    """OpenAI 提供者（使用 `openai` SDK v1）。"""
    def __init__(self, model: str) -> None:
        from openai import OpenAI  # 延迟导入
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def propose_spans(self, text: str, system_prompt: str, max_chars: int) -> List[Tuple[int, int, Optional[str]]]:
        prompt = text[:max_chars]
        msg = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请基于如下文本生成 JSON 切分：\n\n{prompt}\n\n"
                                        f"返回格式示例："
                                        f'[{{"title": "节名或null", "start": 0, "end": 100}}, ...]'}
        ]
        resp = self.client.chat.completions.create(model=self.model, messages=msg, temperature=0)
        content = resp.choices[0].message.content or "[]"
        try:
            data = json.loads(content)
        except Exception:
            data = []
        spans: List[Tuple[int, int, Optional[str]]] = []
        for item in data:
            spans.append((int(item.get("start", 0)), int(item.get("end", 0)), item.get("title")))
        return spans


class HFProvider(BaseLLMProvider):
    """HuggingFace transformers 文本生成（需本地/远端权重）。"""
    def __init__(self, model: str) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # type: ignore
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def propose_spans(self, text: str, system_prompt: str, max_chars: int) -> List[Tuple[int, int, Optional[str]]]:
        prompt = system_prompt.strip() + "\n\n" + text[:max_chars]
        out = self.pipe(prompt, max_new_tokens=512, do_sample=False)[0]["generated_text"]
        # 简单抽取 JSON（防御式解析）
        start = out.find("[")
        end = out.rfind("]")
        js = out[start:end + 1] if start != -1 and end != -1 else "[]"
        try:
            data = json.loads(js)
        except Exception:
            data = []
        spans: List[Tuple[int, int, Optional[str]]] = []
        for item in data:
            spans.append((int(item.get("start", 0)), int(item.get("end", 0)), item.get("title")))
        return spans
