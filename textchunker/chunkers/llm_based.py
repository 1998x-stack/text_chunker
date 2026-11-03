"""LLM-proposed boundaries with finite model-call budgets and safe fallback."""
from __future__ import annotations

from typing import List

from loguru import logger

from ..core.validation import validate_parameters
from ..exceptions import ConfigError, ProviderError
from ..providers import DashScopeProvider, HFProvider
from ..registry import register
from ..types import Chunk
from .base import BaseChunker
from .recursive import _split_by_separators, add_bounded_overlap


def _proposals_are_complete(proposals, length: int) -> bool:
    if not proposals:
        return False
    cursor = 0
    for item in proposals:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            return False
        start, end, title = item
        if (type(start) is not int or type(end) is not int
                or (title is not None and not isinstance(title, str))
                or start != cursor or end <= start or end > length):
            return False
        cursor = end
    return cursor == length


@register("llm")
class LLMChunker(BaseChunker):
    """The provider suggests; the trusted splitter owns all source spans."""

    def __init__(self, cfg, llm_provider=None):
        super().__init__(cfg)
        if llm_provider is not None:
            self.provider = llm_provider
            return
        provider = cfg.llm.get("provider", "dashscope").lower()
        if provider in ("dashscope", "openai"):
            model = cfg.llm.get("llm_model", cfg.llm.get("openai_model", "qwen-max"))
            self.provider = DashScopeProvider(
                model=model, api_key=cfg.llm.get("api_key", ""),
                api_base_url=cfg.llm.get("api_base_url", ""),
                timeout_seconds=float(cfg.llm.get("timeout_seconds", 30)),
            )
        elif provider == "hf":
            self.provider = HFProvider(
                cfg.llm.get("hf_model", "Qwen/Qwen2.5-7B-Instruct")
            )
        else:
            raise ConfigError(f"Unknown provider: {provider}")

    def chunk(self, text: str) -> List[Chunk]:
        if not text:
            return []
        common, options = self.cfg.common, self.cfg.llm
        size = int(common.get("chunk_size", 512))
        overlap = int(common.get("chunk_overlap", 80))
        max_chunks = common.get("max_chunks")
        validate_parameters(size, overlap, max_chunks)
        max_chars = int(options.get("max_chars_per_call", 8000))
        max_calls = int(options.get("max_provider_calls", 16))
        max_prompt_chars = int(options.get("max_total_prompt_chars", 65536))
        attempts = int(options.get("retry_attempts", 1))
        on_failure = options.get("on_failure", "fallback")
        if (max_chars <= 0 or max_calls < 0 or max_prompt_chars < 0
                or attempts < 1 or attempts > 4
                or on_failure not in ("fallback", "error")):
            raise ConfigError("Invalid LLM call budget/retry/failure policy")
        system_prompt = options.get(
            "system_prompt", "Return JSON spans of the complete input."
        )

        chunks: List[Chunk] = []
        calls, charged_chars = 0, 0
        for base in range(0, len(text), max_chars):
            window = text[base:base + max_chars]
            proposals = []
            error_reason = "budget_exhausted"
            # Repeated system instructions also consume model input space.
            call_chars = len(window) + len(system_prompt)
            if calls < max_calls and charged_chars + call_chars <= max_prompt_chars:
                for _ in range(attempts):
                    if calls >= max_calls or (
                            charged_chars + call_chars > max_prompt_chars):
                        break
                    calls += 1
                    charged_chars += call_chars
                    try:
                        proposals = self.provider.propose_spans(
                            window, system_prompt=system_prompt,
                            max_chars=max_chars,
                        )
                    except Exception as exc:
                        logger.warning(
                            "LLM segmentation failed: {}", type(exc).__name__
                        )
                        proposals = []
                    if _proposals_are_complete(proposals, len(window)):
                        break
                    error_reason = "invalid_or_failed_proposal"

            valid = _proposals_are_complete(proposals, len(window))
            if not valid:
                if on_failure == "error":
                    raise ProviderError(
                        f"LLM span proposal unavailable at input offset {base}: "
                        f"{error_reason}"
                    )
                proposals = [(0, len(window), None)]

            for begin, end, title in proposals:
                candidate = window[begin:end]
                parts = (_split_by_separators(
                    candidate, ["\n\n", "\n", " ", ""], size, self.count_tokens
                ) if self.count_tokens(candidate) > size else [candidate])
                cursor = base + begin
                for part in parts:
                    chunks.append(Chunk(
                        id=len(chunks), text=part, start=cursor,
                        end=cursor + len(part),
                        meta={
                            "strategy": "llm", "title": title,
                            "fallback": not valid,
                            "fallback_reason": error_reason if not valid else None,
                            "model_calls": calls,
                        },
                    ))
                    cursor += len(part)
                    if max_chunks is not None and len(chunks) >= max_chunks:
                        break
                if max_chunks is not None and len(chunks) >= max_chunks:
                    break
            if max_chunks is not None and len(chunks) >= max_chunks:
                break

        add_bounded_overlap(
            text, chunks, size, overlap, count=self.count_tokens
        )
        logger.info("LLM segmentation: {} chunks, {} provider calls",
                    len(chunks), calls)
        return chunks
