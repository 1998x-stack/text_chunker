"""Strict, bounded JSON parsing for untrusted LLM-proposed source spans."""
from __future__ import annotations

import json
from typing import Optional


def parse_span_response(
    content: str, *, max_response_chars: int = 65536,
    max_spans: int = 2048,
) -> list[tuple[int, int, Optional[str]]]:
    """Parse a JSON list from plain text or a fenced/wrapped response.

    No coercion from floats, booleans or numeric strings; no regex backtracking,
    arbitrary Python eval, or silent default offset values.
    """
    if (not isinstance(content, str)
            or len(content) > max_response_chars or max_response_chars <= 0
            or max_spans <= 0):
        return []
    decoder = json.JSONDecoder()
    for pos, char in enumerate(content):
        if char != "[":
            continue
        try:
            data, _ = decoder.raw_decode(content[pos:])
        except (ValueError, TypeError):
            continue
        if not isinstance(data, list) or len(data) > max_spans:
            continue
        rows = []
        valid = True
        for item in data:
            if not isinstance(item, dict):
                valid = False
                break
            start, end, title = (
                item.get("start"), item.get("end"), item.get("title")
            )
            if (type(start) is not int or type(end) is not int
                    or start < 0 or end <= start
                    or (title is not None and not isinstance(title, str))):
                valid = False
                break
            rows.append((start, end, title))
        if valid:
            return rows
    return []
