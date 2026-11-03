"""Atomic JSON report writing for inspectable offline evaluations."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def write_report(path: str | Path, report: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(report) if is_dataclass(report) else report
    serialized = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent,
            prefix=".chunker-", suffix=".tmp", delete=False,
        ) as stream:
            temp = stream.name
            stream.write(serialized)
        os.replace(temp, target)
    finally:
        if temp is not None and os.path.exists(temp):
            os.unlink(temp)
