# Documentation index / 文档导航

| Document | Scope | Status |
|---|---|---|
| [README](../README.md) | Installation, quickstart, API | Current |
| [README 中文](README.zh-CN.md) | 中文使用入口 | Current |
| [Architecture V0.3](architecture-v0.3.md) | DocumentIR, spans, provenance, strict evaluator | Current |
| [Usage](usage.md) | CLI flags, examples, tokenizer and format behavior | Current |
| [Development](development.md) | Repository ownership, tests, CI and contribution protocol | Current |
| [Architecture V0.2.1](architecture-v0.2.1.md) | Historical correctness baseline | Historical |
| [Strategy guide](strategy-guide.md) | External strategies and research links | Research notes; not implementation promises |
| [superpowers/](superpowers/) | Prior planning and design exploration | Historical |
| [Change log](../CHANGELOG.md) | Version-level factual changes | Current |

## Source of truth

- **What is implemented:** production Python code in `textchunker/`, verified by tests.
- **How to run:** `README.md` and `docs/usage.md`.
- **What is guaranteed:** `docs/architecture-v0.3.md`.
- **What is only proposed:** historical plans and the roadmap; never infer that a directory or future feature is shipped because it appears in a diagram.

Architecture layering: parser → canonical DocumentIR → chunker (strategy) → independent evaluator → exporter.
