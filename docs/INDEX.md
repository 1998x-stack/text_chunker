# Documentation index / 文档导航

| Document | Scope | Status |
|---|---|---|
| [README](../README.md) | Installation, quickstart, API | Current |
| [README 中文](README.zh-CN.md) | 中文使用入口 | Current |
| [Architecture V0.5–V0.6](architecture-v0.5-v0.6.md) | Model reliability and audited offline retrieval | Current |
| [Evaluation methodology](evaluation.md) | Goldens, metrics, evidence and limitations | Current |
| [Architecture V0.4](architecture-v0.4.md) | Markdown/HTML structure-aware Hybrid Chunking and overflow policies | Current |
| [Architecture V0.3](architecture-v0.3.md) | DocumentIR, spans, provenance, strict evaluator | Previous baseline |
| [Usage](usage.md) | CLI flags, examples, tokenizer and format behavior | Current |
| [Development](development.md) | Repository ownership, tests, CI and contribution protocol | Current |
| [Architecture V0.2.1](architecture-v0.2.1.md) | Historical correctness baseline | Historical |
| [Strategy guide](strategy-guide.md) | External strategies and research links | Research notes; not implementation promises |
| [superpowers/](superpowers/) | Prior planning and design exploration | Historical |
| [Change log](../CHANGELOG.md) | Version-level factual changes | Current |

## Source of truth

- **What is implemented:** production Python code in `textchunker/`, verified by tests.
- **How to run:** `README.md` and `docs/usage.md`.
- **What is guaranteed:** explicit contracts in `docs/architecture-v0.3.md`, `docs/architecture-v0.4.md` and the actual corresponding tests. Offline evaluation and its limits are documented separately.
- **What is only proposed:** historical plans and the roadmap; never infer that a directory or future feature is shipped because it appears in a diagram.

Architecture layering: parser → canonical DocumentIR → chunker (strategy) → independent evaluator → exporter.
