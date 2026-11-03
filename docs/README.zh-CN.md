# Text-Chunker 中文说明

本仓库提供适用于 RAG 数据预处理的可追溯文本分块：固定窗口、递归、标题结构、语义和 LLM 五种策略。V0.3 已支持 `DocumentIR`、来源映射、显式 Tokenizer、独立的完整性评估和稳定 Chunk ID。

**边界说明：** Markdown 当前为保留原文的逐行结构解析，不是完整 Markdown AST；HTML/PDF 的提取文本保留来源 URI，PDF 可保留页码，但不能将字符位置精确映射到原始 DOM/PDF 坐标。不要把这些能力误解为 V0.4 已完成。

## 安装

需要 Python 3.10 及以上：

```bash
pip install -e ".[token,structure,dev]"
```

## 命令行

```bash
python -m textchunker.cli --input sample.txt --output out/chunks.jsonl \
  --strategy recursive --chunk-size 256 --chunk-overlap 32 --tokenizer char
```

可选 `--tokenizer tiktoken:cl100k_base`，也可通过 `--config` 指定 YAML。输出 JSONL 包含旧字段（`text/start/end/meta/source`）和新增的 `chunk_id`、`document_id`、`token_count`、`source_locations`、`complete` 等。

## Python 接口

```python
from textchunker.parsers import parse_text
from textchunker.pipeline import run_chunking
from textchunker.tokenizers import CharacterTokenizer
from textchunker.types import StrategyConfig

doc = parse_text("第一段。\n\n第二段。", "demo.txt")
cfg = StrategyConfig("recursive", common={"chunk_size": 12, "chunk_overlap": 2})
result = run_chunking(doc, cfg, CharacterTokenizer())
assert result.report.ok and result.complete
```

- 完整性用真实字符区间并集计算，不会把重叠算成额外原文覆盖。
- `max_chunks` 的截断结果会被标记为 `complete=False`。
- 原文偏移是 Python 字符索引，不是 UTF-8 字节数。
- 语义 / LLM 依赖和外部模型服务可能引入额外资源或成本，不会默认启用。
- V0.4 目标：结构感知 Hybrid Chunker，优先保护代码块、标题、表格等元素；优先保证可追溯、无数据丢失和分块预算约束。

更多内容：[文档目录](INDEX.md) · [详细用法](usage.md) · [V0.3 架构](architecture-v0.3.md)。
