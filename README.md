
# 使用方法

## 安装

```bash
git clone <your-repo> text-chunker
cd text-chunker
pip install -e .
# 或按需安装可选功能：
# pip install -e .[semantic,token,structure,llm]
```

## 快速试跑

```bash
# 递归分块 + 可视化
python -m textchunker.cli --config configs/default.yaml --input ./docs/sample.md --visualize

# 固定窗口（覆盖 YAML）
python -m textchunker.cli --config configs/default.yaml --strategy fixed --input sample.txt --visualize

# 语义分块（需要 sentence-transformers/torch）
python -m textchunker.cli --config configs/semantic.yaml --input long_article.txt --visualize

# 基于文档结构（Markdown 标题优先）
python -m textchunker.cli --strategy structure --input ./docs/handbook.md --visualize

# 基于 LLM（OpenAI）：需设置 OPENAI_API_KEY
export OPENAI_API_KEY=sk-...
python -m textchunker.cli --config configs/llm_based.yaml --strategy llm --provider openai --input report.md --visualize

# 基于 LLM（HuggingFace 本地/远端模型）
python -m textchunker.cli --strategy llm --provider hf --hf-model Qwen/Qwen2.5-7B-Instruct --input report.md --visualize
```

---

# 设计要点与扩展说明

* **Factory & Registry**：`@register("name")` 装饰器把新策略注册到全局 `_REGISTRY`，`create_chunker` 通过 `name` 实例化。新增策略时仅需在 `chunkers/` 下新建文件并 `@register` 即可热插拔接入。
* **YAML + CLI 覆盖**：`configs/*.yaml` 提供团队级默认，实验时通过 CLI 快速覆盖关键参数（策略名、模型名、阈值等）。
* **LLM 分块**：`providers/llm.py` 抽象了 `BaseLLMProvider`，目前实现了 `OpenAIProvider/HFProvider`；解析采用“防御式 JSON 提取”，稳定性更高。
* **Rich 可视化**：`visualization.py` 用表格展示分块数量、字符长度、跨度与预览，有利于调参与审查。
* **结构化分块**：对 Markdown 按 `#` 标题切段；HTML 已在 reader 中转为纯文本（工程中可替换为更强的 by\_title/DOM 切分）。
* **语义分块**：句子级嵌入 + 相邻相似度断点，参数 `min_similarity` 可结合你语料分布网格搜索；过长时也会受 `chunk_size` 裁断。
* **固定分块**：为稳健性，默认按字符切（生产可把 `tiktoken` 的编码位置信息用于精准 token 切片）。

---

# 下一步可加的插件（留好扩展口）

* **Parent-Child / Auto-Merging**：检索阶段按叶子块召回、返回父级更大上下文。
* **Sentence-Window Retrieval**：检索后扩展上下文窗口。
* **代码/表格感知分块**：以函数/表格为原子单元，避免被切断。
* **版面解析器**：对 PDF 使用版面理解（如 `unstructured`/GROBID）获取更可靠的章节层级。

---

需要我帮你加一个**评测子模块（RAGAS/TruLens 指标）**和**参数网格脚本**吗？我可以在这个骨架上继续补：`evaluators/` + `experiments/`，一键比较 `size/overlap/threshold`，并把可视化做成 `rich` 的仪表板。
