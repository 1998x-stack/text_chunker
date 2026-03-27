# RAG 的 5 大文本分块策略（从图到落地）

下面把图中的五种方法逐一“解剖”：工作原理、适用场景、常见参数、优缺点与实现要点，并在每节后补充前沿做法与参考资料。最后给出一份可执行的实验与评估计划，帮助你在自己的语料上把“分块→召回→答案质量”跑通并调优。

---

## 0）总览与结论先行

在真实项目里，你很少只用**一种**分块法。更现实的做法是：
**结构优先（基于标题/段落） → 递归/语义补洞 → 固定窗口兜底 → 检索阶段做上下文扩展/父子合并**。这条组合拳能同时兼顾“召回精度”“上下文完整性”和“推理成本”。框架层面，LangChain 推荐用 `RecursiveCharacterTextSplitter` 作为通用默认；对于结构化 PDF/HTML，优先用 Unstructured 的“by\_title”或 GROBID 等解析出文档结构再分块；复杂长文可叠加“语义分块”“句窗检索（sentence-window）”“Auto-Merging/Parent-Child 检索”等策略优化生成阶段的上下文供给。([LangChain][1])

---

## 1）Fixed-size chunking（固定大小滑窗）

**怎么做**：按固定长度（字符/词/Token）切分，可设置重叠（overlap）保证边界信息连续；典型起步值是 300–512 或 500–1000 Tokens，重叠 10%–20%。优点是实现简单、吞吐高；缺点是容易切断语义单元、噪声增多。([docs.weaviate.io][2])

**何时用**：
入门基线、内容较为平铺直叙（FAQ、笔记）、或你需要先快速跑通端到端评测。

**关键参数**：
`chunk_size`、`chunk_overlap`、长度度量（字符/词/Token）。Weaviate 与 Databricks 的实践都建议**保留适度重叠**来保护边界语义。([docs.weaviate.io][2])

**小贴士**：
Token 维度更贴近嵌入/LLM 消耗；LangChain/JS 端有 `TokenTextSplitter` 可直接用 tiktoken 切 Token。([datastax.com][3])

---

## 2）Semantic chunking（语义分块）

**怎么做**：先按句/段粗分，再用嵌入相似度自适应地决定断点——相似度显著下降处作为边界，从而让每块内部尽量主题一致。LlamaIndex 的 `SemanticChunker` 与 Chroma/社区里的实现，都是“滑动窗口 + 距离突变阈值”的思路。([docs.llamaindex.ai][4])

**何时用**：
长篇叙事、技术文档、多主题混排的内容；你想提升**检索精度**和**语义完整度**时。

**优点/局限**：
语义更连贯、召回更聚焦；但需要计算嵌入、阈值敏感、对“短表格/代码片段”可能不友好。相关学术根基可追溯到 TextTiling/C99 等话题分割（topic segmentation）方法。([ACL 段落集][5])

---

## 3）Recursive chunking（递归分块）

**怎么做**：按“分隔符优先级”逐层切分：先尝试段落（`\n\n`），不行再句子（`\n`），再词（空格），最后字符，直到满足 `chunk_size`。LangChain 把它做成了 `RecursiveCharacterTextSplitter`，官方也把它作为“**通用推荐**”。([LangChain][1])

**何时用**：
多数文本类型（产品文档、博客、白皮书）作为**默认优选**，尤其是结构并不统一但又希望尽量不破坏语义单元的场景。

**优点/局限**：
比纯固定窗口更“尊重结构”，但仍可能切开表格/代码；遇到复杂 PDF 时建议与“结构化解析”联动。([Unstructured][6])

---

## 4）Document structure-based chunking（基于文档结构）

**怎么做**：先用解析器抽取**标题层级、段落、列表、表格**等元素（如 Unstructured 的 partition、GROBID 对学术 PDF 的 TEI 结构），再按这些自然边界分块；必要时再对超长元素做二次递归切分。([docs.unstructured.io][7])

**何时用**：
PDF/HTML/技术规范/论文/说明书等**结构清晰**的文档。对“标题+正文必须绑定”的检索尤为重要。([docs.unstructured.io][8])

**优点/局限**：
块的语义一致性最好、利于可解释性；但依赖解析质量，复杂版式/扫描件需要更强的版面理解。([Unstructured][9])

---

## 5）LLM-based / Agentic chunking（LLM 代理分块）

**怎么做**：把文本交给 LLM，请它基于**主题/结构/任务**自动判定边界，或先让 LLM 生成**摘要/提纲**再据此切分。适合超复杂或强语义任务。([Medium][10])

**优点/局限**：
分块更贴合任务语义、对非结构化原始资料（访谈、会议纪要）有效；但成本高、稳定性依赖提示词与模型版本，批量一致性需额外保障（例如把决策规则固化为“few-shot 模板”）。([Medium][10])

---

## 进阶组合与检索期增强（强烈建议搭配）

1）**句窗检索（Sentence-Window Retrieval）**：先以句子为检索单位，召回后**自动拼接周边若干句**，既小粒度又保上下文。([glaforge.dev][11])
2）**Chunk Expansion（后处理扩展）**：召回后把**相邻块**一并带上，缓解小块丢上下文的问题。([Pinecone][12])
3）**Parent-Child / Auto-Merging Retriever**：索引时保存**层级关系**（句→段→节），检索时以小块做相似度匹配，再**向上合并父节点**返回更完整片段，实现“召回准、上下文全”的双赢。([docs.llamaindex.ai][13])
4）**与长上下文偏置相关的策略**：LLM 存在“**middle 位置弱势**”的 U 型注意力偏置，“把关键信息尽量置于首尾、或分块时避免把关键信息压到中段”能缓解；近期研究也在通过位置校准改进长上下文利用。([arXiv][14])

---

## 参数与工程侧建议（可直接抄作业）

* **起步网格**：`chunk_size ∈ {300, 512, 800}`（Tokens），`overlap ∈ {0.1, 0.2}`；固定窗 vs 递归 vs 语义 vs 结构化，四类各跑一轮离线评测。([docs.weaviate.io][2])
* **不同语料**：

  * 法规/论文：**结构化优先**，section 级分块 + 递归兜底；召回后做 chunk expansion 或 parent-child 合并。([grobid.readthedocs.io][15])
  * 对话日志：滑窗分块（按轮次，5–7 turns/块，2–3 turns 重叠）。([Dell Technologies Info Hub][16])
  * 博客/白皮书：递归分块默认即可。([LangChain][1])
  * 代码/表格：尽量以**单元为原子**（函数/表格）做结构化切分，再在超长单元内用递归。([Unstructured][6])
* **重叠度**：10%–20% 往往足够，避免把高度重复的块一并喂给 LLM（可在检索后做**去重/相似度抑制**）。([Medium][17])

---

## 参考实现（极简伪代码/实用片段）

**LangChain · 递归分块（默认推荐）**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=80, separators=["\n\n", "\n", " ", ""]
)
docs = splitter.create_documents([raw_text])
```

（官方称之为“通用首选”，按段→句→词→字逐层切，尽量保留语义单元。([LangChain][1])）

**LlamaIndex · 语义分块**

```python
from llama_index.core.node_parser import SemanticSplitterNodeParser
parser = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95)
nodes = parser.get_nodes_from_documents([doc])
```

（基于句级嵌入与“距离突变”找分界。([docs.llamaindex.ai][4])）

**Unstructured · 结构化分块（by\_title）**
先 `partition_pdf/html` 得到元素（Title/NarrativeText/Table…），再 `chunk(by_title, max_characters=1500)`。这类“结构优先”的切分，对 PDF/规范/手册非常稳。([docs.unstructured.io][7])

**检索期扩展 · Parent-Child / Auto-Merging**
在索引中保存节点层级；检索到若干“叶节点”后，按同源父节点合并为更大段落返回。([docs.llamaindex.ai][13])

---

## 怎么“评测好”你的分块（Search & Plan）

### A. 评价维度与工具

* **检索侧**：Recall\@k、MRR、Context Relevance/ Sufficiency。可用 TruLens 的 *RAG Triad* 与 Ragas 的指标体系（faithfulness、answer correctness、context relevance 等）。([trulens.org][18])
* **生成侧**：Answer Relevance/Correctness、Hallucination/Faithfulness。可用 LLM-as-a-Judge 做快速对比，也可参考 RAGChecker 这类细粒度诊断框架。([nb-data.com][19])
* **成本&时延**：索引时间、嵌入费用、每问 Token 输入/输出、延迟 P95。社区经验普遍认为**没有一组万能参数，必须 A/B 与网格实验**。([Unstructured][6])

### B. 实验设计（可直接执行的 Plan）

1）**建立黄金集**：从你的业务 QA/工单里挑 200–500 条问答，标注其**答案所在段落/页面**。
2）**分块策略×参数网格**：

* 策略：固定窗、递归、语义、结构化、结构化+递归。
* 参数：`size={300,512,800}`，`overlap={0,10%,20%}`；语义分块的`阈值/窗口`按默认与±5 个百分点各跑一版。([LangChain][1])
  3）**检索评测**：Top-k∈{3,5,8}，记录**命中率/排名**；对每种策略再加一版**句窗检索**与**chunk expansion**对照。([glaforge.dev][11])
  4）**生成评测**：固定同一 LLM 与提示词，跑 TruLens/Ragas 指标 + 少量人工抽检。([trulens.org][18])
  5）**选择规则**：若**召回已很准但答案缺上下文**，优先上 **Parent-Child/Auto-Merging**；若**召回不稳定**，倾向用**结构化/语义分块**增稳，再微调 size/overlap。([docs.llamaindex.ai][13])
  6）**面向上线的两步走**：
* 第一阶段：结构化（by\_title）→ 递归兜底，size=512, overlap=80；检索后做 chunk-expansion。([docs.unstructured.io][20])
* 第二阶段：对错误案例引入**语义分块**或**LLM-based 分块**做“难例补丁”；在复杂长文加入 **Auto-Merging**。([docs.llamaindex.ai][4])

---

## 常见坑与规避

* **块太大**：嵌入语义“稀释”、相似度不准，召回精度掉；从 300–512 token 起步再按指标放大。([Unstructured][6])
* **块太小**：答案缺证据、生成阶段幻觉上升；用 chunk expansion / parent-child 合并兜底。([Pinecone][12])
* **标题与正文分离**：结构化分块或在节点元数据里把 `title` 绑定进检索与重排特征。([docs.unstructured.io][20])
* **长上下文偏置**：关键线索尽量靠近首尾；或在提示里复写关键信息，减少“丢在中间”的风险。([arXiv][14])

---

## 延伸阅读与可视化参考

* Pinecone 的**分块策略 & 检索后扩展**；LangChain 官方**递归分块**教程；Databricks/Weaviate 的**起步参数**建议。([Pinecone][12])
* 语义分块与社区实现（Chroma 研究记要）；TextTiling/C99 等“话题分割”起源。([research.trychroma.com][21])
* Unstructured/GROBID 的**结构化解析与分块**实践，用于 PDF/论文等复杂版面。([Unstructured][6])

---

### 你可以直接这样落地

1）优先尝试：**by\_title（结构化）→ 递归兜底（512/80）**；
2）检索端开启：**chunk expansion 或句窗检索**；
3）对难例：加一条**语义分块**线路补充；
4）规模化上线：启用**Auto-Merging/Parent-Child**，并用 Ragas/TruLens 做持续回归评测。([docs.unstructured.io][20])

如果你愿意，我可以把上述实验计划转成一份**参数网格 + 指标看板**的小脚本与表格模板，直接拿去跑对比。

[1]: https://python.langchain.com/docs/how_to/recursive_text_splitter/?utm_source=chatgpt.com "How to recursively split text by characters"
[2]: https://docs.weaviate.io/academy/py/standalone/chunking/how_1?utm_source=chatgpt.com "Chunking techniques - 1"
[3]: https://www.datastax.com/blog/how-to-chunk-text-in-javascript-for-rag-applications?utm_source=chatgpt.com "How to Chunk Text in JavaScript for Your RAG Application"
[4]: https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_chunking/?utm_source=chatgpt.com "Semantic Chunker"
[5]: https://aclanthology.org/J97-1003.pdf?utm_source=chatgpt.com "TextTiling: Segmenting Text into Multi-paragraph Subtopic ..."
[6]: https://unstructured.io/blog/chunking-for-rag-best-practices?utm_source=chatgpt.com "Chunking for RAG: best practices"
[7]: https://docs.unstructured.io/open-source/core-functionality/chunking?utm_source=chatgpt.com "Chunking"
[8]: https://docs.unstructured.io/ui/chunking?utm_source=chatgpt.com "Chunking"
[9]: https://unstructured.io/blog/rag-isn-t-so-easy-why-llm-apps-are-challenging-and-how-unstructured-can-help?utm_source=chatgpt.com "RAG Isn't So Easy: Why LLM Apps are Challenging and ..."
[10]: https://masteringllm.medium.com/11-chunking-strategies-for-rag-simplified-visualized-df0dbec8e373?utm_source=chatgpt.com "11 Chunking Strategies for RAG — Simplified & Visualized"
[11]: https://glaforge.dev/posts/2025/02/25/advanced-rag-sentence-window-retrieval/?utm_source=chatgpt.com "Advanced RAG — Sentence Window Retrieval"
[12]: https://www.pinecone.io/learn/chunking-strategies/?utm_source=chatgpt.com "Chunking Strategies for LLM Applications"
[13]: https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/?utm_source=chatgpt.com "Auto Merging Retriever"
[14]: https://arxiv.org/abs/2307.03172?utm_source=chatgpt.com "Lost in the Middle: How Language Models Use Long ..."
[15]: https://grobid.readthedocs.io/en/latest/Principles/?utm_source=chatgpt.com "How GROBID works"
[16]: https://infohub.delltechnologies.com/es-es/p/chunk-twice-retrieve-once-rag-chunking-strategies-optimized-for-different-content-types/?utm_source=chatgpt.com "Chunk Twice, Retrieve Once: RAG Chunking Strategies ..."
[17]: https://medium.com/%40adnanmasood/optimizing-chunking-embedding-and-vectorization-for-retrieval-augmented-generation-ea3b083b68f7?utm_source=chatgpt.com "Optimizing Chunking, Embedding, and Vectorization for ..."
[18]: https://www.trulens.org/getting_started/core_concepts/rag_triad/?utm_source=chatgpt.com "RAG Triad"
[19]: https://www.nb-data.com/p/evaluating-rag-with-llm-as-a-judge?utm_source=chatgpt.com "Evaluating RAG with LLM-as-a-Judge: A Guide to ..."
[20]: https://docs.unstructured.io/api-reference/api-services/chunking?utm_source=chatgpt.com "Chunking strategies"
[21]: https://research.trychroma.com/evaluating-chunking?utm_source=chatgpt.com "Evaluating Chunking Strategies for Retrieval | Chroma Research"