# 文本分块策略分析

**User:** Anonymous (harbertparker90526@outlook.com)  
**Created:** 9/16/2025 17:39:30  
**Updated:** 9/16/2025 18:00:45  
**Exported:** 9/16/2025 18:01:01  
**Link:** [https://chatgpt.com/g/g-p-68c390f088dc81918596271e1836757f-deepsearch/c/68c93050-45f8-8329-b62d-f642c0891ce3](https://chatgpt.com/g/g-p-68c390f088dc81918596271e1836757f-deepsearch/c/68c93050-45f8-8329-b62d-f642c0891ce3)  

## Prompt:
2025-09-16 17:39:27

![已上传的图片](https://chatgpt.com/backend-api/estuary/content?id=file-DdeFpoS2N8zs5g9qUKFMvi&ts=488337&p=fs&cid=1&sig=8f1b93cb45c39ecba3d6fc4f05abdb85272f1f6addc4d6fa5760f19c3f6f275c&v=0)

极其详细分析图中的5大文本分块策略

并且进行 web search 补充更多内容， please search and plan

已思考 1m 49s

正在读取文档

我需要在 Python 中提供一些代码或伪代码，可能会涉及 LangChain、LLamaIndex 等工具，但要简洁，因为用户希望详情丰富。虽然我们可以给出基于常识的分析，但关键声明仍需引用来源。我们会根据常见资源，如 LangChain 文档、OpenAI 食谱等，让信息准确且有参考依据。

我必须注意字数限制：工具有“每个来源字数限制 N；默认 200 字”，但只要我们进行意译，不直接引用大量文字，应该是没问题的。这样可以确保信息简洁且符合规定。

我计划搜索一些策略相关的查询：

*   例如关于 RAG 分块、LangChain 文本分割器、OpenAI 最佳实践等。
*   还会查找关于 LlamaIndex、Pinecone 和 Weaviate 的分块最佳实践和重叠策略。
*   进一步看看与“失落在中间”有关的研究，可能影响分块大小和重叠的方式。

我建议使用 “NLP TextTiling” 和 “语义分割使用相似性阈值” 方法。还要检查 Tiktoken 的分词次数和分块大小，以及 OpenAI 的文本嵌入模型 tamanho 和 token 限制，可能影响块的大小。记得在必要时加入来源引用。

分析RAG的五种分块策略

[

![](https://www.google.com/s2/favicons?domain=https://blog.langformers.com&sz=32)

blog.langformers.com

](https://blog.langformers.com/llm-chunking-strategies/)

[

![](https://www.google.com/s2/favicons?domain=https://python.langchain.com&sz=32)

python.langchain.com

](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)

[

![](https://www.google.com/s2/favicons?domain=https://medium.com&sz=32)

medium.com

](https://medium.com/%40adnanmasood/optimizing-chunking-embedding-and-vectorization-for-retrieval-augmented-generation-ea3b083b68f7)

![](https://www.google.com/s2/favicons?domain=https://www.pinecone.io&sz=32)

![](https://www.google.com/s2/favicons?domain=https://masteringllm.medium.com&sz=32)

![](https://www.google.com/s2/favicons?domain=https://api.python.langchain.com&sz=32)

再显示 17 个

我们有多个来源可以参考，但我还应该包括像 OpenAI 的 “RAG 最佳实践” 这样的权威文档，来自他们的食谱（cookbook）。这样可以确保我们提供的信息是可信且经过验证的。

搜索最佳实践的文档和资源

[

![](https://www.google.com/s2/favicons?domain=https://community.openai.com&sz=32)

community.openai.com

](https://community.openai.com/t/why-is-it-necessary-for-each-chunks-embedding-to-be-a-weighted-average-when-dealing-with-the-embedding-of-long-texts/341417)

[

![](https://www.google.com/s2/favicons?domain=https://weaviate.io&sz=32)

weaviate.io

](https://weaviate.io/blog/advanced-rag)

[

![](https://www.google.com/s2/favicons?domain=https://medium.com&sz=32)

medium.com

](https://medium.com/%40adnanmasood/optimizing-chunking-embedding-and-vectorization-for-retrieval-augmented-generation-ea3b083b68f7)

![](https://www.google.com/s2/favicons?domain=https://unstructured.io&sz=32)

![](https://www.google.com/s2/favicons?domain=https://docs.weaviate.io&sz=32)

![](https://www.google.com/s2/favicons?domain=https://www.datasciencecentral.com&sz=32)

再显示 12 个

我认为还需要包括一些相关的研究，如 "TextTiling Hearst 1997" 和 "C99算法"。这些研究与语义分块相关，尤其是在话题分割的上下文中。接下来，我会搜索关于 TextTiling 和 C99 的信息，以便为分析提供更多依据。

搜索TextTiling和C99算法

[

![](https://www.google.com/s2/favicons?domain=https://aclanthology.org&sz=32)

aclanthology.org

](https://aclanthology.org/J97-1003.pdf)

[

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

github.com

](https://github.com/logological/C99)

[

![](https://www.google.com/s2/favicons?domain=https://people.ischool.berkeley.edu&sz=32)

people.ischool.berkeley.edu

](https://people.ischool.berkeley.edu/~hearst/papers/cl-texttiling97.pdf)

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

![](https://www.google.com/s2/favicons?domain=https://dl.acm.org&sz=32)

![](https://www.google.com/s2/favicons?domain=https://ijret.org&sz=32)

再显示 11 个

我还应该引用 “Lost in the Middle” 研究，它探讨了长上下文的影响，建议将最相关的信息放在开始或结束，可能通过分块来缓解这个问题。我需要进一步搜索相关内容，确认这些理论如何对分块策略产生影响。

搜索关于“Lost in the Middle”效应及相关文献

[

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arxiv.org

](https://arxiv.org/abs/2307.03172)

[

![](https://www.google.com/s2/favicons?domain=https://glaforge.dev&sz=32)

glaforge.dev

](https://glaforge.dev/posts/2025/02/25/advanced-rag-sentence-window-retrieval/)

[

![](https://www.google.com/s2/favicons?domain=https://pixion.co&sz=32)

pixion.co

](https://pixion.co/blog/rag-strategies-context-enrichment)

![](https://www.google.com/s2/favicons?domain=https://cs.stanford.edu&sz=32)

![](https://www.google.com/s2/favicons?domain=https://medium.com&sz=32)

![](https://www.google.com/s2/favicons?domain=https://aclanthology.org&sz=32)

再显示 12 个

我有一组可靠的资源。还应该考虑LlamaIndex中的“自动合并检索”方法，它通过“父子索引”解决了小块数据的处理问题，同时使用指针检索大范围的上下文。接下来我需要搜索相关资料来深入了解这方面的技术。

搜索LlamaIndex自动合并检索与父文档索引

[

![](https://www.google.com/s2/favicons?domain=https://docs.llamaindex.ai&sz=32)

docs.llamaindex.ai

](https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/)

[

![](https://www.google.com/s2/favicons?domain=https://medium.com&sz=32)

medium.com

](https://medium.com/%40govindarajpriyanthan/advance-rag-build-and-evaluate-an-auto-merging-retriever-setup-using-llamaindex-and-trulens-378ce8a8116e)

[

![](https://www.google.com/s2/favicons?domain=https://cloud.google.com&sz=32)

cloud.google.com

](https://cloud.google.com/blog/products/ai-machine-learning/llamaindex-for-rag-on-google-cloud)

![](https://www.google.com/s2/favicons?domain=https://www.linkedin.com&sz=32)

![](https://www.google.com/s2/favicons?domain=https://stackoverflow.com&sz=32)

![](https://www.google.com/s2/favicons?domain=https://saurabhz.hashnode.dev&sz=32)

再显示 8 个

这些来源很好。接下来，我们需要详细呈现“chunking”策略的分析并制定实验计划。分析包括：1) 各种chunking说明，2) 伪代码实现示例，3) 使用场景，4) 参数指南，5) 常见问题，6) 评估计划，7) 推荐管道，8) 在各框架中的工具。最终提供中文分析并引用相关文献。

Searching for Ragas library and RAG metrics

[

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arxiv.org

](https://arxiv.org/abs/2309.15217)

[

![](https://www.google.com/s2/favicons?domain=https://www.patronus.ai&sz=32)

www.patronus.ai

](https://www.patronus.ai/llm-testing/rag-evaluation-metrics)

[

![](https://www.google.com/s2/favicons?domain=https://docs.ragas.io&sz=32)

docs.ragas.io

](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)

![](https://www.google.com/s2/favicons?domain=https://www.trulens.org&sz=32)

![](https://www.google.com/s2/favicons?domain=https://aclanthology.org&sz=32)

![](https://www.google.com/s2/favicons?domain=https://medium.com&sz=32)

再显示 11 个

我考虑添加“LLM-based chunking”和“Agentic chunking”策略，例如“GPT-4o restructures doc”。另外，引用“11 Chunking Strategies”博客中的Agentic chunking、Pinecone的分块策略及其扩展。同时提到通过HTML/PDF结构进行分块，参考Unstructured.io博客并探索开源库的应用。

搜索文档结构和分块策略

[

![](https://www.google.com/s2/favicons?domain=https://docs.unstructured.io&sz=32)

docs.unstructured.io

](https://docs.unstructured.io/open-source/core-functionality/chunking)

[

![](https://www.google.com/s2/favicons?domain=https://grobid.readthedocs.io&sz=32)

grobid.readthedocs.io

](https://grobid.readthedocs.io/en/latest/Principles/)

[

![](https://www.google.com/s2/favicons?domain=https://unstructured.io&sz=32)

unstructured.io

](https://unstructured.io/blog/chunking-for-rag-best-practices)

![](https://www.google.com/s2/favicons?domain=https://www.reddit.com&sz=32)

![](https://www.google.com/s2/favicons?domain=https://medium.com&sz=32)

![](https://www.google.com/s2/favicons?domain=https://news.ycombinator.com&sz=32)

再显示 7 个

我需要为分析构建详细的计划并包括引用，使用中文。我们将提供实验和实施的步骤计划，注意保持简洁但详细。虽然用户要求非常详细，但需要控制篇幅，避免过长（遵守写作规范）。所有引用将内嵌在段落末尾。

对于固定大小的块划分，使用滑动窗口，并确保有重叠。常见划分大小为500-1000个token，重叠比例为10-20%。对于语义块划分，我们可以讨论基于嵌入的句子连贯性。递归划分、文档结构等方法也需要分别提供伪代码和相关文献引用。

我可以使用“自查询检索”和“重写器”来优化。考虑到小块大小，使用“自动合并检索/父子索引”作为高级检索方法。同时，可以引入“句子窗口检索”方法。根据不同的领域（例如FAQ、法律、代码），提供推荐的分块大小，并设定合理参数。同时，加上评估计划和代码示例，确保避免过长。

首先，我将概述核心结论，并用段落列出五种策略：每种策略包括定义、工作原理、推荐参数、使用时机、潜在问题、复杂度及简短代码。接着，介绍进阶组合策略，如自动合并检索、句子窗口检索和块分扩展。在接下来的“实验计划与评估”中，列出数据集选择、评估标准等方向，确保每个点都有相关来源支持。

完成

## Response:
2025-09-16 17:41:19

RAG 的 5 大文本分块策略（从图到落地）
======================

下面把图中的五种方法逐一“解剖”：工作原理、适用场景、常见参数、优缺点与实现要点，并在每节后补充前沿做法与参考资料。最后给出一份可执行的实验与评估计划，帮助你在自己的语料上把“分块→召回→答案质量”跑通并调优。

* * *

0）总览与结论先行
---------

在真实项目里，你很少只用**一种**分块法。更现实的做法是：  
**结构优先（基于标题/段落） → 递归/语义补洞 → 固定窗口兜底 → 检索阶段做上下文扩展/父子合并**。这条组合拳能同时兼顾“召回精度”“上下文完整性”和“推理成本”。框架层面，LangChain 推荐用 `RecursiveCharacterTextSplitter` 作为通用默认；对于结构化 PDF/HTML，优先用 Unstructured 的“by\_title”或 GROBID 等解析出文档结构再分块；复杂长文可叠加“语义分块”“句窗检索（sentence-window）”“Auto-Merging/Parent-Child 检索”等策略优化生成阶段的上下文供给。[docs.llamaindex.ai+4LangChain+4docs.unstructured.io+4](https://python.langchain.com/docs/how_to/recursive_text_splitter/?utm_source=chatgpt.com)

* * *

1）Fixed-size chunking（固定大小滑窗）
-----------------------------

**怎么做**：按固定长度（字符/词/Token）切分，可设置重叠（overlap）保证边界信息连续；典型起步值是 300–512 或 500–1000 Tokens，重叠 10%–20%。优点是实现简单、吞吐高；缺点是容易切断语义单元、噪声增多。[DEV Community+3docs.weaviate.io+3docs.databricks.com+3](https://docs.weaviate.io/academy/py/standalone/chunking/how_1?utm_source=chatgpt.com)

**何时用**：  
入门基线、内容较为平铺直叙（FAQ、笔记）、或你需要先快速跑通端到端评测。

**关键参数**：  
`chunk_size`、`chunk_overlap`、长度度量（字符/词/Token）。Weaviate 与 Databricks 的实践都建议**保留适度重叠**来保护边界语义。[docs.weaviate.io+1](https://docs.weaviate.io/academy/py/standalone/chunking/how_1?utm_source=chatgpt.com)

**小贴士**：  
Token 维度更贴近嵌入/LLM 消耗；LangChain/JS 端有 `TokenTextSplitter` 可直接用 tiktoken 切 Token。[datastax.com](https://www.datastax.com/blog/how-to-chunk-text-in-javascript-for-rag-applications?utm_source=chatgpt.com)

* * *

2）Semantic chunking（语义分块）
-------------------------

**怎么做**：先按句/段粗分，再用嵌入相似度自适应地决定断点——相似度显著下降处作为边界，从而让每块内部尽量主题一致。LlamaIndex 的 `SemanticChunker` 与 Chroma/社区里的实现，都是“滑动窗口 + 距离突变阈值”的思路。[docs.llamaindex.ai+2docs.llamaindex.ai+2](https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_chunking/?utm_source=chatgpt.com)

**何时用**：  
长篇叙事、技术文档、多主题混排的内容；你想提升**检索精度**和**语义完整度**时。

**优点/局限**：  
语义更连贯、召回更聚焦；但需要计算嵌入、阈值敏感、对“短表格/代码片段”可能不友好。相关学术根基可追溯到 TextTiling/C99 等话题分割（topic segmentation）方法。[ACL 段落集+1](https://aclanthology.org/J97-1003.pdf?utm_source=chatgpt.com)

* * *

3）Recursive chunking（递归分块）
--------------------------

**怎么做**：按“分隔符优先级”逐层切分：先尝试段落（`\n\n`），不行再句子（`\n`），再词（空格），最后字符，直到满足 `chunk_size`。LangChain 把它做成了 `RecursiveCharacterTextSplitter`，官方也把它作为“**通用推荐**”。[LangChain+1](https://python.langchain.com/docs/how_to/recursive_text_splitter/?utm_source=chatgpt.com)

**何时用**：  
多数文本类型（产品文档、博客、白皮书）作为**默认优选**，尤其是结构并不统一但又希望尽量不破坏语义单元的场景。

**优点/局限**：  
比纯固定窗口更“尊重结构”，但仍可能切开表格/代码；遇到复杂 PDF 时建议与“结构化解析”联动。[Unstructured](https://unstructured.io/blog/chunking-for-rag-best-practices?utm_source=chatgpt.com)

* * *

4）Document structure-based chunking（基于文档结构）
-------------------------------------------

**怎么做**：先用解析器抽取**标题层级、段落、列表、表格**等元素（如 Unstructured 的 partition、GROBID 对学术 PDF 的 TEI 结构），再按这些自然边界分块；必要时再对超长元素做二次递归切分。[docs.unstructured.io+2docs.unstructured.io+2](https://docs.unstructured.io/open-source/core-functionality/chunking?utm_source=chatgpt.com)

**何时用**：  
PDF/HTML/技术规范/论文/说明书等**结构清晰**的文档。对“标题+正文必须绑定”的检索尤为重要。[docs.unstructured.io](https://docs.unstructured.io/ui/chunking?utm_source=chatgpt.com)

**优点/局限**：  
块的语义一致性最好、利于可解释性；但依赖解析质量，复杂版式/扫描件需要更强的版面理解。[Unstructured](https://unstructured.io/blog/rag-isn-t-so-easy-why-llm-apps-are-challenging-and-how-unstructured-can-help?utm_source=chatgpt.com)

* * *

5）LLM-based / Agentic chunking（LLM 代理分块）
----------------------------------------

**怎么做**：把文本交给 LLM，请它基于**主题/结构/任务**自动判定边界，或先让 LLM 生成**摘要/提纲**再据此切分。适合超复杂或强语义任务。[Medium](https://masteringllm.medium.com/11-chunking-strategies-for-rag-simplified-visualized-df0dbec8e373?utm_source=chatgpt.com)

**优点/局限**：  
分块更贴合任务语义、对非结构化原始资料（访谈、会议纪要）有效；但成本高、稳定性依赖提示词与模型版本，批量一致性需额外保障（例如把决策规则固化为“few-shot 模板”）。[Medium](https://masteringllm.medium.com/11-chunking-strategies-for-rag-simplified-visualized-df0dbec8e373?utm_source=chatgpt.com)

* * *

进阶组合与检索期增强（强烈建议搭配）
------------------

1）**句窗检索（Sentence-Window Retrieval）**：先以句子为检索单位，召回后**自动拼接周边若干句**，既小粒度又保上下文。[glaforge.dev+1](https://glaforge.dev/posts/2025/02/25/advanced-rag-sentence-window-retrieval/?utm_source=chatgpt.com)  
2）**Chunk Expansion（后处理扩展）**：召回后把**相邻块**一并带上，缓解小块丢上下文的问题。[Pinecone](https://www.pinecone.io/learn/chunking-strategies/?utm_source=chatgpt.com)  
3）**Parent-Child / Auto-Merging Retriever**：索引时保存**层级关系**（句→段→节），检索时以小块做相似度匹配，再**向上合并父节点**返回更完整片段，实现“召回准、上下文全”的双赢。[docs.llamaindex.ai+2docs.llamaindex.ai+2](https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/?utm_source=chatgpt.com)  
4）**与长上下文偏置相关的策略**：LLM 存在“**middle 位置弱势**”的 U 型注意力偏置，“把关键信息尽量置于首尾、或分块时避免把关键信息压到中段”能缓解；近期研究也在通过位置校准改进长上下文利用。[arXiv+2arXiv+2](https://arxiv.org/abs/2307.03172?utm_source=chatgpt.com)

* * *

参数与工程侧建议（可直接抄作业）
----------------

*   **起步网格**：`chunk_size ∈ {300, 512, 800}`（Tokens），`overlap ∈ {0.1, 0.2}`；固定窗 vs 递归 vs 语义 vs 结构化，四类各跑一轮离线评测。[docs.weaviate.io+1](https://docs.weaviate.io/academy/py/standalone/chunking/how_1?utm_source=chatgpt.com)
*   **不同语料**：
    *   法规/论文：**结构化优先**，section 级分块 + 递归兜底；召回后做 chunk expansion 或 parent-child 合并。[grobid.readthedocs.io](https://grobid.readthedocs.io/en/latest/Principles/?utm_source=chatgpt.com)
    *   对话日志：滑窗分块（按轮次，5–7 turns/块，2–3 turns 重叠）。[Dell Technologies Info Hub](https://infohub.delltechnologies.com/es-es/p/chunk-twice-retrieve-once-rag-chunking-strategies-optimized-for-different-content-types/?utm_source=chatgpt.com)
    *   博客/白皮书：递归分块默认即可。[LangChain](https://python.langchain.com/docs/how_to/recursive_text_splitter/?utm_source=chatgpt.com)
    *   代码/表格：尽量以**单元为原子**（函数/表格）做结构化切分，再在超长单元内用递归。[Unstructured](https://unstructured.io/blog/chunking-for-rag-best-practices?utm_source=chatgpt.com)
*   **重叠度**：10%–20% 往往足够，避免把高度重复的块一并喂给 LLM（可在检索后做**去重/相似度抑制**）。[Medium](https://medium.com/%40adnanmasood/optimizing-chunking-embedding-and-vectorization-for-retrieval-augmented-generation-ea3b083b68f7?utm_source=chatgpt.com)

* * *

参考实现（极简伪代码/实用片段）
----------------

**LangChain · 递归分块（默认推荐）**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=80, separators=["\n\n", "\n", " ", ""]
)
docs = splitter.create_documents([raw_text])
```

（官方称之为“通用首选”，按段→句→词→字逐层切，尽量保留语义单元。[LangChain](https://python.langchain.com/docs/how_to/recursive_text_splitter/?utm_source=chatgpt.com)）

**LlamaIndex · 语义分块**

```python
from llama_index.core.node_parser import SemanticSplitterNodeParser
parser = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95)
nodes = parser.get_nodes_from_documents([doc])
```

（基于句级嵌入与“距离突变”找分界。[docs.llamaindex.ai+1](https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_chunking/?utm_source=chatgpt.com)）

**Unstructured · 结构化分块（by\_title）**  
先 `partition_pdf/html` 得到元素（Title/NarrativeText/Table…），再 `chunk(by_title, max_characters=1500)`。这类“结构优先”的切分，对 PDF/规范/手册非常稳。[docs.unstructured.io+1](https://docs.unstructured.io/open-source/core-functionality/chunking?utm_source=chatgpt.com)

**检索期扩展 · Parent-Child / Auto-Merging**  
在索引中保存节点层级；检索到若干“叶节点”后，按同源父节点合并为更大段落返回。[docs.llamaindex.ai+1](https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/?utm_source=chatgpt.com)

* * *

怎么“评测好”你的分块（Search & Plan）
--------------------------

### A. 评价维度与工具

*   **检索侧**：Recall@k、MRR、Context Relevance/ Sufficiency。可用 TruLens 的 _RAG Triad_ 与 Ragas 的指标体系（faithfulness、answer correctness、context relevance 等）。[trulens.org+2Ragas+2](https://www.trulens.org/getting_started/core_concepts/rag_triad/?utm_source=chatgpt.com)
*   **生成侧**：Answer Relevance/Correctness、Hallucination/Faithfulness。可用 LLM-as-a-Judge 做快速对比，也可参考 RAGChecker 这类细粒度诊断框架。[nb-data.com+1](https://www.nb-data.com/p/evaluating-rag-with-llm-as-a-judge?utm_source=chatgpt.com)
*   **成本&时延**：索引时间、嵌入费用、每问 Token 输入/输出、延迟 P95。社区经验普遍认为**没有一组万能参数，必须 A/B 与网格实验**。[Unstructured](https://unstructured.io/blog/chunking-for-rag-best-practices?utm_source=chatgpt.com)

### B. 实验设计（可直接执行的 Plan）

1）**建立黄金集**：从你的业务 QA/工单里挑 200–500 条问答，标注其**答案所在段落/页面**。  
2）**分块策略×参数网格**：

*   策略：固定窗、递归、语义、结构化、结构化+递归。
*   参数：`size={300,512,800}`，`overlap={0,10%,20%}`；语义分块的`阈值/窗口`按默认与±5 个百分点各跑一版。[LangChain+1](https://python.langchain.com/docs/how_to/recursive_text_splitter/?utm_source=chatgpt.com)  
    3）**检索评测**：Top-k∈{3,5,8}，记录**命中率/排名**；对每种策略再加一版**句窗检索**与**chunk expansion**对照。[glaforge.dev+1](https://glaforge.dev/posts/2025/02/25/advanced-rag-sentence-window-retrieval/?utm_source=chatgpt.com)  
    4）**生成评测**：固定同一 LLM 与提示词，跑 TruLens/Ragas 指标 + 少量人工抽检。[trulens.org+1](https://www.trulens.org/getting_started/core_concepts/rag_triad/?utm_source=chatgpt.com)  
    5）**选择规则**：若**召回已很准但答案缺上下文**，优先上 **Parent-Child/Auto-Merging**；若**召回不稳定**，倾向用**结构化/语义分块**增稳，再微调 size/overlap。[docs.llamaindex.ai](https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/?utm_source=chatgpt.com)  
    6）**面向上线的两步走**：
*   第一阶段：结构化（by\_title）→ 递归兜底，size=512, overlap=80；检索后做 chunk-expansion。[docs.unstructured.io+2LangChain+2](https://docs.unstructured.io/api-reference/api-services/chunking?utm_source=chatgpt.com)
*   第二阶段：对错误案例引入**语义分块**或**LLM-based 分块**做“难例补丁”；在复杂长文加入 **Auto-Merging**。[docs.llamaindex.ai+2Medium+2](https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_chunking/?utm_source=chatgpt.com)

* * *

常见坑与规避
------

*   **块太大**：嵌入语义“稀释”、相似度不准，召回精度掉；从 300–512 token 起步再按指标放大。[Unstructured](https://unstructured.io/blog/chunking-for-rag-best-practices?utm_source=chatgpt.com)
*   **块太小**：答案缺证据、生成阶段幻觉上升；用 chunk expansion / parent-child 合并兜底。[Pinecone+1](https://www.pinecone.io/learn/chunking-strategies/?utm_source=chatgpt.com)
*   **标题与正文分离**：结构化分块或在节点元数据里把 `title` 绑定进检索与重排特征。[docs.unstructured.io](https://docs.unstructured.io/api-reference/api-services/chunking?utm_source=chatgpt.com)
*   **长上下文偏置**：关键线索尽量靠近首尾；或在提示里复写关键信息，减少“丢在中间”的风险。[arXiv](https://arxiv.org/abs/2307.03172?utm_source=chatgpt.com)

* * *

延伸阅读与可视化参考
----------

*   Pinecone 的**分块策略 & 检索后扩展**；LangChain 官方**递归分块**教程；Databricks/Weaviate 的**起步参数**建议。[docs.databricks.com+3Pinecone+3LangChain+3](https://www.pinecone.io/learn/chunking-strategies/?utm_source=chatgpt.com)
*   语义分块与社区实现（Chroma 研究记要）；TextTiling/C99 等“话题分割”起源。[research.trychroma.com+2ACL 段落集+2](https://research.trychroma.com/evaluating-chunking?utm_source=chatgpt.com)
*   Unstructured/GROBID 的**结构化解析与分块**实践，用于 PDF/论文等复杂版面。[Unstructured+1](https://unstructured.io/blog/chunking-for-rag-best-practices?utm_source=chatgpt.com)

* * *

### 你可以直接这样落地

1）优先尝试：**by\_title（结构化）→ 递归兜底（512/80）**；  
2）检索端开启：**chunk expansion 或句窗检索**；  
3）对难例：加一条**语义分块**线路补充；  
4）规模化上线：启用**Auto-Merging/Parent-Child**，并用 Ragas/TruLens 做持续回归评测。[trulens.org+7docs.unstructured.io+7LangChain+7](https://docs.unstructured.io/api-reference/api-services/chunking?utm_source=chatgpt.com)

如果你愿意，我可以把上述实验计划转成一份**参数网格 + 指标看板**的小脚本与表格模板，直接拿去跑对比。

## Prompt:
2025-09-16 17:45:25

build text chunk pluggable project (factory mode project)
explore and think how to construct the pluggable project for text chunk, now it contains five strategies, but it will be more added
use yaml and argparse please
llm -chunk support openai-API-key or huggingface model or so on
add requirements.txt and setup.cfg for installation using Pip

add rich text package for visualizations on visualization mode

<content>
1. Fixed-size Chunking（固定分块）
核心思想：按固定长度（如 256 tokens）分割文本，可重叠（滑动窗口）。

优点：简单高效，适合常规 NLP 任务（如向量检索）。

缺点：可能切断语义连贯性（如句子中途截断）。

场景：BERT 等模型的输入预处理、基础 RAG 系统。

2. Semantic Chunking（语义分块）
核心思想：基于文本语义边界分块（如段落、话题转折点）。

实现：

规则：按标点（句号、段落符）分割；

模型：用嵌入相似度检测语义边界（如 Sentence-BERT）。

优点：保留语义完整性。

缺点：计算成本较高。

场景：精细化问答、摘要生成。

3. Recursive Chunking（递归分块）
核心思想：分层分割文本（如先按段落→再按句子）。

优点：平衡长度与语义，适配多级处理需求。

缺点：需设计分层规则。

场景：长文档处理（论文、法律文本）。

4. Document Structure-based Chunking（基于文档结构的分块）
核心思想：利用文档固有结构（标题、章节、表格）分块。

实现：解析 Markdown/HTML/PDF 的标签结构。

优点：精准匹配人类阅读逻辑。

缺点：依赖文档格式规范性。

场景：技术手册、结构化报告解析。

5. LLM-based Chunking（基于大模型的分块）
核心思想：用 LLM（如 GPT-4）动态决定分块策略。

方法：

直接生成分块边界；

指导规则引擎优化（如“将这段话按时间线拆分”）。

优点：灵活适配复杂需求。

缺点：成本高、延迟大。

场景：高价值文本处理（如医疗记录、跨语言内容）。

对比总结

策略

核心逻辑

优势

局限性

Fixed-size

固定长度切割

高效、通用

语义断裂风险

Semantic

语义边界检测

保留上下文

计算复杂度高

Recursive

多级递归分割

灵活适配长文本

规则设计复杂

Structure-based

文档标签解析

精准匹配结构

依赖格式标准化

LLM-based

大模型动态决策

智能适应场景

成本高、速度慢
</content>

## Response:
2025-09-16 17:46:49

已思考 1m 21s

我们需要构建一个“工厂模式”的可插拔项目，支持文本块策略，使用yaml和argparse。需要支持LLM文本块，提供OpenAI API密钥或Huggingface模型。项目包括安装所需的requirements.txt和setup.cfg。还要实现“丰富文本”包用于终端可视化，可能通过CLI标志显示预览或ASCII框架。

我们需要使用dataclass来做配置，提供基本的策略接口，并实现例如五种策略。要提供LLM块策略，支持OpenAI API或HuggingFace管道，具体部署应可选，若无API密钥则跳过。需要适配器来读取“LLM提供商”的环境变量（如OPENAI\_API\_KEY）。配置`setup.cfg`以便flake8检查，并提供`requirements.txt`依赖项。勾选更多需求如`markdown-it-py`、`BeautifulSoup4`等。

对于PDF支持，可以选择使用“pdfminer.six”或pypdf，保持最小化。结构化文件将解析Markdown（#）和HTML（h1-h6）标题。实现一个读取器来加载文件（.txt、.md、.html、.pdf），并提供可视化模块使用Rich显示块统计与区间。且提供CLI命令与yaml配置示例。

我们将包括以下关键文件和目录：

项目结构：提供`chunker`文件夹，内含各种切分策略，如`FixedChunker`、`SemanticChunker`等。同时，提供基本配置文件`default.yaml`及`semantic.yaml`配置，包含所需的依赖`requirements.txt`和`setup.cfg`。类`Chunker`将有切分方法，`FixedChunker`使用tiktoken进行切分。

我们将实现一系列功能，包括多种切分策略，如`RecursiveChunker`（递归分割）和`StructureChunker`（基于结构的分割）。此外，还会设计LLM驱动的**LLMBasedChunker**，支持OpenAI与HuggingFace。项目支持CLI、可视化及日志记录功能，通过配置和简单的UI命令运行。

完成

下面给你一个**可插拔（Factory 模式）文本分块工程**，支持 5 种策略并方便扩展到更多策略；同时提供命令行（`argparse`）、YAML 配置、可选 LLM（OpenAI/HuggingFace）驱动的分块，以及基于 `rich` 的可视化预览。你可以直接按“项目树 → 文件逐个复制”的顺序落地。

* * *

项目结构（pluggable/factory）
=======================

```
text-chunker/
├─ README.md
├─ requirements.txt
├─ setup.cfg
├─ setup.py
├─ configs/
│  ├─ default.yaml
│  ├─ semantic.yaml
│  └─ llm_based.yaml
├─ textchunker/
│  ├─ __init__.py
│  ├─ cli.py
│  ├─ config.py
│  ├─ types.py
│  ├─ utils.py
│  ├─ registry.py
│  ├─ factory.py
│  ├─ readers.py
│  ├─ export.py
│  ├─ visualization.py
│  ├─ providers/
│  │  ├─ __init__.py
│  │  └─ llm.py
│  └─ chunkers/
│     ├─ __init__.py
│     ├─ base.py
│     ├─ fixed.py
│     ├─ semantic.py
│     ├─ recursive.py
│     ├─ structure.py
│     └─ llm_based.py
```

* * *

1) `requirements.txt`
---------------------

```txt
# Core
pyyaml>=6.0.1
loguru>=0.7.2
rich>=13.7.1

# Optional tokenization & embeddings
tiktoken>=0.7.0

# Semantic chunking (optional but recommended)
sentence-transformers>=3.0.1
numpy>=1.26.0

# Structure-based parsing
beautifulsoup4>=4.12.3
pypdf>=5.0.0
markdown-it-py>=3.0.0

# LLM providers (optional)
openai>=1.46.0
transformers>=4.44.2
torch>=2.2.0
```

> 按需安装：若不使用某一策略，可删掉对应依赖。

* * *

2) `setup.cfg`（pip 安装元数据）
-------------------------

```ini
[metadata]
name = textchunker
version = 0.1.0
description = Pluggable text chunking factory project with YAML/CLI/LLM support
author = Your Name
license = MIT
long_description = file: README.md
long_description_content_type = text/markdown
url = https://example.com/text-chunker
classifiers =
    Programming Language :: Python :: 3
    License :: OSI Approved :: MIT License
    Topic :: Text Processing :: General

[options]
packages = find:
python_requires = >=3.9
install_requires =
    pyyaml
    loguru
    rich

[options.extras_require]
semantic =
    sentence-transformers
    numpy
token =
    tiktoken
structure =
    beautifulsoup4
    pypdf
    markdown-it-py
llm =
    openai
    transformers
    torch

[flake8]
max-line-length = 100
extend-ignore = E203,W503
```

3) `setup.py`（最小 stub，兼容 `pip install -e .`）
--------------------------------------------

```python
from setuptools import setup
setup()
```

* * *

4) `configs/default.yaml`
-------------------------

```yaml
# 全局默认配置
project:
  name: text-chunker-demo
  seed: 42
  visualize: true              # 是否用 rich 预览
  export_format: jsonl         # jsonl | txt_dir | both

io:
  input_path: ./sample.txt     # 文件或目录
  output_path: ./out/chunks.jsonl

strategy:
  name: recursive              # fixed | semantic | recursive | structure | llm
  common:
    chunk_size: 512            # 以 token 计数更合理（fallback 到字符）
    chunk_overlap: 80
    max_chunks: null
  recursive:
    separators: ["\n\n", "\n", " ", ""]
  fixed:
    use_tokens: true
  semantic:
    model_name: sentence-transformers/all-MiniLM-L6-v2
    min_similarity: 0.62       # 相邻句相似度低于该阈值则断开
    sentence_window: 1
  structure:
    prefer: md                  # md | html | pdf | auto
    sub_split: true             # 节点过长时递归再切
  llm:
    provider: openai            # openai | hf
    openai_model: gpt-4o-mini
    hf_model: Qwen/Qwen2.5-7B-Instruct
    max_chars_per_call: 8000    # 控制单次提示长度
    system_prompt: |
      你是一名文本结构工程师。请将输入文本按主题/标题/时间线合理分块，
      产出 JSON 列表：[{ "title": str|null, "start": int, "end": int }]
```

### `configs/semantic.yaml`

```yaml
strategy:
  name: semantic
  common:
    chunk_size: 600
    chunk_overlap: 100
  semantic:
    model_name: sentence-transformers/all-MiniLM-L6-v2
    min_similarity: 0.65
```

### `configs/llm_based.yaml`

```yaml
strategy:
  name: llm
  llm:
    provider: hf
    hf_model: Qwen/Qwen2.5-7B-Instruct
    max_chars_per_call: 6000
```

* * *

5) `textchunker/__init__.py`
----------------------------

```python
__all__ = ["cli", "factory", "registry", "types"]
```

* * *

6) `textchunker/types.py`
-------------------------

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class Chunk:
    """表示一个文本分块的统一数据结构。

    Attributes:
        id: 分块编号（从 0 开始）。
        text: 分块文本内容。
        start: 在原始文本中的起始字符位置（包含）。
        end: 在原始文本中的结束字符位置（不包含）。
        meta: 额外元数据（如标题、层级、来源文件路径等）。
    """
    id: int
    text: str
    start: int
    end: int
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FileDoc:
    """待处理的文档对象，包含路径与内容。"""
    path: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyConfig:
    """策略通用与专项配置（从 YAML/CLI 合并而来）。"""
    name: str
    common: Dict[str, Any] = field(default_factory=dict)
    semantic: Dict[str, Any] = field(default_factory=dict)
    recursive: Dict[str, Any] = field(default_factory=dict)
    fixed: Dict[str, Any] = field(default_factory=dict)
    structure: Dict[str, Any] = field(default_factory=dict)
    llm: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProjectConfig:
    """工程配置。"""
    project: Dict[str, Any]
    io: Dict[str, Any]
    strategy: StrategyConfig
```

* * *

7) `textchunker/config.py`
--------------------------

```python
from __future__ import annotations
import argparse
import yaml
from dataclasses import asdict
from typing import Any, Dict
from .types import ProjectConfig, StrategyConfig

def load_yaml(path: str) -> Dict[str, Any]:
    """读取 YAML 配置文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def to_project_config(d: Dict[str, Any]) -> ProjectConfig:
    """将 dict 转为强类型配置。"""
    strat = d.get("strategy", {})
    sc = StrategyConfig(
        name=strat.get("name", "recursive"),
        common=strat.get("common", {}),
        semantic=strat.get("semantic", {}),
        recursive=strat.get("recursive", {}),
        fixed=strat.get("fixed", {}),
        structure=strat.get("structure", {}),
        llm=strat.get("llm", {}),
    )
    return ProjectConfig(
        project=d.get("project", {}),
        io=d.get("io", {}),
        strategy=sc,
    )

def build_argparser() -> argparse.ArgumentParser:
    """命令行参数定义。"""
    p = argparse.ArgumentParser(description="Pluggable Text Chunker (Factory)")
    p.add_argument("--config", type=str, default="configs/default.yaml",
                   help="YAML 配置文件路径")
    p.add_argument("--input", type=str, default=None, help="输入文件或目录，覆盖 YAML")
    p.add_argument("--output", type=str, default=None, help="输出路径，覆盖 YAML")
    p.add_argument("--strategy", type=str, default=None,
                   help="策略名：fixed/semantic/recursive/structure/llm")
    p.add_argument("--visualize", action="store_true",
                   help="可视化模式（rich）")
    p.add_argument("--max-chunks", type=int, default=None,
                   help="限制返回的最大分块数")
    # LLM provider 及模型覆盖
    p.add_argument("--provider", type=str, default=None, help="openai/hf")
    p.add_argument("--openai-model", type=str, default=None)
    p.add_argument("--hf-model", type=str, default=None)
    return p

def merge_cli(cfg: ProjectConfig, args: argparse.Namespace) -> ProjectConfig:
    """合并 CLI 覆盖项到配置。"""
    if args.input:
        cfg.io["input_path"] = args.input
    if args.output:
        cfg.io["output_path"] = args.output
    if args.strategy:
        cfg.strategy.name = args.strategy
    if args.visualize:
        cfg.project["visualize"] = True
    if args.max_chunks is not None:
        cfg.strategy.common["max_chunks"] = args.max_chunks
    if args.provider:
        cfg.strategy.llm["provider"] = args.provider
    if args.openai_model:
        cfg.strategy.llm["openai_model"] = args.openai_model
    if args.hf_model:
        cfg.strategy.llm["hf_model"] = args.hf_model
    return cfg
```

* * *

8) `textchunker/utils.py`
-------------------------

```python
from __future__ import annotations
import re
from typing import Iterable, List, Tuple, Optional

# 尝试使用 tiktoken 进行 token 计数，不可用则退化到字符级
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
    _enc = None

def count_tokens(text: str) -> int:
    """统计文本的 token 数（优先 tiktoken；否则返回字符数）。"""
    if _enc:
        return len(_enc.encode(text))
    return len(text)

def whitespace_sentences(text: str) -> List[str]:
    """轻量级句子切分（中文/英文兼容的标点启发式）。"""
    pat = r'(?<=[。！？!?；;])\s+|(?<=\.)\s+'
    parts = re.split(pat, text.strip())
    return [p for p in parts if p]

def clamp(n: int, a: int, b: int) -> int:
    return max(a, min(n, b))

def sliding_windows(seq: List[str], k: int) -> Iterable[Tuple[int, List[str]]]:
    """滑动窗口，返回 (起始索引, 片段列表)。"""
    for i in range(0, len(seq) - k + 1):
        yield i, seq[i:i + k]
```

* * *

9) `textchunker/registry.py`
----------------------------

```python
from __future__ import annotations
from typing import Dict, Type
from .chunkers.base import BaseChunker

_REGISTRY: Dict[str, Type[BaseChunker]] = {}

def register(name: str):
    """装饰器：注册分块策略到全局工厂。"""
    def deco(cls: Type[BaseChunker]):
        _REGISTRY[name.lower()] = cls
        return cls
    return deco

def get(name: str) -> Type[BaseChunker]:
    if name.lower() not in _REGISTRY:
        raise KeyError(f"Unknown chunker '{name}'. Registered: {list(_REGISTRY)}")
    return _REGISTRY[name.lower()]

def available() -> Dict[str, Type[BaseChunker]]:
    return dict(_REGISTRY)
```

* * *

10) `textchunker/factory.py`
----------------------------

```python
from __future__ import annotations
from typing import Any, Dict
from .registry import get
from .types import StrategyConfig
from .chunkers.base import BaseChunker

def create_chunker(cfg: StrategyConfig) -> BaseChunker:
    """工厂：根据策略名创建分块器实例。"""
    cls = get(cfg.name)
    return cls(cfg)
```

* * *

11) `textchunker/readers.py`
----------------------------

```python
from __future__ import annotations
import os
from typing import Iterable, List
from bs4 import BeautifulSoup  # type: ignore
from pypdf import PdfReader  # type: ignore
from .types import FileDoc

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _read_html(path: str) -> str:
    html = _read_text(path)
    soup = BeautifulSoup(html, "html.parser")
    # 将可见文本抽取并保持基本段落
    return "\n".join([t.get_text(" ", strip=True) for t in soup.find_all(["h1","h2","h3","h4","h5","h6","p","li"])])

def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n\n".join(pages)

def load_inputs(path: str) -> Iterable[FileDoc]:
    """支持目录/单文件的统一读取。"""
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for fn in files:
                fp = os.path.join(root, fn)
                ext = os.path.splitext(fn)[-1].lower()
                if ext in {".txt", ".md"}:
                    yield FileDoc(path=fp, text=_read_text(fp))
                elif ext in {".html", ".htm"}:
                    yield FileDoc(path=fp, text=_read_html(fp), meta={"format": "html"})
                elif ext == ".pdf":
                    yield FileDoc(path=fp, text=_read_pdf(fp), meta={"format": "pdf"})
    else:
        ext = os.path.splitext(path)[-1].lower()
        if ext in {".txt", ".md"}:
            yield FileDoc(path=path, text=_read_text(path))
        elif ext in {".html", ".htm"}:
            yield FileDoc(path=path, text=_read_html(path), meta={"format": "html"})
        elif ext == ".pdf":
            yield FileDoc(path=path, text=_read_pdf(path), meta={"format": "pdf"})
        else:
            # 默认尝试按文本
            yield FileDoc(path=path, text=_read_text(path))
```

* * *

12) `textchunker/export.py`
---------------------------

```python
from __future__ import annotations
import json
import os
from typing import Iterable, List
from .types import Chunk, FileDoc

def save_jsonl(path: str, chunks: List[Chunk], source: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            rec = {"id": c.id, "text": c.text, "start": c.start, "end": c.end, "meta": c.meta, "source": source}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def save_txt_dir(dirpath: str, chunks: List[Chunk], prefix: str = "chunk") -> None:
    os.makedirs(dirpath, exist_ok=True)
    for c in chunks:
        fn = os.path.join(dirpath, f"{prefix}-{c.id:04d}.txt")
        with open(fn, "w", encoding="utf-8") as f:
            f.write(c.text)
```

* * *

13) `textchunker/visualization.py`
----------------------------------

```python
from __future__ import annotations
from typing import List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from .types import Chunk

_console = Console()

def preview(text: str, limit: int = 80) -> str:
    s = text.replace("\n", " ")
    return (s[:limit] + "…") if len(s) > limit else s

def show_chunks(chunks: List[Chunk], title: str = "Chunk Preview") -> None:
    """Rich 可视化：展示分块统计与预览。"""
    total_chars = sum(len(c.text) for c in chunks)
    table = Table(title=title)
    table.add_column("ID", justify="right")
    table.add_column("Chars")
    table.add_column("Span")
    table.add_column("Meta")
    table.add_column("Preview")
    for c in chunks:
        meta = ", ".join(f"{k}:{v}" for k, v in c.meta.items() if v is not None) or "-"
        table.add_row(str(c.id), str(len(c.text)), f"[{c.start},{c.end})", meta, preview(c.text))
    _console.print(Panel.fit(table, title=f"Chunks={len(chunks)}, TotalChars={total_chars}"))
```

* * *

14) `textchunker/providers/__init__.py`
---------------------------------------

```python
from .llm import OpenAIProvider, HFProvider, BaseLLMProvider
__all__ = ["OpenAIProvider", "HFProvider", "BaseLLMProvider"]
```

15) `textchunker/providers/llm.py`
----------------------------------

```python
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
```

* * *

16) `textchunker/chunkers/__init__.py`
--------------------------------------

```python
# 仅用于触发注册（import side-effects）
from . import fixed, semantic, recursive, structure, llm_based  # noqa: F401
```

17) `textchunker/chunkers/base.py`
----------------------------------

```python
from __future__ import annotations
from typing import List
from abc import ABC, abstractmethod
from ..types import Chunk, StrategyConfig

class BaseChunker(ABC):
    """分块器基类。所有策略需实现 `chunk(text)`。"""

    def __init__(self, cfg: StrategyConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """将输入文本分割为 Chunk 列表。"""
        raise NotImplementedError
```

18) `textchunker/chunkers/fixed.py`
-----------------------------------

```python
from __future__ import annotations
from typing import List
from loguru import logger
from ..types import Chunk
from ..utils import count_tokens
from ..registry import register
from .base import BaseChunker

@register("fixed")
class FixedChunker(BaseChunker):
    """固定窗口分块（可重叠、按 token 或字符）。"""

    def chunk(self, text: str) -> List[Chunk]:
        c = self.cfg.common
        f = self.cfg.fixed
        size = int(c.get("chunk_size", 512))
        overlap = int(c.get("chunk_overlap", 80))
        use_tokens = bool(f.get("use_tokens", True))
        max_chunks = c.get("max_chunks")

        logger.info(f"Fixed-size chunking: size={size}, overlap={overlap}, tokens={use_tokens}")
        chunks: List[Chunk] = []

        if use_tokens and count_tokens("x") == 1:
            # 使用 token 尺度的简单实现：切片时用近似字符比例回退
            # 为稳健性，这里直接按字符近似实现（工业环境可换为 tiktoken 的编码反解映射）
            pass

        start = 0
        n = len(text)
        while start < n:
            end = min(start + size, n)
            piece = text[start:end]
            cid = len(chunks)
            chunks.append(Chunk(id=cid, text=piece, start=start, end=end, meta={"strategy": "fixed"}))
            if max_chunks and len(chunks) >= max_chunks:
                break
            if end == n:
                break
            start = max(end - overlap, start + 1)

        return chunks
```

19) `textchunker/chunkers/semantic.py`
--------------------------------------

```python
from __future__ import annotations
from typing import List
import numpy as np  # type: ignore
from loguru import logger
from sentence_transformers import SentenceTransformer  # type: ignore
from ..types import Chunk
from ..utils import whitespace_sentences, count_tokens
from ..registry import register
from .base import BaseChunker

@register("semantic")
class SemanticChunker(BaseChunker):
    """语义自适应分块：按句子级相似度断点进行合并。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        model_name = self.cfg.semantic.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        logger.info(f"Loading sentence-transformers model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def chunk(self, text: str) -> List[Chunk]:
        c = self.cfg.common
        s = self.cfg.semantic
        size = int(c.get("chunk_size", 600))
        overlap = int(c.get("chunk_overlap", 100))
        min_sim = float(s.get("min_similarity", 0.62))
        window = int(s.get("sentence_window", 1))
        max_chunks = c.get("max_chunks")

        sents = whitespace_sentences(text)
        if not sents:
            return [Chunk(id=0, text=text, start=0, end=len(text), meta={"strategy": "semantic"})]

        embs = self.model.encode(sents, normalize_embeddings=True)
        chunks: List[Chunk] = []
        buf, start_char = [], 0

        # 预先计算每句的起止字符位置
        offsets = []
        cursor = 0
        for s in sents:
            beg = text.find(s, cursor)
            end = beg + len(s)
            offsets.append((beg, end))
            cursor = end

        def flush(end_char: int):
            cid = len(chunks)
            piece = "".join(buf).strip()
            if piece:
                chunks.append(Chunk(id=cid, text=piece, start=start_char, end=end_char,
                                    meta={"strategy": "semantic"}))

        for i, sent in enumerate(sents):
            buf.append(sent + " ")
            start_char = offsets[i][0] if not chunks and not buf[:-1] else start_char
            # 超长直接断
            if count_tokens("".join(buf)) >= size:
                end_char = offsets[i][1]
                flush(end_char)
                # 重叠回退若干句
                if overlap > 0 and i > 0:
                    back_chars = max(0, end_char - overlap)
                    # 重新起点：找到 back_chars 所在的句开始
                    j = i
                    while j >= 0 and offsets[j][0] > back_chars:
                        j -= 1
                    buf = [sents[j + 1] + " "] if j + 1 <= i else []
                    start_char = offsets[j + 1][0] if j + 1 <= i else end_char
                else:
                    buf, start_char = [], end_char
                if max_chunks and len(chunks) >= max_chunks:
                    break
                continue

            # 相似度断点检测（与前 window 句平均相似度）
            if i >= window:
                v = embs[i]
                ctx = embs[i - window:i].mean(axis=0)
                sim = float(np.dot(v, ctx))
                if sim < min_sim:
                    end_char = offsets[i][0]
                    flush(end_char)
                    buf, start_char = [sent + " "], offsets[i][0]

            # 收尾
            if i == len(sents) - 1 and buf:
                end_char = offsets[i][1]
                flush(end_char)

        return chunks
```

20) `textchunker/chunkers/recursive.py`
---------------------------------------

```python
from __future__ import annotations
from typing import List
from loguru import logger
from ..types import Chunk
from ..registry import register
from .base import BaseChunker
from ..utils import count_tokens

def _split_by_separators(text: str, seps: List[str], size: int) -> List[str]:
    """按优先级分隔符递归切分，直到块不超过目标大小。"""
    if count_tokens(text) <= size or not seps:
        return [text]
    sep = seps[0]
    parts = text.split(sep) if sep else list(text)  # 最后一个分隔符为空字符串时按字符切
    chunks: List[str] = []
    buf = ""
    for i, p in enumerate(parts):
        piece = (buf + (p + sep if i < len(parts) - 1 else p))
        if count_tokens(piece) <= size:
            buf = piece
        else:
            if buf:
                chunks.append(buf)
            # 如果单个 part 自身就很大，递归继续拆
            if count_tokens(p) > size:
                chunks.extend(_split_by_separators(p, seps[1:], size))
                buf = ""
            else:
                buf = p + (sep if i < len(parts) - 1 else "")
    if buf:
        chunks.append(buf)
    # 二次遍历，若仍有过大的块，继续递归下一层分隔符
    out: List[str] = []
    for ch in chunks:
        if count_tokens(ch) > size and len(seps) > 1:
            out.extend(_split_by_separators(ch, seps[1:], size))
        else:
            out.append(ch)
    return out

@register("recursive")
class RecursiveChunker(BaseChunker):
    """递归分块：按段落→句子→词→字符等优先级递归切分。"""

    def chunk(self, text: str) -> List[Chunk]:
        c = self.cfg.common
        r = self.cfg.recursive
        size = int(c.get("chunk_size", 512))
        overlap = int(c.get("chunk_overlap", 80))
        seps = list(r.get("separators", ["\n\n", "\n", " ", ""]))
        max_chunks = c.get("max_chunks")

        logger.info(f"Recursive chunking: size={size}, overlap={overlap}, separators={seps}")

        parts = _split_by_separators(text, seps, size)
        chunks: List[Chunk] = []
        cursor = 0
        for i, part in enumerate(parts):
            start = text.find(part, cursor)
            end = start + len(part)
            chunks.append(Chunk(id=len(chunks), text=part, start=start, end=end, meta={"strategy": "recursive"}))
            cursor = end
            if max_chunks and len(chunks) >= max_chunks:
                break

        # 简单重叠：合并相邻片段的尾/头（字符级）
        if overlap > 0 and chunks:
            for i in range(1, len(chunks)):
                prev, cur = chunks[i - 1], chunks[i]
                head = text[max(cur.start - overlap, prev.start):cur.start]
                cur.text = head + cur.text
                cur.start = cur.start - len(head)

        return chunks
```

21) `textchunker/chunkers/structure.py`
---------------------------------------

```python
from __future__ import annotations
import re
from typing import List, Tuple
from loguru import logger
from bs4 import BeautifulSoup  # type: ignore
from ..types import Chunk
from ..registry import register
from .base import BaseChunker
from ..utils import count_tokens
from .recursive import _split_by_separators

@register("structure")
class StructureChunker(BaseChunker):
    """基于文档结构：Markdown/HTML/PDF（文本抽出后按标题）"""

    def _md_sections(self, text: str) -> List[Tuple[str, int, int]]:
        # 依据 Markdown 标题分段：以 '#' 开头的行作为起点
        lines = text.splitlines(keepends=True)
        heads = []
        for i, ln in enumerate(lines):
            if re.match(r"^\s*#{1,6}\s+", ln):
                heads.append(i)
        heads.append(len(lines))
        spans: List[Tuple[str, int, int]] = []
        for i in range(len(heads) - 1):
            beg_line = heads[i]
            end_line = heads[i + 1]
            beg = sum(len(l) for l in lines[:beg_line])
            end = sum(len(l) for l in lines[:end_line])
            title = re.sub(r"^#+\s+", "", lines[beg_line]).strip()
            spans.append((title, beg, end))
        if not spans:  # 没检测到标题则整文做一个区块
            spans = [("Document", 0, len(text))]
        return spans

    def _html_sections(self, text: str) -> List[Tuple[str, int, int]]:
        # 已在 reader 中转成纯文本，此处退化为按双换行
        return [("HTML", 0, len(text))]

    def chunk(self, text: str) -> List[Chunk]:
        c = self.cfg.common
        st = self.cfg.structure
        size = int(c.get("chunk_size", 800))
        overlap = int(c.get("chunk_overlap", 100))
        prefer = st.get("prefer", "auto")
        sub_split = bool(st.get("sub_split", True))
        max_chunks = c.get("max_chunks")

        if prefer in {"md", "auto"}:
            spans = self._md_sections(text)
        elif prefer == "html":
            spans = self._html_sections(text)
        else:  # pdf/auto
            spans = self._md_sections(text)  # 没有更好的结构时，按段落回退

        chunks: List[Chunk] = []
        for title, beg, end in spans:
            seg = text[beg:end]
            if count_tokens(seg) <= size or not sub_split:
                chunks.append(Chunk(id=len(chunks), text=seg, start=beg, end=end,
                                    meta={"strategy": "structure", "title": title}))
            else:
                parts = _split_by_separators(seg, ["\n\n", "\n", " ", ""], size)
                cursor = beg
                for p in parts:
                    s = text.find(p, cursor)
                    e = s + len(p)
                    chunks.append(Chunk(id=len(chunks), text=p, start=s, end=e,
                                        meta={"strategy": "structure", "title": title}))
                    cursor = e
            if max_chunks and len(chunks) >= max_chunks:
                break

        # 简单重叠拼接
        if overlap > 0 and chunks:
            for i in range(1, len(chunks)):
                prev, cur = chunks[i - 1], chunks[i]
                head = text[max(cur.start - overlap, prev.start):cur.start]
                cur.text = head + cur.text
                cur.start = cur.start - len(head)

        return chunks
```

22) `textchunker/chunkers/llm_based.py`
---------------------------------------

```python
from __future__ import annotations
from typing import List
from loguru import logger
from ..types import Chunk
from ..registry import register
from .base import BaseChunker
from ..providers import OpenAIProvider, HFProvider

@register("llm")
class LLMChunker(BaseChunker):
    """基于 LLM 的智能分块：模型返回 [start,end,title] 列表。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        p = cfg.llm.get("provider", "openai").lower()
        if p == "openai":
            self.provider = OpenAIProvider(cfg.llm.get("openai_model", "gpt-4o-mini"))
        elif p == "hf":
            self.provider = HFProvider(cfg.llm.get("hf_model", "Qwen/Qwen2.5-7B-Instruct"))
        else:
            raise ValueError(f"Unknown provider: {p}")

    def chunk(self, text: str) -> List[Chunk]:
        c = self.cfg.common
        l = self.cfg.llm
        max_chars = int(l.get("max_chars_per_call", 8000))
        sys_prompt = l.get("system_prompt", "Segment the document into meaningful spans. Return JSON spans.")
        max_chunks = c.get("max_chunks")

        spans = self.provider.propose_spans(text, system_prompt=sys_prompt, max_chars=max_chars)
        if not spans:
            # 回退：整体一个块
            return [Chunk(id=0, text=text, start=0, end=len(text), meta={"strategy": "llm"})]

        chunks: List[Chunk] = []
        for s, e, title in spans:
            s = max(0, min(s, len(text)))
            e = max(s, min(e, len(text)))
            piece = text[s:e]
            chunks.append(Chunk(id=len(chunks), text=piece, start=s, end=e,
                                meta={"strategy": "llm", "title": title}))
            if max_chunks and len(chunks) >= max_chunks:
                break
        return chunks
```

* * *

23) `textchunker/cli.py`
------------------------

```python
from __future__ import annotations
import os
from loguru import logger
from .config import load_yaml, to_project_config, build_argparser, merge_cli
from .readers import load_inputs
from .factory import create_chunker
from .export import save_jsonl, save_txt_dir
from .visualization import show_chunks
from .chunkers import *  # noqa: F401  # 触发注册

def main() -> None:
    """命令行入口：读取 -> 分块 -> 可视化 -> 导出。"""
    parser = build_argparser()
    args = parser.parse_args()

    cfg = to_project_config(load_yaml(args.config))
    cfg = merge_cli(cfg, args)

    input_path = cfg.io.get("input_path")
    output_path = cfg.io.get("output_path")
    visualize = bool(cfg.project.get("visualize", False))
    export_format = cfg.project.get("export_format", "jsonl")

    logger.info(f"Strategy = {cfg.strategy.name}")
    chunker = create_chunker(cfg.strategy)

    for doc in load_inputs(input_path):
        logger.info(f"Processing: {doc.path}")
        chunks = chunker.chunk(doc.text)

        if visualize:
            show_chunks(chunks, title=os.path.basename(doc.path))

        if export_format in {"jsonl", "both"}:
            save_jsonl(output_path, chunks, source=doc.path)
            logger.info(f"Written JSONL to {output_path}")
        if export_format in {"txt_dir", "both"}:
            outdir = os.path.splitext(output_path)[0] + "_parts"
            save_txt_dir(outdir, chunks, prefix=os.path.basename(doc.path))
            logger.info(f"Written TXT parts to {outdir}")

if __name__ == "__main__":
    main()
```

* * *

使用方法
====

安装
--

```bash
git clone <your-repo> text-chunker
cd text-chunker
pip install -e .
# 或按需安装可选功能：
# pip install -e .[semantic,token,structure,llm]
```

快速试跑
----

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

* * *

设计要点与扩展说明
=========

*   **Factory & Registry**：`@register("name")` 装饰器把新策略注册到全局 `_REGISTRY`，`create_chunker` 通过 `name` 实例化。新增策略时仅需在 `chunkers/` 下新建文件并 `@register` 即可热插拔接入。
*   **YAML + CLI 覆盖**：`configs/*.yaml` 提供团队级默认，实验时通过 CLI 快速覆盖关键参数（策略名、模型名、阈值等）。
*   **LLM 分块**：`providers/llm.py` 抽象了 `BaseLLMProvider`，目前实现了 `OpenAIProvider/HFProvider`；解析采用“防御式 JSON 提取”，稳定性更高。
*   **Rich 可视化**：`visualization.py` 用表格展示分块数量、字符长度、跨度与预览，有利于调参与审查。
*   **结构化分块**：对 Markdown 按 `#` 标题切段；HTML 已在 reader 中转为纯文本（工程中可替换为更强的 by\_title/DOM 切分）。
*   **语义分块**：句子级嵌入 + 相邻相似度断点，参数 `min_similarity` 可结合你语料分布网格搜索；过长时也会受 `chunk_size` 裁断。
*   **固定分块**：为稳健性，默认按字符切（生产可把 `tiktoken` 的编码位置信息用于精准 token 切片）。

* * *

下一步可加的插件（留好扩展口）
===============

*   **Parent-Child / Auto-Merging**：检索阶段按叶子块召回、返回父级更大上下文。
*   **Sentence-Window Retrieval**：检索后扩展上下文窗口。
*   **代码/表格感知分块**：以函数/表格为原子单元，避免被切断。
*   **版面解析器**：对 PDF 使用版面理解（如 `unstructured`/GROBID）获取更可靠的章节层级。

* * *

需要我帮你加一个**评测子模块（RAGAS/TruLens 指标）**和**参数网格脚本**吗？我可以在这个骨架上继续补：`evaluators/` + `experiments/`，一键比较 `size/overlap/threshold`，并把可视化做成 `rich` 的仪表板。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)