# V0.5–V0.6 — Reliability and evidence-grounded evaluation

Status: implementation included in the consolidated PR #6. Real provider and Docling/OCR E2E still require separate external validation.

## Code boundaries

```text
source → canonical DocumentIR + SourceMap
  → fixed/recursive/structure/hybrid/semantic/LLM
  → independent correctness & structure evaluator → ChunkingResult
  → optional RAG adapters / inspectable evidence evaluation report
```

**Docling:** `parse_docling(native_doc, uri)` visits native document elements in reading order, extracting item text and preserving available `self_ref`, page number and bbox with coordinate origin. Source text is canonical extraction; **raw PDF source character spans are not inferred**. Blank/non-text item refs are recorded in metadata. `parse_docling_file(path)` and CLI `--pdf-parser docling` are opt-in heavy conversion paths. CI uses native-object mocks, not live Docling conversion/OCR. Default PDF ingestion remains pypdf.

**Semantic:** `run_chunking(..., embedding_provider=provider)` injects a mock or real backend. Returned vectors must have matching, finite, nonzero dimensions and are normalized before cosine comparisons. Batch size, model-call limit, and optional per-instance cache are configurable. The default `on_embedding_error: error` fails closed; `recursive` explicitly recovers deterministic source spans. Embedding vectors are never trusted without validation.

**LLM:** Outputs are untrusted JSON, parsed with strict integer offsets, bounded response size and proposal count. Contiguous source coverage is mandatory. `max_provider_calls`, `max_total_prompt_chars`, `retry_attempts` and `timeout_seconds` bound work. `on_failure: fallback` preserves all source windows via a deterministic segmenter and annotates fallback; `error` raises. No model/API calls are made by the offline benchmark.

**Evaluation:** verified canonical source spans and unique evidence quotes define relevance. Binary Recall@k, MRR, nDCG and oracle evidence containment are reported separately from chunk validity and elapsed time. The deterministic default lexical ranker is a testing baseline, not a model-quality claim. See [evaluation methodology](evaluation.md).

**Adapters:** optional `to_langchain_documents` and `to_llamaindex_nodes` provide stable IDs, text, source URI and offsets. Their test doubles validate mapping, not external SDK integration. `contextualize_chunk` prefixes heading context separately and never mutates source-exact `ChunkRecord.text`.

### Not claimed

Real provider uptime/cost, hosted-model quality, real OCR/complex PDF layout fidelity, character glyph coordinates or statistically significant RAG retrieval gains. The five-question corpus is an integrity regression fixture, not a production benchmark.
