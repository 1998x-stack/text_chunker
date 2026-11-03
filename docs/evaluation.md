# Evidence-grounded offline retrieval evaluation

**Purpose:** distinguish whether an annotated answer span survives chunking from whether a ranker retrieves it. This evaluates retrieval candidates only; it does not evaluate generated answers, hallucinations, or live vector indices.

## Contract

Each question has a unique ID, query and one or more `EvidenceAnchor(source_uri, Span)` records. Offsets are half-open Python Unicode indices in canonical `DocumentIR.text`. In JSON, a unique exact quote is resolved to offsets; absent/duplicate quotes, unknown sources and invalid spans fail fast. For HTML/PDF, evidence spans refer to canonical extracted text, **not raw markup or PDF bytes**.

A chunk is relevant only when it comes from the same source URI and **fully contains** one annotated evidence span. Partial overlaps do not count. A result may be valid for chunk integrity yet have poor oracle evidence coverage because a gold answer crosses a boundary.

## Metrics

For N questions and positive rank cutoff k:
- **Recall@k:** fraction of queries with at least one relevant chunk in top k (query-level hit rate).
- **MRR@k:** average reciprocal rank of first relevant chunk; zero when absent.
- **Binary nDCG@k:** sum of 1/log2(rank+1) over relevant candidates, normalized by the ideal DCG given eligible relevant chunks.
- **Oracle recall:** fraction of questions with any fully evidence-containing chunk anywhere in the index, regardless of ranking.
- **Independent source correctness:** complete unique span union, source substring equality and token budget constraints, evaluated before retrieval.

The ranking baseline is lowercase ASCII word and CJK character lexical overlap, with deterministic tie-breaking. A caller may explicitly inject a different scorer; no external embeddings are used in the offline fixture.

## Run

```bash
python -m textchunker.evaluation.benchmark \
  --dataset benchmarks/datasets/golden_v1.json \
  --strategies fixed recursive hybrid --tokenizer char \
  --chunk-size 128 --chunk-overlap 16 --k 3 \
  --output out/retrieval_benchmark.json
```

Output is an atomic JSON report with query rankings, source spans and metric denominators. Wall-clock elapsed time varies. The five-question handcrafted dataset verifies evaluator logic only: **do not claim production search improvement or tune models to this tiny sample**.

For real research, create independently annotated held-out corpora, record document rights and alternative acceptable evidence, validate human adjudication and report confidence intervals.
