# Offline retrieval regression baseline

```bash
python -m textchunker.evaluation.benchmark \
  --dataset benchmarks/datasets/golden_v1.json \
  --strategies fixed recursive hybrid --tokenizer char \
  --output out/retrieval_benchmark.json
```

Five handcrafted questions test the scoring and provenance contract. No
production-representative model accuracy, performance superiority, or real
external model integration is implied. Do not invent historical baseline
numbers; generate inspectable reports with the command above.
