#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m textchunker.evaluation.benchmark \
  --dataset benchmarks/datasets/golden_v1.json \
  --strategies fixed recursive hybrid --tokenizer char \
  --chunk-size 128 --chunk-overlap 16 --k 3 \
  --output out/retrieval_benchmark.json
