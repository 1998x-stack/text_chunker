#!/usr/bin/env bash
set -euo pipefail
source ~/.zshrc
cd "$(dirname "$0")/.."

echo "=== Benchmarking all strategies ==="
for strategy in fixed recursive structure; do
    echo "--- $strategy ---"
    if [ -f sample.txt ]; then
        python -m textchunker.cli --config configs/default.yaml --strategy "$strategy" --input sample.txt --stats 2>&1 | tail -20
    else
        echo "No sample.txt found. Create one to run benchmarks."
    fi
done
echo "Benchmark complete."
