#!/usr/bin/env bash
set -euo pipefail
source ~/.zshrc
cd "$(dirname "$0")/.."

echo "=== Running Recursive Ablation ==="
python -m textchunker.experiments.recursive_ablation \
    --max-docs 32 --chunk-sizes 256 512 800 --chunk-overlaps 0 50 100 \
    --save-json results/recursive_ablation.json

echo ""
echo "=== Running Strategy Comparison ==="
python -m textchunker.experiments.strategy_comparison \
    --strategies fixed recursive --max-docs 32 \
    --save-json results/strategy_comparison.json

echo "Experiments complete."
