#!/usr/bin/env bash
set -euo pipefail
source ~/.zshrc
cd "$(dirname "$0")/.."

echo "=== CI Pipeline ==="
echo "Step 1/3: Lint"
bash scripts/lint.sh

echo ""
echo "Step 2/3: Test"
bash scripts/test.sh

echo ""
echo "Step 3/3: Benchmark (if sample.txt exists)"
bash scripts/benchmark.sh || true

echo ""
echo "CI Pipeline complete."
