#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Correctness checks ==="
bash scripts/lint.sh
bash scripts/test.sh
echo "All required checks passed."
