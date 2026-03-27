#!/usr/bin/env bash
set -euo pipefail
source ~/.zshrc
cd "$(dirname "$0")/.."
python -m isort textchunker/ tests/
python -m black textchunker/ tests/ --line-length 100
echo "Formatting complete."
