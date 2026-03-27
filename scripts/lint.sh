#!/usr/bin/env bash
set -euo pipefail
source ~/.zshrc
cd "$(dirname "$0")/.."
echo "=== flake8 ==="
python -m flake8 textchunker/ tests/ --max-line-length=100 --extend-ignore=E203,W503
echo "=== mypy ==="
python -m mypy textchunker/ --ignore-missing-imports --no-error-summary || true
echo "=== isort check ==="
python -m isort --check-only --diff textchunker/ tests/ || true
echo "Lint complete."
