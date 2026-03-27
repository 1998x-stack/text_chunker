#!/usr/bin/env bash
set -euo pipefail
source ~/.zshrc
cd "$(dirname "$0")/.."
python -m pytest tests/ -v --cov=textchunker --cov-report=term-missing --cov-fail-under=60
