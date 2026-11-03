#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest tests/ -q --cov=textchunker --cov-report=term-missing --cov-fail-under=60
