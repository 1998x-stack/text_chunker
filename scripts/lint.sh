#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Keep pre-existing style debt separate from correctness gating.
# Do not swallow failed checks with '|| true'.
python -m compileall -q textchunker tests
python -m flake8 textchunker/ tests/ --select=E9,F63,F7,F82
