#!/usr/bin/env bash
set -euo pipefail
source ~/.zshrc
cd "$(dirname "$0")/.."
pip install -e ".[all,dev]"
echo "Installation complete."
