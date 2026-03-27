#!/usr/bin/env bash
set -euo pipefail
source ~/.zshrc
cd "$(dirname "$0")/.."

TAG="${1:-textchunker:latest}"
echo "Building Docker image: $TAG"
docker build -t "$TAG" .
echo "Build complete: $TAG"
