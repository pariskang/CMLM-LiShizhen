#!/usr/bin/env bash
set -euo pipefail

# Optional model prefetch when HF_TOKEN is provided
if [[ "${DOWNLOAD_MODELS:-0}" == "1" ]]; then
  HF_TOKEN="${HF_TOKEN:-}" TARGET_DIR="${TARGET_DIR:-/models}" python3 /opt/workspace/scripts/download_models.py
fi

# If CMD starts with an option, default to bash
if [[ "$#" -eq 0 ]]; then
  exec bash
else
  exec "$@"
fi
