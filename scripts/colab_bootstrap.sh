#!/usr/bin/env bash
set -euo pipefail

# Bootstrap script to install dependencies for LangChain/Graph, DB-GPT, SGLang,
# LLaMA-Factory, llama.cpp, DeepSpeed, Unsloth, Auto-PPT, Paper2Any and download
# requested models when credentials are provided. Suitable for Google Colab.

# Usage:
#   HF_TOKEN=... bash scripts/colab_bootstrap.sh
#   HF_TOKEN=... DOWNLOAD_MODELS=1 bash scripts/colab_bootstrap.sh
# Environment variables:
#   HF_TOKEN           Hugging Face token (mandatory for gated models)
#   DOWNLOAD_MODELS    If set to 1, downloads all configured models
#   MODEL_LIST         Optional comma-separated overrides for the model list
#   TARGET_DIR         Base directory for checkouts (default: /content/workspace)

TARGET_DIR="${TARGET_DIR:-/content/workspace}"
CODE_DIR="$TARGET_DIR/src"
MODELS_DIR="$TARGET_DIR/models"
PYTHON_BIN="python3"

mkdir -p "$CODE_DIR" "$MODELS_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python3 is required." >&2
  exit 1
fi

# Install system packages
apt-get update -qq
apt-get install -y -qq git build-essential cmake ninja-build wget curl pkg-config libssl-dev

# Upgrade pip and install core Python deps
$PYTHON_BIN -m pip install --upgrade pip
$PYTHON_BIN -m pip install "torch>=2.2" --extra-index-url https://download.pytorch.org/whl/cu121
$PYTHON_BIN -m pip install \
  "langchain>=0.2" "langgraph>=0.2" "openai>=1.35" "fastapi>=0.110" "uvicorn[standard]>=0.23" \
  "pydantic>=2.6" "pandas" "tiktoken" "huggingface_hub>=0.22" "bitsandbytes>=0.43" \
  "accelerate>=0.30" "transformers>=4.41" "deepspeed>=0.13" "unsloth" "sentencepiece" \
  "safetensors" "gradio>=4.29" "psutil" "shortuuid" "rich" "fire" "omegaconf"

# Clone or update third-party projects
clone_or_update() {
  local repo_url=$1
  local dest=$2
  if [ -d "$dest/.git" ]; then
    git -C "$dest" pull --ff-only
  else
    git clone "$repo_url" "$dest"
  fi
}

clone_or_update https://github.com/limaoyi1/Auto-PPT "$CODE_DIR/auto-ppt"
clone_or_update https://github.com/eosphoros-ai/DB-GPT "$CODE_DIR/db-gpt"
clone_or_update https://github.com/sgl-project/sglang "$CODE_DIR/sglang"
clone_or_update https://github.com/hiyouga/LLaMA-Factory "$CODE_DIR/llama-factory"
clone_or_update https://github.com/ggerganov/llama.cpp "$CODE_DIR/llama.cpp"
clone_or_update https://github.com/OpenDCAI/Paper2Any "$CODE_DIR/paper2any"

# Build llama.cpp (GPU build when CUDA is available)
pushd "$CODE_DIR/llama.cpp" >/dev/null
cmake -B build -DLLAMA_CUBLAS=ON
cmake --build build -j"$(nproc)"
popd >/dev/null

# Install project editable dependencies where applicable
$PYTHON_BIN -m pip install -e "$CODE_DIR/sglang" "$CODE_DIR/llama-factory"

# Optional: download models via helper script
if [[ "${DOWNLOAD_MODELS:-0}" == "1" ]]; then
  HF_TOKEN="${HF_TOKEN:-}" MODEL_LIST="${MODEL_LIST:-}" TARGET_DIR="$MODELS_DIR" \
    $PYTHON_BIN scripts/download_models.py
fi

echo "Setup complete. Workdir: $TARGET_DIR"
