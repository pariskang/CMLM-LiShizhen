# Multi-project LLM/Agent Environment

This guide shows how to bootstrap a GPU-enabled environment (Colab or Docker) that integrates LangChain, LangGraph, Auto-PPT, DB-GPT, SGLang, LLaMA-Factory, llama.cpp, DeepSpeed, Unsloth, and Paper2Any. It also includes a helper to pre-download large models (requires appropriate licenses and Hugging Face tokens).

## Colab setup

```bash
# Clone this repo inside Colab
!git clone https://github.com/<your-org>/CMLM-LiShizhen.git
%cd CMLM-LiShizhen

# Install dependencies and optionally download models
!HF_TOKEN="<hf_token>" DOWNLOAD_MODELS=0 bash scripts/colab_bootstrap.sh
# Set DOWNLOAD_MODELS=1 to prefetch models to /content/workspace/models
```

Key outputs:
- Code checkouts: `/content/workspace/src`
- Models: `/content/workspace/models`
- llama.cpp GPU build: `/content/workspace/src/llama.cpp/build/bin/llama-cli`

You can run LangGraph/LangChain apps with your own scripts, or launch DB-GPT/Gradio services from their respective repos.

## Docker image build

```bash
# From repo root
docker build -t llm-suite:latest -f docker/Dockerfile .

# Run with GPU access and optional model download
docker run --gpus all --rm -it \
  -e HF_TOKEN="<hf_token>" -e DOWNLOAD_MODELS=0 \
  -v $(pwd)/models:/models -v $(pwd)/data:/data -v $(pwd)/configs:/configs \
  -p 8000:8000 -p 7860:7860 \
  llm-suite:latest bash
```

To pre-download models inside the container on first start:

```bash
docker run --gpus all --rm -it \
  -e HF_TOKEN="<hf_token>" -e DOWNLOAD_MODELS=1 \
  -v $(pwd)/models:/models \
  llm-suite:latest
```

## Model download helper

`scripts/download_models.py` uses `huggingface_hub.snapshot_download` to fetch all configured repos. Override the list with `MODEL_LIST="org/model1,org/model2"` or edit `DEFAULT_MODELS` inside the script. Some requested model IDs (qwen-3-next, gpt-oss-20b/120b, minimax-m2) may be gated or under alternative names; placeholders are provided and should be replaced with the exact repo IDs you have permission to access.

## Notes and caveats

- GPU with sufficient VRAM is required for the listed models.
- Hugging Face tokens are mandatory for gated/commercial checkpoints; set `HF_TOKEN` accordingly.
- The Dockerfile installs CUDA 12.1 build tooling; adjust the base image if your host uses a different driver/toolkit.
- Projects are installed in editable mode where feasible to ease customization; refer to each upstream README for service startup.
