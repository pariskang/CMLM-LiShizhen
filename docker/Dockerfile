# Base CUDA image for GPU-accelerated builds
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WORKDIR=/opt/workspace

WORKDIR $WORKDIR

# System deps
RUN apt-get update -qq && apt-get install -y -qq \
    git curl wget build-essential cmake ninja-build pkg-config libssl-dev \
    python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Python packages
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
      "torch>=2.2" --extra-index-url https://download.pytorch.org/whl/cu121 && \
    python3 -m pip install \
      "langchain>=0.2" "langgraph>=0.2" "openai>=1.35" "fastapi>=0.110" "uvicorn[standard]>=0.23" \
      "pydantic>=2.6" "pandas" "tiktoken" "huggingface_hub>=0.22" "bitsandbytes>=0.43" \
      "accelerate>=0.30" "transformers>=4.41" "deepspeed>=0.13" "unsloth" "sentencepiece" \
      "safetensors" "gradio>=4.29" "psutil" "shortuuid" "rich" "fire" "omegaconf"

# Project clones
RUN git clone https://github.com/limaoyi1/Auto-PPT /opt/auto-ppt && \
    git clone https://github.com/eosphoros-ai/DB-GPT /opt/db-gpt && \
    git clone https://github.com/sgl-project/sglang /opt/sglang && \
    git clone https://github.com/hiyouga/LLaMA-Factory /opt/llama-factory && \
    git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp && \
    git clone https://github.com/OpenDCAI/Paper2Any /opt/paper2any

# Build llama.cpp
RUN cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DGGML_CUDA=ON && \
    cmake --build /opt/llama.cpp/build -j$(nproc)

# Install editable packages
RUN python3 -m pip install -e /opt/sglang /opt/llama-factory

COPY scripts/download_models.py /opt/workspace/scripts/download_models.py
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

VOLUME ["/models", "/data", "/configs"]

EXPOSE 8000 7860

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
