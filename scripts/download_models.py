#!/usr/bin/env python3
"""Download curated models for offline inference and fine-tuning.

Env vars:
  HF_TOKEN: optional auth token for gated models.
  MODEL_LIST: optional comma-separated model ids overriding defaults.
  TARGET_DIR: destination directory (default: ./models).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List

from huggingface_hub import snapshot_download

DEFAULT_MODELS: List[str] = [
    "Qwen/Qwen2.5-72B-Instruct",  # surrogate for qwen-3-next
    "openbmb/MiniCPM-V-2_6",      # MiniCPM-V 2.6
    "Qwen/Qwen2.5-VL-7B-Instruct",  # Qwen2.5-VL
    "OpenGVLab/InternVL2_5-20B",  # placeholder for gpt-oss-20b
    "OpenGVLab/InternVL2_5-76B",  # placeholder for gpt-oss-120b scale
    "minimax/minimax-m2-chat"      # minimax-m2
]

@dataclass
class DownloadConfig:
    model_ids: List[str]
    target_dir: str
    token: str | None


def parse_models(raw: str | None) -> List[str]:
    if not raw:
        return DEFAULT_MODELS
    return [item.strip() for item in raw.split(",") if item.strip()]


def download_all(cfg: DownloadConfig) -> None:
    for model_id in cfg.model_ids:
        print(f"\n=== Downloading {model_id} ===")
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=os.path.join(cfg.target_dir, model_id.replace("/", "_")),
                local_dir_use_symlinks=False,
                token=cfg.token,
                resume_download=True,
                max_workers=8,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to download {model_id}: {exc}")


def main() -> None:
    target_dir = os.environ.get("TARGET_DIR", os.path.join(os.getcwd(), "models"))
    model_list = parse_models(os.environ.get("MODEL_LIST"))
    token = os.environ.get("HF_TOKEN")

    os.makedirs(target_dir, exist_ok=True)

    cfg = DownloadConfig(model_ids=model_list, target_dir=target_dir, token=token)
    download_all(cfg)


if __name__ == "__main__":
    main()
