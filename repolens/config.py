"""Settings, read from the environment so the same commands work against any backend."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

INDEX_DIR = ".repolens"
INDEX_FILE = "index.db"


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


@dataclass
class Settings:
    """Where the chat model and the embedder live.

    Any OpenAI-compatible server works: OpenAI, Groq, OpenRouter, Together,
    vLLM, LM Studio or a local Ollama (the default, so nothing needs a key).
    """

    base_url: str = field(
        default_factory=lambda: _env(
            "REPOLENS_BASE_URL", "OPENAI_BASE_URL", default="http://localhost:11434/v1"
        )
    )
    api_key: str = field(
        default_factory=lambda: _env("REPOLENS_API_KEY", "OPENAI_API_KEY", default="ollama")
    )
    model: str = field(
        default_factory=lambda: _env("REPOLENS_MODEL", default="qwen2.5-coder:7b")
    )
    # auto: local embeddings when fastembed is installed, otherwise keyword search only
    embedder: str = field(default_factory=lambda: _env("REPOLENS_EMBEDDER", default="auto"))
    embed_model: str = field(default_factory=lambda: _env("REPOLENS_EMBED_MODEL"))


def index_path(root: Path) -> Path:
    return root / INDEX_DIR / INDEX_FILE
