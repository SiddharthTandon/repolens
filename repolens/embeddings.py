"""Dense embeddings: local (fastembed, ONNX, no GPU needed) or any OpenAI-compatible API."""

from __future__ import annotations

from typing import List, Optional, Sequence

import httpx
import numpy as np

from .config import Settings

DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_API_MODEL = "text-embedding-3-small"


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class Embedder:
    """Turns chunk texts and questions into unit vectors."""

    #: Stored in the index; a different id means the vectors must be rebuilt.
    id: str = "base"

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class FastEmbedEmbedder(Embedder):
    def __init__(self, model: str = DEFAULT_LOCAL_MODEL):
        from fastembed import TextEmbedding  # optional dependency

        self.model_name = model
        self.id = f"fastembed:{model}"
        self._model = TextEmbedding(model_name=model)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return normalize(np.array(list(self._model.embed(list(texts)))))

    def embed_query(self, text: str) -> np.ndarray:
        query_embed = getattr(self._model, "query_embed", None)
        if query_embed is not None:
            return normalize(np.array(list(query_embed([text]))))[0]
        return self.embed_documents([text])[0]


class OpenAIEmbedder(Embedder):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = DEFAULT_API_MODEL,
        batch_size: int = 64,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        self.model_name = model
        self.id = f"openai:{model}"
        self.batch_size = batch_size
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120,
            transport=transport,
        )

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        rows: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            payload = {"model": self.model_name, "input": batch}
            response = self._client.post("/embeddings", json=payload)
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda item: item["index"])
            rows.extend(item["embedding"] for item in data)
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return normalize(np.array(rows))


def make_embedder(settings: Settings, choice: Optional[str] = None) -> Optional[Embedder]:
    """Build the embedder named by `choice` (or the settings).

    ``none`` means keyword search only. ``auto`` uses local embeddings when
    fastembed is installed and falls back to keyword search otherwise.
    """
    choice = (choice or settings.embedder or "auto").lower()
    if choice == "none":
        return None
    if choice in ("local", "fastembed"):
        return FastEmbedEmbedder(settings.embed_model or DEFAULT_LOCAL_MODEL)
    if choice in ("openai", "api"):
        return OpenAIEmbedder(
            settings.base_url, settings.api_key, settings.embed_model or DEFAULT_API_MODEL
        )
    if choice == "auto":
        try:
            import fastembed  # noqa: F401
        except ImportError:
            return None
        return FastEmbedEmbedder(settings.embed_model or DEFAULT_LOCAL_MODEL)
    raise ValueError(f"unknown embedder {choice!r} (use auto, local, openai or none)")
