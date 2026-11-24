from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path

import numpy as np
import pytest

from repolens.embeddings import Embedder, normalize
from repolens.textproc import tokens


class HashingEmbedder(Embedder):
    """Deterministic bag-of-words vectors: close enough to semantic for tests, no model needed."""

    id = "test:hashing"

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokens(text):
                digest = hashlib.md5(token.encode()).digest()
                out[row, int.from_bytes(digest[:4], "little") % self.dim] += 1.0
        return normalize(out)


SAMPLE = {
    "app/retry.py": '''
        """Retry helpers for outgoing HTTP calls."""
        import random
        import time

        DEFAULT_ATTEMPTS = 5


        def backoff_delay(attempt, base=0.5, cap=30.0):
            """Exponential backoff with full jitter."""
            return random.uniform(0, min(cap, base * 2 ** attempt))


        class RetryPolicy:
            def __init__(self, attempts=DEFAULT_ATTEMPTS):
                self.attempts = attempts

            def run(self, func):
                for attempt in range(self.attempts):
                    try:
                        return func()
                    except ConnectionError:
                        time.sleep(backoff_delay(attempt))
                raise RuntimeError("gave up")
    ''',
    "app/auth.py": '''
        import base64


        def basic_auth_header(user, password):
            token = base64.b64encode(f"{user}:{password}".encode()).decode()
            return {"Authorization": "Basic " + token}


        def strip_auth_on_redirect(old_host, new_host, headers):
            if old_host != new_host:
                headers.pop("Authorization", None)
            return headers
    ''',
    "web/server.js": '''
        import express from "express";

        export function createServer(port) {
          const app = express();
          app.get("/health", (req, res) => res.json({ ok: true }));
          return app.listen(port);
        }
    ''',
    "README.md": '''
        # Sample service

        ## Retries
        Outgoing calls are retried with exponential backoff.
    ''',
}


def write_repo(root: Path, files=SAMPLE) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip())
    return root


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return write_repo(tmp_path / "repo")


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()
