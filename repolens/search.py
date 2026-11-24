"""Hybrid retrieval: BM25 and embedding search, merged with reciprocal rank fusion."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .bm25 import BM25
from .chunking import Chunk
from .embeddings import Embedder
from .store import Store
from .textproc import tokens

CANDIDATES = 50
RRF_K = 60
MODES = ("hybrid", "lexical", "dense")


@dataclass
class Hit:
    chunk: Chunk
    score: float
    lexical_rank: Optional[int] = None
    dense_rank: Optional[int] = None


def rrf(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> List[Tuple[int, float]]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def lexical_text(chunk: Chunk) -> str:
    return f"{chunk.path} {chunk.symbol} {chunk.text}"


class Searcher:
    def __init__(self, store: Store, embedder: Optional[Embedder] = None):
        rows = store.all_chunks()
        self.chunks = [chunk for chunk, _ in rows]
        self.bm25 = BM25([tokens(lexical_text(c)) for c in self.chunks])
        self.embedder = embedder
        self.vectors: Optional[np.ndarray] = None
        if embedder is not None and rows and all(v is not None for _, v in rows):
            self.vectors = np.vstack([v for _, v in rows])

    @property
    def has_vectors(self) -> bool:
        return self.vectors is not None and self.embedder is not None

    def _lexical(self, query: str, allowed: Optional[np.ndarray]) -> List[int]:
        scores = self.bm25.scores(tokens(query))
        if allowed is not None:
            scores = np.where(allowed, scores, 0.0)
        order = np.argsort(-scores, kind="stable")[:CANDIDATES]
        return [int(i) for i in order if scores[i] > 0]

    def _dense(self, query: str, allowed: Optional[np.ndarray]) -> List[int]:
        assert self.vectors is not None and self.embedder is not None
        sims = self.vectors @ self.embedder.embed_query(query)
        if allowed is not None:
            sims = np.where(allowed, sims, -np.inf)
        order = np.argsort(-sims, kind="stable")[:CANDIDATES]
        return [int(i) for i in order if np.isfinite(sims[i])]

    def search(
        self,
        query: str,
        k: int = 8,
        mode: str = "hybrid",
        path_glob: Optional[str] = None,
    ) -> List[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not self.chunks:
            return []
        allowed = None
        if path_glob:
            allowed = np.array([fnmatch.fnmatch(c.path, path_glob) for c in self.chunks])
        if mode != "lexical" and not self.has_vectors:
            mode = "lexical"  # no embeddings in this index

        lexical = self._lexical(query, allowed) if mode in ("hybrid", "lexical") else []
        dense = self._dense(query, allowed) if mode in ("hybrid", "dense") else []
        lex_rank = {doc: r for r, doc in enumerate(lexical, start=1)}
        dense_rank = {doc: r for r, doc in enumerate(dense, start=1)}

        if mode == "lexical":
            fused = [(doc, 1.0 / (RRF_K + r)) for doc, r in lex_rank.items()]
        elif mode == "dense":
            fused = [(doc, 1.0 / (RRF_K + r)) for doc, r in dense_rank.items()]
        else:
            fused = rrf([lexical, dense])
        return [
            Hit(self.chunks[doc], score, lex_rank.get(doc), dense_rank.get(doc))
            for doc, score in fused[:k]
        ]
