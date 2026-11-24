"""Okapi BM25 over tokenized chunks, with an inverted index."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np


class BM25:
    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs and self.doc_len.sum() else 1.0
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for doc_id, doc in enumerate(docs):
            for term, freq in Counter(doc).items():
                self.postings[term].append((doc_id, freq))
        self.idf = {
            term: math.log(1.0 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }

    def scores(self, query: Sequence[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float64)
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len / self.avgdl)
        for term in set(query):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            ids = np.fromiter((d for d, _ in postings), dtype=np.int64, count=len(postings))
            tf = np.fromiter((f for _, f in postings), dtype=np.float64, count=len(postings))
            scores[ids] += idf * tf * (self.k1 + 1.0) / (tf + norm[ids])
        return scores

    def top(self, query: Sequence[str], k: int) -> List[Tuple[int, float]]:
        if not self.n_docs:
            return []
        scores = self.scores(query)
        k = min(k, self.n_docs)
        best = np.argpartition(-scores, k - 1)[:k]
        ranked = sorted(best, key=lambda i: (-scores[i], i))
        return [(int(i), float(scores[i])) for i in ranked if scores[i] > 0]
