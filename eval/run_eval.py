"""Retrieval benchmark: does the right file show up in the top results?

Usage:
    python eval/run_eval.py /path/to/requests eval/requests_questions.jsonl --embedder local

For each question the benchmark records the rank of the first retrieved chunk
that comes from one of the expected files, then reports recall@k and mean
reciprocal rank (MRR) for keyword-only, embedding-only and hybrid search.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from repolens.config import Settings, index_path
from repolens.embeddings import make_embedder
from repolens.indexer import index_repo
from repolens.search import Searcher
from repolens.store import Store

KS = (1, 3, 5, 10)


def first_relevant_rank(searcher: Searcher, question: dict, mode: str) -> Optional[int]:
    hits = searcher.search(question["question"], k=max(KS), mode=mode)
    for rank, hit in enumerate(hits, start=1):
        if hit.chunk.path in question["files"]:
            return rank
    return None


def summarize(ranks: List[Optional[int]]) -> Dict[str, float]:
    n = len(ranks)
    out = {f"recall@{k}": sum(1 for r in ranks if r and r <= k) / n for k in KS}
    out["mrr"] = sum(1.0 / r for r in ranks if r) / n
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo", type=Path)
    parser.add_argument("questions", type=Path)
    parser.add_argument("--embedder", default="local", help="local, openai or none")
    parser.add_argument("--misses", action="store_true", help="list questions hybrid missed")
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.questions.read_text().splitlines() if line]
    embedder = make_embedder(Settings(), args.embedder)
    store = Store(index_path(args.repo))
    started = time.perf_counter()
    stats = index_repo(args.repo, store, embedder)
    files, chunks = store.counts()
    print(f"indexed {files} files / {chunks} chunks in {time.perf_counter() - started:.1f}s "
          f"({stats.files_updated} re-embedded) with {embedder.id if embedder else 'no embedder'}")

    searcher = Searcher(store, embedder)
    modes = ["lexical", "dense", "hybrid"] if searcher.has_vectors else ["lexical"]
    print(f"\n{len(questions)} questions\n")
    header = "| search | " + " | ".join(f"recall@{k}" for k in KS) + " | MRR |"
    print(header)
    print("|" + "---|" * (len(KS) + 2))
    results = {}
    for mode in modes:
        ranks = [first_relevant_rank(searcher, q, mode) for q in questions]
        results[mode] = ranks
        s = summarize(ranks)
        cells = " | ".join(f"{s[f'recall@{k}']:.2f}" for k in KS)
        print(f"| {mode} | {cells} | {s['mrr']:.3f} |")

    if args.misses:
        best = "hybrid" if "hybrid" in results else "lexical"
        print(f"\nnot in the top 5 ({best}):")
        for q, rank in zip(questions, results[best]):
            if not rank or rank > 5:
                print(f"  rank {rank or '>10'}: {q['question']}")
    store.close()


if __name__ == "__main__":
    main()
