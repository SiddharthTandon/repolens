"""Building and refreshing the index. Only files whose content changed are re-chunked."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .chunking import Chunk, chunk_file
from .embeddings import Embedder
from .files import discover, read_source
from .store import Store

FLUSH_CHUNKS = 256


@dataclass
class IndexStats:
    files_seen: int = 0
    files_updated: int = 0
    files_removed: int = 0
    chunks_written: int = 0
    seconds: float = 0.0


def embedding_text(chunk: Chunk) -> str:
    header = f"{chunk.path} {chunk.symbol}".strip()
    return f"{header}\n{chunk.text}"


def index_repo(
    root: Path,
    store: Store,
    embedder: Optional[Embedder],
    progress: Optional[Callable[[str], None]] = None,
) -> IndexStats:
    started = time.perf_counter()
    stats = IndexStats()
    embedder_id = embedder.id if embedder else "none"
    if store.get_meta("embedder") != embedder_id:
        store.clear()
        store.set_meta("embedder", embedder_id)

    known = store.file_shas()
    present = set()
    pending: List[Tuple[str, str, List[Chunk]]] = []
    pending_chunks = 0

    def flush() -> None:
        nonlocal pending, pending_chunks
        if not pending:
            return
        vectors = None
        if embedder is not None and pending_chunks:
            texts = [embedding_text(c) for _, _, chunks in pending for c in chunks]
            vectors = embedder.embed_documents(texts)
        offset = 0
        for rel, sha, chunks in pending:
            part = vectors[offset : offset + len(chunks)] if vectors is not None else None
            store.replace_file(rel, sha, chunks, part)
            offset += len(chunks)
            stats.chunks_written += len(chunks)
        store.commit()
        pending, pending_chunks = [], 0

    for rel in discover(root):
        src = read_source(root, rel)
        if src is None:
            continue
        stats.files_seen += 1
        present.add(rel)
        sha = hashlib.sha256(src.text.encode("utf-8")).hexdigest()
        if known.get(rel) == sha:
            continue
        chunks = chunk_file(src)
        pending.append((rel, sha, chunks))
        pending_chunks += len(chunks)
        stats.files_updated += 1
        if progress:
            progress(rel)
        if pending_chunks >= FLUSH_CHUNKS:
            flush()
    flush()

    for rel in set(known) - present:
        store.remove_file(rel)
        stats.files_removed += 1
    store.set_meta("root", str(root.resolve()))
    store.commit()
    stats.seconds = time.perf_counter() - started
    return stats
