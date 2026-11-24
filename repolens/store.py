"""The on-disk index: one SQLite file with chunks, their vectors and file hashes."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .chunking import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, sha TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    language TEXT NOT NULL,
    text TEXT NOT NULL,
    vector BLOB
);
CREATE INDEX IF NOT EXISTS chunks_by_path ON chunks(path);
"""


class Store:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = db_path
        # The web UI serves requests from worker threads; Lens serializes access.
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.executescript(SCHEMA)

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def commit(self) -> None:
        self.conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def clear(self) -> None:
        self.conn.execute("DELETE FROM chunks")
        self.conn.execute("DELETE FROM files")

    def file_shas(self) -> Dict[str, str]:
        return dict(self.conn.execute("SELECT path, sha FROM files"))

    def replace_file(
        self,
        path: str,
        sha: str,
        chunks: Sequence[Chunk],
        vectors: Optional[np.ndarray] = None,
    ) -> None:
        self.conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
        rows = []
        for i, chunk in enumerate(chunks):
            blob = None
            if vectors is not None and len(vectors):
                blob = np.asarray(vectors[i], dtype=np.float32).tobytes()
            rows.append(
                (chunk.path, chunk.start_line, chunk.end_line, chunk.symbol,
                 chunk.language, chunk.text, blob)
            )
        self.conn.executemany(
            "INSERT INTO chunks (path, start_line, end_line, symbol, language, text, vector) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self.conn.execute(
            "INSERT INTO files (path, sha) VALUES (?, ?) "
            "ON CONFLICT(path) DO UPDATE SET sha = excluded.sha",
            (path, sha),
        )

    def remove_file(self, path: str) -> None:
        self.conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
        self.conn.execute("DELETE FROM files WHERE path = ?", (path,))

    def all_chunks(self) -> List[Tuple[Chunk, Optional[np.ndarray]]]:
        out = []
        cursor = self.conn.execute(
            "SELECT path, start_line, end_line, symbol, language, text, vector "
            "FROM chunks ORDER BY path, start_line"
        )
        for path, start, end, symbol, language, text, blob in cursor:
            vector = np.frombuffer(blob, dtype=np.float32) if blob else None
            out.append((Chunk(path, start, end, symbol, language, text), vector))
        return out

    def counts(self) -> Tuple[int, int]:
        files = self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return files, chunks
