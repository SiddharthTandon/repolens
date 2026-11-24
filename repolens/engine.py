"""The pieces put together: refresh the index, retrieve, ask the model, check citations."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

from .citations import Citation, check_citations
from .config import Settings, index_path
from .embeddings import Embedder, make_embedder
from .indexer import IndexStats, index_repo
from .llm import ChatClient
from .prompting import build_messages
from .search import Hit, Searcher
from .store import Store


@dataclass
class Answer:
    question: str
    text: str
    hits: List[Hit]
    citations: List[Citation] = field(default_factory=list)

    @property
    def unsupported(self) -> List[Citation]:
        return [c for c in self.citations if not c.supported]


class Lens:
    def __init__(
        self,
        root: Path,
        settings: Optional[Settings] = None,
        embedder: Optional[Embedder] = None,
        embedder_choice: Optional[str] = None,
        chat: Optional[ChatClient] = None,
    ):
        self.root = Path(root).resolve()
        self.settings = settings or Settings()
        self.embedder = embedder if embedder is not None else make_embedder(
            self.settings, embedder_choice
        )
        self.store = Store(index_path(self.root))
        self._chat = chat
        self._searcher: Optional[Searcher] = None
        self._lock = threading.RLock()

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> Lens:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def chat(self) -> ChatClient:
        if self._chat is None:
            s = self.settings
            self._chat = ChatClient(s.base_url, s.api_key, s.model)
        return self._chat

    def refresh(self, progress: Optional[Callable[[str], None]] = None) -> IndexStats:
        with self._lock:
            stats = index_repo(self.root, self.store, self.embedder, progress)
            if stats.files_updated or stats.files_removed:
                self._searcher = None
            return stats

    @property
    def searcher(self) -> Searcher:
        with self._lock:
            if self._searcher is None:
                self._searcher = Searcher(self.store, self.embedder)
            return self._searcher

    def search(
        self, query: str, k: int = 8, mode: str = "hybrid", path_glob: Optional[str] = None
    ) -> List[Hit]:
        with self._lock:
            return self.searcher.search(query, k=k, mode=mode, path_glob=path_glob)

    def counts(self):
        with self._lock:
            return self.store.counts()

    def meta(self, key: str) -> Optional[str]:
        with self._lock:
            return self.store.get_meta(key)

    def ask(
        self,
        question: str,
        k: int = 8,
        mode: str = "hybrid",
        path_glob: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Answer:
        hits = self.search(question, k=k, mode=mode, path_glob=path_glob)
        text = self.chat.complete(build_messages(question, hits, history))
        return Answer(question, text, hits, check_citations(text, hits))

    def ask_stream(
        self,
        question: str,
        k: int = 8,
        mode: str = "hybrid",
        path_glob: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AnswerStream:
        hits = self.search(question, k=k, mode=mode, path_glob=path_glob)
        pieces = self.chat.stream(build_messages(question, hits, history))
        return AnswerStream(question, hits, pieces)


class AnswerStream:
    """Iterate to receive the answer as it is generated; `answer` holds the result after."""

    def __init__(self, question: str, hits: List[Hit], pieces: Iterator[str]):
        self.question = question
        self.hits = hits
        self._pieces = pieces
        self.answer: Optional[Answer] = None

    def __iter__(self) -> Iterator[str]:
        collected: List[str] = []
        for piece in self._pieces:
            collected.append(piece)
            yield piece
        text = "".join(collected)
        self.answer = Answer(self.question, text, self.hits, check_citations(text, self.hits))
