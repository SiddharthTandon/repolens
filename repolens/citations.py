"""Checking that the citations in an answer point at code the model was actually shown."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

from .search import Hit

_CITATION = re.compile(r"\[([^\[\]\s]+?):(\d+)(?:\s*-\s*(\d+))?\]")


@dataclass(frozen=True)
class Citation:
    path: str
    start: int
    end: int
    supported: bool

    @property
    def location(self) -> str:
        return f"{self.path}:{self.start}-{self.end}"


def check_citations(answer: str, hits: Sequence[Hit]) -> List[Citation]:
    """Every distinct citation in `answer`, marked supported when it overlaps a retrieved chunk."""
    seen = set()
    out: List[Citation] = []
    for match in _CITATION.finditer(answer):
        path = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3) or start)
        if end < start:
            start, end = end, start
        key = (path, start, end)
        if key in seen:
            continue
        seen.add(key)
        supported = any(
            hit.chunk.path == path
            and start <= hit.chunk.end_line
            and end >= hit.chunk.start_line
            for hit in hits
        )
        out.append(Citation(path, start, end, supported))
    return out
