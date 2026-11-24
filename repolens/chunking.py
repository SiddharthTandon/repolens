"""Splitting source files into chunks that follow the structure of the code.

Python files are split along functions, classes and methods with the `ast`
module, so a chunk is usually one whole definition and carries its qualified
name (``Session.request``). Everything else is split into overlapping line
windows, labelled with the first definition or heading found in the window.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .files import SourceFile

MAX_LINES = 80
OVERLAP = 10

_SYMBOL = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:static\s+)?"
    r"(?:def|class|function|func|fn|interface|struct|enum|trait|impl|module|object|type)\s+"
    r"([A-Za-z_$][\w$]*)"
)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")


@dataclass(frozen=True)
class Chunk:
    path: str
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    symbol: str
    language: str
    text: str

    @property
    def location(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


Span = Tuple[int, int, str]  # first line, last line (1-based, inclusive), symbol


def chunk_file(src: SourceFile, max_lines: int = MAX_LINES, overlap: int = OVERLAP) -> List[Chunk]:
    lines = src.text.splitlines()
    if not any(line.strip() for line in lines):
        return []
    spans: Optional[List[Span]] = None
    if src.language == "python":
        try:
            spans = _python_spans(src.text, len(lines), max_lines)
        except (SyntaxError, ValueError, RecursionError):
            spans = None
    if spans is None:
        spans = [(1, len(lines), "")]

    chunks = []
    for start, end, symbol in spans:
        for w_start, w_end in _windows(start, end, max_lines, overlap):
            body = "\n".join(lines[w_start - 1 : w_end])
            if not body.strip():
                continue
            label = symbol or _guess_symbol(lines[w_start - 1 : w_end], src.language)
            chunks.append(Chunk(src.path, w_start, w_end, label, src.language, body))
    return chunks


def _windows(start: int, end: int, size: int, overlap: int) -> List[Tuple[int, int]]:
    if end - start + 1 <= size:
        return [(start, end)]
    step = max(1, size - overlap)
    out = []
    first = start
    while True:
        last = min(end, first + size - 1)
        out.append((first, last))
        if last >= end:
            return out
        first += step


def _first_line(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno] + [d.lineno for d in decorators])


def _python_spans(text: str, n_lines: int, max_lines: int) -> List[Span]:
    tree = ast.parse(text)
    spans: List[Span] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append((_first_line(node), node.end_lineno or node.lineno, node.name))
        elif isinstance(node, ast.ClassDef):
            spans.extend(_class_spans(node, max_lines))

    # Module-level code between definitions (imports, constants, main blocks)
    covered = [False] * (n_lines + 2)
    for start, end, _ in spans:
        for i in range(start, end + 1):
            covered[i] = True
    gap_start = None
    for i in range(1, n_lines + 2):
        if i <= n_lines and not covered[i]:
            if gap_start is None:
                gap_start = i
        elif gap_start is not None:
            spans.append((gap_start, i - 1, ""))
            gap_start = None
    return sorted(spans)


def _class_spans(node: ast.ClassDef, max_lines: int) -> List[Span]:
    start, end = _first_line(node), node.end_lineno or node.lineno
    if end - start + 1 <= max_lines:
        return [(start, end, node.name)]
    methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not methods:
        return [(start, end, node.name)]
    spans: List[Span] = []
    cursor = start
    for method in methods:
        m_start = _first_line(method)
        if m_start > cursor:
            spans.append((cursor, m_start - 1, node.name))
        m_end = method.end_lineno or method.lineno
        spans.append((m_start, m_end, f"{node.name}.{method.name}"))
        cursor = m_end + 1
    if cursor <= end:
        spans.append((cursor, end, node.name))
    return spans


def _guess_symbol(window: List[str], language: str) -> str:
    pattern = _HEADING if language in ("markdown", "rst") else _SYMBOL
    for line in window:
        match = pattern.match(line)
        if match:
            return match.group(1)[:80]
    return ""
