"""Turning a question and the retrieved chunks into chat messages."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .search import Hit

CONTEXT_CHARS = 14000

SYSTEM_PROMPT = """You answer questions about a software repository.

Rules:
- Use only the numbered excerpts below. Do not rely on outside knowledge of the project.
- After every statement about the code, cite the excerpt it comes from by its location in \
square brackets, exactly as written in the excerpt header, for example [src/app/server.py:40-72].
- If the excerpts do not answer the question, say which part is missing instead of guessing.
- Be brief and concrete: name the functions, classes and files involved. Quote code only \
when a few lines make the answer clearer."""


def format_context(hits: Sequence[Hit], budget: int = CONTEXT_CHARS) -> str:
    blocks: List[str] = []
    used = 0
    for i, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        label = chunk.symbol or "module level"
        block = f"[{i}] [{chunk.location}] ({label})\n```{chunk.language}\n{chunk.text}\n```"
        if blocks and used + len(block) > budget:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def build_messages(
    question: str,
    hits: Sequence[Hit],
    history: Optional[Sequence[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    context = format_context(hits) if hits else "(no matching code was found)"
    messages.append(
        {"role": "user", "content": f"Excerpts:\n\n{context}\n\nQuestion: {question}"}
    )
    return messages
