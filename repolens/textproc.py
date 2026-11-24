"""Tokenizing code and questions for keyword search.

Identifiers are split the way people search for them: ``getEnvironProxies``
and ``get_environ_proxies`` both yield ``get``, ``environ`` and ``proxy``, and
the whole identifier is kept as well so exact names still rank first.
"""

from __future__ import annotations

import re
from typing import List

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could do does did for from has have how i if in
    into is it its me my no not of on or our should so than that the their them then there
    these they this those to up us was we were what when where which who why will with would
    you your get gets set use used using via work works
    self cls def return import none true false pass elif else
    """.split()
)


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokens(text: str) -> List[str]:
    out: List[str] = []
    for ident in _IDENT.findall(text):
        pieces = []
        for part in ident.split("_"):
            pieces.extend(_CAMEL.findall(part) or ([part] if part else []))
        words = [p.lower() for p in pieces]
        words = [_stem(w) for w in words if w not in STOPWORDS]
        out.extend(w for w in words if len(w) > 1 and w not in STOPWORDS)
        whole = ident.lower().strip("_")
        if len(pieces) > 1 and whole and whole not in STOPWORDS:
            out.append(whole)
    return out
