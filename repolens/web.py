"""A small web UI and JSON API on top of a `Lens`."""

from __future__ import annotations

from importlib import resources
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .engine import Lens
from .llm import LLMError
from .search import MODES, Hit


class AskRequest(BaseModel):
    question: str
    k: int = 8
    mode: str = "hybrid"
    filter: Optional[str] = None


def _hit(hit: Hit) -> dict:
    chunk = hit.chunk
    return {
        "location": chunk.location,
        "path": chunk.path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "symbol": chunk.symbol,
        "language": chunk.language,
        "text": chunk.text,
        "score": round(hit.score, 5),
        "lexical_rank": hit.lexical_rank,
        "dense_rank": hit.dense_rank,
    }


def create_app(lens: Lens) -> FastAPI:
    app = FastAPI(title="repolens", docs_url="/api/docs")
    page = resources.files("repolens").joinpath("static/index.html").read_text("utf-8")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return page

    @app.get("/api/stats")
    def stats() -> dict:
        files, chunks = lens.counts()
        return {
            "root": str(lens.root),
            "files": files,
            "chunks": chunks,
            "embedder": lens.meta("embedder"),
            "model": lens.settings.model,
        }

    @app.get("/api/search")
    def search(
        q: str,
        k: int = Query(8, ge=1, le=50),
        mode: str = "hybrid",
        filter: Optional[str] = None,
    ) -> dict:
        if mode not in MODES:
            raise HTTPException(400, f"mode must be one of {MODES}")
        lens.refresh()
        return {"hits": [_hit(h) for h in lens.search(q, k=k, mode=mode, path_glob=filter)]}

    @app.post("/api/ask")
    def ask(request: AskRequest) -> dict:
        if not request.question.strip():
            raise HTTPException(400, "question is empty")
        if request.mode not in MODES:
            raise HTTPException(400, f"mode must be one of {MODES}")
        lens.refresh()
        try:
            answer = lens.ask(request.question, k=request.k, mode=request.mode,
                              path_glob=request.filter)
        except LLMError as error:
            raise HTTPException(502, str(error)) from error
        return {
            "answer": answer.text,
            "hits": [_hit(h) for h in answer.hits],
            "citations": [
                {"location": c.location, "path": c.path, "start": c.start, "end": c.end,
                 "supported": c.supported}
                for c in answer.citations
            ],
        }

    return app
