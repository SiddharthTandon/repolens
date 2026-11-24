"""Command line interface: `repolens index | search | ask | chat | serve | stats`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from . import __version__
from .config import Settings
from .engine import Answer, Lens
from .llm import LLMError
from .search import MODES, Hit

console = Console()
err = Console(stderr=True)


def _lens(args: argparse.Namespace) -> Lens:
    return Lens(Path(args.path), Settings(), embedder_choice=args.embedder)


def _refresh(lens: Lens, args: argparse.Namespace, quiet: bool = True) -> None:
    if getattr(args, "no_refresh", False):
        return
    with err.status("Updating index…", spinner="dots"):
        stats = lens.refresh()
    if not quiet or stats.files_updated or stats.files_removed:
        files, chunks = lens.store.counts()
        err.print(
            f"[dim]index: {files} files, {chunks} chunks "
            f"({stats.files_updated} updated, {stats.files_removed} removed, "
            f"{stats.seconds:.1f}s)[/dim]"
        )


def _print_hits(hits: Sequence[Hit], show_code: bool) -> None:
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("location")
    table.add_column("symbol", style="cyan")
    table.add_column("bm25", justify="right", style="dim")
    table.add_column("dense", justify="right", style="dim")
    for i, hit in enumerate(hits, start=1):
        table.add_row(
            str(i),
            hit.chunk.location,
            hit.chunk.symbol or "-",
            str(hit.lexical_rank or "-"),
            str(hit.dense_rank or "-"),
        )
    console.print(table)
    if show_code:
        for hit in hits:
            chunk = hit.chunk
            console.print(
                Panel(
                    Syntax(chunk.text, chunk.language, line_numbers=True,
                           start_line=chunk.start_line, word_wrap=True),
                    title=chunk.location,
                    title_align="left",
                )
            )


def _print_sources(answer: Answer) -> None:
    if not answer.citations:
        err.print("[yellow]The answer cites no code locations.[/yellow]")
        return
    lines = []
    for citation in answer.citations:
        mark = "[green]✓[/green]" if citation.supported else "[red]✗ not in retrieved code[/red]"
        lines.append(f"  {mark} {citation.location}")
    console.print("[bold]Sources[/bold]")
    console.print("\n".join(lines))


def cmd_index(args: argparse.Namespace) -> int:
    with _lens(args) as lens:
        backend = lens.embedder.id if lens.embedder else "keyword search only (no embedder)"
        err.print(f"[dim]embeddings: {backend}[/dim]")
        with err.status("Indexing…", spinner="dots") as status:
            stats = lens.refresh(progress=lambda rel: status.update(f"Indexing {rel}"))
        files, chunks = lens.store.counts()
        console.print(
            f"Indexed [bold]{files}[/bold] files into [bold]{chunks}[/bold] chunks "
            f"({stats.files_updated} updated, {stats.files_removed} removed) "
            f"in {stats.seconds:.1f}s → {lens.store.path}"
        )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    with _lens(args) as lens:
        _refresh(lens, args)
        hits = lens.search(args.query, k=args.k, mode=args.mode, path_glob=args.filter)
        if not hits:
            console.print("No matches.")
            return 1
        _print_hits(hits, show_code=args.code)
    return 0


def _answer(lens: Lens, question: str, args: argparse.Namespace,
            history: Optional[List[Dict[str, str]]] = None) -> Optional[Answer]:
    if args.no_stream:
        with err.status("Thinking…", spinner="dots"):
            answer = lens.ask(question, k=args.k, mode=args.mode, path_glob=args.filter,
                              history=history)
        console.print(Markdown(answer.text))
    else:
        stream = lens.ask_stream(question, k=args.k, mode=args.mode, path_glob=args.filter,
                                 history=history)
        for piece in stream:
            sys.stdout.write(piece)
            sys.stdout.flush()
        sys.stdout.write("\n")
        answer = stream.answer
    if answer is not None:
        if args.show_context:
            _print_hits(answer.hits, show_code=False)
        _print_sources(answer)
    return answer


def cmd_ask(args: argparse.Namespace) -> int:
    with _lens(args) as lens:
        _refresh(lens, args)
        try:
            answer = _answer(lens, args.question, args)
        except LLMError as error:
            err.print(f"[red]error:[/red] {error}")
            return 2
    return 0 if answer and not answer.unsupported else 1


def cmd_chat(args: argparse.Namespace) -> int:
    history: List[Dict[str, str]] = []
    with _lens(args) as lens:
        _refresh(lens, args, quiet=False)
        err.print("[dim]Ask about the code. Empty line or Ctrl-D to quit.[/dim]")
        while True:
            try:
                question = console.input("[bold green]›[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                break
            try:
                answer = _answer(lens, question, args, history)
            except LLMError as error:
                err.print(f"[red]error:[/red] {error}")
                continue
            if answer is not None:
                history += [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer.text},
                ]
                history = history[-8:]  # keep the last few turns
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from .web import create_app
    except ImportError:
        err.print("The web UI needs extra packages: pip install 'repolens[web]'")
        return 2
    lens = _lens(args)
    _refresh(lens, args, quiet=False)
    err.print(f"Serving {lens.root} on http://{args.host}:{args.port}")
    uvicorn.run(create_app(lens), host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    with _lens(args) as lens:
        files, chunks = lens.store.counts()
        console.print(f"index:      {lens.store.path}")
        console.print(f"files:      {files}")
        console.print(f"chunks:     {chunks}")
        console.print(f"embeddings: {lens.store.get_meta('embedder') or 'not built yet'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repolens",
        description="Ask questions about a codebase and get answers with checked citations.",
    )
    parser.add_argument("--version", action="version", version=f"repolens {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--path", default=".", help="repository to work on (default: .)")
        p.add_argument("--embedder", choices=["auto", "local", "openai", "none"],
                       help="embedding backend (default: $REPOLENS_EMBEDDER or auto)")

    def retrieval(p: argparse.ArgumentParser) -> None:
        p.add_argument("-k", type=int, default=8, help="chunks to retrieve (default: 8)")
        p.add_argument("--mode", choices=MODES, default="hybrid")
        p.add_argument("--filter", metavar="GLOB", help="only search paths matching GLOB")
        p.add_argument("--no-refresh", action="store_true",
                       help="skip the incremental index update")

    def answering(p: argparse.ArgumentParser) -> None:
        p.add_argument("--no-stream", action="store_true", help="print the answer when done")
        p.add_argument("--show-context", action="store_true",
                       help="list the chunks given to the model")

    p = sub.add_parser("index", help="build or refresh the index")
    p.add_argument("path", nargs="?", default=".")
    p.add_argument("--embedder", choices=["auto", "local", "openai", "none"])
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="retrieve matching code, no model needed")
    p.add_argument("query")
    common(p)
    retrieval(p)
    p.add_argument("--code", action="store_true", help="print the matching code")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="answer one question")
    p.add_argument("question")
    common(p)
    retrieval(p)
    answering(p)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="interactive session with follow-up questions")
    common(p)
    retrieval(p)
    answering(p)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("serve", help="web UI")
    common(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-refresh", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("stats", help="show what is in the index")
    common(p)
    p.set_defaults(func=cmd_stats)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
