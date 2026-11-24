"""Finding the source files of a repository."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

LANGUAGES = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "shell",
    ".sql": "sql",
    ".md": "markdown",
    ".rst": "rst",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".cfg": "ini",
    ".ini": "ini",
}

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".repolens",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    "target",
    ".tox",
    ".nox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
}

SKIP_NAMES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Cargo.lock"}

MAX_BYTES = 512 * 1024


@dataclass(frozen=True)
class SourceFile:
    path: str  # relative to the repository root, with forward slashes
    language: str
    text: str


def language_for(path: str) -> Optional[str]:
    name = os.path.basename(path)
    if name in SKIP_NAMES or name.endswith((".min.js", ".min.css")):
        return None
    return LANGUAGES.get(os.path.splitext(name)[1].lower())


def _git_files(root: Path) -> Optional[List[str]]:
    """Tracked and untracked-but-not-ignored files, if `root` is a git checkout."""
    try:
        command = ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others",
                   "--exclude-standard"]
        out = subprocess.run(
            command,
            capture_output=True,
            check=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return [p for p in out.decode("utf-8", "replace").split("\0") if p]


def _walk_files(root: Path) -> List[str]:
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            found.append(os.path.relpath(os.path.join(dirpath, name), root))
    return found


def discover(root: Path) -> List[str]:
    """Relative paths of the files worth indexing, sorted."""
    candidates = _git_files(root)
    if candidates is None:
        candidates = _walk_files(root)
    paths = set()
    for rel in candidates:
        rel = rel.replace(os.sep, "/")
        parts = rel.split("/")
        if any(part in SKIP_DIRS for part in parts[:-1]):
            continue
        if language_for(rel) is None:
            continue
        full = root / rel
        try:
            if not full.is_file() or full.stat().st_size > MAX_BYTES:
                continue
        except OSError:
            continue
        paths.add(rel)
    return sorted(paths)


def read_source(root: Path, rel: str) -> Optional[SourceFile]:
    language = language_for(rel)
    if language is None:
        return None
    try:
        data = (root / rel).read_bytes()
    except OSError:
        return None
    if b"\0" in data[:8192]:
        return None  # binary
    return SourceFile(rel, language, data.decode("utf-8", "replace"))
