import textwrap

from repolens.chunking import chunk_file
from repolens.files import SourceFile


def python(text):
    return SourceFile("pkg/mod.py", "python", textwrap.dedent(text).lstrip())


def test_python_definitions_become_their_own_chunks():
    chunks = chunk_file(python('''
        import os

        LIMIT = 3


        def first():
            return 1


        @decorator
        def second():
            return 2


        class Thing:
            def method(self):
                return os.sep
    '''))
    by_symbol = {c.symbol: c for c in chunks}
    assert set(by_symbol) == {"", "first", "second", "Thing"}
    assert by_symbol["second"].text.startswith("@decorator")
    assert by_symbol[""].text.startswith("import os")
    assert by_symbol["first"].location == "pkg/mod.py:6-7"


def test_long_classes_are_split_into_methods():
    methods = "\n".join(
        f"    def m{i}(self):\n" + "".join(f"        x{j} = {j}\n" for j in range(12))
        for i in range(10)
    )
    chunks = chunk_file(python("class Big:\n    '''doc'''\n" + methods), max_lines=40)
    symbols = [c.symbol for c in chunks]
    assert "Big" in symbols  # the header with the docstring
    assert [s for s in symbols if s.startswith("Big.")] == [f"Big.m{i}" for i in range(10)]


def test_line_numbers_match_the_source():
    src = python("\n".join(f"def f{i}():\n    return {i}\n" for i in range(5)))
    lines = src.text.splitlines()
    for chunk in chunk_file(src):
        assert chunk.text == "\n".join(lines[chunk.start_line - 1 : chunk.end_line])


def test_other_languages_use_overlapping_windows_with_a_symbol():
    body = "export function handler(req) {\n" + "  step();\n" * 150 + "}\n"
    chunks = chunk_file(SourceFile("a.js", "javascript", body), max_lines=60, overlap=10)
    assert len(chunks) == 3
    assert chunks[0].symbol == "handler"
    assert chunks[1].start_line == chunks[0].end_line - 9
    assert chunks[-1].end_line == len(body.splitlines())


def test_syntax_errors_fall_back_to_windows():
    chunks = chunk_file(SourceFile("broken.py", "python", "def oops(:\n    pass\n"))
    assert len(chunks) == 1 and chunks[0].symbol == "oops"


def test_markdown_chunks_are_labelled_with_headings():
    chunks = chunk_file(SourceFile("README.md", "markdown", "intro\n\n## Install\npip install x\n"))
    assert chunks[0].symbol == "Install"


def test_empty_files_have_no_chunks():
    assert chunk_file(SourceFile("empty.py", "python", "\n\n")) == []
