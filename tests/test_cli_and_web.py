import pytest

from repolens.cli import main


def test_cli_index_and_search(repo, capsys):
    assert main(["index", str(repo), "--embedder", "none"]) == 0
    assert "Indexed 4 files" in capsys.readouterr().out
    assert main(["search", "basic auth header", "--path", str(repo), "--embedder", "none"]) == 0
    out = capsys.readouterr().out
    assert "app/auth.py" in out and "basic_auth_header" in out


def test_cli_stats(repo, capsys):
    main(["index", str(repo), "--embedder", "none"])
    capsys.readouterr()
    assert main(["stats", "--path", str(repo), "--embedder", "none"]) == 0
    assert "files:      4" in capsys.readouterr().out


def test_cli_ask_reports_an_unreachable_model(repo, monkeypatch, capsys):
    monkeypatch.setenv("REPOLENS_BASE_URL", "http://127.0.0.1:9/v1")
    code = main(["ask", "what retries?", "--path", str(repo), "--embedder", "none", "--no-stream"])
    assert code == 2
    assert "could not reach" in capsys.readouterr().err


def test_web_api(repo, embedder):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from repolens.engine import Lens
    from repolens.web import create_app

    with Lens(repo, embedder=embedder) as lens:
        lens.refresh()
        client = TestClient(create_app(lens))
        assert "repolens" in client.get("/").text
        stats = client.get("/api/stats").json()
        assert stats["files"] == 4 and stats["embedder"] == "test:hashing"
        hits = client.get("/api/search", params={"q": "health endpoint", "k": 2}).json()["hits"]
        assert hits[0]["path"] == "web/server.js"
        assert client.post("/api/ask", json={"question": "  "}).status_code == 400
