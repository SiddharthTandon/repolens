from repolens.config import index_path
from repolens.indexer import index_repo
from repolens.search import Searcher
from repolens.store import Store


def build(repo, embedder):
    store = Store(index_path(repo))
    stats = index_repo(repo, store, embedder)
    return store, stats


def test_indexing_finds_supported_files(repo, embedder):
    store, stats = build(repo, embedder)
    assert stats.files_seen == 4
    assert set(store.file_shas()) == {"app/retry.py", "app/auth.py", "web/server.js", "README.md"}
    files, chunks = store.counts()
    assert files == 4 and chunks > 4


def test_reindexing_only_touches_changed_files(repo, embedder):
    store, _ = build(repo, embedder)
    calls = embedder.calls
    again = index_repo(repo, store, embedder)
    assert again.files_updated == 0 and embedder.calls == calls

    (repo / "app/auth.py").write_text("def login():\n    return True\n")
    (repo / "web/server.js").unlink()
    (repo / "app/new.py").write_text("def added():\n    pass\n")
    changed = index_repo(repo, store, embedder)
    assert changed.files_updated == 2 and changed.files_removed == 1
    assert "web/server.js" not in store.file_shas()
    assert {c.path for c, _ in store.all_chunks()} >= {"app/new.py"}


def test_switching_embedders_rebuilds_the_vectors(repo, embedder):
    store, _ = build(repo, embedder)
    index_repo(repo, store, None)
    assert all(v is None for _, v in store.all_chunks())
    assert store.get_meta("embedder") == "none"


def test_index_ignores_vendored_and_binary_files(repo, embedder):
    (repo / "node_modules/lib").mkdir(parents=True)
    (repo / "node_modules/lib/index.js").write_text("function vendored() {}\n")
    (repo / "logo.py").write_bytes(b"\x89PNG\x00\x00binary")
    store, _ = build(repo, embedder)
    assert "node_modules/lib/index.js" not in store.file_shas()
    assert "logo.py" not in store.file_shas()


def test_hybrid_search_finds_the_right_function(repo, embedder):
    store, _ = build(repo, embedder)
    searcher = Searcher(store, embedder)
    hits = searcher.search("how is the retry backoff delay computed", k=3)
    assert hits[0].chunk.symbol == "backoff_delay"
    assert hits[0].lexical_rank is not None and hits[0].dense_rank is not None

    hits = searcher.search("authorization header removed when redirected to another host", k=3)
    assert hits[0].chunk.symbol == "strip_auth_on_redirect"


def test_lexical_mode_works_without_vectors(repo):
    store, _ = build(repo, None)
    searcher = Searcher(store, None)
    assert not searcher.has_vectors
    hits = searcher.search("RetryPolicy run", mode="hybrid")  # falls back to lexical
    assert hits[0].chunk.symbol == "RetryPolicy"
    assert hits[0].dense_rank is None


def test_path_filter_limits_results(repo, embedder):
    store, _ = build(repo, embedder)
    hits = Searcher(store, embedder).search("server port listen", path_glob="app/*")
    assert hits and all(h.chunk.path.startswith("app/") for h in hits)
