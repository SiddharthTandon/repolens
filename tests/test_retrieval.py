from repolens.bm25 import BM25
from repolens.search import rrf
from repolens.textproc import tokens


def test_identifiers_are_split_and_kept_whole():
    assert tokens("getEnvironProxies") == ["environ", "proxy", "getenvironproxies"]
    assert tokens("get_environ_proxies") == ["environ", "proxy", "get_environ_proxies"]
    assert "http" in tokens("HTTPAdapter") and "adapter" in tokens("HTTPAdapter")


def test_stopwords_are_dropped():
    assert tokens("How does the session handle redirects?") == ["session", "handle", "redirect"]


def test_bm25_prefers_documents_with_rare_query_terms():
    docs = [
        tokens("connection pool size and retries for the adapter"),
        tokens("parse the cookie jar and the cookie header"),
        tokens("the the the adapter"),
    ]
    ranked = BM25(docs).top(tokens("cookie header"), k=3)
    assert ranked[0][0] == 1
    assert all(doc != 0 for doc, _ in ranked)  # no shared terms, no score


def test_bm25_handles_an_empty_corpus():
    assert BM25([]).top(["x"], k=5) == []


def test_rrf_rewards_agreement_between_rankings():
    fused = rrf([[1, 2, 3], [3, 1, 4]])
    order = [doc for doc, _ in fused]
    assert order[0] == 1  # ranked 1st and 2nd
    assert order[1] == 3  # ranked 3rd and 1st
    assert set(order) == {1, 2, 3, 4}
