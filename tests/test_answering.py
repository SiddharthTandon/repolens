import json

import httpx
import pytest

from repolens.chunking import Chunk
from repolens.citations import check_citations
from repolens.engine import Lens
from repolens.llm import ChatClient, LLMError
from repolens.prompting import build_messages, format_context
from repolens.search import Hit


def hit(path, start, end, text="x = 1", symbol="f"):
    return Hit(Chunk(path, start, end, symbol, "python", text), 1.0)


def test_citations_are_checked_against_retrieved_chunks():
    hits = [hit("app/retry.py", 8, 11), hit("app/auth.py", 1, 6)]
    answer = (
        "Backoff is computed in `backoff_delay` [app/retry.py:8-11] and the header is built "
        "in [app/auth.py:4]. Retries live elsewhere [app/policy.py:1-20] [app/retry.py:30-40]."
    )
    found = {c.location: c.supported for c in check_citations(answer, hits)}
    assert found == {
        "app/retry.py:8-11": True,
        "app/auth.py:4-4": True,
        "app/policy.py:1-20": False,
        "app/retry.py:30-40": False,
    }


def test_repeated_citations_are_listed_once():
    hits = [hit("a.py", 1, 5)]
    assert len(check_citations("[a.py:1-5] and again [a.py:1-5]", hits)) == 1


def test_prompt_numbers_excerpts_with_their_locations():
    messages = build_messages("Where?", [hit("a.py", 3, 9, "def f():\n    pass")])
    assert messages[0]["role"] == "system"
    assert "[1] [a.py:3-9] (f)" in messages[-1]["content"]
    assert messages[-1]["content"].endswith("Question: Where?")


def test_context_respects_the_budget():
    hits = [hit(f"f{i}.py", 1, 2, "y" * 1000) for i in range(30)]
    context = format_context(hits, budget=5000)
    assert context.count("```python") < 30
    assert "[1] [f0.py:1-2]" in context


def fake_server(answer_text, captured):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        if body.get("stream"):
            events = "".join(
                f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]})}\n\n"
                for piece in answer_text.split(" ")
                for piece in [piece + " "]
            )
            return httpx.Response(200, text=events + "data: [DONE]\n\n")
        return httpx.Response(200, json={"choices": [{"message": {"content": answer_text}}]})

    return httpx.MockTransport(handler)


def test_chat_client_streams_and_completes():
    captured = []
    client = ChatClient("http://llm/v1", "key", "m", transport=fake_server("one two", captured))
    assert client.complete([{"role": "user", "content": "q"}]) == "one two"
    assert "".join(client.stream([{"role": "user", "content": "q"}])) == "one two "
    assert captured[0]["model"] == "m" and captured[1]["stream"] is True


def test_chat_client_explains_http_errors():
    transport = httpx.MockTransport(lambda r: httpx.Response(401, text="bad key"))
    client = ChatClient("http://llm/v1", "key", "m", transport=transport)
    with pytest.raises(LLMError, match="401"):
        client.complete([{"role": "user", "content": "q"}])


def test_end_to_end_answer_with_checked_citations(repo, embedder):
    captured = []
    text = "Delays use full jitter [app/retry.py:8-10]; see also [app/ghost.py:1-2]."
    chat = ChatClient("http://llm/v1", "key", "m", transport=fake_server(text, captured))
    with Lens(repo, embedder=embedder, chat=chat) as lens:
        lens.refresh()
        answer = lens.ask("how is the backoff delay computed?", k=4)
    assert answer.text == text
    assert [c.supported for c in answer.citations] == [True, False]
    assert [c.location for c in answer.unsupported] == ["app/ghost.py:1-2"]
    prompt = captured[0]["messages"][-1]["content"]
    assert "backoff_delay" in prompt


def test_streamed_answer_is_checked_when_finished(repo, embedder):
    text = "It retries [app/retry.py:14-24]."
    chat = ChatClient("http://llm/v1", "key", "m", transport=fake_server(text, []))
    with Lens(repo, embedder=embedder, chat=chat) as lens:
        lens.refresh()
        stream = lens.ask_stream("what does RetryPolicy.run do?")
        assert "".join(stream).strip() == text
    assert stream.answer is not None and stream.answer.citations[0].supported
