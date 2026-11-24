"""A small client for OpenAI-compatible chat completion endpoints, with streaming."""

from __future__ import annotations

import json
from typing import Dict, Iterator, List, Optional

import httpx


class LLMError(RuntimeError):
    pass


class ChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 180.0,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

    def _payload(self, messages: List[Dict[str, str]], temperature: float, stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }

    def _fail(self, error: Exception) -> LLMError:
        if isinstance(error, httpx.ConnectError):
            return LLMError(
                f"could not reach {self.base_url}. Start a local model with `ollama serve`, "
                "or point REPOLENS_BASE_URL, REPOLENS_API_KEY and REPOLENS_MODEL at a hosted one."
            )
        if isinstance(error, httpx.HTTPStatusError):
            body = error.response.text[:300]
            return LLMError(f"{error.response.status_code} from {self.base_url}: {body}")
        return LLMError(str(error))

    def complete(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
        try:
            response = self._client.post(
                "/chat/completions", json=self._payload(messages, temperature, False)
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise self._fail(error) from error
        return response.json()["choices"][0]["message"]["content"] or ""

    def stream(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> Iterator[str]:
        payload = self._payload(messages, temperature, True)
        try:
            with self._client.stream("POST", "/chat/completions", json=payload) as response:
                if response.is_error:
                    response.read()
                    response.raise_for_status()
                for line in response.iter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    for choice in event.get("choices", []):
                        piece = (choice.get("delta") or {}).get("content")
                        if piece:
                            yield piece
        except httpx.HTTPError as error:
            raise self._fail(error) from error
