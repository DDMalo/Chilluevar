"""Ollama behind the LanguageModel protocol.

We talk to Ollama's native HTTP API with httpx. Ollama answers POST /api/chat
with one JSON object per line (NDJSON): one line per token, and a last line
with `"done": true` and the timings.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from chilluevar.interfaces import Message, Tool, ToolCall

__all__ = ["OllamaLLM"]

# httpx gives up after 5 s without data by default, and loading the model from
# disk took 5.5 s on our laptop (load_duration). Connecting, on the other hand,
# is instant or Ollama is not running at all.
TIMEOUT = httpx.Timeout(120.0, connect=5.0)


class OllamaLLM:
    """Streams answers from a model served by a local Ollama."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        # Only the tests pass a transport: a fake Ollama that answers without
        # a server, so no test ever needs a model or the network.
        self._transport = transport

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[str | ToolCall]:
        body = {"model": self.model, "messages": messages, "stream": True}
        async with (
            httpx.AsyncClient(
                base_url=self.base_url, timeout=TIMEOUT, transport=self._transport
            ) as client,
            client.stream("POST", "/api/chat", json=body) as response,
        ):
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                # The last line carries an empty content: nothing to say.
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
                if chunk.get("done"):
                    break
