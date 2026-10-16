"""Ollama behind the LanguageModel protocol.

We talk to Ollama's native HTTP API with httpx. Ollama answers POST /api/chat
with one JSON object per line (NDJSON): one line per token, and a last line
with `"done": true` and the timings. An error that happens halfway through
arrives as a line with an `"error"` key instead.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from chilluevar.interfaces import Message, Tool, ToolCall
from chilluevar.llm.base import LanguageModelError

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
        try:
            async with (
                httpx.AsyncClient(
                    base_url=self.base_url, timeout=TIMEOUT, transport=self._transport
                ) as client,
                client.stream("POST", "/api/chat", json=body) as response,
            ):
                if response.status_code != httpx.codes.OK:
                    # A streamed response is not read until asked to; the
                    # error message is in the body, so read it first.
                    await response.aread()
                    raise LanguageModelError(self._explain(response))

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = _parse_line(line)
                    # The last line carries an empty content: nothing to say.
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if chunk.get("done"):
                        break
        # The order matters: all of these are kinds of TransportError, so the
        # general case goes last. A connection that takes more than 5 s to open
        # (ConnectTimeout) means the same as one refused: nobody is listening.
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise LanguageModelError(
                f"cannot reach Ollama at {self.base_url}: is it running? "
                "Open the Ollama app or run `ollama serve`."
            ) from exc
        except httpx.TimeoutException as exc:
            raise LanguageModelError(
                f"Ollama sent nothing for {TIMEOUT.read:.0f} s while answering with "
                f"{self.model}: is the model too big for this machine?"
            ) from exc
        except httpx.TransportError as exc:
            raise LanguageModelError(f"lost the connection to Ollama: {exc}") from exc

    def _explain(self, response: httpx.Response) -> str:
        """Turn an error response into a message that says what to do."""
        try:
            error = str(response.json().get("error", response.text))
        except ValueError:
            error = response.text
        if response.status_code == httpx.codes.NOT_FOUND and "not found" in error:
            return f"model {self.model!r} is not installed. Run: ollama pull {self.model}"
        return f"Ollama answered {response.status_code}: {error}"


def _parse_line(line: str) -> dict[str, Any]:
    """Decode one line of the answer, refusing anything that is not a token."""
    try:
        chunk = json.loads(line)
    except json.JSONDecodeError as exc:
        raise LanguageModelError(f"Ollama sent something that is not JSON: {line[:80]!r}") from exc
    if "error" in chunk:
        raise LanguageModelError(f"Ollama stopped with an error: {chunk['error']}")
    return chunk
