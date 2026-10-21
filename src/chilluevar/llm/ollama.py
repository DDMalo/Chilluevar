"""Ollama behind the LanguageModel protocol.

We talk to Ollama's native HTTP API with httpx. Ollama answers POST /api/chat
with one JSON object per line (NDJSON): one line per token, and a last line
with `"done": true` and the timings. An error that happens halfway through
arrives as a line with an `"error"` key instead.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any, Self

import httpx

from chilluevar.config import LLMConfig
from chilluevar.interfaces import Message, Tool, ToolCall
from chilluevar.llm.base import LanguageModelError

__all__ = ["OllamaLLM"]

# httpx gives up after 5 s without data by default, and loading the model from
# disk took 5.5 s on our laptop (load_duration). Connecting, on the other hand,
# is instant or Ollama is not running at all.
TIMEOUT = httpx.Timeout(120.0, connect=5.0)

# Spanish runs at roughly 1.3 to 1.6 tokens per word with these models, so two
# tokens per word is a ceiling that only a runaway answer reaches. Keeping
# answers short is the system prompt's job; this is the safety net behind it.
TOKENS_PER_WORD = 2


class OllamaLLM:
    """Streams answers from a model served by a local Ollama."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.6,
        num_ctx: int = 4096,
        max_reply_tokens: int | None = None,
        keep_alive: str | int = "30m",
        think: bool | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.max_reply_tokens = max_reply_tokens
        self.keep_alive = keep_alive
        self.think = think
        # Only the tests pass a transport: a fake Ollama that answers without
        # a server, so no test ever needs a model or the network.
        self._transport = transport

    @classmethod
    def from_config(cls, config: LLMConfig, **overrides: Any) -> Self:
        """Build the client from the `llm:` section of the configuration.

        `overrides` replaces any setting for this one client, which is what
        lets the model comparison try several models with the same config.
        """
        settings: dict[str, Any] = {
            "model": config.model,
            "base_url": config.base_url,
            "temperature": config.temperature,
            "num_ctx": config.num_ctx,
            "max_reply_tokens": config.max_reply_words * TOKENS_PER_WORD,
            "keep_alive": config.keep_alive,
            "think": config.think,
        }
        settings |= overrides
        return cls(settings.pop("model"), **settings)

    def _client(self) -> httpx.AsyncClient:
        # A fresh client per request. Reusing connections saves nothing worth
        # having on localhost, and this way there is nothing to close at exit.
        return httpx.AsyncClient(base_url=self.base_url, timeout=TIMEOUT, transport=self._transport)

    def _request(self, messages: list[Message]) -> dict[str, Any]:
        """The body of a chat request, with the options that shape the answer."""
        options: dict[str, Any] = {"temperature": self.temperature, "num_ctx": self.num_ctx}
        if self.max_reply_tokens:
            # Ollama stops generating after this many tokens, mid-sentence if
            # need be: a hard ceiling for when the prompt is ignored.
            options["num_predict"] = self.max_reply_tokens
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": options,
        }
        # Only sent when configured: a model that cannot reason may reject it.
        if self.think is not None:
            body["think"] = self.think
        return body

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[str | ToolCall]:
        try:
            async with (
                self._client() as client,
                client.stream("POST", "/api/chat", json=self._request(messages)) as response,
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
        except httpx.TransportError as exc:
            raise self._network_error(exc) from exc

    async def warm_up(self) -> float:
        """Load the model into memory now, and return how many seconds it took.

        Meant to be called once at startup. Without it, the first question
        after starting the assistant waits in silence while the model loads
        (5.5 s on our laptop). Ollama treats a chat request with no messages
        as "just load the model".
        """
        body = {"model": self.model, "messages": [], "stream": False, "keep_alive": self.keep_alive}
        started = time.perf_counter()
        try:
            async with self._client() as client:
                response = await client.post("/api/chat", json=body)
        except httpx.TransportError as exc:
            raise self._network_error(exc) from exc
        if response.status_code != httpx.codes.OK:
            raise LanguageModelError(self._explain(response))
        return time.perf_counter() - started

    async def unload(self) -> None:
        """Free the memory the model is using now, without waiting for keep_alive.

        The model comparison calls this between models: a laptop that tries to
        hold two of them at once runs out of graphics memory and slows down,
        and then it is measuring the lack of memory instead of the model.
        """
        body = {"model": self.model, "messages": [], "stream": False, "keep_alive": 0}
        try:
            async with self._client() as client:
                await client.post("/api/chat", json=body)
        except httpx.TransportError:
            pass  # if Ollama is not even running, there is nothing to free

    def _network_error(self, exc: httpx.TransportError) -> LanguageModelError:
        """Say what a failure to talk to Ollama most likely means.

        The order of the checks matters: all of these are kinds of
        TransportError, so the general case goes last. A connection that takes
        more than 5 s to open (ConnectTimeout) means the same as one refused.
        """
        if isinstance(exc, httpx.ConnectError | httpx.ConnectTimeout):
            return LanguageModelError(
                f"cannot reach Ollama at {self.base_url}: is it running? "
                "Open the Ollama app or run `ollama serve`."
            )
        if isinstance(exc, httpx.TimeoutException):
            return LanguageModelError(
                f"Ollama sent nothing for {TIMEOUT.read:.0f} s with {self.model}: "
                "is the model too big for this machine?"
            )
        return LanguageModelError(f"lost the connection to Ollama: {exc}")

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
