"""The Ollama client, tested against Ollama's real wire format with no server.

`httpx.MockTransport` hands every request to a function instead of the network.
FakeOllama answers with the same NDJSON lines Ollama sends, so the parsing
tested here is the parsing that runs against the real thing.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from chilluevar.interfaces import LanguageModel
from chilluevar.llm import LanguageModelError
from chilluevar.llm.ollama import OllamaLLM

MESSAGES = [{"role": "user", "content": "¿Qué necesito para hacer una paella?"}]


def ndjson(*chunks: dict[str, Any]) -> bytes:
    """Encode chunks the way Ollama sends them: one JSON object per line."""
    return b"".join(json.dumps(chunk).encode() + b"\n" for chunk in chunks)


def say(text: str) -> dict[str, Any]:
    """One token of the answer, as Ollama sends it."""
    return {"model": "qwen2.5:3b", "message": {"role": "assistant", "content": text}, "done": False}


# The last line, copied from a real answer: empty content, done, and timings.
DONE = {
    "model": "qwen2.5:3b",
    "message": {"role": "assistant", "content": ""},
    "done": True,
    "done_reason": "stop",
    "load_duration": 5471269500,
    "eval_count": 403,
    "eval_duration": 11206241000,
}


class FakeOllama:
    """Plays Ollama: records what it is sent and answers with canned lines."""

    def __init__(self, body: bytes, status: int = 200) -> None:
        self.body = body
        self.status = status
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        return httpx.Response(self.status, content=self.body)

    def client(self) -> OllamaLLM:
        return OllamaLLM("qwen2.5:3b", transport=httpx.MockTransport(self))


async def collect(llm: OllamaLLM) -> list[object]:
    return [piece async for piece in llm.stream(MESSAGES)]


def test_satisfies_the_protocol() -> None:
    assert isinstance(OllamaLLM("qwen2.5:3b"), LanguageModel)


async def test_streams_fragments_in_order() -> None:
    # The empty content of the last line must not come out as a fragment.
    ollama = FakeOllama(
        ndjson(say("Para"), say(" hacer"), say(" una"), say(" pa"), say("ella"), DONE)
    )
    assert await collect(ollama.client()) == ["Para", " hacer", " una", " pa", "ella"]


async def test_sends_model_messages_and_stream() -> None:
    ollama = FakeOllama(ndjson(say("Hola"), DONE))
    await collect(ollama.client())
    sent = ollama.requests[0]
    assert sent["model"] == "qwen2.5:3b"
    assert sent["messages"] == MESSAGES
    assert sent["stream"] is True


async def test_skips_blank_lines() -> None:
    # A blank line in the middle must be skipped, not crash json.loads.
    ollama = FakeOllama(ndjson(say("Hola")) + b"\n" + ndjson(say("."), DONE))
    assert await collect(ollama.client()) == ["Hola", "."]


async def test_stops_at_done() -> None:
    # Anything after the "done" line is not part of this answer.
    ollama = FakeOllama(ndjson(say("Hola"), DONE, say("sobra")))
    assert await collect(ollama.client()) == ["Hola"]


# --------------------------------------------------------------------------- #
# Errors: every failure becomes a LanguageModelError that says what to do
# --------------------------------------------------------------------------- #


def failing(error: type[httpx.TransportError]) -> OllamaLLM:
    """A client whose every request fails the way a real network can."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise error("simulated", request=request)

    return OllamaLLM("qwen2.5:3b", transport=httpx.MockTransport(handler))


async def test_ollama_not_running_says_so() -> None:
    with pytest.raises(LanguageModelError, match="is it running"):
        await collect(failing(httpx.ConnectError))


async def test_a_model_too_slow_says_so() -> None:
    with pytest.raises(LanguageModelError, match="too big"):
        await collect(failing(httpx.ReadTimeout))


async def test_missing_model_says_how_to_install_it() -> None:
    # What Ollama answers, with a 404, for a model that was never pulled.
    ollama = FakeOllama(b'{"error":"model \'qwen2.5:3b\' not found"}', status=404)
    with pytest.raises(LanguageModelError, match="Run: ollama pull"):
        await collect(ollama.client())


async def test_error_in_the_middle_of_the_answer() -> None:
    ollama = FakeOllama(ndjson(say("Para"), {"error": "out of memory"}))
    with pytest.raises(LanguageModelError, match="out of memory"):
        await collect(ollama.client())


async def test_garbage_is_an_error_not_a_crash() -> None:
    # What a misconfigured proxy might send back instead of Ollama's answer.
    ollama = FakeOllama(b"<html>Bad Gateway</html>\n")
    with pytest.raises(LanguageModelError, match="not JSON"):
        await collect(ollama.client())
