"""`chilluevar ask`, against a fake Ollama: no server, no model."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from chilluevar import cli
from chilluevar.config import Config
from chilluevar.llm.ollama import OllamaLLM


def fake_ollama(answer: str) -> tuple[httpx.MockTransport, list[dict[str, Any]]]:
    """An Ollama that loads instantly and gives the same answer to everything."""
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if not body["messages"]:  # warm_up
            return httpx.Response(200, json={"done": True, "done_reason": "load"})
        lines = [{"message": {"content": answer}, "done": False}, {"done": True}]
        return httpx.Response(200, content="\n".join(json.dumps(line) for line in lines))

    return httpx.MockTransport(handler), seen


@pytest.fixture
def ollama(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Make every client the CLI builds talk to the fake Ollama instead."""

    def install(answer: str) -> list[dict[str, Any]]:
        transport, seen = fake_ollama(answer)
        real = OllamaLLM.from_config

        def from_config(config: Any, **overrides: Any) -> OllamaLLM:
            return real(config, transport=transport, **overrides)

        monkeypatch.setattr(OllamaLLM, "from_config", from_config)
        return seen

    return install


async def test_ask_prints_the_answer(ollama: Any, capsys: pytest.CaptureFixture[str]) -> None:
    seen = ollama("Son las cinco y media.")
    args = cli.build_parser().parse_args(["ask", "¿Qué hora es?"])

    assert await cli.ask(Config(), args) == 0
    out = capsys.readouterr().out
    assert "Son las cinco y media." in out
    assert "would sound wrong" not in out
    # First the warm-up, then the question with the spoken-answer prompt first.
    assert seen[0]["messages"] == []
    assert seen[1]["messages"][0]["role"] == "system"
    assert "Chilluevar" in seen[1]["messages"][0]["content"]
    assert seen[1]["messages"][1] == {"role": "user", "content": "¿Qué hora es?"}


async def test_ask_flags_an_answer_that_would_sound_wrong(
    ollama: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    ollama("Necesitas:\n- arroz\n- **azafrán**")
    args = cli.build_parser().parse_args(["ask", "¿Qué necesito para una paella?"])

    assert await cli.ask(Config(), args) == 0
    assert "would sound wrong out loud: list, markdown" in capsys.readouterr().out


async def test_ask_uses_the_model_given_on_the_command_line(ollama: Any) -> None:
    seen = ollama("Hola.")
    args = cli.build_parser().parse_args(["ask", "Hola", "--model", "llama3.2:3b"])

    assert await cli.ask(Config(), args) == 0
    assert seen[1]["model"] == "llama3.2:3b"


async def test_ask_explains_errors_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    real = OllamaLLM.from_config
    monkeypatch.setattr(
        OllamaLLM,
        "from_config",
        lambda config, **kw: real(config, transport=httpx.MockTransport(refuse), **kw),
    )
    args = cli.build_parser().parse_args(["ask", "Hola"])

    assert await cli.ask(Config(), args) == 1
    assert "is it running" in capsys.readouterr().out
