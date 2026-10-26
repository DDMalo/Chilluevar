"""The model comparison script, against a fake Ollama: no server, no model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from scripts import bench_llm

from chilluevar.llm.ollama import OllamaLLM


def fake_ollama(answers: dict[str, str]) -> tuple[httpx.MockTransport, list[dict[str, Any]]]:
    """Answers each question with a canned reply; loading and unloading just succeed."""
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if not body["messages"]:  # warm_up or unload
            return httpx.Response(200, json={"done": True})
        question = body["messages"][-1]["content"]
        lines = [{"message": {"content": answers[question]}, "done": False}, {"done": True}]
        return httpx.Response(200, content="\n".join(json.dumps(line) for line in lines))

    return httpx.MockTransport(handler), seen


def test_reads_questions_skipping_comments_and_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "preguntas.txt"
    path.write_text("# grupo\n¿Qué hora es?\n\n  # otro\nHola\n", encoding="utf-8")
    assert bench_llm.read_questions(path) == ["¿Qué hora es?", "Hola"]


def test_the_shipped_questions_are_readable() -> None:
    assert len(bench_llm.read_questions(bench_llm.DEFAULT_QUESTIONS)) == 20


async def test_run_model_scores_each_answer() -> None:
    transport, seen = fake_ollama(
        {
            "¿Qué necesito?": "Necesitas:\n- huevos\n- patatas",  # a list: sounds wrong
            "Hola": "Hola, ¿en qué te ayudo?",  # fine
        }
    )
    llm = OllamaLLM("qwen2.5:3b", transport=transport)
    result = await bench_llm.run_model(llm, ["¿Qué necesito?", "Hola"], "prompt", 60, False)

    assert result["wrong"] == 1
    assert result["problems"] == {"list": 1}
    assert result["answers"][1]["answer"] == "Hola, ¿en qué te ayudo?"
    assert result["first"] <= result["total"]
    # Unloaded first, for a cold load time, and last, to free the memory.
    assert seen[0]["keep_alive"] == 0
    assert seen[-1]["keep_alive"] == 0


def test_table_is_markdown() -> None:
    row = {"load": 2.0, "first": 0.4, "total": 1.5, "words": 18.0, "wrong": 2}
    lines = bench_llm.table({"qwen2.5:3b": row}, asked=20).splitlines()
    assert len(lines) == 3
    assert lines[2] == "| qwen2.5:3b | 2.0 s | 0.40 s | 1.50 s | 18 | 2/20 |"
