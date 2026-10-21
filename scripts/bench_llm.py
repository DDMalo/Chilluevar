#!/usr/bin/env python3
"""Compare language models on speed and on how well they answer out loud (issue #5).

Asks every model the same Spanish questions, with the real system prompt, and
prints a Markdown table you can paste into the issue:

    python scripts/bench_llm.py --models qwen2.5:3b qwen2.5:7b-instruct llama3.2:3b
    python scripts/bench_llm.py --models qwen3:4b qwen3:8b --think false

Pull the models first (`ollama pull qwen2.5:7b-instruct`). Each one is unloaded
before the next is loaded, so they never compete for memory.

Columns:
  load   seconds to load the model from disk. Paid once, at start-up.
  first  median seconds until the first words: the silence the user waits
         through before the assistant starts speaking.
  total  median seconds for the whole answer.
  words  mean words per answer.
  wrong  answers that would sound wrong out loud: lists, markdown, symbols,
         links, or longer than the word limit.

The numbers say nothing about whether the Spanish is natural or the facts are
right. That needs reading, so every answer is saved to benchmarks/results/ (not
committed) and `-v` prints them as they come.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from chilluevar.config import load_config
from chilluevar.llm import LanguageModelError, speech_problems, system_prompt
from chilluevar.llm.ollama import OllamaLLM
from chilluevar.llm.spoken import word_count

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS = ROOT / "benchmarks" / "preguntas.txt"
RESULTS = ROOT / "benchmarks" / "results"


def read_questions(path: Path) -> list[str]:
    """One question per line; blank lines and lines starting with # are skipped."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


async def ask_one(llm: OllamaLLM, prompt: str, question: str) -> dict[str, Any]:
    """Ask one question and time it the way a listener would notice."""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
    ]
    started = time.perf_counter()
    first: float | None = None
    parts: list[str] = []
    async for piece in llm.stream(messages):
        if isinstance(piece, str):
            if first is None:
                first = time.perf_counter() - started
            parts.append(piece)
    total = time.perf_counter() - started
    return {
        "question": question,
        "answer": "".join(parts).strip(),
        "first": first if first is not None else total,
        "total": total,
    }


async def run_model(
    llm: OllamaLLM, questions: list[str], prompt: str, max_words: int, verbose: bool
) -> dict[str, Any]:
    """Every question to one model, and the summary of how it did."""
    await llm.unload()  # so the load time below is a real, cold load
    load = await llm.warm_up()

    answers = []
    for question in questions:
        answer = await ask_one(llm, prompt, question)
        answer["problems"] = speech_problems(answer["answer"], max_words=max_words)
        answers.append(answer)
        if verbose:
            flag = f"  <- {', '.join(answer['problems'])}" if answer["problems"] else ""
            print(f"    > {question}\n      {answer['answer']}{flag}")

    await llm.unload()  # make room for the next model
    return {
        "load": load,
        "first": statistics.median(a["first"] for a in answers),
        "total": statistics.median(a["total"] for a in answers),
        "words": statistics.mean(word_count(a["answer"]) for a in answers),
        "wrong": sum(1 for a in answers if a["problems"]),
        "problems": dict(Counter(p for a in answers for p in a["problems"])),
        "answers": answers,
    }


def table(results: dict[str, dict[str, Any]], asked: int) -> str:
    """The results as a Markdown table, ready to paste into GitHub."""
    lines = [
        "| model | load | first | total | words | wrong |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in results.items():
        lines.append(
            f"| {name} | {row['load']:.1f} s | {row['first']:.2f} s | {row['total']:.2f} s "
            f"| {row['words']:.0f} | {row['wrong']}/{asked} |"
        )
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="+", required=True, help="Ollama model names")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument(
        "--config", type=Path, default=None, help="YAML config to take settings from"
    )
    parser.add_argument(
        "--think",
        choices=["true", "false"],
        default=None,
        help="for models that reason before answering; false is what a voice assistant needs",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="print every answer")
    args = parser.parse_args()

    config = load_config(args.config).llm
    max_words = config.max_reply_words
    prompt = system_prompt(max_words=max_words)
    questions = read_questions(args.questions)
    # Without --think, whatever the configuration says applies.
    overrides = {} if args.think is None else {"think": args.think == "true"}
    print(f"{len(questions)} questions, {len(args.models)} models, limit {max_words} words\n")

    results: dict[str, dict[str, Any]] = {}
    for name in args.models:
        print(f"  {name}...")
        llm = OllamaLLM.from_config(config, model=name, **overrides)
        try:
            results[name] = await run_model(llm, questions, prompt, max_words, args.verbose)
        except LanguageModelError as exc:
            # One missing model should not throw away the measurements of the rest.
            print(f"    skipped: {exc}")

    if not results:
        return 1

    print()
    print(table(results, len(questions)))
    for name, row in results.items():
        if row["problems"]:
            detail = ", ".join(f"{problem} {count}" for problem, count in row["problems"].items())
            print(f"\n{name} broke the rules with: {detail}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"llm-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nEvery answer saved in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
