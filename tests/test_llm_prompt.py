"""The system prompt, and the check that answers sound right out loud."""

from __future__ import annotations

from datetime import date

import pytest

from chilluevar.llm import speech_problems, system_prompt
from chilluevar.llm.prompt import spanish_date
from chilluevar.llm.spoken import Problem, word_count


def test_prompt_carries_name_limit_and_date() -> None:
    prompt = system_prompt("Chilluevar", max_words=40, today=date(2026, 10, 21))
    assert "Eres Chilluevar" in prompt
    assert "40 palabras" in prompt
    assert prompt.endswith("Hoy es miércoles 21 de octubre de 2026.")
    # A placeholder left unfilled would be read to the model literally.
    assert "{" not in prompt


def test_prompt_is_the_same_all_day() -> None:
    # Ollama reuses work on an identical prompt start, so two turns of the
    # same day must send exactly the same prompt.
    assert system_prompt(today=date(2026, 10, 21)) == system_prompt(today=date(2026, 10, 21))


def test_prompt_follows_its_own_rules() -> None:
    # The model imitates the example answer more than it obeys the rules, so
    # the example must pass the same check as the model's answers.
    example = system_prompt().split("Chilluevar: ")[1].splitlines()[0]
    assert speech_problems(example, max_words=60) == []


@pytest.mark.parametrize(
    ("day", "spoken"),
    [
        (date(2026, 10, 19), "lunes 19 de octubre de 2026"),
        (date(2027, 1, 1), "viernes 1 de enero de 2027"),
        (date(2026, 12, 27), "domingo 27 de diciembre de 2026"),
    ],
)
def test_spanish_date(day: date, spoken: str) -> None:
    assert spanish_date(day) == spoken


@pytest.mark.parametrize(
    "answer",
    [
        "Son las cinco y media.",
        "Necesitas huevos, patatas, aceite y sal.",
        "El agua hierve a cien grados al nivel del mar.",
        "Hace 3,5 grados; abrígate bien.",
        "¿Quieres que te lo explique con más detalle?",
        "Es un plato frío, típico de Andalucía, hecho con tomate.",
    ],
)
def test_good_spoken_answers_pass(answer: str) -> None:
    assert speech_problems(answer, max_words=60) == []


@pytest.mark.parametrize(
    ("answer", "problem"),
    [
        ("Necesitas:\n- huevos\n- patatas", Problem.LIST),
        ("Pasos:\n1. Pela las patatas\n2. Fríelas", Problem.LIST),
        ("* Primero esto", Problem.LIST),
        ("Es **muy** fácil.", Problem.MARKDOWN),
        ("## Ingredientes\nHuevos y patatas.", Problem.MARKDOWN),
        ("Usa `pip install`.", Problem.MARKDOWN),
        ("| Planeta | Tamaño |\n| --- | --- |", Problem.MARKDOWN),
        ("Descárgalo de https://python.org", Problem.LINK),
        ("Mira en www.python.org", Problem.LINK),
        ("Es el 12% del total.", Problem.SYMBOL),
        ("Hace 20° fuera.", Problem.SYMBOL),
        ("¡Claro! 😊", Problem.SYMBOL),
        ("Cuesta 5€.", Problem.SYMBOL),
    ],
)
def test_bad_spoken_answers_are_caught(answer: str, problem: str) -> None:
    assert problem in speech_problems(answer)


def test_one_mistake_is_not_counted_twice() -> None:
    # Bold is markdown, not also a stray symbol; a link is not also slashes.
    assert speech_problems("Es **muy** fácil.") == [Problem.MARKDOWN]
    assert speech_problems("Mira https://python.org/downloads") == [Problem.LINK]
    assert speech_problems("- uno\n- dos") == [Problem.LIST]


def test_too_long() -> None:
    answer = " ".join(["palabra"] * 61)
    assert word_count(answer) == 61
    assert speech_problems(answer, max_words=60) == [Problem.TOO_LONG]
    assert speech_problems(answer) == []  # no limit given, no check
