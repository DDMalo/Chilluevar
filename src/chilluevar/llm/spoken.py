"""Does this answer sound right read out loud?

The system prompt asks for short answers with no lists, markdown or symbols.
This module checks whether the model listened. It is what turns "this model
seems to answer well" into a number we can compare across models.

It only detects; it does not repair. Cleaning text for the synthesiser (numbers,
abbreviations, leftover symbols) is the speech-synthesis side's job (issue #6),
and a model that needs repairing all the time is a model to replace, which is
exactly what we want this check to tell us.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["Problem", "speech_problems", "word_count"]


class Problem:
    """Names of the problems, as they appear in the results."""

    LIST = "list"
    MARKDOWN = "markdown"
    SYMBOL = "symbol"
    LINK = "link"
    TOO_LONG = "too_long"


# A line that starts like a list item: "- ", "* ", "• ", "1. ", "2) ".
_LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
# Bold, italics with underscores, inline code, headings and table rows.
_MARKDOWN = re.compile(r"\*\*|__|`|^\s*#{1,6}\s|^\s*\|.*\|\s*$", re.MULTILINE)
_LINK = re.compile(r"https?://|www\.", re.IGNORECASE)
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
# Characters a speech synthesiser either skips or reads aloud by name. Not the
# ordinary punctuation of a sentence, which it uses for intonation.
_SYMBOLS = set("%€$&@/\\=<>+*#_~^{}[]|")
_WORD = re.compile(r"\w+")


def word_count(text: str) -> int:
    return len(_WORD.findall(text))


def _has_symbol(text: str) -> bool:
    # "So" (Symbol, other) is the Unicode category of emojis, and of signs like
    # ° and ™, which should have been written as words.
    return any(char in _SYMBOLS or unicodedata.category(char) == "So" for char in text)


def speech_problems(text: str, *, max_words: int | None = None) -> list[str]:
    """Return what would sound wrong if this text were spoken, or [] if nothing.

    The checks are deliberately strict: a false alarm costs a glance at the
    answer, while a missed list costs the user hearing "guion, guion, guion".
    """
    problems = []
    if _LIST_ITEM.search(text):
        problems.append(Problem.LIST)
    if _MARKDOWN.search(text):
        problems.append(Problem.MARKDOWN)
    if _LINK.search(text):
        problems.append(Problem.LINK)
    # A link is full of slashes and "**bold**" is full of asterisks. Take out
    # what was already reported, so one mistake is not counted twice.
    rest = _MARKDOWN.sub(" ", _LIST_ITEM.sub(" ", _URL.sub(" ", text)))
    if _has_symbol(rest):
        problems.append(Problem.SYMBOL)
    if max_words is not None and word_count(text) > max_words:
        problems.append(Problem.TOO_LONG)
    return problems
