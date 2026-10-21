"""Turning written text into text that sounds right read out loud.

The system prompt asks the model for answers with no lists, markdown or
symbols, and `llm/spoken.py` measures how often it obeys. This module deals
with what gets through anyway, plus the things no prompt can fix: a synthesiser
reads "20:15" as four digits and "15%" as "quince" followed by silence.

Everything here is a pure function on strings, which is why it is the part of
the speech pipeline with the most tests and no model.
"""

from __future__ import annotations

import re

__all__ = ["clean_for_speech", "number_to_words", "time_to_words"]

# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #

_UNITS = (
    "cero",
    "uno",
    "dos",
    "tres",
    "cuatro",
    "cinco",
    "seis",
    "siete",
    "ocho",
    "nueve",
    "diez",
    "once",
    "doce",
    "trece",
    "catorce",
    "quince",
    "dieciséis",
    "diecisiete",
    "dieciocho",
    "diecinueve",
    "veinte",
    "veintiuno",
    "veintidós",
    "veintitrés",
    "veinticuatro",
    "veinticinco",
    "veintiséis",
    "veintisiete",
    "veintiocho",
    "veintinueve",
)
_TENS = (
    "",
    "",
    "",
    "treinta",
    "cuarenta",
    "cincuenta",
    "sesenta",
    "setenta",
    "ochenta",
    "noventa",
)
_HUNDREDS = (
    "",
    "ciento",
    "doscientos",
    "trescientos",
    "cuatrocientos",
    "quinientos",
    "seiscientos",
    "setecientos",
    "ochocientos",
    "novecientos",
)


def _below_thousand(value: int) -> str:
    if value < 30:
        return _UNITS[value]
    if value < 100:
        tens, unit = divmod(value, 10)
        return _TENS[tens] + (f" y {_UNITS[unit]}" if unit else "")
    if value == 100:
        return "cien"
    hundreds, rest = divmod(value, 100)
    return _HUNDREDS[hundreds] + (f" {_below_thousand(rest)}" if rest else "")


def number_to_words(value: int) -> str:
    """Spell out a whole number in Spanish.

    Handles up to the thousands of millions, which is far more than an
    assistant ever says out loud, and stops there rather than pretending to
    cover every case.
    """
    if value < 0:
        return f"menos {number_to_words(-value)}"
    if value < 1000:
        return _below_thousand(value)
    if value < 1_000_000:
        thousands, rest = divmod(value, 1000)
        head = "mil" if thousands == 1 else f"{_below_thousand(thousands)} mil"
        return head + (f" {_below_thousand(rest)}" if rest else "")
    millions, rest = divmod(value, 1_000_000)
    head = "un millón" if millions == 1 else f"{number_to_words(millions)} millones"
    return head + (f" {number_to_words(rest)}" if rest else "")


def _apocopate(words: str) -> str:
    """ "veintiuno grados" is wrong; "veintiún grados" is right.

    Spanish drops the final -o of "uno" before a masculine noun. The
    synthesiser will happily read the wrong form, so we fix it here.
    """
    if words == "uno":
        return "un"
    if words.endswith("veintiuno"):
        return words[: -len("veintiuno")] + "veintiún"
    if words.endswith(" y uno"):
        return words[: -len(" y uno")] + " y un"
    return words


# --------------------------------------------------------------------------- #
# Clock times
# --------------------------------------------------------------------------- #

_HOUR_NAMES = (
    "doce",
    "una",
    "dos",
    "tres",
    "cuatro",
    "cinco",
    "seis",
    "siete",
    "ocho",
    "nueve",
    "diez",
    "once",
)


def time_to_words(hour: int, minute: int) -> str:
    """Say a 24-hour clock time the way a person would.

    We move to the 12-hour form on purpose: "las veinte quince" is how a
    station announcer talks, not how someone tells you the time. The part of
    the day is left out because the assistant is answering about now, and
    nobody says "las ocho y cuarto de la tarde" when you can see it is dark.
    """
    hour %= 24
    minute %= 60
    name = _HOUR_NAMES[hour % 12]
    article = "la" if hour % 12 == 1 else "las"

    if minute == 0:
        return f"{article} {name} en punto"
    if minute == 15:
        return f"{article} {name} y cuarto"
    if minute == 30:
        return f"{article} {name} y media"
    if minute == 45:
        following = (hour + 1) % 24
        next_name = _HOUR_NAMES[following % 12]
        next_article = "la" if following % 12 == 1 else "las"
        return f"{next_article} {next_name} menos cuarto"
    return f"{article} {name} y {number_to_words(minute)}"


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #

# Said as words when they follow a number.
_UNITS_AFTER_NUMBER = {
    "%": ("por ciento", False),
    "€": ("euros", True),
    "$": ("dólares", True),
    "º": ("grados", True),
    "°": ("grados", True),
}

_ABBREVIATIONS = {
    r"\betc\.": "etcétera",
    r"\baprox\.": "aproximadamente",
    r"\bp\.\s*ej\.": "por ejemplo",
    r"\bSr\.": "señor",
    r"\bSra\.": "señora",
    r"\bSrta\.": "señorita",
    r"\bDr\.": "doctor",
    r"\bDra\.": "doctora",
    r"\bpág\.": "página",
    r"\bnúm\.": "número",
    r"\bnº": "número",
    r"\bEE\.\s*UU\.": "Estados Unidos",
}

_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_EMPHASIS = re.compile(r"\*\*|__|\*|_|`")
_MD_HEADING = re.compile(r"^\s*#{1,6}\s*", re.MULTILINE)
_MD_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)

# The article is part of what time_to_words returns, so a "las" already in
# the sentence has to be swallowed or it comes out as "son las las ocho".
_TIME = re.compile(r"(?:\b(?:las|la)\s+)?\b([01]?\d|2[0-3]):([0-5]\d)\b", re.IGNORECASE)
_DECIMAL = re.compile(r"\b(\d+)[.,](\d+)\b")
_THOUSANDS = re.compile(r"\b(\d{1,3})(?:\.(\d{3}))+\b")
_INTEGER = re.compile(r"\d+")
# A number followed by a unit sign, or a currency sign before it.
_NUMBER_UNIT = re.compile(r"(\d+(?:[.,]\d+)?)\s*([%€$º°])")
_UNIT_NUMBER = re.compile(r"([€$])\s*(\d+(?:[.,]\d+)?)")

_LEFTOVER_SYMBOLS = re.compile(r"[&@/\\=<>+~^{}\[\]|#]")
_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{2,}")


# "una hora" but "un minuto": the apocope of "uno" takes the gender of the
# noun that follows. These are the feminine words an assistant actually says.
_FEMININE = frozenset(
    """hora horas semana semanas vez veces persona personas cancion canciones
    alarma alarmas nota notas luz luces puerta puertas ventana ventanas
    pregunta preguntas cosa cosas""".split()
)


def _gendered_one(words: str, following: str) -> str:
    """Pick between "veintiún minutos" and "veintiuna horas"."""
    stripped = "".join(c for c in following.lower() if c.isalpha()).translate(
        str.maketrans("áéíóú", "aeiou")
    )
    if stripped in _FEMININE:
        if words == "uno":
            return "una"
        if words.endswith("veintiuno"):
            return words[: -len("veintiuno")] + "veintiuna"
        if words.endswith(" y uno"):
            return words[: -len(" y uno")] + " y una"
        return words
    return _apocopate(words)


def _spell_number(text: str) -> str:
    """Spell a number that may carry a decimal part."""
    if match := _DECIMAL.fullmatch(text):
        whole, fraction = match.groups()
        return f"{number_to_words(int(whole))} coma {number_to_words(int(fraction))}"
    return number_to_words(int(text))


def clean_for_speech(text: str) -> str:
    """Rewrite text so a speech synthesiser reads it the way a person would."""
    if not text or not text.strip():
        return ""

    # Links first: they are full of slashes and dots that every later rule
    # would otherwise try to interpret.
    text = _MD_LINK.sub(r"\1", text)
    text = _URL.sub("un enlace", text)

    for pattern, replacement in _ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)

    # Markdown that survived the system prompt. A list becomes a sentence per
    # item: read aloud, "guion" before every line is worse than a full stop.
    text = _MD_TABLE_ROW.sub(" ", text)
    text = _MD_HEADING.sub("", text)
    text = _LIST_ITEM.sub("", text)
    text = _MD_EMPHASIS.sub("", text)

    text = _TIME.sub(lambda m: time_to_words(int(m.group(1)), int(m.group(2))), text)

    # 1.500 is a thousand five hundred; 1.5 is one point five. The number of
    # digits after the dot is the only thing that tells them apart.
    text = _THOUSANDS.sub(lambda m: m.group(0).replace(".", ""), text)

    def unit_after(match: re.Match[str]) -> str:
        number, sign = match.groups()
        word, apocopate = _UNITS_AFTER_NUMBER[sign]
        spelled = _spell_number(number)
        return f"{_apocopate(spelled) if apocopate else spelled} {word}"

    def unit_before(match: re.Match[str]) -> str:
        sign, number = match.groups()
        word, _ = _UNITS_AFTER_NUMBER[sign]
        return f"{_apocopate(_spell_number(number))} {word}"

    text = _NUMBER_UNIT.sub(unit_after, text)
    text = _UNIT_NUMBER.sub(unit_before, text)

    text = _DECIMAL.sub(lambda m: _spell_number(m.group(0)), text)

    def integer(match: re.Match[str]) -> str:
        words = number_to_words(int(match.group(0)))
        following = match.string[match.end() :].lstrip().split(" ", 1)[0]
        return _gendered_one(words, following) if following else words

    text = _INTEGER.sub(integer, text)

    # Whatever symbols are left would be read by name or skipped. A full stop
    # turns the end of a list item into a pause instead of a run-on sentence.
    # A hyphen between spaces is punctuation, not part of a word: it becomes a
    # comma so the voice pauses instead of reading "guion" or skipping it.
    text = re.sub("\\s+[-\u2013\u2014]\\s+", ", ", text)
    text = _LEFTOVER_SYMBOLS.sub(" ", text)
    text = _BLANK_LINES.sub(". ", text)
    text = text.replace("\n", ". ")
    text = _SPACES.sub(" ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\.\s*\.", ".", text)
    return text.strip(" .").strip() + ("." if text.strip(" .") else "")
