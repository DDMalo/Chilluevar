"""Speech-synthesis tests.

No voice is downloaded and piper does not have to be installed: `load_voice`
is replaced by a fake, exactly as the speech-to-text tests replace
`load_model`. Most of the file is about the text cleaner, which is pure
functions on strings and where the real work of issue #6 lives.
"""

from __future__ import annotations

from typing import Any

import pytest

from chilluevar.interfaces import TextToSpeech
from chilluevar.llm.spoken import speech_problems
from chilluevar.tts import PiperTTS, clean_for_speech, number_to_words, time_to_words
from chilluevar.tts import piper as piper_module

# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "words"),
    [
        (0, "cero"),
        (1, "uno"),
        (15, "quince"),
        (16, "dieciséis"),
        (21, "veintiuno"),
        (30, "treinta"),
        (31, "treinta y uno"),
        (100, "cien"),
        (101, "ciento uno"),
        (200, "doscientos"),
        (555, "quinientos cincuenta y cinco"),
        (1_000, "mil"),
        (1_001, "mil uno"),
        (2_500, "dos mil quinientos"),
        (1_000_000, "un millón"),
        (2_500_000, "dos millones quinientos mil"),
        (-5, "menos cinco"),
    ],
)
def test_number_to_words(value: int, words: str) -> None:
    assert number_to_words(value) == words


# --------------------------------------------------------------------------- #
# Clock times
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("hour", "minute", "words"),
    [
        (20, 15, "las ocho y cuarto"),
        (13, 0, "la una en punto"),
        (1, 30, "la una y media"),
        (9, 45, "las diez menos cuarto"),
        (23, 45, "las doce menos cuarto"),
        (0, 5, "las doce y cinco"),
        (12, 0, "las doce en punto"),
        (7, 22, "las siete y veintidós"),
    ],
)
def test_time_to_words(hour: int, minute: int, words: str) -> None:
    assert time_to_words(hour, minute) == words


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        # The article belongs to the spoken time, so the one already in the
        # sentence has to go, or it comes out as "son las las ocho".
        ("Son las 20:15.", "Son las ocho y cuarto."),
        ("A las 13:00.", "A la una en punto."),
        ("La batería está al 15%.", "La batería está al quince por ciento."),
        ("Cuesta 20€.", "Cuesta veinte euros."),
        ("Pesa 1.500 gramos.", "Pesa mil quinientos gramos."),
        ("El resultado es 3,5.", "El resultado es tres coma cinco."),
        # Apocope: masculine drops the -o, feminine does not.
        ("Pon el aire a 21°.", "Pon el aire a veintiún grados."),
        ("Faltan 21 minutos.", "Faltan veintiún minutos."),
        ("Faltan 21 horas.", "Faltan veintiuna horas."),
        ("Queda 1 semana.", "Queda una semana."),
        # Markdown the system prompt failed to prevent.
        ("**Importante**: ven.", "Importante: ven."),
        ("## Título\n\nTexto.", "Título. Texto."),
        ("- Uno\n- Dos", "Uno. Dos."),
        ("1. Uno\n2. Dos", "Uno. Dos."),
        ("[la web](https://x.com) dice", "la web dice."),
        ("Mira https://ejemplo.com/a ahora.", "Mira un enlace ahora."),
        # Abbreviations a synthesiser reads letter by letter.
        ("Son 3 aprox.", "Son tres aproximadamente."),
        ("Hola, Sr. Donoso.", "Hola, señor Donoso."),
    ],
)
def test_clean_for_speech(written: str, spoken: str) -> None:
    assert clean_for_speech(written) == spoken


def test_empty_text_stays_empty() -> None:
    assert clean_for_speech("") == ""
    assert clean_for_speech("   \n  ") == ""


def test_plain_sentence_is_left_alone() -> None:
    assert clean_for_speech("Enciende la luz del salón.") == "Enciende la luz del salón."


def test_cleaning_removes_what_the_llm_side_flags() -> None:
    """The two halves of the pipeline have to agree on what sounds wrong.

    `llm/spoken.py` decides whether an answer would sound bad read aloud; this
    module is what fixes it. If something it flags survives the cleaner, one of
    the two is wrong, and this test is what tells us.
    """
    messy = (
        "## Resultados\n\n"
        "- La batería está al 15% y quedan 20:15 horas\n"
        "- Mira **esto** en https://ejemplo.com/datos\n"
        "- Cuesta 1.500€ & pesa 3,5 kg"
    )
    assert speech_problems(messy), "the fixture should look wrong to start with"
    assert speech_problems(clean_for_speech(messy)) == []


# --------------------------------------------------------------------------- #
# The Piper engine
# --------------------------------------------------------------------------- #


class FakeConfig:
    # Deliberately not 22050, so the test proves where the number came from.
    sample_rate = 16_000


class FakeVoice:
    """Stands in for piper's PiperVoice."""

    def __init__(self) -> None:
        self.config = FakeConfig()
        self.spoken: list[str] = []

    def synthesize_stream_raw(self, text: str):
        self.spoken.append(text)
        # Two chunks, to prove they are joined rather than the first returned.
        yield b"\x01\x02"
        yield b"\x03\x04"


@pytest.fixture
def fake_piper(monkeypatch: pytest.MonkeyPatch) -> list[FakeVoice]:
    built: list[FakeVoice] = []

    def factory(voice: str, directory: Any) -> FakeVoice:
        made = FakeVoice()
        built.append(made)
        return made

    monkeypatch.setattr(piper_module, "load_voice", factory)
    return built


async def test_synthesize_joins_the_chunks(fake_piper: list[FakeVoice]) -> None:
    assert await PiperTTS().synthesize("Hola") == b"\x01\x02\x03\x04"


async def test_voice_is_loaded_once_and_reused(fake_piper: list[FakeVoice]) -> None:
    tts = PiperTTS()
    assert not tts.loaded
    await tts.synthesize("una")
    await tts.synthesize("otra")
    assert len(fake_piper) == 1
    assert len(fake_piper[0].spoken) == 2


async def test_text_is_cleaned_before_it_reaches_the_voice(
    fake_piper: list[FakeVoice],
) -> None:
    await PiperTTS().synthesize("Son las 20:15 y queda el 15%.")
    assert fake_piper[0].spoken == ["Son las ocho y cuarto y queda el quince por ciento."]


async def test_cleaning_can_be_turned_off(fake_piper: list[FakeVoice]) -> None:
    await PiperTTS(clean=False).synthesize("Son las 20:15.")
    assert fake_piper[0].spoken == ["Son las 20:15."]


async def test_empty_text_never_loads_the_voice(fake_piper: list[FakeVoice]) -> None:
    tts = PiperTTS()
    assert await tts.synthesize("   ") == b""
    assert not tts.loaded


async def test_sample_rate_comes_from_the_voice(fake_piper: list[FakeVoice]) -> None:
    tts = PiperTTS()
    await tts.synthesize("hola")
    assert tts.sample_rate == 16_000


def test_reading_the_sample_rate_does_not_load_the_voice(
    fake_piper: list[FakeVoice],
) -> None:
    # A property that loads a neural voice is a property that fires on
    # hasattr, which is what the protocol check below does.
    tts = PiperTTS()
    assert tts.sample_rate == piper_module.FALLBACK_SAMPLE_RATE
    assert not tts.loaded
    assert fake_piper == []


def test_satisfies_the_protocol() -> None:
    assert isinstance(PiperTTS(), TextToSpeech)


def test_missing_voice_says_how_to_get_it(tmp_path: Any) -> None:
    with pytest.raises(RuntimeError, match="download_voices"):
        piper_module.load_voice("es_ES-nope-medium", tmp_path)
