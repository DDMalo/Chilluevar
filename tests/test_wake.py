"""The two detectors behind their protocols.

Neither openWakeWord nor the Silero model is installed in CI: `load_detector`
and `load_vad` are module-level functions precisely so a test can replace them,
the same arrangement as `stt.whisper.load_model` and `tts.piper.load_voice`.

What is worth testing here is not the models, which are not ours, but
everything around them: loading once, the threshold, resetting, and what is
said when something is missing.
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Any

import pytest

from chilluevar.audio import vad as vad_module
from chilluevar.audio import wake as wake_module
from chilluevar.audio.vad import FRAME_SAMPLES, SileroVAD
from chilluevar.audio.wake import OpenWakeWord, spoken_form
from chilluevar.interfaces import VoiceActivityDetector, WakeWordDetector

FRAME = b"\x00\x01" * FRAME_SAMPLES  # 480 samples, 30 ms at 16 kHz


# --------------------------------------------------------------------------- #
# The wake word
# --------------------------------------------------------------------------- #


class FakeOpenWakeWord:
    """Stands in for openWakeWord's Model."""

    def __init__(self, score: float = 0.0) -> None:
        self.score = score
        self.calls = 0
        self.resets = 0

    def predict(self, samples: Any) -> dict[str, float]:
        self.calls += 1
        return {"hey_jarvis": self.score}

    def reset(self) -> None:
        self.resets += 1


@pytest.fixture
def fake_wake(monkeypatch: pytest.MonkeyPatch) -> list[FakeOpenWakeWord]:
    built: list[FakeOpenWakeWord] = []

    def factory(model: str, *, framework: str = "onnx") -> FakeOpenWakeWord:
        made = FakeOpenWakeWord(score=0.93)
        built.append(made)
        return made

    monkeypatch.setattr(wake_module, "load_detector", factory)
    return built


def test_wake_word_satisfies_the_protocol() -> None:
    assert isinstance(OpenWakeWord(), WakeWordDetector)


def test_it_reports_the_score_for_the_frame(fake_wake: list[FakeOpenWakeWord]) -> None:
    assert OpenWakeWord().feed(FRAME, 16_000) == pytest.approx(0.93)


def test_the_model_is_loaded_once(fake_wake: list[FakeOpenWakeWord]) -> None:
    # Loading per frame would mean loading it thirty times a second.
    detector = OpenWakeWord()
    assert not detector.loaded
    detector.feed(FRAME, 16_000)
    detector.feed(FRAME, 16_000)
    assert len(fake_wake) == 1
    assert fake_wake[0].calls == 2


def test_resetting_before_anything_was_heard_loads_nothing(
    fake_wake: list[FakeOpenWakeWord],
) -> None:
    # The listener resets on every mute, which happens before the first frame
    # on a machine that has not spoken yet.
    detector = OpenWakeWord()
    detector.reset()
    assert fake_wake == []
    assert not detector.loaded


def test_resetting_clears_the_buffer(fake_wake: list[FakeOpenWakeWord]) -> None:
    detector = OpenWakeWord()
    detector.feed(FRAME, 16_000)
    detector.reset()
    assert fake_wake[0].resets == 1


def pretend_openwakeword(monkeypatch: pytest.MonkeyPatch, *, installed: bool, model: Any) -> None:
    """Decide what `from openwakeword.model import Model` does.

    Without this the result depends on whether the machine running the tests
    happens to have the extra, which is the opposite of what a test is for.
    """
    if not installed:
        monkeypatch.setitem(sys.modules, "openwakeword.model", None)
        return
    module = types.ModuleType("openwakeword.model")
    module.Model = model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openwakeword.model", module)


def test_a_model_that_cannot_be_loaded_says_how_to_get_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The models are fetched once and are not shipped in the package.

    Without this the failure is a stack trace from inside openWakeWord, which
    is the same dead end as the missing Piper voice in #6.
    """

    def explode(**kwargs: Any) -> Any:
        raise FileNotFoundError("hey_jarvis_v0.1.onnx")

    pretend_openwakeword(monkeypatch, installed=True, model=explode)

    with pytest.raises(RuntimeError) as error:
        wake_module.load_detector("hey_jarvis")

    assert "download_models" in str(error.value)


def test_without_the_extra_it_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    pretend_openwakeword(monkeypatch, installed=False, model=None)

    with pytest.raises(RuntimeError, match=r'pip install -e ".\[wake\]"'):
        wake_module.load_detector("hey_jarvis")


@pytest.mark.parametrize(
    ("model", "said"),
    [("hey_jarvis", "hey jarvis"), ("alexa", "alexa"), ("hey_mycroft", "hey mycroft")],
)
def test_the_model_name_is_turned_into_something_sayable(model: str, said: str) -> None:
    assert spoken_form(model) == said


# --------------------------------------------------------------------------- #
# Voice activity
# --------------------------------------------------------------------------- #


class FakeSilero:
    """Stands in for openWakeWord's bundled Silero VAD."""

    def __init__(self, probability: float = 0.0) -> None:
        self.probability = probability
        self.calls = 0
        self.resets = 0

    def predict(self, samples: Any) -> float:
        self.calls += 1
        return self.probability

    def reset_states(self) -> None:
        self.resets += 1


@pytest.fixture
def fake_silero(monkeypatch: pytest.MonkeyPatch) -> list[FakeSilero]:
    built: list[FakeSilero] = []

    def factory() -> FakeSilero:
        made = FakeSilero(probability=0.8)
        built.append(made)
        return made

    monkeypatch.setattr(vad_module, "load_vad", factory)
    return built


def test_vad_satisfies_the_protocol() -> None:
    assert isinstance(SileroVAD(), VoiceActivityDetector)


@pytest.mark.parametrize(("threshold", "speech"), [(0.5, True), (0.9, False)])
def test_the_threshold_decides(
    fake_silero: list[FakeSilero], threshold: float, speech: bool
) -> None:
    assert SileroVAD(threshold=threshold).is_speech(FRAME, 16_000) is speech


def test_the_model_is_loaded_once_too(fake_silero: list[FakeSilero]) -> None:
    vad = SileroVAD()
    vad.is_speech(FRAME, 16_000)
    vad.is_speech(FRAME, 16_000)
    assert len(fake_silero) == 1
    assert fake_silero[0].calls == 2


def test_the_wrong_sample_rate_is_refused(fake_silero: list[FakeSilero]) -> None:
    # No silent resampling here either: it would show up later as sentences
    # ending in odd places, with nothing pointing at the cause.
    with pytest.raises(ValueError, match="16000 Hz"):
        SileroVAD().is_speech(FRAME, 44_100)


def test_a_frame_of_the_wrong_length_says_what_to_change(
    fake_silero: list[FakeSilero], caplog: pytest.LogCaptureFixture
) -> None:
    vad = SileroVAD()
    with caplog.at_level(logging.WARNING, logger="chilluevar.audio.vad"):
        assert vad.is_speech(b"\x00\x01" * 100, 16_000) is False

    assert "frame_ms to 30" in caplog.text
    assert not vad.loaded, "nothing to ask the model about"


def test_a_short_frame_keeps_the_previous_answer(fake_silero: list[FakeSilero]) -> None:
    # Flipping to "not speech" would end the sentence early; keeping the last
    # answer degrades instead of breaking.
    vad = SileroVAD()
    assert vad.is_speech(FRAME, 16_000) is True
    assert vad.is_speech(b"\x00\x01" * 100, 16_000) is True


def test_resetting_clears_the_recurrent_state(fake_silero: list[FakeSilero]) -> None:
    vad = SileroVAD()
    vad.is_speech(FRAME, 16_000)
    vad.reset()
    assert fake_silero[0].resets == 1


def test_resetting_the_voice_detector_early_loads_nothing(
    fake_silero: list[FakeSilero],
) -> None:
    vad = SileroVAD()
    vad.reset()
    assert fake_silero == []


def test_the_voice_detector_without_the_extra_names_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "openwakeword.vad", None)

    with pytest.raises(RuntimeError, match=r'pip install -e ".\[wake\]"'):
        vad_module.load_vad()
