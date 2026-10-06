"""Speech-to-text tests.

No model is ever downloaded and faster-whisper does not even have to be
installed: `load_model` is replaced by a fake that records how it was called.
That is the whole point of keeping the loader as a module-level function.
"""

from __future__ import annotations

from typing import Any

import pytest

from chilluevar.interfaces import SpeechToText
from chilluevar.stt import WhisperSTT
from chilluevar.stt import whisper as whisper_module


class FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeModel:
    """Stands in for faster-whisper's WhisperModel."""

    def __init__(self, name: str, device: str, compute_type: str) -> None:
        self.name = name
        self.device = device
        self.compute_type = compute_type
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append({"samples": len(audio), **kwargs})
        return iter([FakeSegment(" enciende la luz "), FakeSegment(" del salon ")]), None


@pytest.fixture
def fake_whisper(monkeypatch: pytest.MonkeyPatch) -> list[FakeModel]:
    built: list[FakeModel] = []

    def factory(name: str, device: str, compute_type: str) -> FakeModel:
        model = FakeModel(name, device, compute_type)
        built.append(model)
        return model

    monkeypatch.setattr(whisper_module, "load_model", factory)
    return built


def silence(seconds: float = 1.0, sample_rate: int = 16_000) -> bytes:
    return b"\x00" * int(seconds * sample_rate) * 2


async def test_transcribes_and_joins_segments(fake_whisper: list[FakeModel]) -> None:
    stt = WhisperSTT("base")
    assert await stt.transcribe(silence(), 16_000) == "enciende la luz del salon"


async def test_model_is_loaded_once_and_reused(fake_whisper: list[FakeModel]) -> None:
    # Loading per call would add seconds to every turn.
    stt = WhisperSTT("base")
    assert not stt.loaded
    await stt.transcribe(silence(), 16_000)
    await stt.transcribe(silence(), 16_000)
    assert len(fake_whisper) == 1
    assert len(fake_whisper[0].calls) == 2


async def test_model_settings_reach_the_engine(fake_whisper: list[FakeModel]) -> None:
    stt = WhisperSTT("tiny", device="cpu", compute_type="int8", beam_size=1)
    await stt.transcribe(silence(), 16_000)
    model = fake_whisper[0]
    assert (model.name, model.device, model.compute_type) == ("tiny", "cpu", "int8")
    assert model.calls[0]["beam_size"] == 1
    assert model.calls[0]["language"] == "es"


async def test_hints_become_the_initial_prompt(fake_whisper: list[FakeModel]) -> None:
    stt = WhisperSTT("base")
    await stt.transcribe(silence(), 16_000, hints=["Chilluevar", "salon", "cocina"])
    assert fake_whisper[0].calls[0]["initial_prompt"] == "Chilluevar, salon, cocina."


async def test_no_hints_means_no_prompt(fake_whisper: list[FakeModel]) -> None:
    stt = WhisperSTT("base")
    await stt.transcribe(silence(), 16_000)
    assert fake_whisper[0].calls[0]["initial_prompt"] is None


async def test_rejects_the_wrong_sample_rate(fake_whisper: list[FakeModel]) -> None:
    # Resampling behind our back would hide the bug until accuracy dropped.
    stt = WhisperSTT("base")
    with pytest.raises(ValueError, match="16000 Hz"):
        await stt.transcribe(silence(sample_rate=44_100), 44_100)


async def test_empty_audio_returns_empty_text(fake_whisper: list[FakeModel]) -> None:
    stt = WhisperSTT("base")
    assert await stt.transcribe(b"", 16_000) == ""
    assert not stt.loaded  # and does not even load the model


async def test_satisfies_the_protocol() -> None:
    assert isinstance(WhisperSTT("base"), SpeechToText)


def test_prompt_drops_blanks_and_duplicates() -> None:
    assert whisper_module.build_prompt(["", "  ", "luz", "luz"]) == "luz."
    assert whisper_module.build_prompt([]) is None


def test_pcm_is_scaled_to_minus_one_to_one() -> None:
    import numpy as np

    pcm = np.array([0, 32767, -32768], dtype=np.int16).tobytes()
    audio = whisper_module.to_float32(pcm)
    assert audio.dtype == np.float32
    assert audio[0] == 0.0
    assert 0.99 < audio[1] <= 1.0
    assert audio[2] == -1.0
