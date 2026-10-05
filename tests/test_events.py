"""The contract between modules is worth testing: both halves depend on it."""

from __future__ import annotations

import dataclasses

import pytest

from chilluevar.events import (
    AudioCaptured,
    ReplyChunk,
    Stage,
    Transcribed,
    new_turn_id,
)


def test_turn_ids_are_unique() -> None:
    ids = {new_turn_id() for _ in range(1_000)}
    assert len(ids) == 1_000


def test_events_are_immutable() -> None:
    # A module must not be able to edit an event another module still holds.
    # Sample utterances stay in Spanish on purpose: that is what the assistant
    # actually hears, and it keeps the fixtures honest about accents and
    # sentence shape.
    event = Transcribed(turn_id="abc", text="enciende la luz del salon")
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.text = "apaga la luz"  # type: ignore[misc]


def test_audio_defaults_to_16k_mono() -> None:
    # The whole pipeline assumes this format; if the default ever changes,
    # satellite mode would silently send audio the brain cannot read.
    audio = AudioCaptured(turn_id="abc", pcm=b"\x00\x01")
    assert audio.sample_rate == 16_000


def test_every_event_carries_a_turn_id() -> None:
    for event_type in (AudioCaptured, Transcribed, ReplyChunk):
        names = {f.name for f in dataclasses.fields(event_type)}
        assert "turn_id" in names, event_type.__name__


def test_stage_values_are_stable() -> None:
    # Benchmark results are stored under these names, so renaming one would
    # break comparisons against earlier runs.
    assert Stage.SPEECH_TO_TEXT.value == "speech_to_text"
    assert Stage.TEXT_TO_SPEECH.value == "text_to_speech"
