"""Messages exchanged between modules.

This module is the contract between the two halves of the project: the audio
pipeline (capture, wake word, speech to text, speech synthesis) and the
reasoning core (language model, tools, memory). Both sides import from here and
from `interfaces`, and from nothing else of each other, so either half can be
developed against fake data while the other is still unfinished.

Rules for this file:

- Every event is a frozen dataclass, so a module cannot mutate an event another
  module is still holding.
- Every event carries a `turn_id`. One turn is one interaction: from the moment
  the wake word fires until the assistant stops speaking. Correlating events by
  turn is what later lets us measure the latency of each stage.
- No module-specific types leak in here (no numpy arrays, no Ollama objects).
  Audio travels as raw PCM bytes plus its sample rate.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
    "AudioCaptured",
    "Event",
    "ReplyChunk",
    "SpeakRequest",
    "SpeechFinished",
    "Stage",
    "StageTiming",
    "ToolInvoked",
    "Transcribed",
    "TurnFailed",
    "TurnStarted",
    "new_turn_id",
]


def new_turn_id() -> str:
    """Return a short identifier for one interaction."""
    return uuid.uuid4().hex[:12]


class Stage(StrEnum):
    """The pipeline stages we time separately.

    Keeping these as an enum (instead of loose strings) means a typo in a
    benchmark script is caught immediately rather than silently producing an
    extra row in the results table.
    """

    WAKE_WORD = "wake_word"
    CAPTURE = "capture"
    SPEECH_TO_TEXT = "speech_to_text"
    LANGUAGE_MODEL = "language_model"
    TOOL = "tool"
    TEXT_TO_SPEECH = "text_to_speech"


@dataclass(frozen=True, slots=True)
class Event:
    """Fields shared by every event.

    `created_at` uses a monotonic clock: it is meaningless as a wall clock time
    but it is the correct way to measure elapsed time, because it is immune to
    the system clock being adjusted mid-turn.
    """

    turn_id: str
    created_at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True, slots=True)
class TurnStarted(Event):
    """The wake word fired, or the user pressed the push-to-talk key."""

    trigger: str = "wake_word"


@dataclass(frozen=True, slots=True)
class AudioCaptured(Event):
    """Raw audio of a single utterance, already trimmed by the voice detector.

    `pcm` is 16-bit signed little-endian mono audio. We fix the format here
    rather than passing numpy arrays around so that the satellite mode can send
    the very same bytes over the network without converting anything.
    """

    pcm: bytes = b""
    sample_rate: int = 16_000


@dataclass(frozen=True, slots=True)
class Transcribed(Event):
    """What the speech-to-text engine understood."""

    text: str = ""
    language: str = "es"
    audio_duration_ms: int = 0


@dataclass(frozen=True, slots=True)
class ReplyChunk(Event):
    """A piece of the assistant's answer, as the model produces it.

    The assistant speaks sentence by sentence while the model is still writing,
    so the reply arrives in chunks. `is_last` marks the end of the turn's text.
    """

    text: str = ""
    is_last: bool = False


@dataclass(frozen=True, slots=True)
class ToolInvoked(Event):
    """A tool the model decided to call, and the result it returned.

    `ok` is False when the tool raised or was refused (for example, a sensitive
    action the user did not confirm). The model is told either way, so it can
    explain the failure out loud instead of pretending the action happened.
    """

    name: str = ""
    arguments: dict[str, object] = field(default_factory=dict)
    result: str = ""
    ok: bool = True


@dataclass(frozen=True, slots=True)
class SpeakRequest(Event):
    """Text ready to be synthesised, already cleaned of symbols and markup."""

    text: str = ""


@dataclass(frozen=True, slots=True)
class SpeechFinished(Event):
    """The assistant finished speaking this turn, or was interrupted."""

    interrupted: bool = False


@dataclass(frozen=True, slots=True)
class TurnFailed(Event):
    """A stage gave up. The assistant says something short and returns to idle."""

    stage: Stage = Stage.LANGUAGE_MODEL
    reason: str = ""


@dataclass(frozen=True, slots=True)
class StageTiming(Event):
    """How long one stage of one turn took.

    Emitted by every stage from the first version onwards, so that when we get
    to optimising latency we already have a baseline to compare against instead
    of having to guess what it used to be.
    """

    stage: Stage = Stage.LANGUAGE_MODEL
    elapsed_ms: float = 0.0
