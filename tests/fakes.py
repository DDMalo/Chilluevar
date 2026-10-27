"""Stand-ins for the real engines, so the pipeline can be tested without them.

Every protocol in `interfaces.py` has a fake here. They are what lets CI test
the assembled assistant on a machine with no microphone, no speaker, no models
on disk and no network: the thing being tested is the wiring, and the wiring
does not care whether the transcript came from Whisper or from a list.

Each fake takes a `fails` exception. Set it and that engine raises instead of
answering, which is how the failure paths get tested: every one of them is a
real thing that happens (Ollama not running, the voice not downloaded, a
sentence the microphone did not catch), and an assistant that goes silent when
one happens is worse than one that says it did not understand.

This module is imported by the tests, never by `src/`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from chilluevar.bus import EventBus
from chilluevar.events import Event
from chilluevar.interfaces import Message, Tool

__all__ = [
    "QUIET",
    "SAMPLE_RATE",
    "SPEECH",
    "WAKE",
    "FakeAudioSource",
    "FakeLLM",
    "FakeSTT",
    "FakeSink",
    "FakeTTS",
    "FakeToolCall",
    "FakeVAD",
    "FakeWakeWord",
    "collect_events",
    "silence",
]

SAMPLE_RATE = 16_000
SAMPLE_WIDTH_BYTES = 2

# Frames that say what is in them. The listener counts time in frames rather
# than bytes, so a marker carries everything its tests need and reads as the
# scenario it describes: WAKE, SPEECH, SPEECH, QUIET, QUIET is a sentence.
WAKE = b"<wake>"
SPEECH = b"<talk>"
QUIET = b"<quiet>"


def silence(seconds: float = 1.0, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Audio of the right length and format, saying nothing.

    The fakes never listen to it, but the pipeline passes it around and the
    assertions about duration need it to be the size it claims to be.
    """
    return b"\x00" * int(seconds * sample_rate) * SAMPLE_WIDTH_BYTES


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #


class FakeAudioSource:
    """Yields a fixed list of frames and then stops, like a very short file."""

    def __init__(
        self,
        frames_to_yield: Iterable[bytes] = (),
        *,
        sample_rate: int = SAMPLE_RATE,
        frame_ms: int = 30,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self._frames = list(frames_to_yield)

    async def frames(self) -> AsyncIterator[bytes]:
        for frame in self._frames:
            yield frame


class FakeWakeWord:
    """Fires on frames that say they are the wake word.

    A real detector scores every frame against a model of the word. Scoring is
    not what the listener's tests are about: what matters is what it does the
    frame after one fires.
    """

    def __init__(self, *, marker: bytes = WAKE, score: float = 1.0) -> None:
        self.marker = marker
        self.score = score
        self.fed: list[bytes] = []
        self.resets = 0

    def feed(self, frame: bytes, sample_rate: int) -> float:
        self.fed.append(frame)
        return self.score if frame == self.marker else 0.0

    def reset(self) -> None:
        self.resets += 1


class FakeVAD:
    """Says a frame is speech when the frame says so."""

    def __init__(self, *, marker: bytes = SPEECH) -> None:
        self.marker = marker
        self.fed: list[bytes] = []
        self.resets = 0

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        self.fed.append(frame)
        return frame == self.marker

    def reset(self) -> None:
        self.resets += 1


class FakeSink:
    """Keeps what it was told to play instead of opening a sound card.

    `MemorySink` in `audio/sink.py` does the same for the tests of the sink
    itself, but it runs a worker task and has to be started and closed. Here the
    pipeline is what is under test, so the simpler the sink the better.
    """

    def __init__(self) -> None:
        self.played: list[tuple[bytes, int]] = []
        self.drains = 0

    async def play(self, pcm: bytes, sample_rate: int) -> None:
        self.played.append((pcm, sample_rate))

    async def drain(self) -> None:
        self.drains += 1


# --------------------------------------------------------------------------- #
# Speech to text
# --------------------------------------------------------------------------- #


class FakeSTT:
    """Returns scripted transcripts, in order.

    Once the script runs out it keeps returning the last line, so a test about
    something else does not have to script a transcript per turn.
    """

    def __init__(self, *transcripts: str, fails: Exception | None = None) -> None:
        self.transcripts = list(transcripts) or [""]
        self.fails = fails
        self.calls: list[dict[str, Any]] = []

    async def transcribe(
        self,
        pcm: bytes,
        sample_rate: int,
        *,
        language: str = "es",
        hints: Iterable[str] = (),
    ) -> str:
        self.calls.append(
            {
                "bytes": len(pcm),
                "sample_rate": sample_rate,
                "language": language,
                "hints": list(hints),
            }
        )
        if self.fails is not None:
            raise self.fails
        index = min(len(self.calls) - 1, len(self.transcripts) - 1)
        return self.transcripts[index]


# --------------------------------------------------------------------------- #
# The language model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FakeToolCall:
    """A tool call, as the model would ask for it.

    Nothing runs tools yet (issue #10), but `LanguageModel.stream` is typed to
    yield `str | ToolCall`, and a fake that can only yield strings would let the
    pipeline quietly stop handling the other half of its own contract.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


class FakeLLM:
    """Streams a scripted answer, in several pieces, like a real model does.

    The pieces matter: an answer that arrives whole hides every bug in the code
    that joins the fragments back together.
    """

    def __init__(
        self,
        *replies: str | list[str | FakeToolCall],
        fails: Exception | None = None,
        words_per_chunk: int = 3,
    ) -> None:
        self.replies = list(replies) or [""]
        self.fails = fails
        self.words_per_chunk = words_per_chunk
        self.conversations: list[list[Message]] = []

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[str | FakeToolCall]:
        self.conversations.append([dict(message) for message in messages])
        if self.fails is not None:
            raise self.fails
        index = min(len(self.conversations) - 1, len(self.replies) - 1)
        reply = self.replies[index]
        if isinstance(reply, list):
            for piece in reply:
                yield piece
            return
        for chunk in _in_chunks(reply, self.words_per_chunk):
            yield chunk

    @property
    def last_question(self) -> str:
        """What the user said in the most recent conversation."""
        for message in reversed(self.conversations[-1]):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""


def _in_chunks(text: str, words_per_chunk: int) -> list[str]:
    """Cut text into fragments that join back into exactly the original."""
    if not text:
        return []
    words = text.split(" ")
    groups = [words[i : i + words_per_chunk] for i in range(0, len(words), words_per_chunk)]
    chunks = [" ".join(group) for group in groups]
    # The space that split() removed goes back on the front of every piece but
    # the first, so joining the pieces gives exactly the original sentence.
    return [chunk if i == 0 else " " + chunk for i, chunk in enumerate(chunks)]


# --------------------------------------------------------------------------- #
# Speech synthesis
# --------------------------------------------------------------------------- #


class FakeTTS:
    """Turns text into bytes that depend on the text, without a voice.

    The audio is nonsense, but two different sentences give two different
    clips, which is enough for a test to prove the right text was spoken.
    """

    def __init__(self, *, sample_rate: int = 22_050, fails: Exception | None = None) -> None:
        self._sample_rate = sample_rate
        self.fails = fails
        self.spoken: list[str] = []

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def synthesize(self, text: str) -> bytes:
        if self.fails is not None:
            raise self.fails
        self.spoken.append(text)
        # One 16-bit sample per byte of text: deterministic, never empty for a
        # non-empty sentence, and different for different sentences.
        return b"".join(bytes((byte, 0)) for byte in text.encode("utf-8"))


# --------------------------------------------------------------------------- #
# Watching the bus
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def collect_events(bus: EventBus, *types: type[Event]) -> AsyncIterator[list[Event]]:
    """Gather every event of these types published inside the block, in order.

    Used as:

        async with collect_events(bus, Transcribed, TurnFailed) as seen:
            await pipeline.run_turn(...)
        assert [type(event) for event in seen] == [...]

    It records where the events are published rather than by subscribing to
    them. Subscribing looks more natural and gets the order wrong: the bus
    gives each subscriber its own queue, so a reader per type empties its whole
    queue the moment it is scheduled, and events of different types come back
    grouped instead of interleaved as they actually happened. Publishing
    Transcribed, StageTiming, ReplyChunk, StageTiming came back with both
    timings together and the chunk last.

    Real subscribers still receive everything exactly as before; what the bus
    itself does with queues is tested in test_bus.py.
    """
    collected: list[Event] = []
    published = bus.publish

    async def recording(event: Event) -> None:
        # `type(event) in types`, not isinstance: the bus delivers on the exact
        # type, and a helper that reported more than the bus delivered would be
        # lying about the thing it exists to check.
        if type(event) in types:
            collected.append(event)
        await published(event)

    bus.publish = recording  # type: ignore[method-assign]
    try:
        yield collected
    finally:
        bus.publish = published  # type: ignore[method-assign]
