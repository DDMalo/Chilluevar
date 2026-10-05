"""The swappable parts of the assistant, as protocols.

Every engine we might replace later sits behind one of these protocols: speech
to text, the language model, speech synthesis, the wake word detector and the
voice activity detector. Swapping faster-whisper for another engine, or Ollama
for a cloud API, must mean writing one new class and changing one line of
configuration, never touching the pipeline.

We use `typing.Protocol` rather than abstract base classes: an implementation
does not have to import or inherit from anything here, it just has to have the
right methods. That keeps the dependency arrow pointing one way and makes fake
implementations for tests trivial to write.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "AudioSink",
    "AudioSource",
    "LanguageModel",
    "Message",
    "SpeechToText",
    "TextToSpeech",
    "Tool",
    "ToolCall",
    "VoiceActivityDetector",
    "WakeWordDetector",
]

# A chat message, in the shape every provider agrees on.
Message = dict[str, Any]


@runtime_checkable
class AudioSource(Protocol):
    """Where audio comes from: a microphone, or a file when testing.

    Audio arrives as a stream of small frames rather than one finished
    recording. Push-to-talk could manage with a single blob, but the voice
    activity detector has to decide *while you are speaking* where the sentence
    ends, so it needs the frames as they come. Starting here avoids rewriting
    this module later.

    Each frame is `frame_ms` of 16-bit signed little-endian mono PCM.
    """

    sample_rate: int
    frame_ms: int

    def frames(self) -> AsyncIterator[bytes]:
        """Yield audio frames until the source is closed."""
        ...


@runtime_checkable
class AudioSink(Protocol):
    """Where audio goes: the speaker, or a list when testing."""

    async def play(self, pcm: bytes, sample_rate: int) -> None:
        """Queue audio for playback and return immediately."""
        ...

    async def drain(self) -> None:
        """Wait until everything queued has finished playing."""
        ...


@runtime_checkable
class WakeWordDetector(Protocol):
    """Listens continuously and reports when the wake word is heard."""

    async def listen(self) -> AsyncIterator[float]:
        """Yield a confidence score each time the wake word is detected."""
        ...


@runtime_checkable
class VoiceActivityDetector(Protocol):
    """Decides where speech starts and stops inside a stream of audio."""

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        """Return True if this frame contains speech."""
        ...


@runtime_checkable
class SpeechToText(Protocol):
    """Turns captured audio into text."""

    async def transcribe(
        self,
        pcm: bytes,
        sample_rate: int,
        *,
        language: str = "es",
        hints: Iterable[str] = (),
    ) -> str:
        """Transcribe one utterance.

        `hints` carries words the engine is likely to mishear and that matter to
        us: the assistant's own name, the names of rooms and devices, the names
        in the calendar. Passing them raises accuracy noticeably at no cost.
        """
        ...


@runtime_checkable
class ToolCall(Protocol):
    """A tool the model asked to run."""

    name: str
    arguments: dict[str, Any]


@runtime_checkable
class Tool(Protocol):
    """Something the model can do besides talking.

    `schema` is a JSON Schema description of the arguments, which is what the
    model is actually shown. `sensitive` marks the tools that must be confirmed
    out loud before they run: anything that deletes, spends, sends or changes
    the physical state of the flat.
    """

    name: str
    description: str
    schema: dict[str, Any]
    sensitive: bool

    async def run(self, **arguments: Any) -> str:
        """Perform the action and return a short result for the model to read."""
        ...


@runtime_checkable
class LanguageModel(Protocol):
    """Decides what to say and which tools to use."""

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[str | ToolCall]:
        """Yield text fragments as they are produced, and tool calls as requested.

        Streaming is not a refinement we add later: speaking the first sentence
        while the model is still writing the second is the single biggest cut to
        perceived latency, so it belongs in the interface from the start.
        """
        ...


@runtime_checkable
class TextToSpeech(Protocol):
    """Turns text into audio."""

    async def synthesize(self, text: str) -> bytes:
        """Return 16-bit mono PCM for this text."""
        ...

    @property
    def sample_rate(self) -> int:
        """Sample rate of the audio returned by `synthesize`."""
        ...
