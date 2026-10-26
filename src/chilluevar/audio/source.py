"""Audio sources: the microphone, and a WAV file for tests.

Both satisfy the `AudioSource` protocol, so the rest of the pipeline cannot tell
them apart. That is what lets the test suite run with no microphone, no sound
card and no sound at all, which in turn is what lets CI run at all.
"""

from __future__ import annotations

import asyncio
import logging
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType

__all__ = ["MicrophoneSource", "WavFileSource", "collect"]

logger = logging.getLogger(__name__)

# The format every stage of the pipeline assumes. Fixed here on purpose: see
# docs/decisions/0004-satellite-mode-from-day-one.md
SAMPLE_WIDTH_BYTES = 2  # 16-bit
CHANNELS = 1


def _frame_bytes(sample_rate: int, frame_ms: int) -> int:
    """How many bytes one frame takes. 30 ms at 16 kHz mono 16-bit is 960."""
    return int(sample_rate * frame_ms / 1000) * SAMPLE_WIDTH_BYTES


def _require_sounddevice() -> ModuleType:
    """Import sounddevice, or explain which extra is missing.

    The package deliberately does not depend on sounddevice (decision 0003), so
    a bare ImportError here would be confusing. This turns it into an
    instruction.
    """
    try:
        import sounddevice
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(
            'microphone support needs the audio extra: pip install -e ".[audio]"'
        ) from exc
    return sounddevice


class MicrophoneSource:
    """Live audio from an input device."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        frame_ms: int = 30,
        device: str | int | None = None,
        queue_size: int = 100,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self._device = device
        self._queue_size = queue_size
        self.dropped = 0

    async def frames(self) -> AsyncIterator[bytes]:
        """Yield frames from the microphone until the caller stops consuming."""
        sd = _require_sounddevice()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=self._queue_size)

        def hand_over(data: bytes) -> None:
            # Runs on the event loop, so touching the queue here is safe.
            if queue.full():
                queue.get_nowait()
                self.dropped += 1
                logger.warning("audio queue full, dropped a frame")
            queue.put_nowait(data)

        def callback(indata, frames_count, time_info, status) -> None:
            # Careful: PortAudio calls this from its own thread, not from the
            # event loop, so we cannot touch the queue directly. Everything has
            # to be handed over with call_soon_threadsafe. Getting this wrong
            # produces corruption that only shows up under load.
            if status:
                logger.warning("audio input status: %s", status)
            loop.call_soon_threadsafe(hand_over, bytes(indata))

        stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=int(self.sample_rate * self.frame_ms / 1000),
            dtype="int16",
            channels=CHANNELS,
            device=self._device,
            callback=callback,
        )

        with stream:
            logger.info("microphone open: %s", stream.device)
            while True:
                yield await queue.get()


class WavFileSource:
    """Audio read from a .wav file, so tests never need a microphone.

    The file has to be 16-bit mono at the expected sample rate. We refuse
    anything else instead of resampling: a test that quietly converts its own
    fixture is a test that stops telling you the truth.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        sample_rate: int = 16_000,
        frame_ms: int = 30,
    ) -> None:
        self.path = Path(path)
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms

    async def frames(self) -> AsyncIterator[bytes]:
        size = _frame_bytes(self.sample_rate, self.frame_ms)
        with wave.open(str(self.path), "rb") as handle:
            self._check(handle)
            while chunk := handle.readframes(size // SAMPLE_WIDTH_BYTES):
                # The last frame of a file is usually short. Padding it with
                # silence keeps every frame the same length, which the voice
                # detector in issue #8 will rely on.
                if len(chunk) < size:
                    chunk = chunk.ljust(size, b"\x00")
                yield chunk
                # Let the event loop breathe, so a long file cannot starve
                # whatever else is running.
                await asyncio.sleep(0)

    def _check(self, handle: wave.Wave_read) -> None:
        if handle.getnchannels() != CHANNELS:
            raise ValueError(
                f"{self.path.name}: expected mono, got {handle.getnchannels()} channels"
            )
        if handle.getsampwidth() != SAMPLE_WIDTH_BYTES:
            raise ValueError(f"{self.path.name}: expected 16-bit audio")
        if handle.getframerate() != self.sample_rate:
            raise ValueError(
                f"{self.path.name}: expected {self.sample_rate} Hz, got {handle.getframerate()}"
            )


async def collect(
    source: object,
    *,
    stop: asyncio.Event | None = None,
    max_seconds: float = 30.0,
) -> bytes:
    """Gather frames into one recording.

    This is what push-to-talk uses: start consuming, keep the frames, stop when
    the user says so. `max_seconds` is a safety net so a key that never gets
    released cannot fill memory.
    """
    chunks: list[bytes] = []
    limit = int(max_seconds * 1000 / source.frame_ms)  # type: ignore[attr-defined]

    async for frame in source.frames():  # type: ignore[attr-defined]
        chunks.append(frame)
        if stop is not None and stop.is_set():
            break
        if len(chunks) >= limit:
            logger.warning("recording hit the %.0f s limit", max_seconds)
            break
    return b"".join(chunks)
