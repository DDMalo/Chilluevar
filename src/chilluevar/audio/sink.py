"""Audio sinks: the speaker, and a list for tests.

Everything queued is played one clip at a time. Without that, two replies that
arrive close together are played on top of each other and the assistant becomes
unintelligible, which is exactly the kind of bug that only appears once the
model starts streaming sentences in issue #15.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from types import ModuleType, TracebackType
from typing import Self

__all__ = ["MemorySink", "SpeakerSink"]

logger = logging.getLogger(__name__)


def _require_sounddevice() -> ModuleType:
    try:
        import sounddevice
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError('playback needs the audio extra: pip install -e ".[audio]"') from exc
    return sounddevice


class QueuedSink(ABC):
    """Serialises playback through a queue and one worker task.

    `play()` returns as soon as the audio is queued, so the pipeline never
    blocks waiting for the speaker. `drain()` is there for the cases that do
    need to wait: tests, and shutting down without cutting a sentence in half.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[bytes, int]] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Self:
        self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    async def play(self, pcm: bytes, sample_rate: int) -> None:
        self.start()
        await self._queue.put((pcm, sample_rate))

    async def drain(self) -> None:
        """Wait until everything queued has been played."""
        await self._queue.join()

    async def aclose(self) -> None:
        """Finish what is queued, then stop the worker."""
        await self.drain()
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    async def _run(self) -> None:
        while True:
            pcm, sample_rate = await self._queue.get()
            try:
                await self._write(pcm, sample_rate)
            except Exception:
                # One clip failing must not kill the worker, or the assistant
                # goes permanently mute after a single glitch.
                logger.exception("playback failed")
            finally:
                self._queue.task_done()

    @abstractmethod
    async def _write(self, pcm: bytes, sample_rate: int) -> None:
        """Actually produce sound. The only part that differs per sink."""


class SpeakerSink(QueuedSink):
    """Plays through an output device."""

    def __init__(self, *, device: str | int | None = None) -> None:
        super().__init__()
        self._device = device

    async def _write(self, pcm: bytes, sample_rate: int) -> None:
        sd = _require_sounddevice()
        import numpy as np

        samples = np.frombuffer(pcm, dtype=np.int16)
        # sounddevice.play is blocking, so it goes to a worker thread; running
        # it inline would freeze the event loop for the length of the clip.
        await asyncio.to_thread(self._play_blocking, sd, samples, sample_rate)

    def _play_blocking(self, sd: ModuleType, samples: object, sample_rate: int) -> None:
        sd.play(samples, samplerate=sample_rate, device=self._device, blocking=True)


class MemorySink(QueuedSink):
    """Keeps what it was asked to play, for tests.

    Satisfies the same protocol as the speaker, so the queueing behaviour is
    tested for real without a sound card.
    """

    def __init__(self) -> None:
        super().__init__()
        self.played: list[tuple[bytes, int]] = []

    async def _write(self, pcm: bytes, sample_rate: int) -> None:
        self.played.append((pcm, sample_rate))
