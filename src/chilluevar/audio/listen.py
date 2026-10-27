"""Turning a microphone that is always on into one utterance at a time.

Until now a turn began with a key press: open the device, record, close it.
Hands-free there is no key, the device stays open, and the frames have to reach
three different things at different moments: the wake word detector while the
assistant is idle, the voice activity detector while someone is talking, and
the buffer keeping what was said.

So one object owns the source and runs a small state machine over the frames.
See docs/decisions/0007-one-owner-for-the-microphone.md for why the detector is
fed frames instead of opening a source of its own.

The state machine is deliberately synchronous and takes one frame at a time.
Everything that can go wrong here is a transition, and transitions are tested
without a device, without audio and without waiting in real time.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum, auto
from types import TracebackType
from typing import Self

from chilluevar.interfaces import AudioSource, VoiceActivityDetector, WakeWordDetector

__all__ = ["Listener", "ListenerStats"]

logger = logging.getLogger(__name__)

# Frames kept after the last one the voice detector called speech. A fricative
# at the end of a word is quiet enough to be marked silence, and clipping it
# changes the word.
_TAIL_FRAMES = 2


class _State(Enum):
    WAITING = auto()  # idle, feeding the wake word detector
    ARMED = auto()  # the wake word fired, waiting for the sentence to begin
    RECORDING = auto()  # keeping frames until the sentence ends


@dataclass(slots=True)
class ListenerStats:
    """Counters worth seeing, and what the threshold tuning will be read from.

    `false_activations` is the number that matters: the wake word fired and no
    one said anything afterwards. Counting it over an hour of ordinary noise in
    the flat is how the threshold gets chosen, instead of guessed.
    """

    activations: int = 0
    false_activations: int = 0
    utterances: int = 0
    too_short: int = 0
    truncated: int = 0


class Listener:
    """Owns an audio source and yields complete utterances.

    Used as:

        async with Listener(source, wake=detector, vad=vad) as listener:
            async for pcm in listener.utterances():
                ...
    """

    def __init__(
        self,
        source: AudioSource,
        *,
        wake: WakeWordDetector,
        vad: VoiceActivityDetector,
        threshold: float = 0.6,
        start_timeout_ms: int = 3_000,
        silence_ms: int = 800,
        min_utterance_ms: int = 400,
        max_utterance_ms: int = 15_000,
        queue_size: int = 4,
    ) -> None:
        self._source = source
        self._wake = wake
        self._vad = vad
        self.threshold = threshold
        self.start_timeout_ms = start_timeout_ms
        self.silence_ms = silence_ms
        self.min_utterance_ms = min_utterance_ms
        self.max_utterance_ms = max_utterance_ms
        self.stats = ListenerStats()

        self._state = _State.WAITING
        self._frames: list[bytes] = []
        self._waited_ms = 0
        self._silent_ms = 0
        self._muted = False
        self._task: asyncio.Task[None] | None = None
        self._ready: asyncio.Queue[bytes] = asyncio.Queue(maxsize=queue_size)

    # ----------------------------------------------------------------- #
    # Running
    # ----------------------------------------------------------------- #

    async def __aenter__(self) -> Self:
        self._task = asyncio.create_task(self._consume())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def utterances(self) -> AsyncIterator[bytes]:
        """Yield each finished utterance as 16-bit mono PCM."""
        while True:
            yield await self._ready.get()

    @property
    def waiting(self) -> int:
        """Finished utterances not yet collected."""
        return self._ready.qsize()

    def feed(self, frame: bytes) -> None:
        """Hand one frame to the state machine.

        Public because it is the entire behaviour of this class, and because a
        test that calls it directly needs no device, no task and no clock: the
        machine advances a frame at a time and time is counted in frames.
        """
        if self._muted:
            return
        self._step(frame)

    async def _consume(self) -> None:
        """Read the source for as long as the listener is open.

        Reading never stops, not even while muted. A source nobody reads backs
        up: `MicrophoneSource` starts dropping frames and saying so, and the
        audio that arrives after the assistant stops talking is a second old.
        Draining and discarding costs nothing and keeps the stream live.
        """
        async for frame in self._source.frames():
            self.feed(frame)

    # ----------------------------------------------------------------- #
    # Not hearing itself
    # ----------------------------------------------------------------- #

    @property
    def muted(self) -> bool:
        return self._muted

    def mute(self) -> None:
        """Stop detecting, usually because the assistant is about to speak.

        The speaker and the microphone share a room, so without this the
        assistant wakes itself on its own voice. The caller marks the window
        rather than the listener guessing from the bus: the caller is the one
        coroutine that knows when a turn starts and ends, so there is no
        ordering between two subscriptions to get wrong.
        """
        self._muted = True
        self._abandon()

    def unmute(self) -> None:
        self._muted = False
        # The detectors keep a rolling window of the last second or so. Without
        # clearing it, the first frames after speaking are the assistant's own
        # voice, still in the buffer.
        self._wake.reset()

    # ----------------------------------------------------------------- #
    # The state machine
    # ----------------------------------------------------------------- #

    def _step(self, frame: bytes) -> None:
        rate = self._source.sample_rate
        if self._state is _State.WAITING:
            if self._wake.feed(frame, rate) >= self.threshold:
                self._activate()
            return

        speaking = self._vad.is_speech(frame, rate)
        if self._state is _State.ARMED:
            self._armed(frame, speaking)
            return
        self._recording(frame, speaking)

    def _activate(self) -> None:
        self.stats.activations += 1
        logger.info("wake word detected")
        # The wake word itself is left out of the recording on purpose. The
        # benchmark in issue #4 showed Whisper cannot spell an invented name it
        # has never seen, so feeding it "Chilluevar" only adds a wrong word to
        # the transcript. Detecting it here is what replaces transcribing it.
        self._wake.reset()
        self._state = _State.ARMED
        self._frames = []
        self._waited_ms = 0
        self._silent_ms = 0

    def _armed(self, frame: bytes, speaking: bool) -> None:
        if speaking:
            self._state = _State.RECORDING
            self._frames = [frame]
            self._silent_ms = 0
            return

        self._waited_ms += self._frame_ms
        if self._waited_ms >= self.start_timeout_ms:
            # The wake word fired and nobody said anything: the television, a
            # similar sounding word, or a threshold set too low.
            self.stats.false_activations += 1
            logger.info("nothing was said after the wake word")
            self._idle()

    def _recording(self, frame: bytes, speaking: bool) -> None:
        self._frames.append(frame)
        self._silent_ms = 0 if speaking else self._silent_ms + self._frame_ms

        if self._silent_ms >= self.silence_ms:
            self._finish()
        elif self._recorded_ms >= self.max_utterance_ms:
            # Someone reading out a shopping list, or a radio left on. Better a
            # truncated question than a recording that grows until the machine
            # runs out of memory.
            self.stats.truncated += 1
            logger.warning("utterance hit the %.0f s limit", self.max_utterance_ms / 1000)
            self._finish()

    def _finish(self) -> None:
        # Measure the speech, not the recording. The silence that ended the
        # sentence is part of what was captured, and counting it would let a
        # cough followed by the usual pause pass for a question.
        tail_frames = self._silent_ms // self._frame_ms
        voiced = len(self._frames) - tail_frames

        if voiced * self._frame_ms < self.min_utterance_ms:
            # A cough, a door, a chair. Transcribing it wastes seconds and
            # invents a sentence out of noise.
            self.stats.too_short += 1
            logger.info("utterance too short, ignoring it")
            self._idle()
            return

        self.stats.utterances += 1
        # Keep a little past the last speech so a trailing consonant is not
        # clipped, and drop the rest. It is the same silence every single turn,
        # and transcribing it costs time for nothing.
        keep = min(len(self._frames), voiced + _TAIL_FRAMES)
        pcm = b"".join(self._frames[:keep])
        self._idle()
        if self._ready.full():
            # Turns are answered one at a time, so a queue this full means
            # someone is still talking to an assistant that stopped listening.
            # The newest utterance is the one worth keeping.
            self._ready.get_nowait()
            logger.warning("dropped an utterance: the previous one is still being answered")
        self._ready.put_nowait(pcm)

    def _abandon(self) -> None:
        """Throw away a half-recorded utterance and go back to idle."""
        if self._state is not _State.WAITING:
            logger.debug("abandoned an utterance in progress")
        self._idle()

    def _idle(self) -> None:
        self._state = _State.WAITING
        self._frames = []
        self._waited_ms = 0
        self._silent_ms = 0

    @property
    def _frame_ms(self) -> int:
        return self._source.frame_ms

    @property
    def _recorded_ms(self) -> int:
        return len(self._frames) * self._frame_ms
