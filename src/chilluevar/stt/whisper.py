"""faster-whisper behind the SpeechToText protocol.

The model is loaded once and reused. Loading it per call would add seconds to
every single turn, which is the kind of mistake that makes a local assistant
feel broken rather than slow.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterable
from typing import Any

__all__ = ["WhisperSTT", "load_model"]

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
INT16_FULL_SCALE = 32768.0


def load_model(name: str, device: str, compute_type: str) -> Any:
    """Build a faster-whisper model.

    A module-level function rather than a method so tests can replace it
    without a real model, and without faster-whisper being installed at all.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError('speech to text needs the stt extra: pip install -e ".[stt]"') from exc
    return WhisperModel(name, device=device, compute_type=compute_type)


class WhisperSTT:
    """Transcribes 16 kHz mono PCM with a local Whisper model."""

    def __init__(
        self,
        model: str = "small",
        *,
        device: str = "auto",
        compute_type: str = "default",
        beam_size: int = 5,
    ) -> None:
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self._model: Any | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure_model(self) -> Any:
        if self._model is None:
            started = time.monotonic()
            self._model = load_model(self.model_name, self.device, self.compute_type)
            logger.info("loaded whisper %s in %.1f s", self.model_name, time.monotonic() - started)
        return self._model

    async def transcribe(
        self,
        pcm: bytes,
        sample_rate: int,
        *,
        language: str = "es",
        hints: Iterable[str] = (),
    ) -> str:
        if sample_rate != SAMPLE_RATE:
            # No silent resampling: a pipeline that quietly fixes the format is
            # a pipeline that hides the bug until accuracy mysteriously drops.
            raise ValueError(f"whisper expects {SAMPLE_RATE} Hz, got {sample_rate}")
        if not pcm:
            return ""

        audio = to_float32(pcm)
        prompt = build_prompt(hints)

        started = time.monotonic()
        # Transcription is CPU or GPU bound and blocking, so it goes to a
        # worker thread; running it inline would freeze the event loop and stop
        # the microphone mid-sentence.
        text = await asyncio.to_thread(self._run, audio, language, prompt)
        elapsed = time.monotonic() - started

        seconds = len(pcm) / (SAMPLE_RATE * 2)
        logger.info(
            "transcribed %.1f s of audio in %.1f s (%.1fx real time)",
            seconds,
            elapsed,
            seconds / elapsed if elapsed else 0.0,
        )
        return text

    def _run(self, audio: Any, language: str, prompt: str | None) -> str:
        model = self._ensure_model()
        segments, _info = model.transcribe(
            audio,
            language=language,
            beam_size=self.beam_size,
            initial_prompt=prompt,
        )
        # faster-whisper returns a generator; nothing runs until it is consumed.
        return " ".join(segment.text.strip() for segment in segments).strip()


def to_float32(pcm: bytes) -> Any:
    """Convert 16-bit PCM to the float32 in [-1, 1] that Whisper expects."""
    import numpy as np

    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / INT16_FULL_SCALE


def build_prompt(hints: Iterable[str]) -> str | None:
    """Turn the vocabulary hints into Whisper's initial_prompt.

    Whisper conditions on this text as if it were the start of the transcript,
    which biases it towards spelling these words correctly. Cheap, and it is
    what stops the assistant's own name coming out as "chillo ebar".
    """
    words = [hint.strip() for hint in hints if hint and hint.strip()]
    if not words:
        return None
    return ", ".join(dict.fromkeys(words)) + "."
