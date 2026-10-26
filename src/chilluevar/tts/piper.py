"""Piper behind the TextToSpeech protocol.

Piper runs a small neural voice locally and is fast enough to keep up with a
conversation on a plain CPU, which is why it is the synthesiser for a project
whose whole point is not needing a server.

The voice is loaded once and reused, for the same reason the Whisper model is:
paying the load time on every sentence would make the assistant feel broken.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from chilluevar.tts.clean import clean_for_speech

__all__ = ["PiperTTS", "load_voice", "voice_paths"]

logger = logging.getLogger(__name__)

DEFAULT_VOICE_DIR = Path("models/piper")
# What Piper's Spanish voices use. Only reported when the voice's .json is not
# there to be read, which means synthesis is about to fail with a better error.
FALLBACK_SAMPLE_RATE = 22_050


def voice_paths(voice: str, directory: Path = DEFAULT_VOICE_DIR) -> tuple[Path, Path]:
    """Where the two files of a Piper voice live.

    A voice is an .onnx model plus an .onnx.json describing it, including its
    sample rate. Both have to be downloaded; see the README.
    """
    model = directory / f"{voice}.onnx"
    return model, model.with_suffix(".onnx.json")


def load_voice(voice: str, directory: Path = DEFAULT_VOICE_DIR) -> Any:
    """Build a Piper voice.

    A module-level function rather than a method so tests can replace it
    without piper being installed, as `stt.whisper.load_model` does.
    """
    # The file check comes before the import on purpose: a voice that was
    # never downloaded is the more common mistake, and saying so is more useful
    # than complaining about an extra the person may well have installed.
    model, config = voice_paths(voice, directory)
    if not model.exists():
        raise RuntimeError(
            f"voice {voice} not found at {model}. Download it with:\n"
            f"  python -m piper.download_voices {voice} --data-dir {directory}"
        )

    try:
        from piper.voice import PiperVoice
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError('speech synthesis needs the tts extra: pip install -e ".[tts]"') from exc
    return PiperVoice.load(str(model), config_path=str(config))


class PiperTTS:
    """Speaks text with a local Piper voice."""

    def __init__(
        self,
        voice: str = "es_ES-sharvard-medium",
        *,
        directory: Path | str = DEFAULT_VOICE_DIR,
        clean: bool = True,
    ) -> None:
        self.voice = voice
        self.directory = Path(directory)
        self.clean = clean
        self._voice: Any | None = None
        self._sample_rate = 0

    @property
    def loaded(self) -> bool:
        return self._voice is not None

    @property
    def sample_rate(self) -> int:
        """The voice's own rate.

        Not fixed at 16 kHz like the microphone side: Piper voices are usually
        22.05 kHz, and resampling them down would lose quality for nothing. The
        sink is told the rate of each clip, so they can differ.

        Reading this must not load the model. Beyond the obvious waste, a
        property with that side effect makes `hasattr` load a neural voice,
        which is how the protocol check in the tests ended up doing it. The
        rate lives in the voice's .json, which costs nothing to read.
        """
        if self._sample_rate:
            return self._sample_rate
        _, config = voice_paths(self.voice, self.directory)
        if config.exists():
            with config.open(encoding="utf-8") as handle:
                self._sample_rate = int(json.load(handle)["audio"]["sample_rate"])
                return self._sample_rate
        return FALLBACK_SAMPLE_RATE

    def _ensure_voice(self) -> Any:
        if self._voice is None:
            started = time.monotonic()
            self._voice = load_voice(self.voice, self.directory)
            self._sample_rate = int(self._voice.config.sample_rate)
            logger.info(
                "loaded piper voice %s (%d Hz) in %.1f s",
                self.voice,
                self._sample_rate,
                time.monotonic() - started,
            )
        return self._voice

    async def synthesize(self, text: str) -> bytes:
        """Return 16-bit mono PCM for this text."""
        spoken = clean_for_speech(text) if self.clean else text
        if not spoken:
            return b""
        if spoken != text:
            logger.debug("text rewritten for speech")

        started = time.monotonic()
        # Synthesis blocks, so it goes to a worker thread. Inline it would
        # freeze the event loop for as long as the sentence takes.
        pcm = await asyncio.to_thread(self._run, spoken)
        elapsed = time.monotonic() - started

        seconds = len(pcm) / (self._sample_rate * 2) if self._sample_rate else 0.0
        logger.info(
            "spoke %.1f s of audio in %.1f s (%.1fx real time)",
            seconds,
            elapsed,
            seconds / elapsed if elapsed else 0.0,
        )
        return pcm

    def _run(self, text: str) -> bytes:
        voice = self._ensure_voice()
        return b"".join(_raw_chunks(voice, text))


def _raw_chunks(voice: Any, text: str):
    """Yield raw PCM from a Piper voice, whichever API it has.

    Piper renamed this between versions: 1.2 streams bytes from
    `synthesize_stream_raw`, 1.3 yields chunks with the audio in a field. We
    accept both rather than pinning one and breaking on the next install.
    """
    if hasattr(voice, "synthesize_stream_raw"):
        yield from voice.synthesize_stream_raw(text)
        return
    for chunk in voice.synthesize(text):
        yield getattr(chunk, "audio_int16_bytes", None) or bytes(chunk)
