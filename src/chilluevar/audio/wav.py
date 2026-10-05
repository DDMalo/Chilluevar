"""Writing PCM back out as a .wav file.

Used by the `record` command to save what it captured, and by the tests to
build their fixtures, so both agree on the format.
"""

from __future__ import annotations

import wave
from pathlib import Path

__all__ = ["save_wav"]

SAMPLE_WIDTH_BYTES = 2
CHANNELS = 1


def save_wav(path: str | Path, pcm: bytes, sample_rate: int = 16_000) -> Path:
    """Write 16-bit mono PCM to a .wav file."""
    target = Path(path)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH_BYTES)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return target
