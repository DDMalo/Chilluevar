"""Silero behind the VoiceActivityDetector protocol.

It answers one question per frame: is anyone talking in this one. The listener
uses that to find where a sentence ends, which is what replaces letting go of a
key.

We reach Silero through openWakeWord rather than through the `silero-vad`
package. Same model, and it avoids pulling in torch for a 1.8 MB network: the
wake word already needs openWakeWord, which ships this model and runs it on
onnxruntime. It also takes 480-sample frames, which is exactly 30 ms at 16 kHz
and exactly what our sources produce, while the standalone package insists on
512 and would need a buffer in front of it.
"""

from __future__ import annotations

import logging
from typing import Any

__all__ = ["SileroVAD", "load_vad"]

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
# What the bundled model expects, and what a 30 ms frame at 16 kHz contains.
FRAME_SAMPLES = 480


def load_vad() -> Any:
    """Build the voice activity detector.

    A module-level function so the tests replace it without openWakeWord
    installed, like the other engines.
    """
    try:
        from openwakeword.vad import VAD
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(
            'voice activity detection needs the wake extra: pip install -e ".[wake]"'
        ) from exc
    return VAD()


class SileroVAD:
    """Says whether a frame contains speech."""

    def __init__(self, *, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._model: Any | None = None
        self._last = False

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"the voice detector expects {SAMPLE_RATE} Hz, got {sample_rate}")

        samples = _to_int16(frame)
        if len(samples) != FRAME_SAMPLES:
            # A source whose frame_ms is not 30. Rather than resample or buffer
            # silently, keep the previous answer and say so once: the fix is a
            # line of configuration, and guessing here would show up later as
            # sentences cut in odd places.
            logger.warning(
                "the voice detector wants %d samples per frame and got %d; "
                "set audio.frame_ms to 30",
                FRAME_SAMPLES,
                len(samples),
            )
            return self._last

        self._last = float(self._ensure_model().predict(samples)) >= self.threshold
        return self._last

    def reset(self) -> None:
        """Forget the audio heard so far.

        The model is a recurrent network: it carries state from one frame to
        the next, which is what lets it tell a pause inside a word from the end
        of a sentence. After the assistant speaks, that state is the
        assistant's own voice, so it is cleared before listening again.
        """
        if self._model is not None:
            self._model.reset_states()
        self._last = False

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._model = load_vad()
            logger.info("loaded the silero voice detector")
        return self._model


def _to_int16(frame: bytes) -> Any:
    import numpy as np

    return np.frombuffer(frame, dtype=np.int16)
