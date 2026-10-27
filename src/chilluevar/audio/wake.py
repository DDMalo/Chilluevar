"""openWakeWord behind the WakeWordDetector protocol.

It scores every frame against a small model of one word and says how confident
it is that the word just ended. That is the whole job: deciding what to do
after a detection belongs to the listener.

Until we train a model on the assistant's own name (issue #22), the word is one
of the bundled ones. "hey jarvis" is not "Chilluevar", and saying so in the
configuration is better than pretending otherwise.
"""

from __future__ import annotations

import logging
import time
from typing import Any

__all__ = ["DEFAULT_FRAMEWORK", "OpenWakeWord", "load_detector"]

logger = logging.getLogger(__name__)

# openWakeWord defaults to tflite, which needs a runtime that is awkward to
# install on Windows. onnxruntime is a plain wheel everywhere and is already
# pulled in by the same package.
DEFAULT_FRAMEWORK = "onnx"

DOWNLOAD_HINT = 'python -c "import openwakeword.utils as u; u.download_models()"'


def load_detector(model: str, *, framework: str = DEFAULT_FRAMEWORK) -> Any:
    """Build an openWakeWord model.

    A module-level function rather than a method, so the tests replace it
    without openWakeWord installed, exactly as `stt.whisper.load_model` and
    `tts.piper.load_voice` are replaced.
    """
    try:
        from openwakeword.model import Model
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError('the wake word needs the wake extra: pip install -e ".[wake]"') from exc

    try:
        return Model(wakeword_models=[model], inference_framework=framework)
    except Exception as exc:
        # The pre-trained models are not shipped in the package; they are
        # fetched once. Saying which command does that is the difference
        # between a minute and an afternoon.
        raise RuntimeError(
            f"could not load the wake word model {model!r}: {exc}\n"
            f"The pre-trained models are downloaded once with:\n  {DOWNLOAD_HINT}"
        ) from exc


class OpenWakeWord:
    """Scores each frame for one wake word."""

    def __init__(self, model: str = "hey_jarvis", *, framework: str = DEFAULT_FRAMEWORK) -> None:
        self.model_name = model
        self.framework = framework
        self._model: Any | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def feed(self, frame: bytes, sample_rate: int) -> float:
        """Confidence that the wake word ended in this frame.

        Frames go in at whatever size the source produces: openWakeWord keeps
        its own buffer of the last second or so, which is also why `reset`
        exists.
        """
        scores = self._ensure_model().predict(_to_int16(frame))
        # One model is loaded, so there is one score. Taking the maximum keeps
        # this correct if a second word is ever added rather than silently
        # reporting whichever one happens to be first.
        return float(max(scores.values(), default=0.0))

    def reset(self) -> None:
        if self._model is not None:
            self._model.reset()

    def _ensure_model(self) -> Any:
        if self._model is None:
            started = time.monotonic()
            self._model = load_detector(self.model_name, framework=self.framework)
            logger.info(
                "loaded wake word %s (%s) in %.1f s",
                self.model_name,
                self.framework,
                time.monotonic() - started,
            )
        return self._model


def _to_int16(frame: bytes) -> Any:
    import numpy as np

    return np.frombuffer(frame, dtype=np.int16)


def spoken_form(model: str) -> str:
    """How to say a bundled model's name out loud, for the prompt on screen."""
    return model.replace("_", " ").split(" v0")[0]
