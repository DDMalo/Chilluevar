"""Configuration: one YAML file, overridable by environment variables.

Everything that might differ between the two of us, between a laptop and a
satellite, or between a demo and the benchmark script, lives in the YAML file:
which engine to use, which model, which microphone, which tools are allowed.
Nothing here holds a secret. API keys and tokens come from the environment (see
`.env.example`) so that the config file can be committed and shared safely.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Self, get_type_hints

import yaml

__all__ = [
    "AudioConfig",
    "Config",
    "LLMConfig",
    "STTConfig",
    "TTSConfig",
    "WakeWordConfig",
    "load_config",
]

ENV_PREFIX = "CHILLUEVAR_"


@dataclass(slots=True)
class AudioConfig:
    # None means "whatever the operating system considers the default device".
    input_device: str | None = None
    output_device: str | None = None
    sample_rate: int = 16_000
    frame_ms: int = 30


@dataclass(slots=True)
class WakeWordConfig:
    enabled: bool = True
    # Until we train our own model (task 22) we use a bundled one.
    model: str = "hey_jarvis"
    threshold: float = 0.6


@dataclass(slots=True)
class STTConfig:
    engine: str = "faster-whisper"
    model: str = "small"
    language: str = "es"
    # Words the engine tends to mishear and that matter to us.
    hints: list[str] = field(default_factory=lambda: ["Chilluevar"])


@dataclass(slots=True)
class LLMConfig:
    engine: str = "ollama"
    model: str = "qwen2.5:7b-instruct"
    base_url: str = "http://localhost:11434"
    # Spoken answers must be short; this is a hard ceiling, not a suggestion.
    max_reply_words: int = 60
    history_turns: int = 6


@dataclass(slots=True)
class TTSConfig:
    engine: str = "piper"
    voice: str = "es_ES-sharvard-medium"


@dataclass(slots=True)
class Config:
    """The whole configuration.

    `mode` is "local" when one machine does everything and "satellite" when this
    process only handles audio and sends the rest to `brain_url`. Both modes are
    foreseen from the first version so that splitting the assistant in two later
    is a configuration change, not a refactor.
    """

    mode: str = "local"
    brain_url: str | None = None
    offline_only: bool = True
    audio: AudioConfig = field(default_factory=AudioConfig)
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    # Tools allowed to run at all, and those that must be confirmed out loud.
    allowed_tools: list[str] = field(default_factory=list)
    confirm_sensitive: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Self:
        return _build(cls, raw)

    def validate(self) -> None:
        """Fail early on a configuration that cannot work.

        Catching this at startup is much kinder than discovering it halfway
        through a turn, when the assistant would simply go quiet.
        """
        if self.mode not in {"local", "satellite"}:
            raise ValueError(f"mode must be 'local' or 'satellite', got {self.mode!r}")
        if self.mode == "satellite" and not self.brain_url:
            raise ValueError("satellite mode needs brain_url")
        if not 0.0 < self.wake_word.threshold <= 1.0:
            raise ValueError("wake_word.threshold must be between 0 and 1")
        if self.audio.sample_rate not in {8_000, 16_000, 22_050, 44_100, 48_000}:
            raise ValueError(f"unsupported sample rate: {self.audio.sample_rate}")


def _build(cls: type, raw: dict[str, Any]) -> Any:
    """Build a nested dataclass from a plain dict, ignoring unknown keys.

    Unknown keys are ignored on purpose: a config file written for a later
    version should still start an older checkout instead of crashing.

    Note the `get_type_hints` call. Because this module uses
    `from __future__ import annotations`, `Field.type` is the *string*
    "AudioConfig", not the class, so testing it with `is_dataclass` would
    silently never match and every nested section would arrive as a raw dict.
    """
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in known:
            continue
        hint = hints.get(key)
        if is_dataclass(hint) and isinstance(value, dict):
            kwargs[key] = _build(hint, value)  # type: ignore[arg-type]
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _apply_env(config: Config) -> None:
    """Override simple top-level values from the environment.

    Only the handful of values worth changing per run are supported, with
    CHILLUEVAR_ prefixed names, for example CHILLUEVAR_MODE=satellite. Anything
    deeper belongs in the YAML file, where it is readable.
    """
    if (mode := os.getenv(f"{ENV_PREFIX}MODE")) is not None:
        config.mode = mode
    if (url := os.getenv(f"{ENV_PREFIX}BRAIN_URL")) is not None:
        config.brain_url = url
    if (model := os.getenv(f"{ENV_PREFIX}LLM_MODEL")) is not None:
        config.llm.model = model


def load_config(path: str | Path | None = None) -> Config:
    """Load the configuration, or return the defaults when there is no file."""
    if path is None:
        config = Config()
    else:
        text = Path(path).read_text(encoding="utf-8")
        config = Config.from_dict(yaml.safe_load(text) or {})
    _apply_env(config)
    config.validate()
    return config
