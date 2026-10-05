"""Configuration bugs are the kind that waste an afternoon, so they get tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from chilluevar.config import Config, load_config


def test_defaults_are_valid() -> None:
    load_config().validate()


def test_nested_values_come_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "chilluevar.yaml"
    path.write_text(
        "mode: local\nllm:\n  model: llama3.2:3b\nstt:\n  model: base\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.llm.model == "llama3.2:3b"
    assert config.stt.model == "base"
    # Values the file did not mention keep their defaults.
    assert config.tts.engine == "piper"


def test_unknown_keys_are_ignored(tmp_path: Path) -> None:
    # A config written for a later version must still start an older checkout.
    path = tmp_path / "chilluevar.yaml"
    path.write_text("mode: local\nsomething_from_the_future: 42\n", encoding="utf-8")
    assert load_config(path).mode == "local"


def test_satellite_mode_requires_a_brain_url() -> None:
    with pytest.raises(ValueError, match="brain_url"):
        Config(mode="satellite").validate()


def test_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        Config(mode="cerebro").validate()


def test_rejects_an_impossible_threshold() -> None:
    config = Config()
    config.wake_word.threshold = 1.5
    with pytest.raises(ValueError, match="threshold"):
        config.validate()


def test_environment_overrides_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "chilluevar.yaml"
    path.write_text("mode: local\n", encoding="utf-8")
    monkeypatch.setenv("CHILLUEVAR_MODE", "satellite")
    monkeypatch.setenv("CHILLUEVAR_BRAIN_URL", "ws://192.168.1.50:8765")
    config = load_config(path)
    assert config.mode == "satellite"
    assert config.brain_url == "ws://192.168.1.50:8765"


def test_example_config_is_loadable() -> None:
    # The file we ship must actually work; otherwise the quickstart in the
    # README is wrong and nobody can run the project.
    load_config(Path(__file__).parent.parent / "config" / "chilluevar.example.yaml")
