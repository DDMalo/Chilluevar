# Changelog

Based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org).

## [Unreleased]

## [0.1.0] - 2026-10-26

The first version that is an assistant: press Enter, speak, and hear the answer
out loud, with everything running on this machine.

### Added

- Microphone capture and queued playback, with `chilluevar record` and
  `chilluevar devices` (#27).
- Speech to text with faster-whisper, on a CPU or a GPU, with vocabulary hints
  so the assistant's own name is not misheard, and `chilluevar transcribe` (#29,
  #30).
- `scripts/bench_stt.py`, which compares Whisper model sizes on accuracy and
  speed against our own recordings (#31).
- A client for Ollama's native API that keeps the model loaded between
  questions, a system prompt for short spoken answers in Spanish, and
  `chilluevar ask` (#32).
- `scripts/bench_llm.py` and 20 Spanish test questions, used to choose
  `qwen3:8b`, with thinking off, as the first language model (#32).
- Speech synthesis with a local Piper voice, and a cleaner that turns written
  Spanish into spoken Spanish (times, numbers, units, abbreviations, leftover
  markdown), with `chilluevar say` (#33).
- Fake engines for every protocol, so the whole pipeline is tested in CI with no
  microphone, no models and no network (#34).
- The pipeline that joins the four engines into one turn, publishing its events
  on the bus and timing every stage, and `chilluevar run`. When a stage fails,
  it says so out loud instead of going silent (#35).
- Decision records 0005 (Ollama's native API) and 0006 (the first language
  model).
- 196 tests, still with no models and no network.

### Fixed

- A failed turn is no longer buried under the stack trace of the apology that
  also failed (#36).

### Known limitations

- The answer is spoken only once the model has written all of it. Speaking it
  sentence by sentence is #15.
- There is no wake word yet: a turn starts and ends with Enter.
- The model cannot use tools yet.

## [0.0.1] - 2026-10-01

First version: the project scaffolding. There is no assistant yet, but
everything that lets two people build one in parallel is in place.

### Added

- Contracts between modules (`events.py`) and protocols for the swappable
  engines (`interfaces.py`).
- In-process event bus (`bus.py`), dropping the oldest event when a subscriber
  falls behind.
- YAML configuration with environment overrides, validation at startup and both
  local and satellite modes foreseen.
- Logging that does not record transcripts unless explicitly asked to.
- An 18-test suite with no models and no network, and CI running `ruff` and
  `pytest` on Python 3.11 and 3.12.
- Issue and pull request templates, and four decision records in
  `docs/decisions/`.
