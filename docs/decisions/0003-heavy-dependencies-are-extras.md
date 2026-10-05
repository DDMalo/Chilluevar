# 0003 · Heavy dependencies are optional extras

- **Date:** 2026-10-01
- **Status:** accepted

## Problem

`faster-whisper`, `sounddevice`, `piper` and `openwakeword` drag in PyTorch,
native audio bindings and models of several hundred megabytes. As ordinary
dependencies they make installing the project take minutes, make CI download
models on every run, and make the tests impossible to run on a machine with no
sound card.

## Decision

The base package depends only on `pyyaml`. The heavy parts are extras:
`[audio]`, `[stt]`, `[tts]`, `[wake]`, plus `[dev]` for the tooling.

CI installs `[dev]` only and never downloads a model or reaches the network.

## Consequences

**For**

- `pip install -e ".[dev]"` takes seconds, and CI is fast and free.
- Tests are deterministic: they do not depend on a model version or on a service
  being up.
- The satellite installs `[audio]` and `[wake]` and skips PyTorch entirely,
  which is exactly what we need on a small machine.

**Against**

- You have to remember to install the matching extra when starting each feature,
  and an `ImportError` from a missing extra is easy to misread. Engines must say
  clearly which extra to install.

## Rejected alternative

**Everything as a required dependency.** Simpler to explain, but it makes CI
slow and expensive and breaks the idea of a lightweight satellite.
