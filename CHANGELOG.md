# Changelog

Based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org).

## [Unreleased]

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
