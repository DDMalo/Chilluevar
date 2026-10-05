# 0002 · Every engine lives behind a `Protocol`

- **Date:** 2026-10-01
- **Status:** accepted

## Problem

We are going to swap engines many times: comparing Whisper model sizes, trying
several models in Ollama, maybe replacing Piper. If the pipeline knows about
faster-whisper, every comparison is a change to the pipeline.

## Decision

Everything swappable is declared in `interfaces.py` as a `typing.Protocol`:
speech to text, language model, speech synthesis, wake word, voice activity
detection and tools. Adding an engine means writing one class and changing one
line of configuration.

We use `Protocol` rather than abstract base classes: an implementation does not
have to inherit from or import anything here, it just needs the right methods.
That keeps the dependency arrow pointing one way and makes a fake engine for a
test trivial to write.

## Consequences

**For**

- Comparing models is a configuration change, and that is precisely the
  experiment we want to repeat many times.
- CI can run without models: tests use fakes that satisfy the protocols.
- `LanguageModel.stream()` yields fragments as they are produced. Speaking the
  first sentence while the model writes the second is the single biggest cut to
  perceived latency, so it is in the interface from the start rather than being
  retrofitted.

**Against**

- An abstraction layer before a single implementation exists, which is exactly
  the mistake that usually goes wrong. We accept it because we know there will
  be several implementations of each: comparing engines *is* the project.

## Rejected alternative

**Call each library directly and refactor when needed.** Faster at first, but
the first model comparison already forces the refactor, and it would leave CI
depending on downloading models.
