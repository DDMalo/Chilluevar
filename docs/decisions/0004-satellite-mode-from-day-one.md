# 0004 · Satellite mode is designed in from day one

- **Date:** 2026-10-01
- **Status:** accepted

## Problem

We want the assistant to end up living on a small device in the flat, but that
device cannot run Whisper and a language model at an acceptable speed. The heavy
work has to happen on a stronger machine.

An old desktop with 8 GB and integrated graphics is useless as a brain but
perfectly good as a satellite, so we want to choose without touching code.

## Decision

The configuration has a `mode` field with two values:

- `local`: one machine does everything.
- `satellite`: this process only handles the microphone, the wake word and
  playback, and sends the rest to `brain_url`.

The field exists and is validated from v0.0, even though the implementation
arrives in v1.0. Because modules already talk through events (decision 0001) and
audio travels as raw 16 kHz PCM, splitting the pipeline is inserting a transport
in the middle of the bus, not redesigning it.

## Consequences

**For**

- Choosing where each half runs is configuration, not a refactor.
- The audio format is pinned from the start, so satellite and brain cannot drift
  apart on it.
- It works the same for a Raspberry Pi 5 and for a repurposed old PC.

**Against**

- A configuration field that for months has only one useful value, and the
  temptation to leave it half-done. It is validated (`satellite` requires
  `brain_url`) so it cannot sit in a silently broken state.

## Rejected alternative

**Build local mode only and split it when the time comes.** It would have locked
in incompatible details along the way, above all passing numpy objects between
modules instead of bytes, which is exactly what stops them being sent over the
network as they are.
