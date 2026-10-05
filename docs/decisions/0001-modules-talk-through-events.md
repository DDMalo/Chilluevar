# 0001 · Modules talk through events, not calls

- **Date:** 2026-10-01
- **Status:** accepted

## Problem

Two of us are working in parallel on one pipeline: audio, transcription, model,
speech. If each module calls the next one directly, neither of us can move until
the other has finished their half, and any change to a signature breaks the
other person's work.

## Decision

Modules do not call each other. They publish immutable events on an in-process
bus (`bus.py`) and subscribe to the ones they need. The catalogue of events
(`events.py`) is the contract, and it is the first thing we agreed on.

Every event carries a `turn_id` identifying one complete interaction, from the
wake word firing until the assistant stops speaking.

## Consequences

**For**

- Each of us can develop against fake events while the other finishes.
- Per-stage latency measurement comes for free: correlate by `turn_id`.
- Putting a network hop in the middle (satellite mode) rewrites nothing.
- Tests do not need the whole pipeline: publish one event, assert the next.

**Against**

- One more indirection to follow when reading the code for the first time.
- A contract mismatch is not caught by the interpreter; it shows up as an event
  nobody consumes. We offset that with tests over the contract itself.

## Rejected alternative

**Direct calls between modules.** Shorter to write and easier to follow in a
debugger, but it forces us to work in series and turns satellite mode into a
rewrite. With two people and a latency target, it does not pay off.

## Detail worth remembering

When a subscriber falls behind, the bus drops the **oldest** event, not the new
one, and counts it in `bus.dropped`. For audio this is correct: a stale frame is
worthless, and a bus that blocked here would stall the microphone.
