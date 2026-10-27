# 7. One owner for the microphone

Date: 2026-10-27

## Status

Accepted. Changes the `WakeWordDetector` protocol agreed in decision 0002.

## Context

Until now the microphone was opened for the length of one recording and closed
again: `chilluevar run` pressed a key, called `collect()`, and got a block of
audio back. Issue #8 removes the key, and with it that shape.

Hands-free, the microphone is open all the time and three things want the same
frames at different moments:

- the wake word detector, while the assistant is idle,
- the voice activity detector, while someone is talking, to decide where the
  sentence ended,
- the buffer that keeps what was said, at the same time as the detector above.

The protocol we wrote in issue #2 cannot express that:

```python
async def listen(self) -> AsyncIterator[float]:
    """Yield a confidence score each time the wake word is detected."""
```

It takes no audio, so an implementation has to open a source of its own. Two
consumers would then mean opening the device twice, which on most machines
fails outright and on the rest gives each of them half the frames.

The protocol was written before anyone had an engine to put behind it. It
survived four issues because only one thing was ever listening.

## Decision

The detector is fed frames instead of fetching them:

```python
def feed(self, frame: bytes, sample_rate: int) -> float: ...
def reset(self) -> None: ...
```

One object, `audio.listen.Listener`, owns the source. It reads frames in a
background task and runs a small state machine over them: waiting for the wake
word, armed and waiting for speech to begin, then recording until the voice
activity detector says the sentence ended. It hands finished utterances to
whoever is iterating it.

`reset()` exists because these detectors keep a rolling window of the last
second or so of audio. Without it, the frames that triggered a detection are
still in the window on the next frame and trigger it again.

## Consequences

The state machine is where the behaviour lives, and it is synchronous and takes
a frame at a time, so every transition is tested without audio, without a
device and without waiting in real time.

Not hearing itself falls out of the same ownership. The listener keeps
consuming while the assistant speaks, because a source nobody reads backs up
and starts dropping frames, but it feeds nothing to the detectors and resets
them afterwards. The caller marks that window; it is the one coroutine that
knows when a turn starts and ends, so there is no ordering to get wrong.

What this does not solve is interrupting the assistant mid-answer, which needs
the opposite: listening precisely while the speaker is playing. That is its own
issue, and it gets much cheaper after replies are streamed sentence by
sentence, because stopping then means not queueing the next piece rather than
aborting a clip already handed to the sound card.

The old protocol had no implementation, so nothing but the fakes had to change.
Deciding this in issue #4 or #5 would have been guessing; the shape only became
obvious once a second consumer existed.
