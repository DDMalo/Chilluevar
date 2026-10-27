"""The listener, frame by frame.

The state machine counts time in frames, so with 30 ms frames a test spells out
a scenario as a list and never waits for anything: WAKE, SPEECH, SPEECH, then
enough QUIET to end the sentence.

No microphone, no models, no clock.
"""

from __future__ import annotations

import asyncio

import pytest

from chilluevar.audio.listen import Listener
from chilluevar.interfaces import VoiceActivityDetector, WakeWordDetector
from fakes import QUIET, SAMPLE_RATE, SPEECH, WAKE, FakeAudioSource, FakeVAD, FakeWakeWord

FRAME_MS = 30


def build(frames: list[bytes] | None = None, **options: object) -> Listener:
    """A listener whose source, detector and voice detector are all fakes."""
    source = FakeAudioSource(frames or [], sample_rate=SAMPLE_RATE, frame_ms=FRAME_MS)
    settings: dict[str, object] = {
        "wake": FakeWakeWord(),
        "vad": FakeVAD(),
        "silence_ms": 90,  # three quiet frames
        "min_utterance_ms": 60,  # two frames
        "start_timeout_ms": 120,  # four frames
        "max_utterance_ms": 300,  # ten frames
    }
    settings |= options
    return Listener(source, **settings)  # type: ignore[arg-type]


def play(listener: Listener, *frames: bytes) -> None:
    for frame in frames:
        listener.feed(frame)


async def spoken(listener: Listener) -> bytes:
    """The next finished utterance, or a failure rather than a hung test."""
    return await asyncio.wait_for(anext(listener.utterances()), 1)


# --------------------------------------------------------------------------- #
# The protocols
# --------------------------------------------------------------------------- #


def test_the_fakes_satisfy_the_protocols() -> None:
    assert isinstance(FakeWakeWord(), WakeWordDetector)
    assert isinstance(FakeVAD(), VoiceActivityDetector)


# --------------------------------------------------------------------------- #
# A sentence
# --------------------------------------------------------------------------- #


async def test_a_sentence_from_wake_word_to_silence() -> None:
    listener = build()

    play(listener, WAKE, SPEECH, SPEECH, SPEECH, QUIET, QUIET, QUIET)

    assert await spoken(listener) == SPEECH * 3 + QUIET * 2
    assert listener.stats.utterances == 1
    assert listener.stats.false_activations == 0


async def test_the_wake_word_is_not_in_the_recording() -> None:
    """Whisper must never be shown the wake word.

    The benchmark in issue #4 found it cannot spell a name it has never seen,
    so feeding it "Chilluevar" only adds a wrong word to the transcript.
    Detecting the word here is what replaces transcribing it.
    """
    listener = build()

    play(listener, WAKE, SPEECH, SPEECH, QUIET, QUIET, QUIET)

    assert WAKE not in await spoken(listener)


async def test_nothing_happens_until_the_wake_word() -> None:
    listener = build()

    play(listener, SPEECH, SPEECH, SPEECH, QUIET, QUIET, QUIET)

    assert listener.waiting == 0
    assert listener.stats.activations == 0


async def test_a_pause_before_speaking_is_allowed() -> None:
    # People say the word, think, and then ask. Three quiet frames here is
    # under the four-frame start timeout.
    listener = build()

    play(listener, WAKE, QUIET, QUIET, QUIET, SPEECH, SPEECH, QUIET, QUIET, QUIET)

    assert await spoken(listener) == SPEECH * 2 + QUIET * 2


async def test_a_pause_inside_a_sentence_does_not_end_it() -> None:
    # Two quiet frames is under the three that end a sentence, so the second
    # half of the sentence belongs to the same utterance.
    listener = build()

    play(listener, WAKE, SPEECH, QUIET, QUIET, SPEECH, QUIET, QUIET, QUIET)

    assert await spoken(listener) == SPEECH + QUIET * 2 + SPEECH + QUIET * 2


async def test_two_sentences_in_a_row() -> None:
    listener = build()

    play(listener, WAKE, SPEECH, SPEECH, QUIET, QUIET, QUIET)
    play(listener, WAKE, SPEECH, SPEECH, SPEECH, QUIET, QUIET, QUIET)

    assert await spoken(listener) == SPEECH * 2 + QUIET * 2
    assert await spoken(listener) == SPEECH * 3 + QUIET * 2
    assert listener.stats.utterances == 2


# --------------------------------------------------------------------------- #
# Going back to idle without a sentence
# --------------------------------------------------------------------------- #


async def test_a_wake_word_with_nothing_after_it_is_counted() -> None:
    """The number the threshold tuning is read from.

    An hour of ordinary noise in the flat with this at zero means the
    threshold is right; a handful an hour means it is too low.
    """
    listener = build()

    play(listener, WAKE, QUIET, QUIET, QUIET, QUIET, QUIET)

    assert listener.stats.false_activations == 1
    assert listener.waiting == 0


async def test_it_still_works_after_a_false_activation() -> None:
    listener = build()

    play(listener, WAKE, QUIET, QUIET, QUIET, QUIET)
    play(listener, WAKE, SPEECH, SPEECH, QUIET, QUIET, QUIET)

    assert listener.stats.false_activations == 1
    assert await spoken(listener) == SPEECH * 2 + QUIET * 2


async def test_the_silence_that_ended_the_sentence_is_trimmed() -> None:
    """It is the same silence every turn, and Whisper transcribes all of it.

    Only the tail is cut: a pause in the middle is part of the sentence, and
    two frames are kept past the last speech so a trailing consonant, which a
    voice detector often calls silence, is not clipped.
    """
    listener = build(silence_ms=180)  # six quiet frames end a sentence

    play(listener, WAKE, SPEECH, SPEECH, *[QUIET] * 6)

    assert await spoken(listener) == SPEECH * 2 + QUIET * 2


async def test_a_cough_is_not_a_question() -> None:
    # One frame of noise, under the two-frame minimum. Transcribing it wastes
    # seconds and invents a sentence out of nothing.
    listener = build()

    play(listener, WAKE, SPEECH, QUIET, QUIET, QUIET)

    assert listener.waiting == 0
    assert listener.stats.too_short == 1
    assert listener.stats.utterances == 0


async def test_a_radio_left_on_is_cut_off() -> None:
    listener = build()

    play(listener, WAKE, *[SPEECH] * 20)

    assert listener.stats.truncated == 1
    assert len(await spoken(listener)) == len(SPEECH) * 10


# --------------------------------------------------------------------------- #
# Not hearing itself
# --------------------------------------------------------------------------- #


async def test_it_does_not_wake_on_its_own_voice() -> None:
    """The classic failure once the speaker and the microphone share a room."""
    listener = build()

    listener.mute()
    play(listener, WAKE, SPEECH, SPEECH, QUIET, QUIET, QUIET)

    assert listener.stats.activations == 0
    assert listener.waiting == 0


async def test_unmuting_clears_what_the_detector_was_holding() -> None:
    # These detectors keep a rolling window of the last second or so. Without
    # clearing it, the first frames after speaking are the assistant's own
    # voice, still in the buffer.
    wake = FakeWakeWord()
    listener = build(wake=wake)

    listener.mute()
    listener.unmute()

    assert wake.resets == 1


async def test_muting_mid_sentence_throws_it_away() -> None:
    # Half a question is worse than none: it transcribes to something the
    # person never asked.
    listener = build()

    play(listener, WAKE, SPEECH, SPEECH)
    listener.mute()
    listener.unmute()
    play(listener, QUIET, QUIET, QUIET)

    assert listener.waiting == 0


async def test_it_listens_again_after_speaking() -> None:
    listener = build()

    listener.mute()
    play(listener, WAKE, SPEECH)
    listener.unmute()
    play(listener, WAKE, SPEECH, SPEECH, QUIET, QUIET, QUIET)

    assert await spoken(listener) == SPEECH * 2 + QUIET * 2


# --------------------------------------------------------------------------- #
# Reading a source for real
# --------------------------------------------------------------------------- #


async def test_it_reads_the_source_on_its_own() -> None:
    frames = [QUIET, WAKE, SPEECH, SPEECH, QUIET, QUIET, QUIET]

    async with build(frames) as listener:
        assert await spoken(listener) == SPEECH * 2 + QUIET * 2


async def test_closing_stops_reading() -> None:
    listener = build([QUIET] * 3)
    async with listener:
        await asyncio.sleep(0)
    assert listener._task is None


async def test_the_oldest_utterance_is_dropped_when_nobody_collects() -> None:
    """Turns are answered one at a time.

    A queue this full means someone kept talking to an assistant that had
    stopped listening. The newest question is the one they are waiting on.
    """
    listener = build(queue_size=1)

    play(listener, WAKE, SPEECH, SPEECH, QUIET, QUIET, QUIET)
    play(listener, WAKE, SPEECH, SPEECH, SPEECH, QUIET, QUIET, QUIET)

    assert listener.waiting == 1
    assert await spoken(listener) == SPEECH * 3 + QUIET * 2


@pytest.mark.parametrize("threshold", [0.5, 0.9])
async def test_the_threshold_decides_what_counts_as_a_detection(threshold: float) -> None:
    listener = build(wake=FakeWakeWord(score=0.7), threshold=threshold)

    play(listener, WAKE, SPEECH, SPEECH, QUIET, QUIET, QUIET)

    assert listener.stats.activations == (1 if threshold < 0.7 else 0)
