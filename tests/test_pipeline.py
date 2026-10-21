"""The assembled assistant, end to end, with fake engines.

Every test here runs a whole turn: audio in, speech out. Nothing is downloaded
and no socket is opened, because the engines come from `fakes.py` and the
pipeline was built never to know which engines it has.

These are the tests that would have caught the mistakes the separate modules
could not: a transcript that never reaches the model, a reply played at the
microphone's sample rate instead of the voice's, an assistant that goes silent
instead of saying it did not understand.
"""

from __future__ import annotations

import logging

import pytest

from chilluevar.bus import EventBus
from chilluevar.events import (
    AudioCaptured,
    Event,
    ReplyChunk,
    SpeakRequest,
    SpeechFinished,
    Stage,
    StageTiming,
    Transcribed,
    TurnFailed,
    TurnStarted,
)
from chilluevar.llm import LanguageModelError
from chilluevar.pipeline import FAILURE_PHRASES, Pipeline
from fakes import SAMPLE_RATE, FakeLLM, FakeSink, FakeSTT, FakeTTS, collect_events, silence

PROMPT = "Eres Chilluevar, un asistente de voz."


def build(
    *,
    stt: FakeSTT | None = None,
    llm: FakeLLM | None = None,
    tts: FakeTTS | None = None,
    sink: FakeSink | None = None,
    **options: object,
) -> tuple[Pipeline, EventBus, FakeSTT, FakeLLM, FakeTTS, FakeSink]:
    """A pipeline with fakes for whatever the test did not provide."""
    bus = EventBus()
    stt = stt or FakeSTT("enciende la luz del salon")
    llm = llm or FakeLLM("Vale, enciendo la luz del salon.")
    tts = tts or FakeTTS()
    sink = sink or FakeSink()
    pipeline = Pipeline(
        bus,
        stt=stt,
        llm=llm,
        tts=tts,
        sink=sink,
        system_prompt=PROMPT,
        **options,  # type: ignore[arg-type]
    )
    return pipeline, bus, stt, llm, tts, sink


def only(events: list[Event], kind: type[Event]) -> list[Event]:
    return [event for event in events if isinstance(event, kind)]


# --------------------------------------------------------------------------- #
# A turn that works
# --------------------------------------------------------------------------- #


async def test_a_whole_turn() -> None:
    pipeline, _, _, _, tts, sink = build()

    turn = await pipeline.run_turn(silence(2.0), SAMPLE_RATE)

    assert turn.ok
    assert turn.transcript == "enciende la luz del salon"
    assert turn.reply == "Vale, enciendo la luz del salon."
    assert tts.spoken == ["Vale, enciendo la luz del salon."]
    assert len(sink.played) == 1
    assert sink.drains == 1


async def test_the_question_reaches_the_model_with_the_system_prompt() -> None:
    pipeline, _, _, llm, _, _ = build()

    await pipeline.run_turn(silence(), SAMPLE_RATE)

    assert llm.conversations[0][0] == {"role": "system", "content": PROMPT}
    assert llm.last_question == "enciende la luz del salon"


async def test_the_language_and_hints_reach_the_transcriber() -> None:
    # The wake word is the word Whisper is least likely to know, so losing the
    # hints on the way through the pipeline costs real accuracy.
    pipeline, _, stt, _, _, _ = build(language="es", hints=["Chilluevar", "salon"])

    await pipeline.run_turn(silence(1.5), SAMPLE_RATE)

    assert stt.calls[0]["language"] == "es"
    assert stt.calls[0]["hints"] == ["Chilluevar", "salon"]
    assert stt.calls[0]["sample_rate"] == SAMPLE_RATE


async def test_the_reply_is_played_at_the_voice_rate_not_the_microphone_rate() -> None:
    # Audio in is 16 kHz and the Piper voices are 22.05 kHz. Playing the answer
    # at the microphone's rate makes it come out slow and deep.
    pipeline, _, _, _, _, sink = build(tts=FakeTTS(sample_rate=22_050))

    await pipeline.run_turn(silence(), SAMPLE_RATE)

    _, rate = sink.played[0]
    assert rate == 22_050


async def test_every_turn_gets_its_own_identifier() -> None:
    pipeline, _, _, _, _, _ = build()

    first = await pipeline.run_turn(silence(), SAMPLE_RATE)
    second = await pipeline.run_turn(silence(), SAMPLE_RATE)

    assert first.turn_id != second.turn_id


# --------------------------------------------------------------------------- #
# What it puts on the bus
# --------------------------------------------------------------------------- #


async def test_a_turn_publishes_the_whole_contract() -> None:
    pipeline, bus, _, _, _, _ = build()

    async with collect_events(
        bus, TurnStarted, AudioCaptured, Transcribed, SpeakRequest, SpeechFinished
    ) as seen:
        turn = await pipeline.run_turn(silence(1.0), SAMPLE_RATE)

    for kind in (TurnStarted, AudioCaptured, Transcribed, SpeakRequest, SpeechFinished):
        assert len(only(seen, kind)) == 1, f"{kind.__name__} was not published exactly once"
    # One turn is one identifier, or none of the timings can be correlated.
    assert {event.turn_id for event in seen} == {turn.turn_id}


async def test_the_transcript_event_carries_the_audio_duration() -> None:
    pipeline, bus, _, _, _, _ = build()

    async with collect_events(bus, Transcribed) as seen:
        await pipeline.run_turn(silence(2.0), SAMPLE_RATE)

    transcribed = only(seen, Transcribed)[0]
    assert isinstance(transcribed, Transcribed)
    assert transcribed.audio_duration_ms == 2000
    assert transcribed.text == "enciende la luz del salon"


async def test_the_reply_is_published_in_pieces_and_then_closed() -> None:
    # Issue #15 speaks these as they arrive. Publishing them now means only the
    # speaking side has to change then.
    pipeline, bus, _, _, _, _ = build(llm=FakeLLM("uno dos tres cuatro", words_per_chunk=2))

    async with collect_events(bus, ReplyChunk) as seen:
        await pipeline.run_turn(silence(), SAMPLE_RATE)

    chunks = [event for event in seen if isinstance(event, ReplyChunk)]
    assert len(chunks) > 2
    assert "".join(chunk.text for chunk in chunks) == "uno dos tres cuatro"
    assert [chunk.is_last for chunk in chunks] == [False] * (len(chunks) - 1) + [True]


async def test_every_stage_is_timed() -> None:
    pipeline, bus, _, _, _, _ = build()

    async with collect_events(bus, StageTiming) as seen:
        turn = await pipeline.run_turn(silence(), SAMPLE_RATE)

    timed = {event.stage for event in seen if isinstance(event, StageTiming)}
    assert timed == {Stage.SPEECH_TO_TEXT, Stage.LANGUAGE_MODEL, Stage.TEXT_TO_SPEECH}
    assert set(turn.timings) == timed
    assert turn.total_ms == pytest.approx(sum(turn.timings.values()))


async def test_the_stages_happen_in_order() -> None:
    pipeline, bus, _, _, _, _ = build()

    async with collect_events(bus, Transcribed, SpeakRequest, SpeechFinished) as seen:
        await pipeline.run_turn(silence(), SAMPLE_RATE)

    heard = only(seen, Transcribed)[0]
    asked = only(seen, SpeakRequest)[0]
    done = only(seen, SpeechFinished)[0]
    assert heard.created_at <= asked.created_at <= done.created_at


# --------------------------------------------------------------------------- #
# Turns that fail
# --------------------------------------------------------------------------- #


async def test_nothing_understood_is_not_sent_to_the_model() -> None:
    # Silence, background noise, or a key pressed by accident. Asking the model
    # to answer an empty question wastes seconds and invents a reply.
    pipeline, bus, _, llm, tts, _ = build(stt=FakeSTT(""))

    async with collect_events(bus, TurnFailed, Transcribed) as seen:
        turn = await pipeline.run_turn(silence(), SAMPLE_RATE)

    assert not turn.ok
    assert turn.failed is Stage.SPEECH_TO_TEXT
    assert llm.conversations == []
    assert only(seen, Transcribed) == []
    assert tts.spoken == [FAILURE_PHRASES[Stage.SPEECH_TO_TEXT]]


async def test_a_transcriber_that_raises_ends_the_turn() -> None:
    pipeline, _, _, llm, _, _ = build(stt=FakeSTT(fails=RuntimeError("no model on disk")))

    turn = await pipeline.run_turn(silence(), SAMPLE_RATE)

    assert turn.failed is Stage.SPEECH_TO_TEXT
    assert "no model on disk" in turn.reason
    assert llm.conversations == []


async def test_a_model_that_is_not_running_is_said_out_loud() -> None:
    # The single most likely failure in daily use: Ollama is not started.
    pipeline, bus, _, _, tts, sink = build(
        llm=FakeLLM(fails=LanguageModelError("cannot reach Ollama at localhost:11434"))
    )

    async with collect_events(bus, TurnFailed) as seen:
        turn = await pipeline.run_turn(silence(), SAMPLE_RATE)

    failed = only(seen, TurnFailed)[0]
    assert isinstance(failed, TurnFailed)
    assert failed.stage is Stage.LANGUAGE_MODEL
    assert "Ollama" in turn.reason
    assert tts.spoken == [FAILURE_PHRASES[Stage.LANGUAGE_MODEL]]
    assert len(sink.played) == 1


async def test_a_model_that_says_nothing_is_a_failure() -> None:
    pipeline, _, _, _, _, _ = build(llm=FakeLLM(""))

    turn = await pipeline.run_turn(silence(), SAMPLE_RATE)

    assert turn.failed is Stage.LANGUAGE_MODEL
    assert turn.reason == "the model said nothing"


async def test_a_voice_that_is_missing_does_not_try_to_apologise_out_loud() -> None:
    # Apologising would need the engine that just broke. The turn still fails
    # cleanly, and the reason is in the log and in the result.
    pipeline, bus, _, _, _, sink = build(tts=FakeTTS(fails=RuntimeError("voice not downloaded")))

    async with collect_events(bus, TurnFailed, SpeechFinished) as seen:
        turn = await pipeline.run_turn(silence(), SAMPLE_RATE)

    assert turn.failed is Stage.TEXT_TO_SPEECH
    assert "voice not downloaded" in turn.reason
    assert sink.played == []
    assert only(seen, SpeechFinished) == []


async def test_a_failed_apology_does_not_bury_the_real_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Both engines broken at once, which is what the first real run looked like.

    Whisper could not load its CUDA libraries and the Piper voice had never
    been downloaded, so the turn failed and then the apology failed too. The
    apology is a secondary failure and gets one line; a traceback for it is
    longer than everything else on screen and pushes the real reason off it.
    """
    pipeline, _, _, _, _, _ = build(
        stt=FakeSTT(fails=RuntimeError("Library cublas64_12.dll is not found")),
        tts=FakeTTS(
            fails=RuntimeError(
                "voice not found at models/piper/es_ES.onnx. Download it with:\n"
                "  python -m piper.download_voices es_ES --data-dir models/piper"
            )
        ),
    )

    with caplog.at_level(logging.WARNING, logger="chilluevar.pipeline"):
        turn = await pipeline.run_turn(silence(), SAMPLE_RATE)

    assert turn.failed is Stage.SPEECH_TO_TEXT
    assert "cublas" in turn.reason

    apologies = [record for record in caplog.records if "apology" in record.getMessage()]
    assert len(apologies) == 1
    assert apologies[0].exc_info is None, "a traceback here buries the real reason"
    # The "download it with" line is help for whoever asked for speech, not for
    # a warning about a turn that had already failed for another reason.
    assert "download_voices" not in apologies[0].getMessage()


async def test_a_failed_stage_is_still_timed() -> None:
    # A stage that gave up after four seconds is exactly the one worth a number.
    pipeline, _, _, _, _, _ = build(llm=FakeLLM(fails=LanguageModelError("down")))

    turn = await pipeline.run_turn(silence(), SAMPLE_RATE)

    assert Stage.LANGUAGE_MODEL in turn.timings
    assert Stage.TEXT_TO_SPEECH not in turn.timings


async def test_a_failure_does_not_stop_the_next_turn() -> None:
    # The assistant goes back to listening. A crash here would mean restarting
    # it every time Ollama hiccups.
    pipeline, _, _, _, _, _ = build(stt=FakeSTT("", "enciende la luz"))

    first = await pipeline.run_turn(silence(), SAMPLE_RATE)
    second = await pipeline.run_turn(silence(), SAMPLE_RATE)

    assert not first.ok
    assert second.ok
    assert second.reply
