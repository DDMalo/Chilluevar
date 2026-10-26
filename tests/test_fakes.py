"""Tests for the fakes themselves.

Testing a test double looks circular, but these are the contract's stunt
doubles: every later test about the assembled assistant believes whatever they
say. `runtime_checkable` protocols only compare method names, so a fake whose
signature has drifted from `interfaces.py` still passes `isinstance` and
quietly tests the wrong thing. These tests are what notices.
"""

from __future__ import annotations

import pytest

from chilluevar.bus import EventBus
from chilluevar.events import (
    ReplyChunk,
    SpeechFinished,
    Stage,
    StageTiming,
    Transcribed,
    TurnStarted,
)
from chilluevar.interfaces import (
    AudioSink,
    AudioSource,
    LanguageModel,
    SpeechToText,
    TextToSpeech,
    ToolCall,
)
from fakes import (
    SAMPLE_RATE,
    FakeAudioSource,
    FakeLLM,
    FakeSink,
    FakeSTT,
    FakeToolCall,
    FakeTTS,
    collect_events,
    silence,
)

# --------------------------------------------------------------------------- #
# The protocols
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("fake", "protocol"),
    [
        (FakeAudioSource(), AudioSource),
        (FakeSink(), AudioSink),
        (FakeSTT(), SpeechToText),
        (FakeLLM(), LanguageModel),
        (FakeTTS(), TextToSpeech),
        (FakeToolCall("light"), ToolCall),
    ],
)
def test_fakes_satisfy_their_protocol(fake: object, protocol: type) -> None:
    assert isinstance(fake, protocol)


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #


def test_silence_is_the_length_it_claims() -> None:
    # Two bytes per sample, or every duration computed from it is half of what
    # it should be.
    assert len(silence(1.0)) == SAMPLE_RATE * 2
    assert len(silence(0.5)) == SAMPLE_RATE


async def test_source_yields_its_frames_and_stops() -> None:
    source = FakeAudioSource([b"one", b"two"])
    assert [frame async for frame in source.frames()] == [b"one", b"two"]


async def test_sink_keeps_what_it_played() -> None:
    sink = FakeSink()
    await sink.play(b"\x01\x02", 22_050)
    await sink.drain()
    assert sink.played == [(b"\x01\x02", 22_050)]
    assert sink.drains == 1


# --------------------------------------------------------------------------- #
# Speech to text
# --------------------------------------------------------------------------- #


async def test_stt_follows_the_script_and_then_repeats_the_last_line() -> None:
    stt = FakeSTT("enciende la luz", "apaga la luz")
    assert await stt.transcribe(silence(), SAMPLE_RATE) == "enciende la luz"
    assert await stt.transcribe(silence(), SAMPLE_RATE) == "apaga la luz"
    assert await stt.transcribe(silence(), SAMPLE_RATE) == "apaga la luz"


async def test_stt_records_how_it_was_called() -> None:
    stt = FakeSTT("hola")
    await stt.transcribe(silence(0.5), SAMPLE_RATE, language="es", hints=["Chilluevar"])
    assert stt.calls == [
        {
            "bytes": SAMPLE_RATE,
            "sample_rate": SAMPLE_RATE,
            "language": "es",
            "hints": ["Chilluevar"],
        }
    ]


async def test_stt_can_be_made_to_fail() -> None:
    stt = FakeSTT(fails=RuntimeError("no model"))
    with pytest.raises(RuntimeError, match="no model"):
        await stt.transcribe(silence(), SAMPLE_RATE)


# --------------------------------------------------------------------------- #
# The language model
# --------------------------------------------------------------------------- #


async def test_llm_answer_arrives_in_pieces_that_join_back() -> None:
    llm = FakeLLM("uno dos tres cuatro cinco", words_per_chunk=2)
    pieces = [piece async for piece in llm.stream([])]
    assert len(pieces) > 1, "an answer that arrives whole hides the joining bugs"
    assert "".join(str(piece) for piece in pieces) == "uno dos tres cuatro cinco"


async def test_llm_records_the_conversation_it_was_given() -> None:
    llm = FakeLLM("vale")
    messages = [
        {"role": "system", "content": "eres Chilluevar"},
        {"role": "user", "content": "enciende la luz"},
    ]
    async for _ in llm.stream(messages):
        pass
    assert llm.conversations == [messages]
    assert llm.last_question == "enciende la luz"


async def test_llm_can_ask_for_a_tool() -> None:
    call = FakeToolCall("light", {"room": "salon"})
    llm = FakeLLM(["Vale, ", call, "ya esta."])
    pieces = [piece async for piece in llm.stream([])]
    assert pieces[1] is call


async def test_llm_can_be_made_to_fail() -> None:
    llm = FakeLLM(fails=RuntimeError("ollama is not running"))
    with pytest.raises(RuntimeError, match="ollama"):
        async for _ in llm.stream([]):
            pass


# --------------------------------------------------------------------------- #
# Speech synthesis
# --------------------------------------------------------------------------- #


async def test_tts_is_deterministic_and_tells_sentences_apart() -> None:
    tts = FakeTTS()
    first = await tts.synthesize("hola")
    assert first == await FakeTTS().synthesize("hola")
    assert first != await tts.synthesize("adios")
    assert tts.spoken == ["hola", "adios"]


async def test_tts_reports_the_rate_it_was_built_with() -> None:
    # Not 16 kHz: a pipeline that assumes the microphone's rate for playback
    # would pass against a fake that happened to share it.
    assert FakeTTS().sample_rate == 22_050
    assert FakeTTS(sample_rate=16_000).sample_rate == 16_000


async def test_tts_can_be_made_to_fail() -> None:
    tts = FakeTTS(fails=RuntimeError("voice not found"))
    with pytest.raises(RuntimeError, match="voice"):
        await tts.synthesize("hola")


# --------------------------------------------------------------------------- #
# Watching the bus
# --------------------------------------------------------------------------- #


async def test_collect_events_keeps_types_it_was_asked_for_and_drops_the_rest() -> None:
    bus = EventBus()
    async with collect_events(bus, TurnStarted, Transcribed) as seen:
        await bus.publish(TurnStarted("abc"))
        await bus.publish(Transcribed("abc", text="hola"))
        await bus.publish(SpeechFinished("abc"))  # not asked for
    assert [type(event) for event in seen] == [TurnStarted, Transcribed]


async def test_collect_events_interleaves_types_as_they_were_published() -> None:
    """The order of a real turn, where the types alternate.

    The first version of this helper read one queue per type with a task each,
    and a task empties its whole queue as soon as it is scheduled. This exact
    sequence came back with both timings together and the chunk last, which is
    an order no turn ever produced.
    """
    bus = EventBus()
    async with collect_events(bus, Transcribed, StageTiming, ReplyChunk) as seen:
        await bus.publish(Transcribed("t", text="hola"))
        await bus.publish(StageTiming("t", stage=Stage.SPEECH_TO_TEXT))
        await bus.publish(ReplyChunk("t", text="vale"))
        await bus.publish(StageTiming("t", stage=Stage.LANGUAGE_MODEL))

    assert [type(event) for event in seen] == [Transcribed, StageTiming, ReplyChunk, StageTiming]


async def test_collect_events_still_delivers_to_real_subscribers() -> None:
    # Watching must not mean intercepting: the pipeline under test has to see
    # the same bus it would see in production.
    bus = EventBus()
    async with (
        bus.subscribe(TurnStarted) as stream,
        collect_events(bus, TurnStarted) as seen,
    ):
        await bus.publish(TurnStarted("abc"))
        delivered = await anext(stream)

    assert delivered.turn_id == "abc"
    assert [event.turn_id for event in seen] == ["abc"]


async def test_collect_events_ignores_what_happens_outside_the_block() -> None:
    bus = EventBus()
    await bus.publish(TurnStarted("before"))
    async with collect_events(bus, TurnStarted) as seen:
        await bus.publish(TurnStarted("during"))
    await bus.publish(TurnStarted("after"))
    assert [event.turn_id for event in seen] == ["during"]
