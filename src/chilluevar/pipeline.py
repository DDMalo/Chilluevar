"""One turn, from captured audio to spoken answer.

This is the first module that makes the others into an assistant. Until now
each engine worked on its own and nothing joined them; here a turn is driven
through transcribe, think and speak, publishing the events from `events.py` as
it goes.

Two decisions shape the whole file:

**It is handed audio, it never captures any.** Recording belongs to whoever
owns the microphone, which in satellite mode is a different machine from the
one running this. Taking PCM in means the satellite split (issue #23) is a
network hop in front of `run_turn`, not a rewrite of it.

**It publishes events and also returns a result.** The events are for anything
watching (the logs, a dashboard, the satellite bridge); the return value is for
whoever asked for the turn and is waiting to print it. Making the caller
subscribe to the bus to learn what it just asked for would be architecture for
its own sake.

What it does not do yet is speak while the model is still writing. The reply is
collected whole and then synthesised, which is simple and noticeably slower
than it has to be; cutting that wait is issue #15, and the `ReplyChunk` events
are already published here so that only the speaking side has to change.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from chilluevar.bus import EventBus
from chilluevar.events import (
    AudioCaptured,
    ReplyChunk,
    SpeakRequest,
    SpeechFinished,
    Stage,
    StageTiming,
    Transcribed,
    TurnFailed,
    TurnStarted,
    new_turn_id,
)
from chilluevar.interfaces import AudioSink, LanguageModel, Message, SpeechToText, TextToSpeech
from chilluevar.llm import speech_problems

__all__ = ["FAILURE_PHRASES", "Pipeline", "TurnResult"]

logger = logging.getLogger(__name__)

# Said out loud when a stage gives up. An assistant that goes quiet is
# indistinguishable from one that crashed, and the person is left wondering
# whether to repeat themselves. These are short on purpose: the point is to
# hand the turn back, not to explain.
FAILURE_PHRASES = {
    Stage.SPEECH_TO_TEXT: "No te he entendido.",
    Stage.LANGUAGE_MODEL: "Ahora mismo no puedo responder.",
    # Nothing is said when synthesis itself failed: saying it would need the
    # engine that just broke.
    Stage.TEXT_TO_SPEECH: "",
}


def _first_line(message: str) -> str:
    """The first line of an error message.

    Our own errors are written to be read: they say what is wrong and then, on
    the next lines, the command that fixes it. That is right in front of the
    person who caused it and wrong in a warning about something that already
    failed for another reason.
    """
    return message.strip().splitlines()[0] if message.strip() else ""


@dataclass(slots=True)
class TurnResult:
    """What happened in one turn."""

    turn_id: str
    transcript: str = ""
    reply: str = ""
    failed: Stage | None = None
    reason: str = ""
    timings: dict[Stage, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.failed is None

    @property
    def total_ms(self) -> float:
        return sum(self.timings.values())


class Pipeline:
    """Drives one turn through the engines it was given.

    Everything it uses arrives through the constructor as a protocol from
    `interfaces.py`, so this module imports no engine and knows about neither
    Whisper nor Ollama nor Piper. That is what lets the tests run it end to end
    with no model on disk, and what will let an engine be swapped later by
    changing a line of configuration.
    """

    def __init__(
        self,
        bus: EventBus,
        *,
        stt: SpeechToText,
        llm: LanguageModel,
        tts: TextToSpeech,
        sink: AudioSink,
        system_prompt: str,
        language: str = "es",
        hints: Iterable[str] = (),
        max_reply_words: int | None = None,
    ) -> None:
        self._bus = bus
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._sink = sink
        self._system_prompt = system_prompt
        self._language = language
        self._hints = list(hints)
        self._max_reply_words = max_reply_words

    async def run_turn(
        self,
        pcm: bytes,
        sample_rate: int,
        *,
        trigger: str = "push_to_talk",
    ) -> TurnResult:
        """Transcribe this audio, answer it and say the answer out loud."""
        turn = TurnResult(turn_id=new_turn_id())
        await self._bus.publish(TurnStarted(turn.turn_id, trigger=trigger))
        await self._bus.publish(AudioCaptured(turn.turn_id, pcm=pcm, sample_rate=sample_rate))

        if not await self._listen(turn, pcm, sample_rate):
            return turn
        if not await self._think(turn):
            return turn
        await self._speak(turn, turn.reply)
        return turn

    # ----------------------------------------------------------------- #
    # The stages
    # ----------------------------------------------------------------- #

    async def _listen(self, turn: TurnResult, pcm: bytes, sample_rate: int) -> bool:
        duration_ms = int(len(pcm) / (sample_rate * 2) * 1000)
        try:
            async with self._timed(turn, Stage.SPEECH_TO_TEXT):
                text = await self._stt.transcribe(
                    pcm,
                    sample_rate,
                    language=self._language,
                    hints=self._hints,
                )
        except Exception as error:  # broad on purpose: see _fail
            return await self._fail(turn, Stage.SPEECH_TO_TEXT, str(error))

        turn.transcript = text.strip()
        if not turn.transcript:
            # Silence, background noise, or a sentence that started before the
            # key was pressed. Common enough that it is a normal outcome of a
            # turn rather than an error.
            return await self._fail(turn, Stage.SPEECH_TO_TEXT, "nothing was understood")

        await self._bus.publish(
            Transcribed(
                turn.turn_id,
                text=turn.transcript,
                language=self._language,
                audio_duration_ms=duration_ms,
            )
        )
        return True

    async def _think(self, turn: TurnResult) -> bool:
        messages: list[Message] = [
            {"role": "system", "content": self._system_prompt},
            # No history yet: every turn is a fresh conversation. Remembering
            # what was said before is issue #9.
            {"role": "user", "content": turn.transcript},
        ]
        pieces: list[str] = []
        try:
            async with self._timed(turn, Stage.LANGUAGE_MODEL):
                async for piece in self._llm.stream(messages):
                    if not isinstance(piece, str):
                        # A tool call. Running tools is issue #10; until then
                        # the model is told about none, so this should not
                        # happen, and ignoring it is safer than guessing.
                        logger.warning("ignoring a tool call: tools arrive with issue #10")
                        continue
                    pieces.append(piece)
                    await self._bus.publish(ReplyChunk(turn.turn_id, text=piece))
        except Exception as error:  # broad on purpose: see _fail
            return await self._fail(turn, Stage.LANGUAGE_MODEL, str(error))

        # A final empty chunk closes the reply. The alternative, marking the
        # last real piece, means holding each piece back until the next one
        # arrives, which is exactly the delay issue #15 exists to remove.
        await self._bus.publish(ReplyChunk(turn.turn_id, text="", is_last=True))

        turn.reply = "".join(pieces).strip()
        if not turn.reply:
            return await self._fail(turn, Stage.LANGUAGE_MODEL, "the model said nothing")

        problems = speech_problems(turn.reply, max_words=self._max_reply_words)
        if problems:
            # Not a failure: the cleaner in tts/clean.py fixes most of it. Worth
            # a line in the log because a prompt the model keeps ignoring is a
            # prompt to change.
            logger.warning("the reply would sound wrong out loud: %s", ", ".join(problems))
        return True

    async def _speak(self, turn: TurnResult, text: str) -> bool:
        try:
            async with self._timed(turn, Stage.TEXT_TO_SPEECH):
                pcm = await self._tts.synthesize(text)
        except Exception as error:  # broad on purpose: see _fail
            return await self._fail(turn, Stage.TEXT_TO_SPEECH, str(error))

        if not pcm:
            return await self._fail(turn, Stage.TEXT_TO_SPEECH, "the voice produced no audio")

        await self._bus.publish(SpeakRequest(turn.turn_id, text=text))
        # Playing is outside the timing on purpose: how long a clip takes to
        # play is the length of the clip, not a cost anyone could optimise.
        # Counting it would bury the number issue #14 actually cares about.
        await self._sink.play(pcm, self._tts.sample_rate)
        await self._sink.drain()
        await self._bus.publish(SpeechFinished(turn.turn_id))
        return True

    # ----------------------------------------------------------------- #
    # Failing, and timing
    # ----------------------------------------------------------------- #

    async def _fail(self, turn: TurnResult, stage: Stage, reason: str) -> bool:
        """Record the failure, announce it, and say something short out loud.

        Every stage catches `Exception` rather than a list of its engine's own
        errors, which is usually a smell and here is the point: an orchestrator
        whose entire job is to hand the turn back cannot have a failure mode
        that depends on which engine is plugged in. `asyncio.CancelledError`
        does not inherit from `Exception`, so a cancelled turn still cancels.
        """
        turn.failed = stage
        turn.reason = reason
        logger.error("turn failed at %s: %s", stage.value, reason, extra={"turn_id": turn.turn_id})
        await self._bus.publish(TurnFailed(turn.turn_id, stage=stage, reason=reason))
        await self._apologise(turn, stage)
        return False

    async def _apologise(self, turn: TurnResult, stage: Stage) -> None:
        phrase = FAILURE_PHRASES.get(stage, "")
        if not phrase:
            return
        try:
            pcm = await self._tts.synthesize(phrase)
            if pcm:
                await self._sink.play(pcm, self._tts.sample_rate)
                await self._sink.drain()
        except Exception as error:
            # One line, not a traceback. The turn already logged why it failed,
            # and a stack trace for the apology is longer than everything else
            # on screen put together: the first run on a machine with no voice
            # downloaded buried the real reason under twenty lines of it.
            logger.warning(
                "could not speak the apology: %s",
                _first_line(str(error)),
                extra={"turn_id": turn.turn_id},
            )

    @asynccontextmanager
    async def _timed(self, turn: TurnResult, stage: Stage) -> AsyncIterator[None]:
        """Time a stage and publish it, whether it succeeded or not.

        A stage that failed after four seconds is exactly the one worth having
        a number for, so the timing is recorded in a `finally`.
        """
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            turn.timings[stage] = elapsed_ms
            await self._bus.publish(StageTiming(turn.turn_id, stage=stage, elapsed_ms=elapsed_ms))
