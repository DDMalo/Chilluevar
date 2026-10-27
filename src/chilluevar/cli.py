"""Entry point.

`chilluevar run` is the assistant: press Enter, speak, hear the answer. The
other commands are the same pipeline taken apart, which is how each stage was
built and how each is still debugged on its own: `record` is the microphone and
the speaker (issue #3), `transcribe` is speech to text (#4), `ask` is the
language model in writing (#5) and `say` is the voice (#6).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from chilluevar import __version__
from chilluevar.audio import (
    Listener,
    MicrophoneSource,
    OpenWakeWord,
    SileroVAD,
    SpeakerSink,
    WavFileSource,
    collect,
)
from chilluevar.audio.wake import spoken_form
from chilluevar.audio.wav import save_wav
from chilluevar.bus import EventBus
from chilluevar.config import Config, load_config
from chilluevar.events import Stage
from chilluevar.llm import LanguageModelError, speech_problems, system_prompt
from chilluevar.logging_setup import setup_logging
from chilluevar.pipeline import Pipeline, TurnResult
from chilluevar.stt import WhisperSTT
from chilluevar.tts import PiperTTS, clean_for_speech

if TYPE_CHECKING:
    # Only for the type annotations: importing it for real needs the llm extra.
    from chilluevar.llm.ollama import OllamaLLM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chilluevar", description="Local voice assistant")
    parser.add_argument("--config", type=Path, default=None, help="path to a YAML config file")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--log-transcripts",
        action="store_true",
        help="write what was said into the log (off by default)",
    )
    parser.add_argument("--version", action="version", version=f"chilluevar {__version__}")

    sub = parser.add_subparsers(dest="command")

    record = sub.add_parser("record", help="record from the microphone and play it back")
    record.add_argument(
        "--output", type=Path, default=Path("recording.wav"), help="where to save the recording"
    )
    record.add_argument(
        "--from-file",
        type=Path,
        default=None,
        help="read from a .wav instead of the microphone, to try this without a mic",
    )
    record.add_argument("--no-playback", action="store_true", help="record only, do not play back")

    transcribe = sub.add_parser("transcribe", help="turn speech into text")
    transcribe.add_argument(
        "file", type=Path, nargs="?", default=None, help="a .wav; omit it to use the microphone"
    )
    transcribe.add_argument("--model", default=None, help="override the model from the config")
    transcribe.add_argument(
        "--save",
        type=Path,
        default=None,
        help="keep the audio and the transcript as <name>.wav and <name>.txt",
    )

    ask = sub.add_parser("ask", help="ask the language model something, in writing")
    ask.add_argument(
        "question", nargs="?", default=None, help="omit it to keep asking until an empty line"
    )
    ask.add_argument("--model", default=None, help="override the model from the config")

    say = sub.add_parser("say", help="read text out loud")
    say.add_argument("text", nargs="?", default=None, help="omit it to read from stdin")
    say.add_argument("--voice", default=None, help="override the voice from the config")
    say.add_argument("--output", type=Path, default=None, help="save to a .wav instead of playing")
    say.add_argument(
        "--raw", action="store_true", help="speak the text as written, without cleaning it"
    )
    say.add_argument(
        "--show", action="store_true", help="print what will actually be spoken and stop"
    )

    run = sub.add_parser("run", help="the whole assistant: speak, and hear the answer")
    run.add_argument("--once", action="store_true", help="answer one question and exit")
    run.add_argument(
        "--hands-free",
        action="store_true",
        help="start a turn with the wake word instead of the Enter key",
    )
    run.add_argument(
        "--from-file",
        type=Path,
        default=None,
        help="take the question from a .wav instead of the microphone (implies --once)",
    )
    run.add_argument("--model", default=None, help="override the language model from the config")

    sub.add_parser("devices", help="list the audio devices this machine has")
    return parser


def show_config(config: Config) -> int:
    print(f"chilluevar {__version__}")
    print(f"  mode:  {config.mode}")
    print(f"  stt:   {config.stt.engine} ({config.stt.model})")
    print(f"  llm:   {config.llm.engine} ({config.llm.model})")
    print(f"  tts:   {config.tts.engine} ({config.tts.voice})")
    print("\nRun `chilluevar run` to talk to it, or `record`, `transcribe`, `ask` and")
    print("`say` to exercise one stage at a time.")
    return 0


def show_devices() -> int:
    try:
        import sounddevice as sd
    except ImportError:
        print('install the audio extra first:  pip install -e ".[audio]"')
        return 1
    print(sd.query_devices())
    return 0


async def capture(config: Config, from_file: Path | None) -> bytes:
    """Audio from a file, or push to talk: Enter starts, Enter stops."""
    if from_file is not None:
        source = WavFileSource(
            from_file,
            sample_rate=config.audio.sample_rate,
            frame_ms=config.audio.frame_ms,
        )
        print(f"reading {from_file}")
        return await collect(source)

    source = MicrophoneSource(
        sample_rate=config.audio.sample_rate,
        frame_ms=config.audio.frame_ms,
        device=config.audio.input_device,
    )
    # input() blocks, so every call to it goes to a worker thread. Reading
    # stdin on the event loop would stop audio frames arriving while we
    # wait for a key. (ruff's ASYNC250 rule catches exactly this.)
    await asyncio.to_thread(input, "press Enter to start recording...")
    print("recording, press Enter again to stop")
    stop = asyncio.Event()
    asyncio.get_running_loop().create_task(_wait_for_enter(stop))
    return await collect(source, stop=stop)


async def record_once(config: Config, args: argparse.Namespace) -> int:
    pcm = await capture(config, args.from_file)
    seconds = len(pcm) / (config.audio.sample_rate * 2)
    print(f"captured {seconds:.1f} s")

    if not pcm:
        print("nothing was recorded")
        return 1

    save_wav(args.output, pcm, config.audio.sample_rate)
    print(f"saved to {args.output}")

    if not args.no_playback:
        print("playing it back")
        async with SpeakerSink(device=config.audio.output_device) as sink:
            await sink.play(pcm, config.audio.sample_rate)
            await sink.drain()
    return 0


async def transcribe_once(config: Config, args: argparse.Namespace) -> int:
    pcm = await capture(config, args.file)
    if not pcm:
        print("nothing to transcribe")
        return 1

    stt = WhisperSTT(
        args.model or config.stt.model,
        device=config.stt.device,
        compute_type=config.stt.compute_type,
        beam_size=config.stt.beam_size,
    )
    print(f"transcribing with {stt.model_name} (the first run downloads the model)")
    text = await stt.transcribe(
        pcm,
        config.audio.sample_rate,
        language=config.stt.language,
        hints=config.stt.hints,
    )
    print(f"\n  {text or '(nothing recognised)'}\n")

    if args.save is not None:
        wav_path = args.save.with_suffix(".wav")
        txt_path = args.save.with_suffix(".txt")
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        save_wav(wav_path, pcm, config.audio.sample_rate)
        txt_path.write_text(text + "\n", encoding="utf-8")
        print(f"saved {wav_path} and {txt_path}")
        print("the .txt holds what the model heard: correct it to what you actually said,")
        print("or the benchmark will score itself against its own mistakes.")
    return 0


async def ask(config: Config, args: argparse.Namespace) -> int:
    """Talk to the model in writing, with the same prompt it will get by voice."""
    try:
        from chilluevar.llm.ollama import OllamaLLM
    except ImportError:
        print('the language model needs the llm extra:  pip install -e ".[llm]"')
        return 1

    overrides = {"model": args.model} if args.model else {}
    llm = OllamaLLM.from_config(config.llm, **overrides)
    prompt = system_prompt(max_words=config.llm.max_reply_words)

    try:
        print(f"loading {llm.model}...")
        print(f"ready in {await llm.warm_up():.1f} s\n")
        while question := args.question or await _read_question():
            await _answer(llm, prompt, question, config.llm.max_reply_words)
            if args.question:
                break
    except LanguageModelError as exc:
        print(f"error: {exc}")
        return 1
    return 0


async def _read_question() -> str:
    """One question from the keyboard; an empty line or Ctrl+Z/Ctrl+D ends."""
    try:
        return (await asyncio.to_thread(input, "> ")).strip()
    except EOFError:
        return ""


async def _answer(llm: OllamaLLM, prompt: str, question: str, max_words: int) -> None:
    # No memory between questions yet: every question starts a fresh
    # conversation. Keeping the history is issue #9.
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
    ]
    started = time.perf_counter()
    first_words: float | None = None
    parts: list[str] = []
    async for piece in llm.stream(messages):
        if not isinstance(piece, str):
            continue  # a tool call; tools arrive with issue #10
        if first_words is None:
            first_words = time.perf_counter() - started
        parts.append(piece)
        print(piece, end="", flush=True)

    total = time.perf_counter() - started
    problems = speech_problems("".join(parts), max_words=max_words)
    print(f"\n  [first words after {first_words or total:.2f} s, {total:.2f} s in total]")
    if problems:
        print(f"  [would sound wrong out loud: {', '.join(problems)}]")
    print()


async def _wait_for_enter(stop: asyncio.Event) -> None:
    await asyncio.to_thread(input)
    stop.set()


async def say_once(config: Config, args: argparse.Namespace) -> int:
    text = args.text if args.text is not None else sys.stdin.read()
    if not text.strip():
        print("nothing to say")
        return 1

    if args.show:
        print(clean_for_speech(text) if not args.raw else text)
        return 0

    tts = PiperTTS(
        args.voice or config.tts.voice,
        directory=config.tts.voice_dir,
        clean=config.tts.clean_text and not args.raw,
    )
    try:
        pcm = await tts.synthesize(text)
    except RuntimeError as error:
        # A missing voice or a missing extra is something the person can fix.
        # A stack trace buries the one line that says how.
        print(error)
        return 1
    if not pcm:
        print("the synthesiser produced no audio")
        return 1

    if args.output is not None:
        save_wav(args.output, pcm, tts.sample_rate)
        print(f"saved to {args.output}")
        return 0

    async with SpeakerSink(device=config.audio.output_device) as sink:
        await sink.play(pcm, tts.sample_rate)
        await sink.drain()
    return 0


async def run_assistant(config: Config, args: argparse.Namespace) -> int:
    """The whole thing: press Enter, speak, hear the answer.

    This function only assembles the engines and feeds the pipeline audio. The
    turn itself lives in `pipeline.py`, which knows nothing about Whisper,
    Ollama or Piper, so swapping any of them is a change to these few lines.
    """
    try:
        from chilluevar.llm.ollama import OllamaLLM
    except ImportError:
        print('the language model needs the llm extra:  pip install -e ".[llm]"')
        return 1

    stt = WhisperSTT(
        config.stt.model,
        device=config.stt.device,
        compute_type=config.stt.compute_type,
        beam_size=config.stt.beam_size,
    )
    overrides = {"model": args.model} if args.model else {}
    llm = OllamaLLM.from_config(config.llm, **overrides)
    tts = PiperTTS(config.tts.voice, directory=config.tts.voice_dir, clean=config.tts.clean_text)
    print(f"stt: {stt.model_name}   llm: {llm.model}   tts: {tts.voice}")

    try:
        # Loading the model now rather than inside the first turn: otherwise
        # the first question sits in silence for several seconds and looks
        # like the assistant did not hear it.
        print(f"model ready in {await llm.warm_up():.1f} s\n")
    except LanguageModelError as error:
        print(f"error: {error}")
        return 1

    bus = EventBus()
    async with SpeakerSink(device=config.audio.output_device) as sink:
        pipeline = Pipeline(
            bus,
            stt=stt,
            llm=llm,
            tts=tts,
            sink=sink,
            system_prompt=system_prompt(max_words=config.llm.max_reply_words),
            language=config.stt.language,
            hints=config.stt.hints,
            max_reply_words=config.llm.max_reply_words,
        )
        try:
            if args.hands_free:
                return await _hands_free(pipeline, config, args)
            return await _turns(pipeline, config, args)
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            return 0


async def _hands_free(pipeline: Pipeline, config: Config, args: argparse.Namespace) -> int:
    """Wait for the wake word, answer, and go back to waiting."""
    if args.from_file is not None:
        print("--from-file is push to talk; drop --hands-free to use it")
        return 1

    source = MicrophoneSource(
        sample_rate=config.audio.sample_rate,
        frame_ms=config.audio.frame_ms,
        device=config.audio.input_device,
    )
    listener = Listener(
        source,
        wake=OpenWakeWord(config.wake_word.model, framework=config.wake_word.framework),
        vad=SileroVAD(threshold=config.listen.vad_threshold),
        threshold=config.wake_word.threshold,
        start_timeout_ms=config.listen.start_timeout_ms,
        silence_ms=config.listen.silence_ms,
        min_utterance_ms=config.listen.min_utterance_ms,
        max_utterance_ms=config.listen.max_utterance_ms,
    )

    print(f'say "{spoken_form(config.wake_word.model)}" and then ask (Ctrl+C to stop)')
    async with listener:
        async for pcm in listener.utterances():
            # Muted for the whole turn, not only while the speaker is playing:
            # the pauses in between are when the wake word would fire on the
            # assistant's own last word.
            listener.mute()
            try:
                turn = await pipeline.run_turn(pcm, config.audio.sample_rate, trigger="wake_word")
            finally:
                listener.unmute()

            _report(turn)
            if args.once:
                return 0 if turn.ok else 1
    return 0  # pragma: no cover - only reached if the source ends


async def _turns(pipeline: Pipeline, config: Config, args: argparse.Namespace) -> int:
    """Keep answering questions until asked to stop."""
    # A file holds one question, so looping over it would answer the same thing
    # for ever.
    once = args.once or args.from_file is not None
    while True:
        pcm = await capture(config, args.from_file)
        if not pcm:
            print("nothing was recorded")
            return 1

        turn = await pipeline.run_turn(pcm, config.audio.sample_rate)
        _report(turn)
        if once:
            return 0 if turn.ok else 1


def _report(turn: TurnResult) -> None:
    print()
    if turn.transcript:
        print(f"  you:        {turn.transcript}")
    if turn.reply:
        print(f"  chilluevar: {turn.reply}")
    if turn.failed is not None:
        print(f"  (gave up at {turn.failed.value}: {turn.reason})")
    print(f"  [{_timings(turn)}]\n")


def _timings(turn: TurnResult) -> str:
    """One line of timings, which is also the number the README quotes.

    Printed every turn rather than behind a flag: a latency nobody sees is a
    latency nobody fixes, and issue #14 starts from whatever this says today.
    """
    names = {
        Stage.SPEECH_TO_TEXT: "heard in",
        Stage.LANGUAGE_MODEL: "thought for",
        Stage.TEXT_TO_SPEECH: "spoke in",
    }
    parts = [
        f"{names[stage]} {ms / 1000:.1f} s" for stage, ms in turn.timings.items() if stage in names
    ]
    parts.append(f"{turn.total_ms / 1000:.1f} s in total")
    return ", ".join(parts)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, log_transcripts=args.log_transcripts)

    if args.command == "devices":
        return show_devices()

    config = load_config(args.config)
    if args.command == "record":
        return asyncio.run(record_once(config, args))
    if args.command == "transcribe":
        return asyncio.run(transcribe_once(config, args))
    if args.command == "ask":
        return asyncio.run(ask(config, args))
    if args.command == "say":
        return asyncio.run(say_once(config, args))
    if args.command == "run":
        return asyncio.run(run_assistant(config, args))
    return show_config(config)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
