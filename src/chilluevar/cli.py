"""Entry point.

`chilluevar` prints the configuration it would run with. `chilluevar record` is
the push-to-talk loop from issue #3: record, save, play back. The rest of the
pipeline arrives with issues #4 to #7.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from chilluevar import __version__
from chilluevar.audio import MicrophoneSource, SpeakerSink, WavFileSource, collect
from chilluevar.audio.wav import save_wav
from chilluevar.config import Config, load_config
from chilluevar.logging_setup import setup_logging
from chilluevar.stt import WhisperSTT


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

    sub.add_parser("devices", help="list the audio devices this machine has")
    return parser


def show_config(config: Config) -> int:
    print(f"chilluevar {__version__}")
    print(f"  mode:  {config.mode}")
    print(f"  stt:   {config.stt.engine} ({config.stt.model})")
    print(f"  llm:   {config.llm.engine} ({config.llm.model})")
    print(f"  tts:   {config.tts.engine} ({config.tts.voice})")
    print("\nTry `chilluevar record`. The full pipeline is not wired yet.")
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


async def _wait_for_enter(stop: asyncio.Event) -> None:
    await asyncio.to_thread(input)
    stop.set()


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
    return show_config(config)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
