"""Choose the wake word threshold by measuring it, not by guessing.

The threshold trades two failures against each other:

- too low and the assistant wakes up on the television, a similar sounding
  word, or someone saying something else entirely in the next room;
- too high and it ignores you, which is worse, because you then say it louder
  and feel ridiculous.

Neither number can be reasoned about. They are measured, over our own flat and
our own voices, the same way issue #4 chose the Whisper model size:

    # once: leave it recording the room while you do something else
    python scripts/tune_wake.py --record 3600 --output recordings/noise/evening.wav

    # then: try every threshold against the same audio
    python scripts/tune_wake.py --noise recordings/noise/*.wav --spoken recordings/wake/*.wav

Each file under --spoken is a clip of one of us saying the wake word and then
asking something. Each file under --noise is the flat with nobody talking to
the assistant: television, conversation, the kitchen, whatever an evening
sounds like.

The recordings stay out of the repository. They are our voices and our flat;
only the table goes in the issue.

Every pass runs the real `Listener` with the real engines, so the numbers come
from the same code that runs live rather than from an approximation of it.
A pass is about 22 times faster than real time, so an hour of audio costs
roughly three minutes per threshold.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path

from chilluevar.audio.listen import Listener
from chilluevar.audio.source import MicrophoneSource, WavFileSource
from chilluevar.audio.vad import SileroVAD
from chilluevar.audio.wake import OpenWakeWord
from chilluevar.audio.wav import save_wav
from chilluevar.config import Config, load_config

DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SAMPLE_WIDTH_BYTES = 2


@dataclass(slots=True)
class Result:
    threshold: float
    woke: int = 0
    answered: int = 0
    detected: int = 0
    clips: int = 0
    noise_seconds: float = 0.0

    @property
    def woke_per_hour(self) -> float:
        return self.woke * 3600 / self.noise_seconds if self.noise_seconds else 0.0

    @property
    def answered_per_hour(self) -> float:
        return self.answered * 3600 / self.noise_seconds if self.noise_seconds else 0.0

    @property
    def detection_rate(self) -> float:
        return self.detected / self.clips if self.clips else 0.0


def build_listener(config: Config, threshold: float, source: object) -> Listener:
    """A listener exactly as `chilluevar run --hands-free` builds it."""
    return Listener(
        source,  # type: ignore[arg-type]
        wake=OpenWakeWord(config.wake_word.model, framework=config.wake_word.framework),
        vad=SileroVAD(threshold=config.listen.vad_threshold),
        threshold=threshold,
        start_timeout_ms=config.listen.start_timeout_ms,
        silence_ms=config.listen.silence_ms,
        min_utterance_ms=config.listen.min_utterance_ms,
        max_utterance_ms=config.listen.max_utterance_ms,
    )


async def run_over(config: Config, threshold: float, path: Path) -> Listener:
    """Feed one file through a listener and return it, counters and all."""
    source = WavFileSource(
        path,
        sample_rate=config.audio.sample_rate,
        frame_ms=config.audio.frame_ms,
    )
    listener = build_listener(config, threshold, source)
    async for frame in source.frames():
        listener.feed(frame)
    return listener


def seconds_in(path: Path, sample_rate: int) -> float:
    return path.stat().st_size / (sample_rate * SAMPLE_WIDTH_BYTES)


async def measure(
    config: Config,
    threshold: float,
    noise: list[Path],
    spoken: list[Path],
) -> Result:
    result = Result(threshold=threshold, clips=len(spoken))

    for path in noise:
        listener = await run_over(config, threshold, path)
        result.woke += listener.stats.activations
        # Something that got all the way through as a question. Worse than a
        # bare activation: the assistant actually answers, out loud, at night.
        result.answered += listener.stats.utterances
        result.noise_seconds += seconds_in(path, config.audio.sample_rate)

    for path in spoken:
        listener = await run_over(config, threshold, path)
        result.detected += 1 if listener.stats.utterances else 0

    return result


def pick(results: list[Result], budget: float) -> Result | None:
    """The lowest threshold that stays within the false-wake budget.

    Lowest rather than highest because missing you is the failure you notice
    every time, and a stray activation at three in the morning is the one you
    only notice once. Among those within budget, the lowest threshold is the
    one most likely to hear you.
    """
    if results and results[0].clips and not any(r.detected for r in results):
        # Not a threshold problem. Recommending the lowest one here would read
        # as "this is the best we can do" when the truth is that the clips do
        # not contain the word, or contain a different one.
        return None

    affordable = [r for r in results if r.woke_per_hour <= budget]
    if not affordable:
        return None
    best = max(r.detection_rate for r in affordable)
    return min(
        (r for r in affordable if r.detection_rate == best),
        key=lambda r: r.threshold,
    )


def table(results: list[Result]) -> str:
    lines = [
        "| threshold | woke/hour | answered/hour | detected |",
        "| --- | --- | --- | --- |",
    ]
    for r in results:
        detected = f"{r.detected}/{r.clips}" if r.clips else "-"
        lines.append(
            f"| {r.threshold:.2f} | {r.woke_per_hour:.1f} | "
            f"{r.answered_per_hour:.1f} | {detected} |"
        )
    return "\n".join(lines)


async def record(seconds: float, output: Path, config: Config) -> int:
    """Keep the room for a while, so every threshold is tried on the same audio."""
    source = MicrophoneSource(
        sample_rate=config.audio.sample_rate,
        frame_ms=config.audio.frame_ms,
        device=config.audio.input_device,
    )
    wanted = int(seconds * 1000 / config.audio.frame_ms)
    chunks: list[bytes] = []

    print(f"recording {seconds / 60:.0f} min to {output}; Ctrl+C to stop early")
    try:
        async for frame in source.frames():
            chunks.append(frame)
            if len(chunks) % (1000 // config.audio.frame_ms * 60) == 0:
                print(f"  {len(chunks) * config.audio.frame_ms / 60000:.0f} min", flush=True)
            if len(chunks) >= wanted:
                break
    except KeyboardInterrupt:
        print("\nstopped early")

    if not chunks:
        print("nothing was recorded")
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    save_wav(output, b"".join(chunks), config.audio.sample_rate)
    print(f"saved {len(chunks) * config.audio.frame_ms / 60000:.1f} min to {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--noise", type=Path, nargs="*", default=[])
    parser.add_argument("--spoken", type=Path, nargs="*", default=[])
    parser.add_argument("--thresholds", type=float, nargs="*", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument(
        "--budget",
        type=float,
        default=1.0,
        help="false wake-ups per hour we are willing to live with",
    )
    parser.add_argument("--record", type=float, default=None, metavar="SECONDS")
    parser.add_argument("--output", type=Path, default=Path("recordings/noise/room.wav"))
    args = parser.parse_args(argv)

    config = load_config(args.config)

    if args.record is not None:
        return asyncio.run(record(args.record, args.output, config))

    noise = [p for p in args.noise if p.exists()]
    spoken = [p for p in args.spoken if p.exists()]
    if not noise and not spoken:
        print("nothing to measure: pass --noise and/or --spoken, or --record first")
        return 1

    minutes = sum(seconds_in(p, config.audio.sample_rate) for p in noise) / 60
    print(f"wake word: {config.wake_word.model} ({config.wake_word.framework})")
    print(f"noise: {minutes:.1f} min over {len(noise)} file(s)")
    print(f"spoken: {len(spoken)} clip(s)\n")

    results = []
    for threshold in sorted(args.thresholds):
        print(f"  threshold {threshold:.2f}...", end="", flush=True)
        result = asyncio.run(measure(config, threshold, noise, spoken))
        results.append(result)
        print(" done", flush=True)

    print()
    print(table(results))

    chosen = pick(results, args.budget)
    print()
    if chosen is None and spoken and not any(r.detected for r in results):
        print(
            f"the word was not heard once in {len(spoken)} clip(s) at any threshold. "
            f"That is not a threshold to tune: check the clips really say "
            f'"{config.wake_word.model.replace("_", " ")}" and that they are '
            "16 kHz mono."
        )
        return 1
    if chosen is None:
        print(
            f"no threshold stays under {args.budget:.1f} false wake-ups an hour. "
            "Either the room is loud or the word is too common: try another "
            "bundled model, or train one on our own name (issue #22)."
        )
        return 0
    print(
        f"pick {chosen.threshold:.2f}: {chosen.woke_per_hour:.1f} false wake-ups an hour "
        f"and {chosen.detected}/{chosen.clips} heard. Put it in wake_word.threshold."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
