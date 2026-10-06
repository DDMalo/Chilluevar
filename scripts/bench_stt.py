#!/usr/bin/env python3
"""Compare Whisper model sizes on accuracy and speed.

This is the harness for the second half of issue #4. Point it at a folder of
recordings and what was actually said, and it prints a table you can paste into
the issue.

    recordings/
      001.wav
      001.txt      <- what you actually said, one line
      002.wav
      002.txt

    python scripts/bench_stt.py recordings --models tiny base small

Accuracy is word error rate: the proportion of words that have to be inserted,
deleted or substituted to turn the transcript into the reference. Lower is
better, and 0 means a perfect match.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import time
import unicodedata
import wave
from pathlib import Path

from chilluevar.stt import WhisperSTT


def normalise(text: str) -> list[str]:
    """Lower-case, strip accents and punctuation, split into words.

    Without this, "enciende la luz." and "Enciende la luz" would count as a
    mistake, and the numbers would say more about punctuation than about the
    model.
    """
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return re.findall(r"[a-z0-9]+", text)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over words, divided by the reference length."""
    ref, hyp = normalise(reference), normalise(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    # One row of the edit-distance matrix at a time: the full matrix is not
    # needed when only the final number matters.
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            current.append(
                previous[j - 1]
                if ref_word == hyp_word
                else 1 + min(previous[j - 1], previous[j], current[j - 1])
            )
        previous = current
    return previous[-1] / len(ref)


def read_wav(path: Path) -> tuple[bytes, int, float]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"{path.name}: expected 16-bit mono")
        pcm = handle.readframes(handle.getnframes())
        rate = handle.getframerate()
    return pcm, rate, len(pcm) / (rate * 2)


async def run_model(name: str, clips: list[Path], args: argparse.Namespace) -> dict[str, float]:
    stt = WhisperSTT(name, device=args.device, compute_type=args.compute_type)
    hints = args.hints.split(",") if args.hints else []

    errors: list[float] = []
    audio_seconds = 0.0
    spent = 0.0

    for clip in clips:
        pcm, rate, seconds = read_wav(clip)
        reference = clip.with_suffix(".txt").read_text(encoding="utf-8").strip()

        started = time.monotonic()
        text = await stt.transcribe(pcm, rate, language=args.language, hints=hints)
        spent += time.monotonic() - started
        audio_seconds += seconds

        errors.append(word_error_rate(reference, text))
        if args.verbose:
            print(f"    {clip.name}: {text}")

    return {
        "wer": sum(errors) / len(errors),
        "worst": max(errors),
        "seconds": spent,
        "speed": audio_seconds / spent if spent else 0.0,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="folder with .wav files and matching .txt")
    parser.add_argument("--models", nargs="+", default=["tiny", "base", "small"])
    parser.add_argument("--language", default="es")
    parser.add_argument("--hints", default="Chilluevar")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="default")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every transcript")
    args = parser.parse_args()

    clips = sorted(args.folder.glob("*.wav"))
    clips = [clip for clip in clips if clip.with_suffix(".txt").exists()]
    if not clips:
        print(f"no .wav with a matching .txt in {args.folder}")
        return 1
    print(f"{len(clips)} clips, {len(args.models)} models\n")

    results: dict[str, dict[str, float]] = {}
    for name in args.models:
        print(f"  {name}...")
        results[name] = await run_model(name, clips, args)

    print(f"\n| {'model':8} | {'WER':>7} | {'worst':>7} | {'time':>8} | {'speed':>8} |")
    print(f"| {'-' * 8} | {'-' * 7} | {'-' * 7} | {'-' * 8} | {'-' * 8} |")
    for name, row in results.items():
        print(
            f"| {name:8} | {row['wer']:6.1%} | {row['worst']:6.1%} "
            f"| {row['seconds']:7.1f}s | {row['speed']:7.1f}x |"
        )
    print("\nWER: lower is better. speed: how many seconds of audio per second of compute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
