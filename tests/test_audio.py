"""Audio tests. They never open a sound card, so they run anywhere, CI included."""

from __future__ import annotations

import asyncio
import wave
from pathlib import Path

import pytest

from chilluevar.audio import MemorySink, WavFileSource, collect
from chilluevar.audio.wav import save_wav
from chilluevar.interfaces import AudioSink, AudioSource


def write_wav(
    path: Path,
    *,
    seconds: float = 0.3,
    sample_rate: int = 16_000,
    channels: int = 1,
    width: int = 2,
) -> Path:
    """Write a silent wav file to test against."""
    silence = b"\x00" * int(sample_rate * seconds) * width * channels
    if channels == 1 and width == 2:
        return save_wav(path, silence, sample_rate)
    # The odd formats the source must refuse still need writing by hand.
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(sample_rate)
        handle.writeframes(silence)
    return path


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #


async def test_wav_source_yields_fixed_size_frames(tmp_path: Path) -> None:
    source = WavFileSource(write_wav(tmp_path / "a.wav", seconds=0.3), frame_ms=30)
    sizes = {len(frame) async for frame in source.frames()}
    # 30 ms at 16 kHz, 16-bit mono = 480 samples = 960 bytes, every frame.
    assert sizes == {960}


async def test_wav_source_pads_the_last_short_frame(tmp_path: Path) -> None:
    # 0.25 s is not a whole number of 30 ms frames, so the last one is short
    # and must be padded rather than handed over at the wrong length.
    source = WavFileSource(write_wav(tmp_path / "b.wav", seconds=0.25), frame_ms=30)
    frames = [frame async for frame in source.frames()]
    assert all(len(frame) == 960 for frame in frames)
    assert len(frames) == 9  # ceil(250 / 30)


async def test_wav_source_rejects_stereo(tmp_path: Path) -> None:
    source = WavFileSource(write_wav(tmp_path / "c.wav", channels=2))
    with pytest.raises(ValueError, match="mono"):
        [frame async for frame in source.frames()]


async def test_wav_source_rejects_the_wrong_sample_rate(tmp_path: Path) -> None:
    source = WavFileSource(write_wav(tmp_path / "d.wav", sample_rate=44_100))
    with pytest.raises(ValueError, match="16000 Hz"):
        [frame async for frame in source.frames()]


async def test_wav_source_satisfies_the_protocol(tmp_path: Path) -> None:
    assert isinstance(WavFileSource(write_wav(tmp_path / "e.wav")), AudioSource)


# --------------------------------------------------------------------------- #
# collect (what push-to-talk uses)
# --------------------------------------------------------------------------- #


async def test_collect_joins_every_frame(tmp_path: Path) -> None:
    source = WavFileSource(write_wav(tmp_path / "f.wav", seconds=0.3), frame_ms=30)
    audio = await collect(source)
    assert len(audio) == 10 * 960


async def test_collect_stops_when_asked(tmp_path: Path) -> None:
    source = WavFileSource(write_wav(tmp_path / "g.wav", seconds=5), frame_ms=30)
    stop = asyncio.Event()
    stop.set()
    audio = await collect(source, stop=stop)
    # The stop flag is checked after the first frame, so exactly one arrives.
    assert len(audio) == 960


async def test_collect_has_a_safety_limit(tmp_path: Path) -> None:
    # A push-to-talk key that never gets released must not fill memory.
    source = WavFileSource(write_wav(tmp_path / "h.wav", seconds=5), frame_ms=30)
    audio = await collect(source, max_seconds=1.0)
    assert len(audio) == 33 * 960  # int(1000 / 30) frames


# --------------------------------------------------------------------------- #
# Sinks
# --------------------------------------------------------------------------- #


async def test_sink_plays_in_order() -> None:
    async with MemorySink() as sink:
        await sink.play(b"one", 16_000)
        await sink.play(b"two", 16_000)
        await sink.drain()
        assert [pcm for pcm, _ in sink.played] == [b"one", b"two"]


async def test_play_does_not_block_the_caller() -> None:
    sink = MemorySink()
    sink.start()
    # Queueing three clips must return immediately; only drain() waits.
    await asyncio.wait_for(
        asyncio.gather(*(sink.play(b"x", 16_000) for _ in range(3))),
        timeout=0.5,
    )
    await sink.aclose()
    assert len(sink.played) == 3


async def test_a_failing_clip_does_not_kill_the_sink() -> None:
    class Flaky(MemorySink):
        async def _write(self, pcm: bytes, sample_rate: int) -> None:
            if pcm == b"bad":
                raise RuntimeError("device went away")
            await super()._write(pcm, sample_rate)

    async with Flaky() as sink:
        await sink.play(b"bad", 16_000)
        await sink.play(b"good", 16_000)
        await sink.drain()
        # The assistant must not go permanently mute after one glitch.
        assert [pcm for pcm, _ in sink.played] == [b"good"]


async def test_sink_satisfies_the_protocol() -> None:
    assert isinstance(MemorySink(), AudioSink)
