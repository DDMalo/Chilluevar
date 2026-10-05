"""Microphone capture, playback, wake word and voice activity detection.

Sources and sinks arrive with issue #3; the wake word and voice activity
detector with issue #8.
"""

from chilluevar.audio.sink import MemorySink, SpeakerSink
from chilluevar.audio.source import MicrophoneSource, WavFileSource, collect

__all__ = [
    "MemorySink",
    "MicrophoneSource",
    "SpeakerSink",
    "WavFileSource",
    "collect",
]
