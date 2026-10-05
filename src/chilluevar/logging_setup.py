"""Logging that never writes down what was said.

A voice assistant that logs transcripts is a voice assistant that keeps a diary
of its owner. By default the logs record that a turn happened, which stage ran
and how long it took, but not the text of the conversation. Setting
`log_transcripts=True` turns that on explicitly, for debugging, and the log line
says so.
"""

from __future__ import annotations

import logging
import sys

__all__ = ["setup_logging"]

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"


class _RedactTranscripts(logging.Filter):
    """Drop the fields that would contain what the user or assistant said."""

    SENSITIVE_FIELDS = ("text", "transcript", "reply", "arguments")

    def filter(self, record: logging.LogRecord) -> bool:
        for name in self.SENSITIVE_FIELDS:
            if hasattr(record, name):
                setattr(record, name, "<redacted>")
        return True


def setup_logging(level: str = "INFO", *, log_transcripts: bool = False) -> None:
    """Configure the root logger once, at startup."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    if not log_transcripts:
        handler.addFilter(_RedactTranscripts())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    if log_transcripts:
        root.warning("transcript logging is ON: this log will contain what was said")
