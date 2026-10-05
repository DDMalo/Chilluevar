"""Entry point.

For now it loads the configuration and prints what it would run, which is enough
to prove the wiring works end to end and gives the first version something to
build on. The pipeline itself arrives with issues #3 to #7.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chilluevar import __version__
from chilluevar.config import load_config
from chilluevar.logging_setup import setup_logging


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, log_transcripts=args.log_transcripts)
    config = load_config(args.config)

    print(f"chilluevar {__version__}")
    print(f"  mode:  {config.mode}")
    print(f"  stt:   {config.stt.engine} ({config.stt.model})")
    print(f"  llm:   {config.llm.engine} ({config.llm.model})")
    print(f"  tts:   {config.tts.engine} ({config.tts.voice})")
    print("\nThe pipeline is not wired yet. See the open issues for v0.1.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
