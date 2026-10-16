"""Language model clients behind the LanguageModel protocol.

The Ollama client is imported from `chilluevar.llm.ollama` rather than from
here, because it needs httpx (the `[llm]` extra) and this package must keep
importing on an install without it.
"""

from chilluevar.llm.base import LanguageModelError

__all__ = ["LanguageModelError"]
