"""What every language model client shares, whatever the provider."""

from __future__ import annotations

__all__ = ["LanguageModelError"]


class LanguageModelError(RuntimeError):
    """The model could not answer, with a message that says how to fix it.

    Every client turns its own failures (a refused connection, a missing model,
    a broken reply) into this one exception. The pipeline then only has to
    catch one thing to say a short apology out loud and go back to listening,
    whichever engine is behind it.

    The message is written for us, not for the user: "is Ollama running?" is
    far more useful in a log than a raw connection traceback.
    """
