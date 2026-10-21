# 0005 · Talk to Ollama through its native API, with `httpx`

- **Date:** 2026-10-21
- **Status:** accepted

## Problem

The language model runs in Ollama, a separate local server. Python has to send
it the conversation and read the answer back as it is written, so the
assistant can start speaking before the model has finished. There are three
ways to do that, and the choice decides what we can tune, what we can test and
what we depend on.

## Decision

`OllamaLLM` (`src/chilluevar/llm/ollama.py`) calls Ollama's native `/api/chat`
endpoint directly with `httpx`, and reads the answer as newline-delimited JSON,
one object per token. `httpx` is the only new dependency, in an `[llm]` extra
(decision 0003).

## Consequences

**For**

- The native API takes the settings that matter most for a voice assistant:
  `keep_alive`, which keeps the model in memory between questions instead of
  unloading it after five idle minutes, and `options.num_ctx`, the context
  size, whose small default silently drops the start of a long conversation,
  system prompt included.
- It also takes `think`, which turns off the hidden reasoning of models such
  as qwen3. With it on, the first spoken word arrives seconds later.
- Tests replay Ollama's exact wire format with `httpx.MockTransport`: the
  parsing that runs in CI is the parsing that runs at home, with no server, no
  model and no network.
- About two hundred lines, half of them comments, and nothing hidden. We can
  explain every one of them.
- Closing the stream early (the user interrupts) closes the connection, and
  Ollama stops generating straight away.

**Against**

- We maintain the parsing ourselves. If Ollama changes its format, that is our
  problem, not a library's. The format has been stable, and the tests pin it.
- Moving to a cloud provider means writing a second client. The `LanguageModel`
  protocol (decision 0002) already makes that a new class plus one line of
  configuration, and the project is local first anyway.

## Rejected alternatives

**The `ollama` Python package.** The official client, and a perfectly good one.
But it wraps the same HTTP calls in its own types, pulls in pydantic, and to
test against it we would have to fake the library rather than the protocol. It
saves little code and hides the part worth understanding.

**Ollama's OpenAI-compatible endpoint (`/v1/chat/completions`).** Tempting,
because one client would then also speak to Groq, OpenRouter or llama.cpp. But
that endpoint does not take `keep_alive` or the context size, which are exactly
the two settings that decide how fast and how reliable a local assistant feels.

## Measured

On our laptop (RTX 4060, 8 GB) with `qwen2.5:3b`:

| | Without these settings | With them |
| --- | --- | --- |
| First question after starting | 5.0 s loading in silence | `warm_up()` pays it at start-up |
| Next question | 0.5 s to be ready | 0.5 s to be ready |
| After 5 idle minutes | unloaded: 5 s again | still loaded for 30 minutes |
| "¿Qué necesito para hacer una paella?" | 403 tokens, 16.8 s | 13 words, 0.84 s |

The last row is the system prompt as much as the client, but it needs both: the
prompt asks for short answers and `num_predict` stops the ones that ignore it.
