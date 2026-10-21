# 0006 · `qwen3:8b`, with thinking off, as the first language model

- **Date:** 2026-10-21
- **Status:** accepted, to be confirmed on the brain machine

## Problem

Issue #5 asks for "a first small model that answers well in Spanish". Small
enough to run next to Whisper on one laptop graphics card; good enough that
the assistant does not make things up out loud; and obedient to the system
prompt, because an answer full of lists and symbols cannot be spoken.

## How we measured

`scripts/bench_llm.py` asked five models the same 20 Spanish questions
(`benchmarks/preguntas.txt`) with the real system prompt, on an RTX 4060 laptop
GPU with 8 GB, Ollama 0.40.4. Each group of questions tempts the model to break
one rule: a list, a long answer, a symbol, or an invented fact it cannot know.

The script measures speed and format. Whether the facts are right was checked
by reading all 100 answers. The first run:

| model | size | load | first words | total | words | breaks format | wrong facts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen2.5:3b | 1.9 GB | 6.1 s | 0.47 s | 0.68 s | 13 | 0/20 | 6/20 |
| llama3.2:3b | 2.0 GB | 5.1 s | 0.46 s | 0.74 s | 24 | 1/20 | 5/20 |
| qwen2.5:7b-instruct | 4.7 GB | 7.2 s | 0.48 s | 0.87 s | 15 | 0/20 | 1/20 |
| qwen3:4b | 2.5 GB | 6.1 s | 0.47 s | 1.95 s | 88 | 20/20 | — |
| **qwen3:8b** | 5.2 GB | 7.1 s | 0.49 s | 1.35 s | 26 | **0/20** | **1/20** |

What the "wrong facts" were:

- **qwen2.5:3b:** monkeys ("monos") in a paella, seven planets instead of
  eight, and it invented tomorrow's weather and today's exchange rate instead
  of saying it could not know them. Two answers were not even sentences.
- **llama3.2:3b:** 15 % of 80 is "twenty", today is "the eighteenth" when the
  prompt says the twenty-first, Python has to be installed "from a CD", the
  paella has chickpeas, and its history of Spain is wrong and was cut off by
  the length ceiling. It also says "papas", not "patatas", although the
  prompt asks for Spain.
- **qwen2.5:7b-instruct:** water boils at "eighty-five degrees". One error,
  but in the most basic fact of the set.
- **qwen3:4b:** disqualified. With `think: false` it still wrote its reasoning,
  in English, as the answer ("Okay, the user is asking..."), so every answer ran
  into the length ceiling. `think: false` worked for qwen3:8b on the same
  Ollama.
- **qwen3:8b:** recommended places in Jaén that probably do not exist. It also
  put chorizo in the paella, which is a matter of taste rather than of fact.

## Second run, with the prompt fixed

The first run showed qwen3:8b ending over half of its answers with "¿quieres
saber más?", because the prompt told it to "offer to expand". The prompt was
changed to ask for no closing offer, and both finalists were measured again
with the same 20 questions.

| model | first words | total | words | breaks format | wrong facts |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen2.5:7b-instruct | 0.48 s | 0.86 s | 16 | 1/20 | 2/20 |
| **qwen3:8b** | 0.48 s | 1.12 s | **20** (was 26) | 0/20 | 2/20 |

- **qwen2.5:7b-instruct** invented today's exchange rate ("about 1.10
  dollars"), which it had declined to do in the first run. It ended its answer
  about the weather with a sentence in Chinese, which a Spanish voice cannot
  read and `speech_problems()` does not catch. Its history of Spain ran into
  the length ceiling.
- **qwen3:8b** said 3 to the fourth is "thirty-six", which it had got right in
  the first run. It again recommended places in Jaén that are not there, this
  time a museum that is in Tenerife. It used American Spanish twice
  ("camarones", "vegetales"), and it offered to "check it with a web search",
  which it cannot do.
- **`think: false` sent to qwen2.5:7b-instruct**, a model that does not reason,
  was accepted and ignored. Sending it by default is therefore safe.

Over the two runs each finalist got 3 facts wrong in 40 answers, and each got
right in one run a basic fact it got wrong in the other (100 degrees for
qwen2.5, 81 for qwen3). At temperature 0.6, a single run cannot tell them apart
on facts.

## Decision

`qwen3:8b` with `think: false` is the default model in `config.py` and in the
example configuration.

## Why

- **The two 3B models make things up**, and a voice assistant that confidently
  says something false is worse than one that says nothing. They are out.
- **On facts the finalists are tied**, 3 wrong in 40 each. What tips it is how
  they fail:
  - qwen3:8b broke the spoken format in none of its 40 answers;
    qwen2.5:7b-instruct broke it once and, worse, switched to Chinese
    mid-sentence, a failure the detector cannot see and the user would hear.
  - qwen3:8b adds the useful nuance (water boils at less than 100 degrees on a
    mountain) and, when it cannot know something, suggests where to find it.
- **It is a little slower to finish** (1.12 s against 0.86 s) and says a
  little more (20 words against 16), but the first words arrive at the same
  time, 0.48 s. Until replies are spoken sentence by sentence (#15, v0.4), the
  whole answer is what the user waits for, so this is a real cost of about a
  quarter of a second.
- **qwen3 is the usual recommendation among small models for tool calling,**
  which matters from #10 onwards. We have not measured that yet; #13 will.

## Costs and risks

- **5.2 GB of the 8 GB of graphics memory,** with Whisper still to fit beside
  it. If the two do not fit together, qwen2.5:7b-instruct (4.7 GB) is the
  fallback: it scored the same on facts.
- **`think: false` must stay on.** Without it qwen3 reasons before every
  answer, and the first words arrive seconds later. Nothing stops someone
  setting it back to null, so the example file says why it is there.
- **Random mistakes on basic facts.** Both finalists got a basic fact right in
  one run and wrong in the other. A lower temperature should make that rarer;
  measuring it belongs with the model optimisation in #16.
- **The detector cannot see a change of language.** Text in Chinese passed as
  a good answer. Flagging letters outside the Latin alphabet is a small
  addition to `speech_problems()` for a follow-up.
- **Two runs of twenty questions.** Enough to rule out the 3B models, not
  enough to call the two finalists apart with certainty. The comparison is a
  script so that it can be rerun cheaply, with more questions, whenever the
  prompt changes.
- **Measured on B's laptop.** If the brain ends up being A's laptop, the
  comparison has to be rerun there before this is final.

## Rejected alternatives

**qwen2.5:7b-instruct.** Tied on facts and a little faster, but it once
switched to Chinese mid-answer. The fallback if memory is short.

**Any 3B model.** Faster to load and lighter, but they invent facts, which no
amount of prompting fixes.

**Larger models (14B and up).** They do not fit in 8 GB next to Whisper.
