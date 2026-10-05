## What this does

<!-- One or two sentences. If it closes an issue: "Closes #12". -->

## Why this way

<!-- The decision you took and the alternative you rejected.
     If it is an architectural decision, add a file in docs/decisions/. -->

## How you checked it

<!-- What you ran or tried by hand. If it affects latency, paste the benchmark
     before and after. -->

## Before asking for review

- [ ] `ruff check .` and `ruff format .` pass
- [ ] `pytest` passes, with new tests for what you added
- [ ] No test calls a real model or the network
- [ ] No secrets, models or recordings committed
- [ ] README or example config updated if needed
