# How we work

There are two of us, so the process matters as much as the code.

## Split

| Role | Area |
| --- | --- |
| **A** | Audio, wake word, speech to text, speech synthesis, external integrations, security, satellite mode, deployment |
| **B** | Language model, tools, memory, model comparison, tool-selection tests |
| **Both** | Architecture, performance, evaluation, documentation |

Every issue carries its role label. We both review each other's pull requests,
and each of us should occasionally implement something from the other's area so
neither gets boxed in.

## Getting set up

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
cp config/chilluevar.example.yaml config/chilluevar.yaml
```

## The loop

1. Pick an issue from the current milestone and assign it to yourself.
2. Branch from `main`: `feat/<short-thing>`, `fix/<short-thing>`, `docs/<thing>`.
3. Commit with [Conventional Commits](https://www.conventionalcommits.org):
   `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`. One per coherent
   change, not one per day.
4. Open the pull request, even as a draft. The other one reviews it.
5. Once approved, squash and merge with a message in the same format.

Nothing goes straight into `main`.

## Rules we do not bend

- **Tests never call a real model or the network.** That is what the protocols
  in `interfaces.py` are for: write a fake engine. A test that needs to download
  something is a test that is set up wrong.
- **Every issue lands with its tests.** They are not left for the end.
- **No secrets, models or recordings in the repository.** Secrets live in
  `.env`, which is gitignored.
- **Recordings of our voices are not pushed**, except the short fixtures in
  `tests/fixtures/` that we both agree on.
- **An architectural decision gets written down.** A new numbered file in
  `docs/decisions/`, with the rejected alternative and why. That is what later
  lets us explain the project in an interview.

## Versions

One milestone per version. When it closes: an entry in `CHANGELOG.md`, a
`vX.Y.Z` tag and a GitHub release.
