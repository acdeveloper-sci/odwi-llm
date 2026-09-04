# odwi-llm — context for Claude Code

## What this project is

A provider-agnostic LLM access layer: a thin port (`LLMPort`) plus
interchangeable adapters. Applications depend only on `LLMPort` and its
types; LiteLLM, Any-LLM or a native SDK sit behind the port.

Stage 1 (contract + adapters + test suite) is **implemented**. See
[`CHANGELOG.md`](CHANGELOG.md) for what exists and what is left for
Stage 2.

## Repo layout

```
src/odwi_llm/
  core/          contract only, no provider library
    types.py         §4 types (Message, LLMRequest, LLMResponse, StreamChunk, …)
    requirements.py  LLMRequirements, LLMCapabilities, CapabilityError
    errors.py        LLMError hierarchy — the domain never sees a raw library error
    port.py          LLMPort (ABC): generate · structured · stream · chat_with_tools
    sync.py          SyncLLM — blocking facade over any LLMPort
    composition.py   FallbackLLM(primary, fallback=None)
  adapters/      the only layer that imports a provider library
    litellm_adapter.py   LiteLLMAdapter — primary
    anyllm_adapter.py    AnyLLMAdapter — fallback
    config.py            ProviderConfig (env-var API key, base_url, timeout)
    pricing.py           static cost table (no library is a reliable cost source)
    _shared.py           helpers + call_with_retry (bounded technical retry, §5.3)
tests/
  contract/        exact assertions vs a FakeAdapter — always offline
  contract_shape/  structural assertions; replays tests/cassettes/ by default,
                   --live hits real providers, --live --record rewrites cassettes
  architecture/    import-boundary AST check
  adapters/        pricing, error mapping, retry, real-fallback tests
  cassettes/       recorded LLMPort-boundary responses (see its README.md)
llm_lab/           separate uv project — Stage 1 exploration, kept in the repo
```

## Where the documentation lives

Design and task plans are **private local notes under `.docs/`**, not part
of this repository and not linkable from committed files. As of now there
are four plans:

- `llm_agnostic_connector_design.md` — architecture, the §4 contract, closed decisions
- `odwi_llm_experiments_task_plan.md` — Stage 1 lab (`llm_lab/experiments/`)
- `odwi_llm_core_task_plan.md` — `core/` + adapters + contract tests
- `odwi_llm_repo_hygiene_task_plan.md` — preparing the public repo

If you do not have `.docs/` locally, ask the user for the relevant plan
rather than guessing.

## Project rules

- All code (names, comments, docstrings, strings) is written in English,
  regardless of the conversation language. Everything committed to this
  repo outside `.docs/` is in English — `README.md`, `CHANGELOG.md`, this
  file, `llm_lab/`.
- Package manager: `uv`. Do not use `pip` directly.
- Follow the active task plan in order. Each task has a "Done when"
  ("Hecho cuando") criterion — verify it before moving to the next.
- One commit per task that changed files; run `uv run pytest` and
  `uv run mypy src tests` before proposing a commit.
- Anthropic is excluded from the Stage 1 lab (no API key). Do not add it
  unless the user says so.

## After finishing each task

Summarize in 2-3 lines what was done and which file(s) changed, then
stop and let the user review before any commit.

## Best practices / security

- **Never commit secrets or API keys.** Credentials are always read from
  an environment variable (`GEMINI_API_KEY`, `GROQ_API_KEY`; local
  providers via `ProviderConfig(base_url=…)`).
- **Never log or print the value of an API key or an auth header.**
- **`.docs/` is never committed.** Enforced by `.gitignore` and a local
  `pre-commit` hook that rejects any staged path under `.docs/` (even a
  forced `git add -f`). It was removed from all git history before the
  first public push.
- **Cassettes never capture a raw HTTP request or the `raw` field of a
  response.** A cassette is the adapter's final, retry-resolved,
  reasoning-stripped object, serialized with `model_dump(exclude={"raw"})`.
  Recording happens at the `LLMPort` boundary, not at the HTTP layer — so
  a cassette can never contain an API key.
- **Before adding a dependency**, weigh its maintenance and popularity
  against the real need. Do not add one for momentary convenience.

## Stage 2 maintenance rule

When a Stage 2 (or later) task adds or changes public surface of the
package, update `README.md` and/or add an example under `examples/` in the
same commit — do not leave it for the end of the stage.
