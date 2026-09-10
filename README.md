# odwi-llm

[![CI](https://github.com/acdeveloper-sci/odwi-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/acdeveloper-sci/odwi-llm/actions/workflows/ci.yml)

Provider-agnostic LLM access layer. Applications depend only on `LLMPort`
and its types; LiteLLM, Any-LLM or a native SDK sit behind the port as
interchangeable adapters. On top sits the **AI Core** (Stage 2): policy,
context and orchestration ports an app composes with `Composer`.

- **Architecture:** [`ARCHITECTURE.md`](ARCHITECTURE.md) — the port/adapter
  design, the two stages, the `Workflow` lifecycle, the five extension points.
- **Status & where to resume:** [`CHANGELOG.md`](CHANGELOG.md) — what's
  implemented (Stage 1 + Stage 2 AI Core), the public API, what's left.
- Runnable examples: [`examples/`](examples/) — start at
  `01_basic_generate.py`; no paid API keys, just a local Ollama.
- Stage-1 lab (separate `uv` project): [`llm_lab/`](llm_lab/) — raw provider
  behaviour, `experiments/FINDINGS.md`, filled test matrix (§9.3).

Detailed design documentation and task plans are kept in private local
notes, not part of this repository.

Package manager: `uv`. Python: 3.13+. Tests: `uv run pytest` (offline,
replays `tests/cassettes/`); `--live` hits real providers.

Contributor setup: run `uv run pre-commit install` once after cloning so
the formatting hooks run on every commit.
