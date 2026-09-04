# odwi-llm

[![CI](https://github.com/acdeveloper-sci/odwi-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/acdeveloper-sci/odwi-llm/actions/workflows/ci.yml)

Provider-agnostic LLM access layer. Applications depend only on `LLMPort`
and its types; LiteLLM, Any-LLM or a native SDK sit behind the port as
interchangeable adapters.

- **Status & where to resume:** [`CHANGELOG.md`](CHANGELOG.md) — what's
  implemented (Stage 1), the public API, what's left for Stage 2.
- Stage-1 lab (separate `uv` project): [`llm_lab/`](llm_lab/) — raw provider
  behaviour, `experiments/FINDINGS.md`, filled test matrix (§9.3).

Detailed design documentation and task plans are kept in private local
notes, not part of this repository.

Package manager: `uv`. Python: 3.13+. Tests: `uv run pytest` (offline,
replays `tests/cassettes/`); `--live` hits real providers.

Contributor setup: run `uv run pre-commit install` once after cloning so
the formatting hooks run on every commit.
