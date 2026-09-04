# odwi-llm — status and resume point

Summary for anyone who did **not** follow the implementation phases.
Detailed design documentation and task plans are kept in private local
notes, not part of this repository. Lab evidence:
[`llm_lab/experiments/FINDINGS.md`](llm_lab/experiments/FINDINGS.md).

---

## 0.1.0 — Stage 1: contract + adapters + suite (2026-09)

Executed plans: the Stage-1 lab plan and the core plan (both kept as
private local notes).

### What got implemented

- **`odwi_llm.core`** — the §4 contract, with no provider library:
  - `types.py` — `Role`, `Message`, `Intent`, `LLMRequest`, `Usage`,
    `FinishReason`, `LLMResponse`, `StructuredResponse[T]`, `ToolSpec`,
    `ToolCall`, `ToolResult`, `ChatTurnResponse`, `StreamChunk`.
  - `requirements.py` — `LLMRequirements`, `LLMCapabilities`,
    `CapabilityError` (fail-fast when the adapter is constructed, §4.4).
  - `errors.py` — the `LLMError` hierarchy (§4.5). The domain never sees a
    raw `litellm`/`any_llm` error.
  - `port.py` — `LLMPort` (ABC): `generate` · `structured` · `stream` ·
    `chat_with_tools` + `capabilities`. Async core.
  - `sync.py` — `SyncLLM` + `_LoopRunner`: a blocking facade over **any**
    `LLMPort`, works even when an event loop is already running
    (Streamlit/Jupyter). Does not use `asyncio.run()`.
  - `composition.py` — `FallbackLLM(primary, fallback=None)` (§5.4).
- **`odwi_llm.adapters`** — the only layer that imports a provider library:
  - `litellm_adapter.LiteLLMAdapter` — **primary** (decision #4).
  - `anyllm_adapter.AnyLLMAdapter` — **fallback**. Cannot tool-call
    against LM Studio → `CapabilityError` at construction.
  - `config.ProviderConfig` — API key via env var, `base_url`, timeout.
  - `pricing.py` — static cost table (decision #3). No library is a
    reliable cost source; the package maintains it.
  - `_shared.py` — helpers + `call_with_retry` (§5.3: 429/5xx/timeout,
    bounded to 3, does not retry auth/context/filter/schema).
- **Tests** — `uv run pytest`: 109 passed, 4-6 skipped, **no network, no quota**.
  - `tests/contract/` — exact suite against `FakeAdapter`, always offline.
  - `tests/contract_shape/` — structural suite; by default it replays
    `tests/cassettes/` (31 real recorded JSON files), `--live` goes
    against providers, `--live --record` re-records.
  - `tests/architecture/test_import_boundaries.py` — checks §3.2.
  - `mypy --strict` clean.

### API the package exposes to a consuming app

```python
from odwi_llm.adapters.litellm_adapter import LiteLLMAdapter
from odwi_llm.adapters.anyllm_adapter import AnyLLMAdapter
from odwi_llm.core.composition import FallbackLLM
from odwi_llm.core.requirements import LLMRequirements
from odwi_llm.core.sync import SyncLLM
from odwi_llm.core.types import LLMRequest, Message, Role, Intent

adapter = LiteLLMAdapter(
    "groq", "openai/gpt-oss-120b",
    LLMRequirements(structured_output=True),   # fail-fast if unmet
)
llm = FallbackLLM(primary=adapter)             # fallback=None in Stage 1

# async app   -> use the LLMPort directly:     await llm.generate(req)
# sync app    -> wrap once per process:
sync = SyncLLM(llm)
resp = sync.generate(LLMRequest(messages=[Message(role=Role.USER, content="…")]))
```

The app depends **only** on `LLMPort` / `SyncLLM` / `FallbackLLM` and the
types. Swapping `LiteLLMAdapter` for `AnyLLMAdapter` touches no app.
Credentials via environment variable (`GEMINI_API_KEY`, `GROQ_API_KEY`);
Ollama/LM Studio via `ProviderConfig(base_url=…)`.

### What is explicitly left for Stage 2

- **`odwi_llm.guardrails`** — input policy (minimization, allowlists),
  tool policy (arg validation, read-only, row limits), output policy
  (schema, citation). Optionally `any-guardrail` for toxicity/jailbreak
  (decision #5). Folder not created yet.
- **`odwi_llm.orchestration`** — multi-turn loop, tool execution,
  retrieval (`search_docs`), history. Folder not created yet.
- **Enabling the fallback:** today every app builds
  `FallbackLLM(primary=…)` with no `fallback`. Stage 2 passes
  `FallbackLLM(primary=LiteLLMAdapter(...), fallback=AnyLLMAdapter(...))`
  when availability justifies it (verified in
  `tests/adapters/test_fallback_real.py`).
- **AI Insights app** — first consuming app (roadmap §10, milestone `l7`).
- **`NativeAnthropicAdapter`** — no API key; added when one exists,
  without touching the rest (criterion in §5.2.1).
- **Stage 2 proper** — `stream_chat_with_tools` (if evaluation justifies
  it), RAG, a table catalog over DuckDB. Each one starts with its own
  short `Specify` before design.

### How to re-record cassettes (§9.5)

When the pinned `litellm` / `any-llm-sdk` version changes, or observed
behaviour drifts:

```bash
uv run pytest tests/contract_shape/ --live --record --adapter litellm
uv run pytest tests/contract_shape/ --live --record --adapter anyllm
```

Needs `GEMINI_API_KEY` / `GROQ_API_KEY` in the environment and Ollama +
LM Studio running locally.
