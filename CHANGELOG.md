# odwi-llm — status and resume point

Summary for anyone who did **not** follow the implementation phases.
Detailed design documentation and task plans are kept in private local
notes, not part of this repository. Lab evidence:
[`llm_lab/experiments/FINDINGS.md`](llm_lab/experiments/FINDINGS.md).

---

## 0.2.0 — Stage 2: AI Core (2026-09)

Executed plan: the AI Core task plan (private local notes). Four new
modules, **ports and types only — no real policy rules**; those are each
consuming app's job, never this library's.

### What got implemented

- **`odwi_llm.guardrails`** — the policy contract:
  - `types.py` — `PolicyContext` (frozen at composition, never re-read
    silently mid-conversation) and `Decision[T] = Allow | Deny | Redact[T]`,
    a discriminated union. `Allow.grounded` distinguishes "allowed,
    grounded in data" from "allowed, general knowledge". `Redact[T]` is
    generic — `Decision[str]` for input/output text,
    `Decision[dict[str, Any]]` for sanitized tool arguments.
  - `errors.py` — `PolicyError` → `PolicyConfigError` (composition built
    wrong, fails at assembly), `PolicyDeniedError`. Separate hierarchy
    from `LLMError`.
  - `port.py` — `InputGuardrail` / `ToolGuardrail` / `OutputGuardrail`
    protocols (one per phase; I/O methods are `async`; `ToolGuardrail.covers`
    is a sync property, `None` = catch-all) and `GuardrailSet`
    (`input_guardrails` / `tool_guardrails` / `output_guardrails`).
- **`odwi_llm.observability`** — `ObservabilityPort` (`emit(event, **fields)`,
  sync) in `port.py`, `NullObservability` (no-op default) in `null.py`.
  No backend.
- **`odwi_llm.context`** — `ContextBundle` (`knowledge` near-static /
  `data` per-run / optional `prior_output` snapshot) and `ContextPort`
  (`async select(*, message=None, ctx)` — single-shot or per-turn).
- **`odwi_llm.orchestration`** — the execution mechanism:
  - `types.py` — `WorkflowTask`, `WorkflowResult` (carries `grounded`,
    `tool_calls_made`).
  - `port.py` — `Orchestrator` (`async run(task, ctx) -> WorkflowResult`)
    and `ToolExecutor` (`async execute(call, ctx) -> ToolResult` — the app
    runs tools; `Workflow` only invokes).
  - `workflow.py` — `Workflow`, the thin reference `Orchestrator`: input
    policy → context → LLM → output policy, emitting on each point and
    propagating `grounded`. With `tools`, a bounded tool-calling loop:
    covering `ToolGuardrail.before` (may `Deny` / `Redact` args) →
    `tool_executor.execute` → covering `.after` (may `Deny` / `Redact` the
    result) → feed back, repeat until no more tool calls or
    `max_tool_iterations` (then `WorkflowResult(finish_reason=OTHER,
    "stopped: …")`, never an exception). Fail-fast at construction: a tool
    with no covering guardrail, or tools with no executor → `PolicyConfigError`.
  - `composition.py` — `Composer`, the composition root. Without
    `orchestrator=` it builds the default `Workflow`, forwarding
    everything (a custom orchestrator is used as-is).
- **Tests** — `tests/ai_core/` (fakes + suites). The import-boundary check
  now covers the four new modules and a third boundary: agentic engine
  libraries (`langgraph`, `langchain`, `crewai`, …) may not appear in the
  Core. `uv run pytest`: 150 passed, 6 skipped, offline. `mypy --strict`
  clean.

### API a consuming app uses

```python
from odwi_llm.guardrails.port import GuardrailSet
from odwi_llm.guardrails.types import PolicyContext
from odwi_llm.orchestration.composition import Composer
from odwi_llm.orchestration.types import WorkflowTask

comp = Composer(
    llm=llm,                       # any LLMPort from Stage 1
    context=my_context,            # a ContextPort
    guardrails=GuardrailSet(       # the app's real guardrails go here
        input_guardrails=[...],
        tool_guardrails=[...],
        output_guardrails=[...],
    ),
    tools=[...], tool_executor=my_executor,   # optional
    observability=my_obs,                     # optional, default no-op
)
ctx = PolicyContext(
    session_id="s1", provider="groq", model="…", resolved_at=now
)
result = await comp.orchestrator.run(
    WorkflowTask(session_id="s1", input={"message": "…"}), ctx
)
```

The Core provides the *shape* — phases, the `Decision` type, the loop, the
fail-fasts. The app provides the *content* — the actual guardrails, the
context selection, the tool executor, the prompt assembly.

### What is explicitly left

- **Real guardrails / context / tool executors** — domain-specific, each
  consuming app's job, never this library's.
- **A real agentic `Orchestrator` adapter** (LangGraph / Agents SDK / …) —
  built only when a real case needs multi-step tool chaining the reference
  `Workflow` cannot express. It is another adapter of the same
  `Orchestrator` port; the import-boundary test already forbids those
  libraries in the Core.
- **AI Insights app** — still the first consuming app / acceptance
  criterion (its own short `Specify` when picked up).

### Where the code diverged from the plan

- **`tests/ai_core/`**, not `tests/contract/` (which the design doc named)
  — a dedicated package with its own `__init__.py`, to avoid pytest
  basename collisions with the Stage 1 contract suite and keep concerns
  separate.
- **The design doc** (`..._etapa2_design_v0.4.md`, filename unchanged)
  moved internally to **v0.6**: v0.5 made the guardrail and `ContextPort`
  protocols `async` (a sync one reaching an LLM via `SyncLLM` would block
  the `Workflow` event loop); v0.6 added the `ToolExecutor` protocol and
  the `tool_executor` parameter on `Workflow` / `Composer`.
- **Reference `Workflow` loop choices** (the design left the loop body to
  this stage): a `before` `Deny` does not abort the run — it feeds a
  synthetic error `ToolResult` back and the loop continues (deliberately
  asymmetric with input/output `Deny`, which does abort: the orchestrator
  is mechanism, not policy — Specify §4 — so a single failed tool call
  among possibly several in one turn is left for the model to react to,
  not treated as grounds to abort the whole run); between turns the
  reference appends one `Role.ASSISTANT` message plus one `Role.TOOL`
  message per result; observability events `policy_decision`
  (`phase="tool_before"` / `"tool_after"`), `tool_call`, and a per-turn
  `llm_call` were added.

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

- **`odwi_llm.guardrails` / `observability` / `context` / `orchestration`**
  — the AI Core. Done in 0.2.0 above (ports and types; real rules stay in
  each app).
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
