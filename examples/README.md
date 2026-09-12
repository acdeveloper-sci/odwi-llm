# Examples

Runnable, in reading order. Each is self-contained and needs no paid API
keys — only a local [Ollama](https://ollama.com) daemon with the model
pulled:

```bash
ollama pull qwen3:0.6b
```

Run any of them with `uv run python examples/<file>`.

This series teaches three things, not just one:

1. **Direct use of the API** — `Composer`, guardrails, tools, context.
   Most examples below are this.
2. **The line between what `odwi-llm` resolves and what stays on the
   application side.** [`08_ai_core_context_patterns.py`](08_ai_core_context_patterns.py)
   is a real case of that line getting tested, not just stated — see the
   note near the end of its docstring.
3. **When to recognize you need to build something deliberately left out
   today** (an agentic orchestrator adapter, for instance). This is
   conceptual guidance, not something a runnable example can demonstrate
   on its own: the working line is whether the model picks one tool from
   a fixed menu once per turn (still a `Workflow`) or chains multiple
   tool decisions with no sequence anticipated by whoever built the
   workflow (that crosses into `Agent`).

| # | File | Shows |
|---|---|---|
| 01 | [`01_basic_generate.py`](01_basic_generate.py) | The smallest use: one `LiteLLMAdapter`, one async `generate()` call. |
| 02 | [`02_sync_script.py`](02_sync_script.py) | The same call without `async`/`await`, via `SyncLLM`. |
| 03 | [`03_structured_output.py`](03_structured_output.py) | `structured()` with a Pydantic model; `response.data` is a validated instance, not text. |
| 04 | [`04_fallback.py`](04_fallback.py) | `FallbackLLM` — a primary port with a fallback port for transient provider errors. |
| 05 | [`05_ai_core_minimal.py`](05_ai_core_minimal.py) | The smallest `Composer` path: permissive guardrails, a fixed `ContextPort`, an `ObservabilityPort` that prints the lifecycle. No tools. |
| 06 | [`06_ai_core_guardrails.py`](06_ai_core_guardrails.py) | Guardrails that decide: an input `Deny` (→ `CONTENT_FILTER`) and an output `Redact` (→ text rewritten), across three example messages. |
| 07 | [`07_ai_core_tools.py`](07_ai_core_tools.py) | The tool-calling loop: a `ToolGuardrail` with `covers`, a `tool_executor`, and `tool_call` / `tool_before` / `tool_after` events. |
| 08 | [`08_ai_core_context_patterns.py`](08_ai_core_context_patterns.py) | A follow-up chat over a prior report: `ContextBundle.prior_output` built once, manual prompt assembly, `Allow(grounded=True/False)` plus chained output guardrails. |
| 09 | [`09_ai_core_tool_coverage.py`](09_ai_core_tool_coverage.py) | `ToolGuardrail.covers`: a catch-all and a tool-specific guardrail both running on the same tool, the specific one denying based on a structured tool-call argument. |
| 10 | [`10_ai_core_tool_redact.py`](10_ai_core_tool_redact.py) | One `ToolGuardrail` implementing both `before` (clamps args) and `after` (redacts a leaked contact pattern in the tool's own returned content — the most reliable guardrail input of the series). |
| 11 | [`11_ai_core_max_iterations.py`](11_ai_core_max_iterations.py) | `Workflow`'s `max_tool_iterations` worst-case bound, with a scripted `LLMPort` that always requests a tool — deterministic by construction, needs no model at all. |
| 12 | [`12_ai_core_composition_seams.py`](12_ai_core_composition_seams.py) | `Composer`'s own seam, in two parts: a custom `Orchestrator` that skips every policy phase (a narrow, honestly-scoped case, not a recommendation), and the default `Workflow` running over a Stage 1 `FallbackLLM` with no friction. |
| 13 | [`13_custom_llm_adapter.py`](13_custom_llm_adapter.py) | A real, minimal `LLMPort` implemented outside `odwi_llm.adapters` — plain HTTP against Ollama's native API, no `litellm` — including its own fail-fast against `LLMRequirements`. Stage 1 only, no `Composer`. |

`04_fallback.py` names a local LM Studio model as the fallback, but in the
happy path the fallback is never contacted, so it still runs with only
Ollama up. `07_ai_core_tools.py` relies on a 0.6b model choosing to call
the tool (`tool_choice="auto"`); if a run shows `tool_calls=0`, run it
again. `11_ai_core_max_iterations.py` is the only example in this series
that does not need Ollama (or any model) running at all — its `LLMPort`
is hand-scripted in the script itself. `12_ai_core_composition_seams.py`
reuses `04`'s fallback pair and is happy-path only in the same way —
LM Studio does not need to be running.
