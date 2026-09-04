# FINDINGS — Stage 1 lab (Tasks 6–11)

An **operational** summary, not the design. Detail and evidence:
`experiments/0*.py` (`FIELD MAP` / `OUTCOMES` / `COMPARISON` blocks),
`experiments/10_lib_matrix.py`, `experiments/dumps/`, and the §9.3 matrix
of the design document.

Lab providers: **Gemini** `gemini-3.5-flash-lite`, **Groq**
`openai/gpt-oss-120b`, **Ollama** `qwen3:0.6b`, **LM Studio**
`llama-3.2-3b-instruct`. Anthropic is out of scope for this phase.

---

## 1. What each adapter must absorb — §5.1 steps 4 and 5

### 1.1 Discard reasoning (step 4)

Reasoning **never** goes away on its own; structured mode does not silence
it either (Task 7). It shows up in different places depending on the path:

| Path | Where the reasoning arrives |
|---|---|
| Native SDK | `message.reasoning` (Groq, Ollama) · `message.reasoning_content` (LM Studio) · `part.thought_signature` bytes (Gemini) |
| LiteLLM | `message.reasoning` (`groq/` route) · `message.reasoning_content` (`ollama_chat/` route) · `message.provider_specific_fields.thought_signatures` (`gemini/` route) |
| Any-LLM | `message.reasoning` as a `Reasoning(content=…)` **object**, across all providers; in streaming, `Reasoning(...)` deltas |

The adapter strips all of these before building `LLMResponse`. In
streaming it must also filter `delta.reasoning` / `delta.reasoning_content`
(Groq 30 chunks, Ollama 160 chunks in Task 9) or they leak through as text.

### 1.2 Parse tool args (step 5)

`ToolCall.arguments` in the contract (§4.3) is an already-parsed dict.
Reality:

- **Native SDK**: Gemini delivers `args` as a **dict** (`{'city': 'Paris'}`); Groq / Ollama / LM Studio as a **JSON string** (`'{"city":"Paris"}'`).
- **LiteLLM and Any-LLM**: **always a JSON string**, Gemini included (litellm does `json.dumps` on the dict).
- ⇒ the adapter does `json.loads` except when it is already a dict.
- **LiteLLM + Gemini + tools**: puts the `thought_signature` **inside the `id`** of the tool call: `id = "call_1984494__thought__El4K…"`. The adapter must cut the `__thought__…` suffix to recover the real `id`.

### 1.3 Normalize `finish_reason`

| Situation | Native Gemini | Native OpenAI-shaped | LiteLLM | Any-LLM |
|---|---|---|---|---|
| normal end | `STOP` | `"stop"` | `"stop"` | `"stop"` |
| truncated output | `MAX_TOKENS` | `"length"` | `"length"` | `"length"` (Any-LLM also **raises** `LengthFinishReasonError` if `response_format` was requested) |
| turn with a tool call | **`STOP`** (no value of its own!) | `"tool_calls"` | `"tool_calls"` (litellm already infers it) | `"tool_calls"` |

⇒ The adapter maps `STOP`/`MAX_TOKENS` → `FinishReason.STOP`/`LENGTH`, and
to detect a tool turn with the native Gemini SDK it **cannot** look at
`finish_reason`: it checks whether there is a `function_call` in the
`parts`. Empty text + `LENGTH` is a valid result (reasoning models spend
the budget thinking), not an error.

### 1.4 Map errors → the §4.5 hierarchy

| Observed case | Origin | Map to |
|---|---|---|
| `exceed_context_size_error` HTTP 400 | LM Studio (Task 9b) | `LLMContextLengthError` |
| HTTP 400 `INVALID_ARGUMENT` "input token count exceeds…" | Gemini (docs) | `LLMContextLengthError` |
| HTTP 400 `context_length_exceeded` | Groq (docs) | `LLMContextLengthError` |
| **prompt silently truncated, HTTP 200** | Ollama (Task 9b) | *no signal* — the Ollama adapter **cannot** detect input overflow; document the gap |
| HTTP 400 `peg-native format` (malformed tool call) | LM Studio (Task 8) | `LLMProviderError` (server-side generation failure, **not** `LLMSchemaError`) |
| HTTP 503 `UNAVAILABLE` "high demand" | Gemini free tier (recurring) | `LLMProviderError` + bounded technical retry (§5.3) — implemented in `_dump_utils.call_with_retry` |
| `NotImplementedError` LM Studio tools | Any-LLM (Task 10) | `CapabilityError` at adapter construction (fail-fast, §4.4) |

Non-obvious detail: LM Studio's error body is **double-wrapped and
non-standard** — `exc.code` ends up `None`; you have to parse `exc.body` /
the message text. Any-LLM already ships its own typed hierarchy
(`ContextLengthExceededError`, `RateLimitError`, `AuthenticationError`,
`LengthFinishReasonError`, …) that maps almost 1:1 to the §4.5 one.

### 1.5 Streaming → `StreamChunk` (§4.3)

- `text_delta`: non-empty `parts[].text` (Gemini) / `delta.content` (the rest). Discard `""`/`None` deltas; the final empty delta is the `done` marker, not text.
- `done`: on `finish_reason`; normalize `STOP` → `stop`.
- `usage`: **native SDK** → Groq/Ollama/LM Studio send it in a final `choices:[]` chunk (Groq also on the `stop` chunk); Gemini puts it, cumulative, in **every** chunk (there is no separate usage chunk). **LiteLLM** drops the `choices:[]` chunk → **no usage in streaming** unless you pass `stream_options={"include_usage": True}` explicitly. **Any-LLM** does preserve usage.
- The usage chunk carries `choices:[]` → accessing `chunk.choices[0]` blows up; always guard.

---

## 2. Adjustments to the `Intent` map — §5.2

Probe `experiments/06_intent_probe.py` (`dumps/intent_probe.txt`): the same
request twice at `temperature=0` and once at `temperature=1`, via native
SDK.

| Provider | `temperature=0` accepted | 2× temp=0 identical | Note |
|---|---|---|---|
| Gemini `gemini-3.5-flash-lite` | yes | **no** | flash-lite is not deterministic even at temp=0; the free tier exposes no `seed` |
| Groq `openai/gpt-oss-120b` | yes | **yes** | it is a reasoning model and still honors `temperature=0` and reproduces |
| Ollama `qwen3:0.6b` | yes | **no** | varies; would need a fixed `options.seed` (and no thinking) to reproduce |
| LM Studio `llama-3.2-3b-instruct` | yes | **yes** | |

**Conclusion:**

- The §5.2 draft stands: `DETERMINISTIC → temperature=0` for all four.
  **None reject or ignore `temperature=0`**, and no model with "adaptive
  thinking" that requires `temperature=1` showed up in this set (gpt-oss
  accepted 0) — the §5.2 note about Fable stays hypothetical here.
- Important nuance: **`temperature=0` ≠ reproducible**. Gemini and Ollama
  still vary output to output. If Stage 1 needs real reproducibility
  (recording cassettes, §9.5), the adapter must also pin `seed` where it
  exists (`seed` on Groq, `options.seed` on Ollama) and assume the Gemini
  free tier does not allow it.
- `BALANCED` = omit the parameter; `CREATIVE` = `temperature=1`. No
  changes: in the probe `temperature=1` always differed from `0`, i.e. the
  parameter arrives and takes effect on all four.

---

## 3. Primary adapter for Stage 1: **LiteLLM**

| Criterion | LiteLLM | Any-LLM |
|---|---|---|
| Matrix cells that pass | **16 / 16** | 15 / 16 |
| Hard blockers | — | **LM Studio + tools = `NotImplementedError`** (wraps the native `lmstudio-python` SDK; tools only via its agentic `.act()` API) |
| Per-provider extras | no | yes (`any-llm-sdk[ollama,lmstudio]`) |
| `structured` | JSON string in `content` only (the adapter parses it) | **Pydantic instance** in `message.parsed` |
| `usage` in streaming | no, unless `stream_options` is explicit | yes |
| Normalization | aggressive and **inconsistent** (reasoning in 3 fields depending on route; `thought_signature` inside the `id`) | cleaner; its own error hierarchy |
| Dependency weight | high (boto3, tiktoken, tokenizers…) | lower |

**Recommendation:** primary **LiteLLM**. Decisive reason: Stage 2 needs
tool calling against the local models, and Any-LLM cannot do it with
LM Studio. LiteLLM's normalization warts (§1.1–§1.5) are all absorbable in
the adapter; a capability blocker is not.

Any-LLM stays as a **`fallback` candidate** (§5.4, decision #4) for
Gemini/Groq, where it does cover and adds `message.parsed` and typed
errors. Both adapters are built on the same §4 contract; switching from one
to the other touches no app.

---

## 4. Input for §11 — open decisions

### Decision #2 — does `stream` + tool calling enter the Stage 2 contract?

**Recommendation: NO in the base contract.** `stream()` stays text-only as
in §4.6; `chat_with_tools()` returns the full turn (does not stream).

- stream+tools was not measured in the lab (Tasks 9 and 10 are text-only
  on purpose — §9.4).
- What was seen: the **text-only** stream already diverges hard — Gemini
  sends coarse chunks with cumulative `usage` per chunk; Groq sends the
  `usage` in two chunks; LiteLLM eats the `usage` chunk; reasoning arrives
  as its own stream of deltas. Adding tool-call deltas on top multiplies
  that divergence.
- Besides, `gpt-oss-120b` does not emit parallel tool calls even without
  streaming, and LM Studio blows up with two tool calls (Task 8). The
  ground is not stable even for the non-stream case.
- Concrete proposal for §11 #2: keep `stream()` text-only in the Stage 2
  contract. If an app needs "fast first token" in tool-augmented chat, a
  separate operation (`stream_chat_with_tools`) is evaluated in a later
  lab; it is not committed to the contract now.

### Decision #3 — who computes `estimated_cost_usd`?

**Recommendation: a price table owned by the package.** The adapter fills
`Usage.estimated_cost_usd` only if the model is in that table; otherwise
`None` (the type already allows it, §4.1).

- **Any-LLM**: reports no cost at all (no field, `model_extra` empty).
- **LiteLLM**: `response._hidden_params["response_cost"]` gives a float
  (`0.0` for local, correct), but (a) it is **outside** `model_dump()` and
  `repr()`, (b) `litellm.completion_cost()` **raises** "model isn't mapped
  yet" for `openai/gpt-oss-120b` — the price table LiteLLM ships lags
  behind on new models.
- No native SDK returns cost (Groq gives *timings*, not price).
- ⇒ Do not delegate cost to the library. The package maintains its own
  `{provider, model, input_usd_per_1M, output_usd_per_1M}` table and the
  adapter computes `input_tokens·pᵢ + output_tokens·pₒ`. Models with no
  entry → `estimated_cost_usd = None`.
