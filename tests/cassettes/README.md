# tests/cassettes/

Recorded real provider outputs for `tests/contract_shape/`. `uv run pytest`
(no flag) replays these — offline, no quota. `--live` hits real providers.

## Recording at the `LLMPort` boundary is intentional

A cassette is **the adapter's final output object** (`LLMResponse`,
`StructuredResponse`, `ChatTurnResponse`, or a list of `StreamChunk`) —
captured *above* `_acompletion`, so after `call_with_retry` has already
resolved any transient 503/429 and after reasoning has been stripped.
Recording is **not** done at the HTTP layer (no VCR / respx).

Two reasons:

1. **Simpler and stable.** No brittle request-matching; a cassette can't
   carry "noise" from retries that won't reproduce identically. Retry
   behaviour is verified separately, with mocks, in
   `tests/adapters/test_retry.py` — never against a cassette.
2. **A cassette can never contain an API key.** The raw HTTP request
   (which carries `Authorization` / `x-goog-api-key` headers) is never
   captured — only our own response types are serialized. The `raw`
   field of the response types is also explicitly excluded
   (`model_dump(exclude={"raw"})`): adapters don't fill it today, but if
   one ever does it could hold provider account data, and it must not be
   committed.

## File format

One file per `(adapter, provider, scenario)`:
`<adapter>__<provider>__<scenario>.json`

```json
{
  "meta": {
    "recorded_utc": "2026-09-03T14:00:00+00:00",
    "adapter": "litellm",
    "adapter_lib": "litellm",
    "adapter_lib_version": "1.99.0",
    "provider": "gemini",
    "model": "gemini-3.5-flash-lite",
    "scenario": "generate"
  },
  "kind": "LLMResponse",
  "payload": { ... }
}
```

`scenario` ∈ `generate | structured | tools | stream`.

## (Re)recording

Needs the real credentials / local servers. Per adapter:

```bash
uv run pytest tests/contract_shape/ --live --record --adapter litellm
uv run pytest tests/contract_shape/ --live --record --adapter anyllm
```

Re-record when the pinned `litellm` / `any-llm-sdk` version changes
(§9.5) — the version in `meta.adapter_lib_version` tells you which
snapshot a cassette is from.
