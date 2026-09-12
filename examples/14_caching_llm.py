"""14 - CachingLLM: an LLMPort decorator that caches generate().

No `ai_core` prefix - Stage 1 surface, same level as `01`/`04`/`13`. Wraps
`LiteLLMAdapter` (not `13`'s `OllamaNativeAdapter` - the focus here is the
decorator pattern itself, not writing another adapter).

`CachingLLM` wraps any other `LLMPort` and only caches `generate()`.
`structured()` / `chat_with_tools()` / `stream()` delegate straight
through, uncached - not because caching them is impossible, but because
each needs different invalidation semantics that are out of scope here:
a cached `structured()` result would need to key on the schema too, a
cached tool-calling turn would need to account for which tools were
registered, and a cached stream would need to replay chunks rather than
return one shot. `generate()` is the one operation simple enough to cache
correctly in a small example.

The cache key deliberately excludes `LLMRequest.metadata`. This is not a
style preference - it follows directly from what `core/types.py` already
declares about that field: "Opaque metadata for logging/tracing only.
Never sent to the provider." If it never reaches the provider, it cannot
affect the response, so keying on it would only cause spurious misses for
two requests that are identical from the provider's point of view. The
key is built from `(messages, intent, max_output_tokens)` instead - the
three fields that actually reach the wire.

The cache itself is a plain in-memory dict, no expiration, no size limit -
a demo simplification, the same kind `08`/`10` used for their thresholds.
A production cache would need at least a TTL and a bound on size.

Verification below covers three cases, not just the happy path:

  1. The same `LLMRequest` twice -> the second call is a cache hit,
     confirmed both by the hit counter and by wall-clock time (a real
     generation against Ollama takes a perceptible fraction of a second;
     a cache hit should be close to instant).
  2. A different `LLMRequest` (different message) -> a miss.
  3. The SAME request as case 1, but with different `metadata` -> still a
     hit. This is the case that actually proves excluding `metadata` from
     the key was correct, not just asserted - it is checked here, not
     assumed.

No API keys. Only needs a running Ollama daemon with the model pulled:

    ollama pull qwen3:0.6b

Run:

    uv run python examples/14_caching_llm.py
"""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from odwi_llm.adapters.litellm_adapter import LiteLLMAdapter
from odwi_llm.core.port import LLMPort
from odwi_llm.core.requirements import LLMCapabilities, LLMRequirements
from odwi_llm.core.types import (
    ChatTurnResponse,
    LLMRequest,
    LLMResponse,
    Message,
    Role,
    StreamChunk,
    StructuredResponse,
    T,
    ToolSpec,
)

# The wire-relevant slice of a Message: content that actually reaches the
# provider. tool_call_id / name are included since a TOOL-role message's
# meaning depends on them too.
_MessageKey = tuple[str, str, str | None, str | None]
_CacheKey = tuple[tuple[_MessageKey, ...], str, int | None]


def _message_key(message: Message) -> _MessageKey:
    return (
        message.role.value,
        message.content,
        message.tool_call_id,
        message.name,
    )


def _cache_key(request: LLMRequest) -> _CacheKey:
    # metadata is excluded on purpose - see the module docstring. Only
    # messages, intent and max_output_tokens ever reach the provider.
    return (
        tuple(_message_key(m) for m in request.messages),
        request.intent.value,
        request.max_output_tokens,
    )


class CachingLLM(LLMPort):
    """Wraps any `LLMPort` and caches `generate()` only."""

    def __init__(self, wrapped: LLMPort) -> None:
        self._wrapped = wrapped
        self._cache: dict[_CacheKey, LLMResponse] = {}
        self.hits = 0
        self.misses = 0

    @property
    def capabilities(self) -> LLMCapabilities:
        # Pass-through - caching does not change what the wrapped port
        # can do.
        return self._wrapped.capabilities

    async def generate(self, request: LLMRequest) -> LLMResponse:
        key = _cache_key(request)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            print(f"  [cache] hit  ({self.hits} hits, {self.misses} misses)")
            return cached

        self.misses += 1
        print(f"  [cache] miss ({self.hits} hits, {self.misses} misses)")
        response = await self._wrapped.generate(request)
        self._cache[key] = response
        return response

    async def structured(
        self, request: LLMRequest, schema: type[T]
    ) -> StructuredResponse[T]:
        # Not cached - see the module docstring on why this is a scope
        # cut, not a limitation of the pattern.
        return await self._wrapped.structured(request, schema)

    async def chat_with_tools(
        self, request: LLMRequest, tools: list[ToolSpec]
    ) -> ChatTurnResponse:
        return await self._wrapped.chat_with_tools(request, tools)

    def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        return self._wrapped.stream(request)


async def main() -> None:
    inner = LiteLLMAdapter("ollama", "qwen3:0.6b", LLMRequirements())
    llm = CachingLLM(inner)

    request_a = LLMRequest(
        messages=[Message(role=Role.USER, content="Name three primary colors.")],
        metadata={"trace_id": "run-1"},
    )

    print("case 1: same request twice -> second call should be a hit")
    t0 = time.perf_counter()
    r1 = await llm.generate(request_a)
    t1 = time.perf_counter()
    r2 = await llm.generate(request_a)
    t2 = time.perf_counter()
    print(f"  first call:  {t1 - t0:.3f}s -> {r1.text!r}")
    print(f"  second call: {t2 - t1:.3f}s -> {r2.text!r}")
    assert r1.text == r2.text, "cached response must match the original"

    print("\ncase 2: a different request -> miss")
    request_b = LLMRequest(
        messages=[Message(role=Role.USER, content="Name two secondary colors.")]
    )
    r3 = await llm.generate(request_b)
    print(f"  -> {r3.text!r}")

    print(
        "\ncase 3: same request as case 1, different metadata -> still a "
        "hit (the actual proof metadata belongs out of the key)"
    )
    request_a_different_metadata = request_a.model_copy(
        update={"metadata": {"trace_id": "totally-different-run"}}
    )
    t3 = time.perf_counter()
    r4 = await llm.generate(request_a_different_metadata)
    t4 = time.perf_counter()
    print(f"  call: {t4 - t3:.3f}s -> {r4.text!r}")
    assert r4.text == r1.text, "differing metadata must not change the cache key"

    print(f"\nfinal count: hits={llm.hits} misses={llm.misses}")
    assert llm.hits == 2 and llm.misses == 2, (
        f"expected 2 hits / 2 misses, got hits={llm.hits} misses={llm.misses}"
    )


if __name__ == "__main__":
    asyncio.run(main())
