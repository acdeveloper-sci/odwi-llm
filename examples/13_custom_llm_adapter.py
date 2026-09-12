"""13 - a custom LLMPort adapter, written outside odwi-llm's own adapters/.

No `ai_core` prefix, unlike `05`-`12`: this is Stage 1 surface (`LLMPort`),
not the AI Core. No `Composer`, no guardrails, no context - same level as
`01`/`04`, consumed directly.

`OllamaNativeAdapter` talks to Ollama's own `/api/chat` endpoint over
plain HTTP (`httpx` - already a transitive dependency here via `litellm`/
`ollama`/`openai`, so nothing new was added), with no `litellm` in
between. It shows the actual work a third-party adapter author has to do:
subclass `LLMPort`, translate the wire format on the way in and out, map
the provider's own vocabulary to the shared error hierarchy, and fail
fast at construction against declared `LLMRequirements` - all without any
access to this package's own internal helpers.

Scope of what is real here:

  * `generate()` - complete. Builds the `/api/chat` request with
    `"stream": false`, reads `message.content` as the response text, maps
    `prompt_eval_count` / `eval_count` to `Usage`, and `done_reason` to
    `FinishReason` through a small mapping table written in this file
    (not imported from anywhere - `adapters/_shared.py` is a private
    module, not public surface a third-party author could reach for even
    if they wanted to).
  * `chat_with_tools()` - complete, same trivial `get_time` tool used by
    `07`/`09`/`10`, showing `ChatTurnResponse.tool_calls` populated from
    Ollama's own tool-call shape.
  * `structured()` / `stream()` - stubs (`raise NotImplementedError`).
    This is a scope cut of THIS EXAMPLE, not a provider limitation:
    Ollama supports both (`format` for JSON-schema-constrained output,
    NDJSON for streaming) - a real adapter would implement them the same
    way `LiteLLMAdapter` does, just against this wire format instead.
  * `capabilities` - `LLMCapabilities(structured_output=False,
    tool_calling=True, streaming=False, vision=False,
    context_tokens=None)`, honest about what is actually implemented
    above, not what the provider could support in principle.

Fail-fast is re-implemented here, in a few lines, rather than imported
from `adapters/_shared.unmet_requirements` - that function is a private
helper for this package's own adapters, invisible (rightly) to an
external adapter author. The demonstration below builds this adapter
twice: once with `LLMRequirements(tool_calling=True)` (met, so
`generate()` / `chat_with_tools()` proceed), and once with
`LLMRequirements(structured_output=True)` (not met, so `__init__` raises
`CapabilityError` immediately) - proving the Stage 1 fail-fast guarantee
holds for an adapter built entirely outside this package, not just the
two shipped here.

Findings, checked empirically, not assumed:

  * `done_reason` came back `"stop"` every time a tool call was made
    (several runs of `chat_with_tools()`), never the literal
    `"tool_calls"` some OpenAI-shaped APIs use. The `_FINISH` table still
    maps `"tool_calls"` defensively (Ollama's docs do not commit to never
    sending it), but the orchestrator loop
    (`orchestration/workflow.py`) never actually branches on
    `ChatTurnResponse.finish_reason` - it looks at `tool_calls` being
    non-empty instead, so this quirk does not affect correctness here.
  * `tool_calls[].function.arguments` came back as an already-parsed
    dict every time, never a JSON string - checked directly (a temporary
    `print(type(...))` against a real response) before writing
    `chat_with_tools()`, because this project's own lab notes recorded a
    JSON-string form from a different tool-calling path, and assuming
    the same shape here without checking would have been exactly the
    kind of unverified claim this series avoids.
  * `tool_calls[].id` IS present on this Ollama version (e.g.
    `"call_5nrs97iu"`), confirmed across several runs with and without
    arguments - `ToolCall.id` uses it directly, with a positional
    fallback (`f"call-{i}"`) only as a defensive default for an older
    daemon that might omit it.

Error handling here is deliberately collapsed to one generic
`LLMProviderError` for any HTTP/connection failure. This is a scope cut
of this example, not what a production adapter should look like: a real
one would distinguish `LLMAuthError` (401/403), `LLMRateLimitError` (429,
carrying `retry_after_seconds`), and reserve `LLMProviderError` for
5xx/timeouts - see `LiteLLMAdapter`/`AnyLLMAdapter` (`adapters/`) for that
full mapping. Everything collapses to one category here because a local
Ollama daemon has no auth and no billing to distinguish between.

No API keys. Only needs a running Ollama daemon with the model pulled:

    ollama pull qwen3:0.6b

Run:

    uv run python examples/13_custom_llm_adapter.py
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx

from odwi_llm.core.errors import LLMProviderError
from odwi_llm.core.port import LLMPort
from odwi_llm.core.requirements import (
    CapabilityError,
    LLMCapabilities,
    LLMRequirements,
)
from odwi_llm.core.types import (
    ChatTurnResponse,
    FinishReason,
    LLMRequest,
    LLMResponse,
    Message,
    Role,
    StreamChunk,
    StructuredResponse,
    T,
    ToolCall,
    ToolSpec,
    Usage,
)

BASE_URL = "http://localhost:11434"

# Ollama's own done_reason vocabulary -> the shared FinishReason. Small and
# local to this file on purpose - see the module docstring on why this does
# not reuse adapters/_shared.map_finish_reason.
_FINISH = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
}


def _map_finish_reason(raw: str | None) -> FinishReason:
    if raw is None:
        return FinishReason.STOP
    return _FINISH.get(raw.lower(), FinishReason.OTHER)


class OllamaNativeAdapter(LLMPort):
    """A minimal but real `LLMPort` adapter over Ollama's native
    `/api/chat`, written entirely from outside `odwi_llm.adapters`.
    """

    def __init__(
        self, model: str, requirements: LLMRequirements, base_url: str = BASE_URL
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._caps = LLMCapabilities(
            structured_output=False,
            tool_calling=True,
            streaming=False,
            vision=False,
            context_tokens=None,
        )

        # Fail-fast at construction (design §4.4's pattern): only the
        # booleans this adapter actually claims are checked, in a few
        # lines - not a full reimplementation of unmet_requirements.
        missing = []
        if requirements.structured_output and not self._caps.structured_output:
            missing.append("structured_output")
        if requirements.tool_calling and not self._caps.tool_calling:
            missing.append("tool_calling")
        if requirements.streaming and not self._caps.streaming:
            missing.append("streaming")
        if requirements.vision and not self._caps.vision:
            missing.append("vision")
        if missing:
            raise CapabilityError(
                f"ollama/{model} does not meet requirements: {', '.join(missing)}"
            )

    @property
    def capabilities(self) -> LLMCapabilities:
        return self._caps

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"{self._base_url}/api/chat", json=payload)
                resp.raise_for_status()
                result: dict[str, Any] = resp.json()
                return result
        except httpx.HTTPError as exc:
            # Collapsed to one category on purpose - see the module
            # docstring's error-handling note.
            raise LLMProviderError(f"ollama request failed: {exc}") from exc

    def _usage(self, raw: dict[str, Any]) -> Usage:
        return Usage(
            input_tokens=int(raw.get("prompt_eval_count", 0) or 0),
            output_tokens=int(raw.get("eval_count", 0) or 0),
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        payload = {
            "model": self._model,
            "messages": [
                {"role": m.role.value, "content": m.content} for m in request.messages
            ],
            "stream": False,
        }
        raw = await self._post(payload)
        return LLMResponse(
            text=raw.get("message", {}).get("content", ""),
            model=raw.get("model", self._model),
            provider="ollama",
            finish_reason=_map_finish_reason(raw.get("done_reason")),
            usage=self._usage(raw),
        )

    async def structured(
        self, request: LLMRequest, schema: type[T]
    ) -> StructuredResponse[T]:
        # A scope cut of this example, not a provider limitation - see the
        # module docstring. Ollama supports this via `format=<json schema>`.
        raise NotImplementedError(
            "structured() is out of scope for this example adapter; "
            "Ollama supports it via the 'format' request field"
        )

    def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        # Same note as structured(): Ollama supports NDJSON streaming,
        # this example just does not implement it. `def`, not `async def`
        # - same reason as LLMPort.stream itself (core/port.py): an
        # abstract async-generator method is typed as a plain function
        # returning the iterator, not a coroutine that returns one.
        raise NotImplementedError(
            "stream() is out of scope for this example adapter; "
            "Ollama supports it via 'stream': true and NDJSON chunks"
        )

    async def chat_with_tools(
        self, request: LLMRequest, tools: list[ToolSpec]
    ) -> ChatTurnResponse:
        payload = {
            "model": self._model,
            "messages": [
                {"role": m.role.value, "content": m.content} for m in request.messages
            ],
            "stream": False,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters_schema,
                    },
                }
                for tool in tools
            ],
        }
        raw = await self._post(payload)
        message = raw.get("message", {})
        # Checked empirically, not assumed: this Ollama version's native
        # tool_calls DO carry their own "id" (e.g. "call_5nrs97iu"),
        # confirmed across several runs with and without arguments - the
        # positional fallback below is defensive for an older daemon that
        # might omit it, not evidence that it is normally missing.
        # `arguments` also came back as an already-parsed dict every time
        # (confirmed the same way), unlike a JSON-string form seen from
        # other tool-calling APIs in this project's own lab notes - no
        # json.loads needed here.
        calls = [
            ToolCall(
                id=str(tc.get("id") or f"call-{i}"),
                name=tc["function"]["name"],
                arguments=tc["function"].get("arguments", {}),
            )
            for i, tc in enumerate(message.get("tool_calls") or [])
        ]
        return ChatTurnResponse(
            text=message.get("content", ""),
            model=raw.get("model", self._model),
            provider="ollama",
            finish_reason=_map_finish_reason(raw.get("done_reason")),
            usage=self._usage(raw),
            tool_calls=calls,
        )


GET_TIME = ToolSpec(
    name="get_time",
    description="Return the current UTC time as an ISO 8601 string.",
    parameters_schema={"type": "object", "properties": {}},
)


async def main() -> None:
    print("=== generate() + chat_with_tools() over OllamaNativeAdapter ===")
    llm = OllamaNativeAdapter("qwen3:0.6b", LLMRequirements(tool_calling=True))

    response = await llm.generate(
        LLMRequest(
            messages=[Message(role=Role.USER, content="Name three primary colors.")]
        )
    )
    print(f"generate(): {response.text!r}")
    print(
        f"  [model={response.model} finish={response.finish_reason.value} "
        f"tokens={response.usage.input_tokens}+{response.usage.output_tokens}]"
    )

    turn = await llm.chat_with_tools(
        LLMRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=(
                        "You have a tool called get_time. You MUST call "
                        "get_time to answer. What is the current time?"
                    ),
                )
            ]
        ),
        tools=[GET_TIME],
    )
    print(f"\nchat_with_tools(): tool_calls={turn.tool_calls}")
    print(f"  [finish={turn.finish_reason.value}]")

    print("\n=== fail-fast: LLMRequirements the adapter cannot meet ===")
    try:
        OllamaNativeAdapter("qwen3:0.6b", LLMRequirements(structured_output=True))
    except CapabilityError as exc:
        print(f"CapabilityError raised as expected: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
