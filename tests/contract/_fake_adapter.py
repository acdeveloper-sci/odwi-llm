"""Task 9 — a deterministic, network-free LLMPort for the contract suite.

The contract suite (Fase B) is written against this first, so it speaks
the language of §4, not of any particular library. Every knob below maps
to a cell of the §9.3 matrix; nothing here touches a real provider.

Not collected by pytest (leading underscore); imported by the
`test_*.py` files in this directory.
"""

from collections.abc import AsyncIterator

from pydantic import BaseModel, ValidationError

from odwi_llm.core.errors import LLMError, LLMSchemaError
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
    StreamChunk,
    StructuredResponse,
    T,
    ToolCall,
    ToolSpec,
    Usage,
)

FULL_CAPS = LLMCapabilities(
    structured_output=True,
    tool_calling=True,
    streaming=True,
    vision=True,
    context_tokens=200_000,
)

_DEFAULT_USAGE = Usage(input_tokens=3, output_tokens=5, estimated_cost_usd=None)


def _unmet(req: LLMRequirements, caps: LLMCapabilities) -> list[str]:
    missing: list[str] = []
    if req.structured_output and not caps.structured_output:
        missing.append("structured_output")
    if req.tool_calling and not caps.tool_calling:
        missing.append("tool_calling")
    if req.streaming and not caps.streaming:
        missing.append("streaming")
    if req.vision and not caps.vision:
        missing.append("vision")
    if req.min_context_tokens is not None and (
        caps.context_tokens is None or caps.context_tokens < req.min_context_tokens
    ):
        missing.append("min_context_tokens")
    return missing


def _default_stream(text: str, usage: Usage, finish: FinishReason) -> list[StreamChunk]:
    return [
        StreamChunk(kind="text_delta", text=part)
        for part in (text[: len(text) // 2], text[len(text) // 2 :])
        if part
    ] + [
        StreamChunk(kind="usage", usage=usage),
        StreamChunk(kind="done", finish_reason=finish),
    ]


class FakeAdapter(LLMPort):
    """In-memory LLMPort. Deterministic. No network.

    Knobs (all keyword-only):
      capabilities        - reported by `.capabilities` (default FULL_CAPS)
      requirements        - if given, checked against capabilities at
                            construction; unmet -> CapabilityError (§4.4)
      error               - an LLMError instance every operation raises
                            (covers the 6 §4.5 types)
      text                - text of the non-streaming responses
      finish_reason       - finish_reason of the responses / done chunk
      usage               - Usage attached to responses
      structured_payload  - dict validated against the requested schema;
                            a bad shape surfaces as LLMSchemaError, like a
                            real adapter after its retries (§4.2)
      tool_calls          - ToolCalls returned by chat_with_tools
                            (arguments are dicts, never strings - §4.3)
      stream_chunks       - overrides the default StreamChunk sequence
      hidden_reasoning    - reasoning the "provider" produced; the adapter
                            must never surface it (§ principle 3). Kept
                            here only so a test can assert it never leaks.
    """

    def __init__(
        self,
        *,
        capabilities: LLMCapabilities | None = None,
        requirements: LLMRequirements | None = None,
        error: LLMError | None = None,
        text: str = "fake response",
        finish_reason: FinishReason = FinishReason.STOP,
        usage: Usage | None = None,
        structured_payload: dict[str, object] | None = None,
        tool_calls: list[ToolCall] | None = None,
        stream_chunks: list[StreamChunk] | None = None,
        hidden_reasoning: str | None = None,
    ) -> None:
        self._caps = capabilities or FULL_CAPS
        if requirements is not None:
            unmet = _unmet(requirements, self._caps)
            if unmet:
                raise CapabilityError(
                    f"adapter does not meet requirements: {', '.join(unmet)}"
                )
        self._error = error
        self._text = text
        self._finish = finish_reason
        self._usage = usage or _DEFAULT_USAGE
        self._structured_payload: dict[str, object] = structured_payload or {}
        self._tool_calls = list(tool_calls or [])
        self._stream_chunks = stream_chunks
        self.hidden_reasoning = hidden_reasoning

    @property
    def capabilities(self) -> LLMCapabilities:
        return self._caps

    def _raise_if_configured(self) -> None:
        if self._error is not None:
            raise self._error

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self._raise_if_configured()
        return LLMResponse(
            text=self._text,
            model="fake-model",
            provider="fake",
            finish_reason=self._finish,
            usage=self._usage,
        )

    async def structured(
        self, request: LLMRequest, schema: type[T]
    ) -> StructuredResponse[T]:
        self._raise_if_configured()
        try:
            data = schema.model_validate(self._structured_payload)
        except ValidationError as exc:
            raise LLMSchemaError(
                f"structured output did not validate after retries: {exc}"
            ) from exc
        return StructuredResponse[T](
            text=self._text,
            model="fake-model",
            provider="fake",
            finish_reason=self._finish,
            usage=self._usage,
            data=data,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        self._raise_if_configured()
        chunks = self._stream_chunks
        if chunks is None:
            chunks = _default_stream(self._text, self._usage, self._finish)
        for chunk in chunks:
            yield chunk

    async def chat_with_tools(
        self, request: LLMRequest, tools: list[ToolSpec]
    ) -> ChatTurnResponse:
        self._raise_if_configured()
        finish = FinishReason.TOOL_CALLS if self._tool_calls else self._finish
        return ChatTurnResponse(
            text="" if self._tool_calls else self._text,
            model="fake-model",
            provider="fake",
            finish_reason=finish,
            usage=self._usage,
            tool_calls=list(self._tool_calls),
        )
