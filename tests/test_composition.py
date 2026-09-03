"""Task 7 — FallbackLLM (design §5.4).

Inline LLMPort stubs (no FakeAdapter yet — that is Fase B / Task 9).
"""

from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel

from odwi_llm.core.composition import FallbackLLM
from odwi_llm.core.errors import (
    LLMAuthError,
    LLMContentFilteredError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMSchemaError,
)
from odwi_llm.core.port import LLMPort
from odwi_llm.core.requirements import LLMCapabilities
from odwi_llm.core.sync import SyncLLM
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
    ToolSpec,
    Usage,
)

_USAGE = Usage(input_tokens=1, output_tokens=1)
_REQ = LLMRequest(messages=[Message(role=Role.USER, content="hi")])


class _Echo(BaseModel):
    value: str = "x"


def _caps() -> LLMCapabilities:
    return LLMCapabilities(
        structured_output=True, tool_calling=True, streaming=True,
        vision=False, context_tokens=None,
    )


class _OkPort(LLMPort):
    """Succeeds at everything; marks its identity in the text field."""

    def __init__(self, tag: str) -> None:
        self._tag = tag

    @property
    def capabilities(self) -> LLMCapabilities:
        return _caps()

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text=self._tag, model="m", provider=self._tag,
            finish_reason=FinishReason.STOP, usage=_USAGE,
        )

    async def structured(
        self, request: LLMRequest, schema: type[T]
    ) -> StructuredResponse[T]:
        return StructuredResponse[T](
            text=self._tag, model="m", provider=self._tag,
            finish_reason=FinishReason.STOP, usage=_USAGE, data=schema(),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(kind="text_delta", text=self._tag)
        yield StreamChunk(kind="done", finish_reason=FinishReason.STOP)

    async def chat_with_tools(
        self, request: LLMRequest, tools: list[ToolSpec]
    ) -> ChatTurnResponse:
        return ChatTurnResponse(
            text=self._tag, model="m", provider=self._tag,
            finish_reason=FinishReason.TOOL_CALLS, usage=_USAGE,
        )


class _FailPort(LLMPort):
    """Raises the given exception from every operation."""

    def __init__(self, exc: LLMError) -> None:
        self._exc = exc

    @property
    def capabilities(self) -> LLMCapabilities:
        return _caps()

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise self._exc

    async def structured(
        self, request: LLMRequest, schema: type[T]
    ) -> StructuredResponse[T]:
        raise self._exc

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        raise self._exc
        yield  # pragma: no cover  (makes this an async generator)

    async def chat_with_tools(
        self, request: LLMRequest, tools: list[ToolSpec]
    ) -> ChatTurnResponse:
        raise self._exc


class _PartialStreamPort(LLMPort):
    """Yields one chunk, then raises — the mid-stream failure case."""

    @property
    def capabilities(self) -> LLMCapabilities:
        return _caps()

    async def generate(self, request: LLMRequest) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError

    async def structured(  # pragma: no cover
        self, request: LLMRequest, schema: type[T]
    ) -> StructuredResponse[T]:
        raise NotImplementedError

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(kind="text_delta", text="partial")
        raise LLMProviderError("died mid-stream")

    async def chat_with_tools(  # pragma: no cover
        self, request: LLMRequest, tools: list[ToolSpec]
    ) -> ChatTurnResponse:
        raise NotImplementedError


# --- fallback=None: transparent -----------------------------------
async def test_no_fallback_passes_through() -> None:
    llm = FallbackLLM(primary=_OkPort("primary"))
    assert (await llm.generate(_REQ)).text == "primary"
    assert llm.capabilities == _caps()


async def test_no_fallback_reraises_transient() -> None:
    llm = FallbackLLM(primary=_FailPort(LLMProviderError("5xx")))
    with pytest.raises(LLMProviderError):
        await llm.generate(_REQ)


# --- fallback delegates on transient errors ----------------------
@pytest.mark.parametrize(
    "exc", [LLMProviderError("5xx"), LLMRateLimitError("429", retry_after_seconds=1.0)]
)
async def test_fallback_delegates_on_transient(exc: LLMError) -> None:
    llm = FallbackLLM(primary=_FailPort(exc), fallback=_OkPort("backup"))
    assert (await llm.generate(_REQ)).text == "backup"
    assert (await llm.structured(_REQ, _Echo)).provider == "backup"
    assert (await llm.chat_with_tools(_REQ, [])).provider == "backup"


# --- fallback does NOT delegate on these (§5.4 "qué no cubre") ---
@pytest.mark.parametrize(
    "exc",
    [
        LLMAuthError("bad key"),
        LLMSchemaError("invalid after retries"),
        LLMContentFilteredError("blocked"),
    ],
)
async def test_fallback_does_not_delegate_on_hard_errors(exc: LLMError) -> None:
    llm = FallbackLLM(primary=_FailPort(exc), fallback=_OkPort("backup"))
    with pytest.raises(type(exc)):
        await llm.generate(_REQ)


# --- streaming -------------------------------------------------
async def test_stream_falls_back_before_first_chunk() -> None:
    llm = FallbackLLM(
        primary=_FailPort(LLMProviderError("down")), fallback=_OkPort("backup")
    )
    chunks = [c async for c in llm.stream(_REQ)]
    assert [c.text for c in chunks] == ["backup", None]


async def test_stream_does_not_fall_back_after_partial_output() -> None:
    llm = FallbackLLM(primary=_PartialStreamPort(), fallback=_OkPort("backup"))
    seen: list[str | None] = []
    with pytest.raises(LLMProviderError):
        async for c in llm.stream(_REQ):
            seen.append(c.text)
    assert seen == ["partial"]  # consumer kept the partial output, no silent swap


# --- SyncLLM wraps FallbackLLM "sin cambios" (§5.4) -------------
def test_syncllm_wraps_fallbackllm() -> None:
    fb = FallbackLLM(
        primary=_FailPort(LLMProviderError("down")), fallback=_OkPort("backup")
    )
    with SyncLLM(fb) as sync_llm:
        assert sync_llm.generate(_REQ).text == "backup"
        assert [c.text for c in sync_llm.stream(_REQ)] == ["backup", None]
