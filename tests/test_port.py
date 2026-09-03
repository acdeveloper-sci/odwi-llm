"""Task 5 — LLMPort is a real ABC (design §4.6)."""

from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel

from odwi_llm.core.port import LLMPort
from odwi_llm.core.requirements import LLMCapabilities
from odwi_llm.core.types import (
    ChatTurnResponse,
    FinishReason,
    LLMRequest,
    LLMResponse,
    StreamChunk,
    StructuredResponse,
    ToolSpec,
    Usage,
)

_USAGE = Usage(input_tokens=1, output_tokens=1)
_CAPS = LLMCapabilities(
    structured_output=False,
    tool_calling=False,
    streaming=False,
    vision=False,
    context_tokens=None,
)


class _MinimalPort(LLMPort):
    """Every abstract member implemented with a trivial body."""

    @property
    def capabilities(self) -> LLMCapabilities:
        return _CAPS

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text="", model="m", provider="p",
            finish_reason=FinishReason.STOP, usage=_USAGE,
        )

    async def structured[T: BaseModel](
        self, request: LLMRequest, schema: type[T]
    ) -> StructuredResponse[T]:
        return StructuredResponse[T](
            text="", model="m", provider="p",
            finish_reason=FinishReason.STOP, usage=_USAGE,
            data=schema(),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(kind="done", finish_reason=FinishReason.STOP)

    async def chat_with_tools(
        self, request: LLMRequest, tools: list[ToolSpec]
    ) -> ChatTurnResponse:
        return ChatTurnResponse(
            text="", model="m", provider="p",
            finish_reason=FinishReason.STOP, usage=_USAGE,
        )


def test_minimal_subclass_instantiates() -> None:
    port = _MinimalPort()
    assert isinstance(port, LLMPort)


def test_incomplete_subclass_raises_typeerror() -> None:
    class _Incomplete(LLMPort):
        async def generate(self, request: LLMRequest) -> LLMResponse:  # pragma: no cover
            raise NotImplementedError

    with pytest.raises(TypeError):
        _Incomplete()  # type: ignore[abstract]
