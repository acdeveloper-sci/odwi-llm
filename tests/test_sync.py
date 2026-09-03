"""Task 6 — SyncLLM works with or without an already-running event loop (§4.7).

The load-bearing test is `test_generate_inside_running_loop`: calling the
blocking facade from inside a live asyncio loop must NOT raise
`RuntimeError: asyncio.run() cannot be called from a running event loop`
(nor "This event loop is already running"). That is what justifies
`_LoopRunner` over a naive facade.
"""

import asyncio
from collections.abc import AsyncIterator

from pydantic import BaseModel

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


class _StubPort(LLMPort):
    """In-memory LLMPort. Every op awaits once, then returns canned data."""

    @property
    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            structured_output=True,
            tool_calling=True,
            streaming=True,
            vision=False,
            context_tokens=None,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        await asyncio.sleep(0)
        return LLMResponse(
            text="pong", model="stub", provider="stub",
            finish_reason=FinishReason.STOP, usage=_USAGE,
        )

    async def structured(
        self, request: LLMRequest, schema: type[T]
    ) -> StructuredResponse[T]:
        await asyncio.sleep(0)
        return StructuredResponse[T](
            text="", model="stub", provider="stub",
            finish_reason=FinishReason.STOP, usage=_USAGE, data=schema(),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        for token in ("he", "llo"):
            await asyncio.sleep(0)
            yield StreamChunk(kind="text_delta", text=token)
        yield StreamChunk(kind="done", finish_reason=FinishReason.STOP)

    async def chat_with_tools(
        self, request: LLMRequest, tools: list[ToolSpec]
    ) -> ChatTurnResponse:
        await asyncio.sleep(0)
        return ChatTurnResponse(
            text="", model="stub", provider="stub",
            finish_reason=FinishReason.TOOL_CALLS, usage=_USAGE,
        )


def test_generate_plain_sync() -> None:
    with SyncLLM(_StubPort()) as sync_llm:
        assert sync_llm.generate(_REQ).text == "pong"


def test_structured_and_tools_plain_sync() -> None:
    with SyncLLM(_StubPort()) as sync_llm:
        structured = sync_llm.structured(_REQ, _Echo)
        turn = sync_llm.chat_with_tools(_REQ, [])
    assert isinstance(structured.data, _Echo)
    assert turn.finish_reason is FinishReason.TOOL_CALLS


def test_stream_plain_sync() -> None:
    with SyncLLM(_StubPort()) as sync_llm:
        chunks = list(sync_llm.stream(_REQ))
    assert [c.text for c in chunks] == ["he", "llo", None]
    assert chunks[-1].kind == "done"


def test_generate_inside_running_loop() -> None:
    async def driver() -> LLMResponse:
        sync_llm = SyncLLM(_StubPort())
        try:
            # blocking sync call, made from within a live event loop
            return sync_llm.generate(_REQ)
        finally:
            sync_llm.close()

    result = asyncio.run(driver())  # would raise if the facade used asyncio.run()
    assert result.text == "pong"


def test_stream_inside_running_loop() -> None:
    async def driver() -> list[str | None]:
        sync_llm = SyncLLM(_StubPort())
        try:
            return [c.text for c in sync_llm.stream(_REQ)]
        finally:
            sync_llm.close()

    assert asyncio.run(driver()) == ["he", "llo", None]
