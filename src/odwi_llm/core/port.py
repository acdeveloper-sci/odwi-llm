"""The port interface — design §4.6.

One turn with a language model. No state, no history, no tool execution,
no business retries. Adapters implement this; applications depend only on
it.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from odwi_llm.core.requirements import LLMCapabilities
from odwi_llm.core.types import (
    ChatTurnResponse,
    LLMRequest,
    LLMResponse,
    StreamChunk,
    StructuredResponse,
    T,
    ToolSpec,
)


class LLMPort(ABC):
    """One turn with a language model. No state, no history, no tool execution."""

    @property
    @abstractmethod
    def capabilities(self) -> LLMCapabilities: ...

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse: ...

    @abstractmethod
    async def structured(
        self, request: LLMRequest, schema: type[T]
    ) -> StructuredResponse[T]: ...

    # `def`, not `async def`: subclasses implement this as an async generator
    # (`async def ... yield`). An abstract async-generator method must be typed
    # `def -> AsyncIterator[...]` — `async def -> AsyncIterator[...]` is a
    # coroutine returning the iterator (mypy docs: LauncherIncorrect vs
    # LauncherCorrect). Design §4.6 shows `async def`; see plan Task 16.
    @abstractmethod
    def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]: ...

    @abstractmethod
    async def chat_with_tools(
        self, request: LLMRequest, tools: list[ToolSpec]
    ) -> ChatTurnResponse: ...
