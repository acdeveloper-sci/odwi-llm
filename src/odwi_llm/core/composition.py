"""Fallback composition — design §5.4.

`FallbackLLM` wraps a primary `LLMPort` and an optional fallback one. The
fallback lives here, in a thin composition layer — never inside an
adapter (an adapter does not know another exists). With `fallback=None`
(the Stage 1 default) it behaves exactly like the primary, at no cost.

Only transient / provider-side failures fall through:
`LLMRateLimitError` and `LLMProviderError`. Auth errors, content-filter
blocks and schema failures are NOT retried against the fallback — they
would very likely repeat on the second provider, and masking them with an
automatic retry hurts diagnosis (§5.4).
"""

from collections.abc import AsyncIterator

from odwi_llm.core.errors import LLMProviderError, LLMRateLimitError
from odwi_llm.core.port import LLMPort
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

_FALLBACK_ON = (LLMRateLimitError, LLMProviderError)


class FallbackLLM(LLMPort):
    """Wraps a primary port and an optional fallback port.

    If fallback is None, behaves exactly like primary (Stage 1 default).
    """

    def __init__(self, primary: LLMPort, fallback: LLMPort | None = None) -> None:
        self._primary = primary
        self._fallback = fallback

    @property
    def capabilities(self) -> LLMCapabilities:
        # The primary is what a caller gets by default; report its caps.
        return self._primary.capabilities

    async def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            return await self._primary.generate(request)
        except _FALLBACK_ON:
            if self._fallback is None:
                raise
            return await self._fallback.generate(request)

    async def structured(
        self, request: LLMRequest, schema: type[T]
    ) -> StructuredResponse[T]:
        try:
            return await self._primary.structured(request, schema)
        except _FALLBACK_ON:
            if self._fallback is None:
                raise
            return await self._fallback.structured(request, schema)

    async def chat_with_tools(
        self, request: LLMRequest, tools: list[ToolSpec]
    ) -> ChatTurnResponse:
        try:
            return await self._primary.chat_with_tools(request, tools)
        except _FALLBACK_ON:
            if self._fallback is None:
                raise
            return await self._fallback.chat_with_tools(request, tools)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        yielded = False
        try:
            async for chunk in self._primary.stream(request):
                yielded = True
                yield chunk
        except _FALLBACK_ON:
            # Only switch if the primary failed before emitting anything.
            # Mid-stream we cannot fall back without garbling the output.
            if self._fallback is None or yielded:
                raise
            async for chunk in self._fallback.stream(request):
                yield chunk
