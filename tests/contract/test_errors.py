"""Contract: errors — the 6 §4.5 LLMError types can be forced and caught
through the port, and LLMRateLimitError carries retry_after_seconds.
"""

import pytest

from odwi_llm.core.errors import (
    LLMAuthError,
    LLMContentFilteredError,
    LLMContextLengthError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMSchemaError,
)
from odwi_llm.core.types import LLMRequest, Message, Role

from conftest import AdapterFactory

_REQ = LLMRequest(messages=[Message(role=Role.USER, content="hi")])

_ERROR_CASES: list[LLMError] = [
    LLMAuthError("bad or missing key"),
    LLMRateLimitError("429", retry_after_seconds=1.5),
    LLMContextLengthError("prompt exceeds context window"),
    LLMContentFilteredError("blocked by content filter"),
    LLMSchemaError("did not validate after retries"),
    LLMProviderError("provider 5xx / timeout"),
]


@pytest.mark.parametrize("exc", _ERROR_CASES, ids=lambda e: type(e).__name__)
async def test_error_is_raised_through_the_port(
    adapter_factory: AdapterFactory, exc: LLMError
) -> None:
    adapter = adapter_factory(error=exc)
    with pytest.raises(type(exc)):
        await adapter.generate(_REQ)
    # same error on every operation, and always an LLMError subclass
    with pytest.raises(LLMError):
        await adapter.chat_with_tools(_REQ, [])
    with pytest.raises(LLMError):
        [c async for c in adapter.stream(_REQ)]


async def test_rate_limit_retry_after_seconds(
    adapter_factory: AdapterFactory,
) -> None:
    adapter = adapter_factory(
        error=LLMRateLimitError("429", retry_after_seconds=1.5)
    )
    with pytest.raises(LLMRateLimitError) as caught:
        await adapter.generate(_REQ)
    assert caught.value.retry_after_seconds == 1.5


async def test_rate_limit_retry_after_defaults_to_none(
    adapter_factory: AdapterFactory,
) -> None:
    adapter = adapter_factory(error=LLMRateLimitError("429"))
    with pytest.raises(LLMRateLimitError) as caught:
        await adapter.generate(_REQ)
    assert caught.value.retry_after_seconds is None
