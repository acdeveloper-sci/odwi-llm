"""Task 9 "Hecho cuando" — the FakeAdapter can produce or force any
§9.3 matrix case, offline. (Not the contract suite itself; that is Task 10.)
"""

import pytest
from pydantic import BaseModel

from odwi_llm.core.errors import (
    LLMAuthError,
    LLMContentFilteredError,
    LLMContextLengthError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMSchemaError,
)
from odwi_llm.core.port import LLMPort
from odwi_llm.core.requirements import CapabilityError, LLMCapabilities, LLMRequirements
from odwi_llm.core.types import (
    ChatTurnResponse,
    FinishReason,
    LLMRequest,
    LLMResponse,
    Message,
    Role,
    StructuredResponse,
    ToolCall,
)

from _fake_adapter import FULL_CAPS, FakeAdapter

_REQ = LLMRequest(messages=[Message(role=Role.USER, content="hi")])

_ALL_ERRORS: list[LLMError] = [
    LLMAuthError("bad key"),
    LLMRateLimitError("429", retry_after_seconds=2.5),
    LLMContextLengthError("too long"),
    LLMContentFilteredError("blocked"),
    LLMSchemaError("bad schema"),
    LLMProviderError("5xx"),
]


class _Simple(BaseModel):
    name: str = "x"
    count: int = 0


def test_is_a_real_llmport() -> None:
    assert isinstance(FakeAdapter(), LLMPort)


async def test_four_ops_return_contract_types() -> None:
    fake = FakeAdapter(text="hello")
    assert isinstance(await fake.generate(_REQ), LLMResponse)
    sr = await fake.structured(_REQ, _Simple)
    assert isinstance(sr, StructuredResponse) and isinstance(sr.data, _Simple)
    turn = await fake.chat_with_tools(_REQ, [])
    assert isinstance(turn, ChatTurnResponse)
    chunks = [c async for c in fake.stream(_REQ)]
    assert chunks[-1].kind == "done"


def test_capabilities_reported() -> None:
    assert FakeAdapter().capabilities == FULL_CAPS
    caps = LLMCapabilities(
        structured_output=False, tool_calling=False, streaming=True,
        vision=False, context_tokens=1000,
    )
    assert FakeAdapter(capabilities=caps).capabilities == caps


def test_requirements_unmet_raise_capability_error_at_construction() -> None:
    no_tools = LLMCapabilities(
        structured_output=True, tool_calling=False, streaming=True,
        vision=False, context_tokens=1000,
    )
    with pytest.raises(CapabilityError):
        FakeAdapter(
            capabilities=no_tools,
            requirements=LLMRequirements(tool_calling=True),
        )
    with pytest.raises(CapabilityError):
        FakeAdapter(
            capabilities=no_tools,
            requirements=LLMRequirements(min_context_tokens=8000),
        )


@pytest.mark.parametrize("exc", _ALL_ERRORS, ids=lambda e: type(e).__name__)
async def test_any_error_can_be_forced(exc: LLMError) -> None:
    fake = FakeAdapter(error=exc)
    with pytest.raises(type(exc)):
        await fake.generate(_REQ)
    with pytest.raises(type(exc)):
        await fake.chat_with_tools(_REQ, [])
    with pytest.raises(type(exc)):
        [c async for c in fake.stream(_REQ)]


async def test_rate_limit_retry_after_is_accessible() -> None:
    fake = FakeAdapter(error=LLMRateLimitError("429", retry_after_seconds=2.5))
    with pytest.raises(LLMRateLimitError) as caught:
        await fake.generate(_REQ)
    assert caught.value.retry_after_seconds == 2.5


async def test_tool_calls_come_back_as_dicts() -> None:
    calls = [
        ToolCall(id="c1", name="get_weather", arguments={"city": "Paris"}),
        ToolCall(id="c2", name="get_weather", arguments={"city": "Tokyo"}),
    ]
    turn = await FakeAdapter(tool_calls=calls).chat_with_tools(_REQ, [])
    assert turn.finish_reason is FinishReason.TOOL_CALLS
    assert [tc.arguments for tc in turn.tool_calls] == [
        {"city": "Paris"}, {"city": "Tokyo"}
    ]
    assert all(isinstance(tc.arguments, dict) for tc in turn.tool_calls)


async def test_structured_bad_payload_surfaces_as_schema_error() -> None:
    ok = await FakeAdapter(structured_payload={"name": "a", "count": 2}).structured(
        _REQ, _Simple
    )
    assert ok.data == _Simple(name="a", count=2)
    with pytest.raises(LLMSchemaError):
        await FakeAdapter(structured_payload={"count": "not-an-int"}).structured(
            _REQ, _Simple
        )


async def test_stream_default_sequence_and_no_reasoning_leak() -> None:
    fake = FakeAdapter(text="abcd", hidden_reasoning="SECRET THOUGHTS")
    chunks = [c async for c in fake.stream(_REQ)]
    kinds = [c.kind for c in chunks]
    assert kinds[-1] == "done"
    assert "usage" in kinds
    assert chunks[-1].finish_reason is FinishReason.STOP
    text_deltas = [c.text for c in chunks if c.kind == "text_delta"]
    assert "".join(t or "" for t in text_deltas) == "abcd"
    assert all("SECRET" not in (t or "") for t in text_deltas)
