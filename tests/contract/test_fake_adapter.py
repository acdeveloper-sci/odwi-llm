"""Smoke test for the test double itself (Task 9). The thorough
scenario coverage lives in the contract suite (test_generate.py …).
"""

import pytest
from pydantic import BaseModel

from odwi_llm.core.errors import LLMAuthError
from odwi_llm.core.port import LLMPort
from odwi_llm.core.requirements import CapabilityError, LLMCapabilities, LLMRequirements
from odwi_llm.core.types import (
    ChatTurnResponse,
    LLMRequest,
    LLMResponse,
    Message,
    Role,
    StructuredResponse,
)

from _fake_adapter import FULL_CAPS, FakeAdapter

_REQ = LLMRequest(messages=[Message(role=Role.USER, content="hi")])


class _Simple(BaseModel):
    name: str = "x"


def test_is_a_real_llmport() -> None:
    assert isinstance(FakeAdapter(), LLMPort)
    assert FakeAdapter().capabilities == FULL_CAPS


async def test_four_ops_return_contract_types_offline() -> None:
    fake = FakeAdapter(text="hello")
    assert isinstance(await fake.generate(_REQ), LLMResponse)
    assert isinstance(await fake.structured(_REQ, _Simple), StructuredResponse)
    assert isinstance(await fake.chat_with_tools(_REQ, []), ChatTurnResponse)
    assert [c async for c in fake.stream(_REQ)][-1].kind == "done"


def test_capability_error_knob() -> None:
    no_tools = LLMCapabilities(
        structured_output=True, tool_calling=False, streaming=True,
        vision=False, context_tokens=1000,
    )
    with pytest.raises(CapabilityError):
        FakeAdapter(capabilities=no_tools, requirements=LLMRequirements(tool_calling=True))


async def test_error_knob() -> None:
    with pytest.raises(LLMAuthError):
        await FakeAdapter(error=LLMAuthError("bad key")).generate(_REQ)
