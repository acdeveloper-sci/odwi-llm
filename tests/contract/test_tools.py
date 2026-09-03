"""Contract: chat_with_tools — one call, many calls, args are dicts never
strings (§4.3, FINDINGS §1.2), and an unmet tool_calling requirement is a
construction-time CapabilityError (§4.4, FINDINGS §1 LM Studio + Any-LLM).
"""

import pytest

from odwi_llm.core.requirements import CapabilityError, LLMCapabilities, LLMRequirements
from odwi_llm.core.types import (
    ChatTurnResponse,
    FinishReason,
    LLMRequest,
    Message,
    Role,
    ToolCall,
)

from conftest import AdapterFactory

_MSGS = [Message(role=Role.USER, content="what's the weather in Paris and Tokyo?")]

_NO_TOOLS = LLMCapabilities(
    structured_output=True,
    tool_calling=False,
    streaming=True,
    vision=False,
    context_tokens=100_000,
)


async def test_single_tool_call_args_are_dict(
    adapter_factory: AdapterFactory,
) -> None:
    calls = [ToolCall(id="c1", name="get_weather", arguments={"city": "Paris"})]
    adapter = adapter_factory(tool_calls=calls)
    turn = await adapter.chat_with_tools(LLMRequest(messages=_MSGS), [])
    assert isinstance(turn, ChatTurnResponse)
    assert turn.finish_reason is FinishReason.TOOL_CALLS
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].arguments == {"city": "Paris"}
    assert isinstance(turn.tool_calls[0].arguments, dict)


async def test_multiple_tool_calls(adapter_factory: AdapterFactory) -> None:
    calls = [
        ToolCall(id="c1", name="get_weather", arguments={"city": "Paris"}),
        ToolCall(id="c2", name="get_weather", arguments={"city": "Tokyo"}),
    ]
    adapter = adapter_factory(tool_calls=calls)
    turn = await adapter.chat_with_tools(LLMRequest(messages=_MSGS), [])
    assert [tc.arguments["city"] for tc in turn.tool_calls] == ["Paris", "Tokyo"]
    assert all(isinstance(tc.arguments, dict) for tc in turn.tool_calls)


def test_unmet_tool_requirement_raises_capability_error(
    adapter_factory: AdapterFactory,
) -> None:
    with pytest.raises(CapabilityError):
        adapter_factory(
            capabilities=_NO_TOOLS,
            requirements=LLMRequirements(tool_calling=True),
        )
