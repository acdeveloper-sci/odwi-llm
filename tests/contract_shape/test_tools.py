"""Shape: tool-call arguments are always dicts, never JSON strings (§4.3)."""

from odwi_llm.core.port import LLMPort
from odwi_llm.core.types import (
    ChatTurnResponse,
    LLMRequest,
    Message,
    Role,
    ToolSpec,
)

_WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
)

_REQ = LLMRequest(
    messages=[
        Message(
            role=Role.USER,
            content="What is the weather in Paris? You must call the get_weather tool.",
        )
    ],
    # no max_output_tokens (see test_generate.py)
)


async def test_tool_call_arguments_are_dicts(adapter: LLMPort) -> None:
    turn = await adapter.chat_with_tools(_REQ, [_WEATHER_TOOL])
    assert isinstance(turn, ChatTurnResponse)
    for call in turn.tool_calls:
        assert isinstance(call.arguments, dict)
        assert not isinstance(call.arguments, str)
        assert isinstance(call.id, str)
        assert call.name
