"""Shape: structured() returns data that validates against the requested model."""

from pydantic import BaseModel

from odwi_llm.core.port import LLMPort
from odwi_llm.core.types import LLMRequest, Message, Role, StructuredResponse


class Weather(BaseModel):
    # all fields defaulted so the fake's empty payload also validates
    city: str = "unknown"
    temperature_c: float = 0.0


_REQ = LLMRequest(
    messages=[
        Message(
            role=Role.USER,
            content=(
                "Return the current weather for Paris as JSON with keys "
                '"city" (string) and "temperature_c" (number).'
            ),
        )
    ],
    # no max_output_tokens (see test_generate.py)
)


async def test_structured_data_validates_against_schema(adapter: LLMPort) -> None:
    resp = await adapter.structured(_REQ, Weather)
    assert isinstance(resp, StructuredResponse)
    assert isinstance(resp.data, Weather)  # already validated by the adapter
