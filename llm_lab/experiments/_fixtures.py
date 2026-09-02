"""Shared prompts / schema / tool used by both Fase B (native SDK) and
Fase C (litellm, any-llm). The values here mirror the ones defined inline
in 01_raw_text.py .. 04_raw_stream.py so the two phases are comparable.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

# Task 6 / 10 - simple text
TEXT_PROMPT = "In one sentence, list the three primary colors of light in the additive model."

# Task 9 / 10 - streaming
STREAM_PROMPT = "Write three short sentences about why the sky is blue."


# Task 7 / 10 - structured output
class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Assignee(BaseModel):
    name: str
    email: str


class TaskItem(BaseModel):
    title: str
    priority: Priority
    assignee: Assignee
    done: bool


STRUCTURED_PROMPT = (
    "Create a task item for reviewing the Q3 budget, assigned to Dana Lee "
    "(dana@example.com), high priority, not done yet. Respond with JSON only."
)

SCHEMA = TaskItem.model_json_schema()


# Task 8 / 10 - tool calling
TOOL_NAME = "get_weather"
TOOL_DESCRIPTION = "Get the current weather for a city."
TOOL_PARAMS = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "description": "City name, e.g. 'Paris'"},
    },
    "required": ["city"],
    "additionalProperties": False,
}
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": TOOL_DESCRIPTION,
            "parameters": TOOL_PARAMS,
        },
    }
]

TOOL_PROMPT_SINGLE = (
    "What is the current weather in Paris? You must call the get_weather tool."
)
