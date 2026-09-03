"""Contract: structured — simple schema, nested+enum+bool schema, and a
payload that fails validation surfaces as LLMSchemaError (§4.2, FINDINGS §1.2).
"""

from enum import Enum

from pydantic import BaseModel
import pytest

from odwi_llm.core.errors import LLMSchemaError
from odwi_llm.core.types import LLMRequest, Message, Role, StructuredResponse

from conftest import AdapterFactory

_MSGS = [Message(role=Role.USER, content="give me the object")]


class _Simple(BaseModel):
    name: str
    count: int


class _Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class _Assignee(BaseModel):
    name: str
    email: str


class _TaskItem(BaseModel):
    title: str
    priority: _Priority
    assignee: _Assignee
    done: bool


async def test_simple_schema(adapter_factory: AdapterFactory) -> None:
    adapter = adapter_factory(structured_payload={"name": "a", "count": 3})
    resp = await adapter.structured(LLMRequest(messages=_MSGS), _Simple)
    assert isinstance(resp, StructuredResponse)
    assert resp.data == _Simple(name="a", count=3)


async def test_nested_enum_bool_schema(adapter_factory: AdapterFactory) -> None:
    payload = {
        "title": "Review Q3 budget",
        "priority": "high",
        "assignee": {"name": "Dana Lee", "email": "dana@example.com"},
        "done": False,
    }
    adapter = adapter_factory(structured_payload=payload)
    resp = await adapter.structured(LLMRequest(messages=_MSGS), _TaskItem)
    assert resp.data.priority is _Priority.HIGH
    assert resp.data.assignee.name == "Dana Lee"
    assert resp.data.done is False


async def test_invalid_payload_raises_schema_error(
    adapter_factory: AdapterFactory,
) -> None:
    adapter = adapter_factory(structured_payload={"name": "a", "count": "NaN"})
    with pytest.raises(LLMSchemaError):
        await adapter.structured(LLMRequest(messages=_MSGS), _Simple)
