"""Smoke test for the guardrail fakes themselves (Task 4). Thorough
Workflow-level coverage lives in the orchestration suite (Fase D).
"""

import inspect
from datetime import datetime
from typing import Any

from odwi_llm.core.types import (
    FinishReason,
    LLMResponse,
    Message,
    Role,
    ToolCall,
    ToolResult,
    Usage,
)
from odwi_llm.guardrails.port import (
    InputGuardrail,
    OutputGuardrail,
    ToolGuardrail,
)
from odwi_llm.guardrails.types import Allow, Deny, PolicyContext, Redact

from ._fake_guardrails import (
    FakeInputGuardrail,
    FakeOutputGuardrail,
    FakeToolGuardrail,
)

_CTX = PolicyContext(
    session_id="s1",
    provider="fake",
    model="fake-model",
    resolved_at=datetime(2026, 1, 1),
)
_MSG = Message(role=Role.USER, content="hi")
_RESPONSE = LLMResponse(
    text="hi",
    model="fake-model",
    provider="fake",
    finish_reason=FinishReason.STOP,
    usage=Usage(input_tokens=1, output_tokens=1),
)
_CALL = ToolCall(id="c1", name="search_docs", arguments={"q": "x"})
_RESULT = ToolResult(tool_call_id="c1", content="{}")


def test_fakes_satisfy_the_protocols() -> None:
    i: InputGuardrail = FakeInputGuardrail()
    o: OutputGuardrail = FakeOutputGuardrail()
    t: ToolGuardrail = FakeToolGuardrail()
    assert inspect.iscoroutinefunction(i.check)
    assert inspect.iscoroutinefunction(o.check)
    assert inspect.iscoroutinefunction(t.before)
    assert inspect.iscoroutinefunction(t.after)
    assert t.covers is None


async def test_input_guardrail_forces_each_decision_variant() -> None:
    redacted: Redact[str] = Redact(reason="pii", redacted="[x]")
    for decision in (
        Allow(),
        Allow(grounded=False),
        Deny(reason="off topic"),
        redacted,
    ):
        g = FakeInputGuardrail(decision)
        assert await g.check(_MSG, _CTX) is decision
    # grounded flag survives untouched
    g = FakeInputGuardrail(Allow(grounded=False))
    out = await g.check(_MSG, _CTX)
    assert isinstance(out, Allow) and out.grounded is False


async def test_output_guardrail_forces_each_decision_variant() -> None:
    redacted: Redact[str] = Redact(reason="pii", redacted="[x]")
    for decision in (Allow(), Allow(grounded=False), Deny(reason="no"), redacted):
        g = FakeOutputGuardrail(decision)
        assert await g.check(_RESPONSE, _CTX) is decision


async def test_tool_guardrail_before_and_after_decisions() -> None:
    sanitized: Redact[dict[str, Any]] = Redact(
        reason="pii-arg", redacted={"q": "[x]"}
    )
    g = FakeToolGuardrail(before=sanitized, after=Deny(reason="bad result"))
    assert await g.before(_CALL, _CTX) is sanitized
    assert await g.after(_CALL, _RESULT, _CTX) == Deny(reason="bad result")
    # defaults are Allow on both sides
    d = FakeToolGuardrail()
    assert isinstance(await d.before(_CALL, _CTX), Allow)
    assert isinstance(await d.after(_CALL, _RESULT, _CTX), Allow)


def test_tool_guardrail_covers_is_configurable_including_none() -> None:
    assert FakeToolGuardrail().covers is None  # catch-all default
    specific = FakeToolGuardrail(covers=frozenset({"search_docs"}))
    assert specific.covers == frozenset({"search_docs"})


async def test_fakes_record_their_calls() -> None:
    i = FakeInputGuardrail()
    await i.check(_MSG, _CTX)
    assert i.calls == [(_MSG, _CTX)]

    t = FakeToolGuardrail()
    await t.before(_CALL, _CTX)
    await t.after(_CALL, _RESULT, _CTX)
    assert t.before_calls == [(_CALL, _CTX)]
    assert t.after_calls == [(_CALL, _RESULT, _CTX)]
