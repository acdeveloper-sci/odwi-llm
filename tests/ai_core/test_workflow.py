"""Task 12 — Workflow.run() against fakes, no tool-calling loop yet.

Covers: allowed message responds normally; a Deny in input short-circuits
before the LLM; Deny / Redact / Allow(grounded=False) in output are
reflected in WorkflowResult; the happy-path observability event order.
Plus the four `_validate_tool_coverage` cases (design §12).
"""

from datetime import datetime

import pytest

from odwi_llm.core.types import FinishReason, ToolSpec
from odwi_llm.guardrails.errors import PolicyConfigError
from odwi_llm.guardrails.port import GuardrailSet
from odwi_llm.guardrails.types import Allow, Deny, PolicyContext, Redact
from odwi_llm.orchestration.types import WorkflowTask
from odwi_llm.orchestration.workflow import Workflow, _validate_tool_coverage

from _fake_adapter import FakeAdapter

from ._fake_context import FakeContext
from ._fake_guardrails import (
    FakeInputGuardrail,
    FakeOutputGuardrail,
    FakeToolGuardrail,
)
from ._fake_observability import FakeObservability

_CTX = PolicyContext(
    session_id="s1",
    provider="ollama",
    model="qwen3:0.6b",
    resolved_at=datetime(2026, 1, 1),
)
_TASK = WorkflowTask(session_id="s1", input={"message": "hello"})


def _tool(name: str) -> ToolSpec:
    return ToolSpec(name=name, description=name, parameters_schema={})


# --- run() ---------------------------------------------------------------


async def test_allowed_message_responds_normally() -> None:
    obs = FakeObservability()
    wf = Workflow(
        llm=FakeAdapter(text="hi there"),
        context=FakeContext(),
        guardrails=GuardrailSet(),
        observability=obs,
    )
    result = await wf.run(_TASK, _CTX)

    assert result.text == "hi there"
    assert result.finish_reason is FinishReason.STOP
    assert result.grounded is True
    assert result.tool_calls_made == 0
    assert obs.names == ["execution_started", "llm_call", "result"]


async def test_input_deny_short_circuits_before_the_llm() -> None:
    obs = FakeObservability()
    ctx_fake = FakeContext()
    out_guard = FakeOutputGuardrail()
    wf = Workflow(
        llm=FakeAdapter(text="should not be used"),
        context=ctx_fake,
        guardrails=GuardrailSet(
            input_guardrails=[FakeInputGuardrail(Deny(reason="off topic"))],
            output_guardrails=[out_guard],
        ),
        observability=obs,
    )
    result = await wf.run(_TASK, _CTX)

    assert result.text == "off topic"
    assert result.finish_reason is FinishReason.CONTENT_FILTER
    assert "llm_call" not in obs.names
    assert ("error", {"reason": "off topic"}) in obs.events
    assert ctx_fake.calls == []  # never reached context selection
    assert out_guard.calls == []  # never reached output policy


async def test_output_deny_is_reflected() -> None:
    obs = FakeObservability()
    wf = Workflow(
        llm=FakeAdapter(text="raw answer"),
        context=FakeContext(),
        guardrails=GuardrailSet(
            output_guardrails=[FakeOutputGuardrail(Deny(reason="blocked"))]
        ),
        observability=obs,
    )
    result = await wf.run(_TASK, _CTX)

    assert result.text == "blocked"
    assert result.finish_reason is FinishReason.CONTENT_FILTER
    assert ("error", {"reason": "blocked"}) in obs.events


async def test_output_redact_rewrites_the_text() -> None:
    redacted: Redact[str] = Redact(reason="pii", redacted="[clean]")
    wf = Workflow(
        llm=FakeAdapter(text="answer with a secret in it"),
        context=FakeContext(),
        guardrails=GuardrailSet(
            output_guardrails=[FakeOutputGuardrail(redacted)]
        ),
        observability=FakeObservability(),
    )
    result = await wf.run(_TASK, _CTX)

    assert result.text == "[clean]"
    assert result.finish_reason is FinishReason.STOP  # not a Deny


async def test_output_allow_not_grounded_is_reflected() -> None:
    wf = Workflow(
        llm=FakeAdapter(text="general knowledge answer"),
        context=FakeContext(),
        guardrails=GuardrailSet(
            output_guardrails=[FakeOutputGuardrail(Allow(grounded=False))]
        ),
        observability=FakeObservability(),
    )
    result = await wf.run(_TASK, _CTX)

    assert result.text == "general knowledge answer"
    assert result.grounded is False


async def test_happy_path_observability_event_order() -> None:
    obs = FakeObservability()
    wf = Workflow(
        llm=FakeAdapter(text="ok"),
        context=FakeContext(),
        guardrails=GuardrailSet(
            input_guardrails=[FakeInputGuardrail()],
            output_guardrails=[FakeOutputGuardrail()],
        ),
        observability=obs,
    )
    await wf.run(_TASK, _CTX)

    assert obs.names == [
        "execution_started",
        "policy_decision",
        "llm_call",
        "policy_decision",
        "result",
    ]
    assert obs.fields_for("policy_decision") == [
        {"phase": "input", "decision": "allow"},
        {"phase": "output", "decision": "allow"},
    ]


# --- _validate_tool_coverage (design §12) ------------------------------


def test_coverage_tools_but_no_tool_guardrail() -> None:
    with pytest.raises(PolicyConfigError, match="no ToolGuardrail"):
        _validate_tool_coverage([_tool("search_docs")], [])


def test_coverage_one_tool_uncovered_while_others_covered() -> None:
    with pytest.raises(PolicyConfigError, match=r"\bquery_table\b"):
        _validate_tool_coverage(
            [_tool("search_docs"), _tool("query_table")],
            [FakeToolGuardrail(covers=frozenset({"search_docs"}))],
        )


def test_coverage_catch_all_covers_everything() -> None:
    _validate_tool_coverage(
        [_tool("search_docs"), _tool("query_table")],
        [FakeToolGuardrail(covers=None)],
    )  # no raise


def test_coverage_catch_all_and_specific_coexist() -> None:
    _validate_tool_coverage(
        [_tool("search_docs")],
        [
            FakeToolGuardrail(covers=None),
            FakeToolGuardrail(covers=frozenset({"search_docs"})),
        ],
    )  # no raise


def test_coverage_fail_fast_is_wired_into_workflow_init() -> None:
    with pytest.raises(PolicyConfigError, match="no ToolGuardrail"):
        Workflow(
            llm=FakeAdapter(),
            context=FakeContext(),
            guardrails=GuardrailSet(),
            tools=[_tool("search_docs")],
        )
