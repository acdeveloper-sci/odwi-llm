"""Task 14 — Composer assembles a default Workflow with everything
forwarded, and construction fails fast on a bad tool setup.
"""

from datetime import datetime

import pytest

from odwi_llm.core.types import (
    FinishReason,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from odwi_llm.guardrails.errors import PolicyConfigError
from odwi_llm.guardrails.port import GuardrailSet
from odwi_llm.guardrails.types import PolicyContext
from odwi_llm.observability.null import NullObservability
from odwi_llm.orchestration.composition import Composer
from odwi_llm.orchestration.types import WorkflowResult, WorkflowTask
from odwi_llm.orchestration.workflow import Workflow

from _fake_adapter import FakeAdapter

from ._fake_context import FakeContext
from ._fake_guardrails import FakeToolGuardrail
from ._fake_observability import FakeObservability

_CTX = PolicyContext(
    session_id="s1",
    provider="fake",
    model="fake-model",
    resolved_at=datetime(2026, 1, 1),
)


def _tool(name: str = "search_docs") -> ToolSpec:
    return ToolSpec(name=name, description=name, parameters_schema={})


class _Executor:
    async def execute(
        self, call: ToolCall, ctx: PolicyContext
    ) -> ToolResult:
        return ToolResult(tool_call_id=call.id, content="ok")


def test_default_orchestrator_is_a_workflow_with_everything_forwarded() -> None:
    llm = FakeAdapter()
    context = FakeContext()
    guardrails = GuardrailSet()
    obs = FakeObservability()

    comp = Composer(
        llm=llm, context=context, guardrails=guardrails, observability=obs
    )

    assert isinstance(comp.orchestrator, Workflow)
    assert comp.orchestrator._llm is llm
    assert comp.orchestrator._context is context
    assert comp.orchestrator._guardrails is guardrails
    assert comp.orchestrator._obs is obs
    assert comp.observability is obs


def test_default_observability_is_null_and_shared_with_the_workflow() -> None:
    comp = Composer(
        llm=FakeAdapter(), context=FakeContext(), guardrails=GuardrailSet()
    )
    assert isinstance(comp.observability, NullObservability)
    assert isinstance(comp.orchestrator, Workflow)
    assert comp.orchestrator._obs is comp.observability


def test_a_custom_orchestrator_is_used_as_is() -> None:
    class _Stub:
        async def run(
            self, task: WorkflowTask, ctx: PolicyContext
        ) -> WorkflowResult:
            return WorkflowResult(text="x", finish_reason=FinishReason.STOP)

    stub = _Stub()
    comp = Composer(
        llm=FakeAdapter(),
        context=FakeContext(),
        guardrails=GuardrailSet(),
        orchestrator=stub,
    )
    assert comp.orchestrator is stub


def test_tools_without_covering_guardrail_fails_at_composer_construction() -> None:
    with pytest.raises(PolicyConfigError, match="no ToolGuardrail"):
        Composer(
            llm=FakeAdapter(),
            context=FakeContext(),
            guardrails=GuardrailSet(),
            tools=[_tool()],
            tool_executor=_Executor(),
        )


def test_tools_without_executor_fails_at_composer_construction() -> None:
    with pytest.raises(PolicyConfigError, match="no tool_executor"):
        Composer(
            llm=FakeAdapter(),
            context=FakeContext(),
            guardrails=GuardrailSet(tool_guardrails=[FakeToolGuardrail()]),
            tools=[_tool()],
        )


def test_tools_with_guardrail_and_executor_are_forwarded() -> None:
    executor = _Executor()
    comp = Composer(
        llm=FakeAdapter(),
        context=FakeContext(),
        guardrails=GuardrailSet(tool_guardrails=[FakeToolGuardrail()]),
        tools=[_tool()],
        tool_executor=executor,
    )
    assert isinstance(comp.orchestrator, Workflow)
    assert [t.name for t in comp.orchestrator._tools] == ["search_docs"]
    assert comp.orchestrator._tool_executor is executor
