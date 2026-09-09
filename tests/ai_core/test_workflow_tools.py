"""Task 13 — Workflow.run() tool-calling loop.

Covers: one tool call with a catch-all guardrail is allowed and executed;
a Deny in `before` skips execution; exceeding max_tool_iterations returns
a clear WorkflowResult (not an exception); a `tool_call` event per call.
Plus before/after Redact on args and result.
"""

from datetime import datetime
from typing import Any

import pytest

from odwi_llm.core.types import (
    ChatTurnResponse,
    FinishReason,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)
from odwi_llm.guardrails.errors import PolicyConfigError
from odwi_llm.guardrails.port import GuardrailSet
from odwi_llm.guardrails.types import Deny, PolicyContext, Redact
from odwi_llm.orchestration.types import WorkflowResult, WorkflowTask
from odwi_llm.orchestration.workflow import Workflow

from _fake_adapter import FakeAdapter

from ._fake_context import FakeContext
from ._fake_guardrails import FakeToolGuardrail
from ._fake_observability import FakeObservability

_CTX = PolicyContext(
    session_id="s1",
    provider="ollama",
    model="qwen3:0.6b",
    resolved_at=datetime(2026, 1, 1),
)
_TASK = WorkflowTask(session_id="s1", input={"message": "look it up"})
_USAGE = Usage(input_tokens=1, output_tokens=1)


def _tool(name: str = "search_docs") -> ToolSpec:
    return ToolSpec(name=name, description=name, parameters_schema={})


def _turn(text: str, *calls: ToolCall) -> ChatTurnResponse:
    return ChatTurnResponse(
        text=text,
        model="fake-model",
        provider="fake",
        finish_reason=(
            FinishReason.TOOL_CALLS if calls else FinishReason.STOP
        ),
        usage=_USAGE,
        tool_calls=list(calls),
    )


class ScriptedToolLLM(FakeAdapter):
    """`chat_with_tools` returns each scripted turn in order (the last one
    repeats). Everything else is inherited from FakeAdapter.
    """

    def __init__(self, turns: list[ChatTurnResponse]) -> None:
        super().__init__()
        self._turns = turns
        self.chat_calls = 0

    async def chat_with_tools(
        self, request: Any, tools: Any
    ) -> ChatTurnResponse:
        turn = self._turns[min(self.chat_calls, len(self._turns) - 1)]
        self.chat_calls += 1
        return turn


class RecordingExecutor:
    """Records each execute() call; returns a fixed ToolResult."""

    def __init__(self, content: str = "tool result") -> None:
        self._content = content
        self.calls: list[tuple[ToolCall, PolicyContext]] = []

    async def execute(
        self, call: ToolCall, ctx: PolicyContext
    ) -> ToolResult:
        self.calls.append((call, ctx))
        return ToolResult(tool_call_id=call.id, content=self._content)


def _wf(
    *,
    llm: ScriptedToolLLM,
    executor: RecordingExecutor,
    tool_guardrails: list[FakeToolGuardrail] | None = None,
    max_tool_iterations: int = 6,
    obs: FakeObservability | None = None,
) -> Workflow:
    return Workflow(
        llm=llm,
        context=FakeContext(),
        guardrails=GuardrailSet(
            tool_guardrails=list(
                tool_guardrails or [FakeToolGuardrail()]  # catch-all, allow
            )
        ),
        tools=[_tool()],
        tool_executor=executor,
        max_tool_iterations=max_tool_iterations,
        observability=obs or FakeObservability(),
    )


# --- construction fail-fast -------------------------------------------


def test_tools_without_executor_is_rejected_at_construction() -> None:
    with pytest.raises(PolicyConfigError, match="no tool_executor"):
        Workflow(
            llm=FakeAdapter(),
            context=FakeContext(),
            guardrails=GuardrailSet(tool_guardrails=[FakeToolGuardrail()]),
            tools=[_tool()],
        )


# --- the loop --------------------------------------------------------


async def test_one_tool_call_catch_all_allows_and_executes() -> None:
    call = ToolCall(id="c1", name="search_docs", arguments={"q": "x"})
    llm = ScriptedToolLLM([_turn("", call), _turn("final answer")])
    executor = RecordingExecutor()
    obs = FakeObservability()
    result = await _wf(llm=llm, executor=executor, obs=obs).run(_TASK, _CTX)

    assert result.text == "final answer"
    assert result.finish_reason is FinishReason.STOP
    assert result.tool_calls_made == 1
    assert [c.name for c, _ in executor.calls] == ["search_docs"]
    assert llm.chat_calls == 2
    assert ("tool_call", {"name": "search_docs", "id": "c1"}) in obs.events


async def test_two_tool_calls_in_one_turn_emit_two_tool_call_events() -> None:
    calls = (
        ToolCall(id="c1", name="search_docs", arguments={"q": "a"}),
        ToolCall(id="c2", name="search_docs", arguments={"q": "b"}),
    )
    llm = ScriptedToolLLM([_turn("", *calls), _turn("done")])
    executor = RecordingExecutor()
    obs = FakeObservability()
    result = await _wf(llm=llm, executor=executor, obs=obs).run(_TASK, _CTX)

    assert result.tool_calls_made == 2
    assert len(executor.calls) == 2
    assert obs.names.count("tool_call") == 2


async def test_deny_in_before_skips_execution() -> None:
    call = ToolCall(id="c1", name="search_docs", arguments={"q": "x"})
    llm = ScriptedToolLLM([_turn("", call), _turn("recovered")])
    executor = RecordingExecutor()
    obs = FakeObservability()
    guard = FakeToolGuardrail(before=Deny(reason="tool blocked"))
    result = await _wf(
        llm=llm, executor=executor, tool_guardrails=[guard], obs=obs
    ).run(_TASK, _CTX)

    assert executor.calls == []  # never executed
    assert result.text == "recovered"  # loop continued with an error result
    assert {
        "phase": "tool_before",
        "tool": "search_docs",
        "decision": "deny",
    } in obs.fields_for("policy_decision")


async def test_redact_in_before_sanitizes_the_args_passed_to_executor() -> None:
    call = ToolCall(id="c1", name="search_docs", arguments={"q": "secret"})
    llm = ScriptedToolLLM([_turn("", call), _turn("ok")])
    executor = RecordingExecutor()
    sanitized: Redact[dict[str, Any]] = Redact(
        reason="pii", redacted={"q": "[redacted]"}
    )
    guard = FakeToolGuardrail(before=sanitized)
    await _wf(llm=llm, executor=executor, tool_guardrails=[guard]).run(
        _TASK, _CTX
    )

    assert executor.calls[0][0].arguments == {"q": "[redacted]"}


async def test_exceeding_max_tool_iterations_stops_cleanly() -> None:
    call = ToolCall(id="c1", name="search_docs", arguments={})
    # only tool-call turns, forever
    llm = ScriptedToolLLM([_turn("", call)])
    executor = RecordingExecutor()
    obs = FakeObservability()
    result = await _wf(
        llm=llm, executor=executor, max_tool_iterations=2, obs=obs
    ).run(_TASK, _CTX)

    assert isinstance(result, WorkflowResult)
    assert result.finish_reason is FinishReason.OTHER
    assert "exceeded max_tool_iterations (2)" in result.text
    assert result.tool_calls_made == 2
    assert ("error", {"reason": "exceeded max_tool_iterations"}) in obs.events
