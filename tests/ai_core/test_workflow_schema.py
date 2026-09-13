"""Task 18 — Workflow schema dispatch (design v0.7).

Covers the four `tools`/`schema` dispatch paths (design §10 table) and the
Redact-against-structured-response fail-closed rule. Exceeding
`max_tool_iterations` with `schema` configured must still never call
`structured()` — verified with a call-counting spy, not just by
inspecting the result.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel

from odwi_llm.core.types import (
    ChatTurnResponse,
    FinishReason,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)
from odwi_llm.guardrails.port import GuardrailSet
from odwi_llm.guardrails.types import PolicyContext, Redact
from odwi_llm.orchestration.types import WorkflowTask
from odwi_llm.orchestration.workflow import Workflow

from _fake_adapter import FakeAdapter

from ._fake_context import FakeContext
from ._fake_guardrails import FakeOutputGuardrail, FakeToolGuardrail
from ._fake_observability import FakeObservability

_CTX = PolicyContext(
    session_id="s1",
    provider="ollama",
    model="qwen3:0.6b",
    resolved_at=datetime(2026, 1, 1),
)
_TASK = WorkflowTask(session_id="s1", input={"message": "hello"})
_USAGE = Usage(input_tokens=1, output_tokens=1)


class Answer(BaseModel):
    value: str


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


class RecordingExecutor:
    async def execute(self, call: ToolCall, ctx: PolicyContext) -> ToolResult:
        return ToolResult(tool_call_id=call.id, content="tool result")


class SpyStructuredLLM(FakeAdapter):
    """Scripts `chat_with_tools` (when `turns` is given) and counts
    `structured()` calls — the max_tool_iterations+schema case needs to
    prove `structured()` was never reached, not just infer it from the
    result.
    """

    def __init__(
        self, *, turns: list[ChatTurnResponse] | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._turns = turns
        self.chat_calls = 0
        self.structured_calls = 0

    async def chat_with_tools(
        self, request: Any, tools: Any
    ) -> ChatTurnResponse:
        assert self._turns is not None
        turn = self._turns[min(self.chat_calls, len(self._turns) - 1)]
        self.chat_calls += 1
        return turn

    async def structured(self, request: Any, schema: Any) -> Any:
        self.structured_calls += 1
        return await super().structured(request, schema)


# --- the four dispatch paths (design §10 table) -------------------------


async def test_no_tools_no_schema_uses_generate_unchanged() -> None:
    llm = SpyStructuredLLM(text="plain text")
    wf = Workflow(llm=llm, context=FakeContext(), guardrails=GuardrailSet())
    result = await wf.run(_TASK, _CTX)

    assert result.text == "plain text"
    assert result.data is None
    assert llm.structured_calls == 0


async def test_schema_only_calls_structured_directly_no_loop() -> None:
    llm = SpyStructuredLLM(structured_payload={"value": "42"})
    wf = Workflow(
        llm=llm, context=FakeContext(), guardrails=GuardrailSet(), schema=Answer
    )
    result = await wf.run(_TASK, _CTX)

    assert isinstance(result.data, Answer)
    assert result.data.value == "42"
    assert llm.structured_calls == 1
    assert llm.chat_calls == 0


async def test_tools_only_is_unchanged_no_data() -> None:
    call = ToolCall(id="c1", name="search_docs", arguments={})
    llm = SpyStructuredLLM(turns=[_turn("", call), _turn("final answer")])
    wf = Workflow(
        llm=llm,
        context=FakeContext(),
        guardrails=GuardrailSet(tool_guardrails=[FakeToolGuardrail()]),
        tools=[_tool()],
        tool_executor=RecordingExecutor(),
    )
    result = await wf.run(_TASK, _CTX)

    assert result.text == "final answer"
    assert result.data is None
    assert llm.structured_calls == 0


async def test_tools_and_schema_calls_structured_after_clean_close() -> None:
    call = ToolCall(id="c1", name="search_docs", arguments={})
    llm = SpyStructuredLLM(
        turns=[_turn("", call), _turn("ignored free text")],
        structured_payload={"value": "from tools"},
    )
    wf = Workflow(
        llm=llm,
        context=FakeContext(),
        guardrails=GuardrailSet(tool_guardrails=[FakeToolGuardrail()]),
        tools=[_tool()],
        tool_executor=RecordingExecutor(),
        schema=Answer,
    )
    result = await wf.run(_TASK, _CTX)

    assert isinstance(result.data, Answer)
    assert result.data.value == "from tools"
    assert llm.structured_calls == 1
    # the tool-call turn, then the clean-close (no more tool_calls) turn
    assert llm.chat_calls == 2


async def test_max_tool_iterations_with_schema_never_calls_structured() -> None:
    call = ToolCall(id="c1", name="search_docs", arguments={})
    # only tool-call turns, forever -> loop never closes cleanly
    llm = SpyStructuredLLM(
        turns=[_turn("", call)], structured_payload={"value": "unused"}
    )
    wf = Workflow(
        llm=llm,
        context=FakeContext(),
        guardrails=GuardrailSet(tool_guardrails=[FakeToolGuardrail()]),
        tools=[_tool()],
        tool_executor=RecordingExecutor(),
        schema=Answer,
        max_tool_iterations=2,
    )
    result = await wf.run(_TASK, _CTX)

    assert result.finish_reason is FinishReason.OTHER
    assert result.data is None
    assert llm.structured_calls == 0


# --- Redact against a structured final response (design §10, v0.7) ------


async def test_redact_against_structured_response_is_fail_closed() -> None:
    redacted: Redact[str] = Redact(reason="pii", redacted="[clean]")
    llm = SpyStructuredLLM(structured_payload={"value": "42"})
    obs = FakeObservability()
    wf = Workflow(
        llm=llm,
        context=FakeContext(),
        guardrails=GuardrailSet(
            output_guardrails=[FakeOutputGuardrail(redacted)]
        ),
        schema=Answer,
        observability=obs,
    )
    result = await wf.run(_TASK, _CTX)

    assert result.finish_reason is FinishReason.CONTENT_FILTER
    assert "redact not supported on structured output" in result.text
    assert "pii" in result.text
    assert result.data is None
    assert any(name == "error" for name in obs.names)


async def test_redact_against_non_structured_response_still_rewrites_text() -> None:
    # No schema configured -> response is a plain LLMResponse, so the
    # v0.7 fail-closed rule must NOT apply here; Redact behaves as before.
    redacted: Redact[str] = Redact(reason="pii", redacted="[clean]")
    llm = SpyStructuredLLM(text="answer with a secret")
    wf = Workflow(
        llm=llm,
        context=FakeContext(),
        guardrails=GuardrailSet(
            output_guardrails=[FakeOutputGuardrail(redacted)]
        ),
    )
    result = await wf.run(_TASK, _CTX)

    assert result.text == "[clean]"
    assert result.finish_reason is FinishReason.STOP
