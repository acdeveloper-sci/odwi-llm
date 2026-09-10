"""07 - AI Core tool-calling loop.

A `ToolGuardrail` (with `covers` naming the one tool it governs), a
`tool_executor` that actually runs it, and permissive input/output
guardrails. The observability sink prints the full loop:

    llm_call -> tool_call -> policy_decision(phase=tool_before)
             -> policy_decision(phase=tool_after) -> llm_call (next turn)
             -> ... -> result

The tool is `get_time` — no arguments, no external dependency, just the
clock — so `before` has nothing to sanitize and just Allows.

Note: `qwen3:0.6b` is a small model and `chat_with_tools` uses
`tool_choice="auto"`, so it is not 100% reliable at deciding to call the
tool. If the output shows `tool_calls=0`, just run it again. (The lab
confirmed this model *can* tool-call via Ollama; see
`llm_lab/experiments/FINDINGS.md`.)

No API keys. It only needs a running Ollama daemon with the model pulled:

    ollama pull qwen3:0.6b

Run:

    uv run python examples/07_ai_core_tools.py
"""

import asyncio
from datetime import UTC, datetime
from typing import Any

from odwi_llm.adapters.litellm_adapter import LiteLLMAdapter
from odwi_llm.context.types import ContextBundle
from odwi_llm.core.requirements import LLMRequirements
from odwi_llm.core.types import (
    ChatTurnResponse,
    LLMResponse,
    Message,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from odwi_llm.guardrails.port import GuardrailSet
from odwi_llm.guardrails.types import Allow, Decision, PolicyContext
from odwi_llm.orchestration.composition import Composer
from odwi_llm.orchestration.types import WorkflowTask

PROVIDER = "ollama"
MODEL = "qwen3:0.6b"

GET_TIME = ToolSpec(
    name="get_time",
    description="Return the current UTC time as an ISO 8601 string.",
    parameters_schema={"type": "object", "properties": {}},
)


class AllowInput:
    async def check(
        self, message: Message, ctx: PolicyContext
    ) -> Decision[str]:
        return Allow()


class AllowOutput:
    async def check(
        self, response: LLMResponse | ChatTurnResponse, ctx: PolicyContext
    ) -> Decision[str]:
        return Allow()


class TimeToolGuardrail:
    @property
    def covers(self) -> frozenset[str] | None:
        # frozenset -> this guardrail governs only get_time. Return None
        # instead to make it a catch-all over every registered tool.
        return frozenset({"get_time"})

    async def before(
        self, call: ToolCall, ctx: PolicyContext
    ) -> Decision[dict[str, Any]]:
        # before() may Deny the call or Redact its arguments; get_time
        # takes none, so there is nothing to sanitize.
        return Allow()

    async def after(
        self, call: ToolCall, result: ToolResult, ctx: PolicyContext
    ) -> Decision[str]:
        # after() sees the executed result and could Deny or Redact it.
        return Allow()


class TimeExecutor:
    async def execute(
        self, call: ToolCall, ctx: PolicyContext
    ) -> ToolResult:
        # The app runs its own tools; the Workflow only invokes this.
        return ToolResult(
            tool_call_id=call.id, content=datetime.now(UTC).isoformat()
        )


class FixedContext:
    async def select(
        self, *, message: str | None = None, ctx: PolicyContext
    ) -> ContextBundle:
        return ContextBundle()


class PrintObservability:
    def emit(self, event: str, **fields: Any) -> None:
        detail = f"  {fields}" if fields else ""
        print(f"  * {event}{detail}")


async def main() -> None:
    llm = LiteLLMAdapter(PROVIDER, MODEL, LLMRequirements())

    comp = Composer(
        llm=llm,
        context=FixedContext(),
        guardrails=GuardrailSet(
            input_guardrails=[AllowInput()],
            tool_guardrails=[TimeToolGuardrail()],  # must cover every tool
            output_guardrails=[AllowOutput()],
        ),
        tools=[GET_TIME],
        tool_executor=TimeExecutor(),
        observability=PrintObservability(),
    )

    ctx = PolicyContext(
        session_id="ex-07",
        provider=PROVIDER,
        model=MODEL,
        resolved_at=datetime.now(UTC),
    )
    task = WorkflowTask(
        session_id="ex-07",
        input={
            "message": (
                "You have a tool called get_time. You MUST call get_time "
                "to answer. What is the current time?"
            )
        },
    )

    print("events:")
    result = await comp.orchestrator.run(task, ctx)

    print(f"\ntext: {result.text}")
    print(
        f"[finish={result.finish_reason.value} "
        f"tool_calls={result.tool_calls_made}]"
    )


if __name__ == "__main__":
    asyncio.run(main())
