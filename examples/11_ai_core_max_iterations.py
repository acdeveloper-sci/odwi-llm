"""11 - AI Core max_tool_iterations: the worst-case bound on the tool loop.

Unlike `01`-`10`, this example does NOT talk to a real model - no Ollama,
no API key. What it tests is whether `Workflow` honors the
`max_tool_iterations` cap and always returns a `WorkflowResult` (never an
exception, never a real infinite loop) - that is a property of `Workflow`
itself, not of any model's behavior. A real model would add noise
(whether it cooperates or not) to something that has nothing to do with
that.

`temperature=0` would not help either. Greedy determinism only guarantees
the same next token given the exact same context - but the context
changes every turn of the loop (the previous tool result is appended to
it). A perfectly deterministic model can still keep asking for
variations of the same tool forever if the tool never returns something
that satisfies it. `max_tool_iterations` is a worst-case bound, the same
kind of thing as an HTTP timeout - it does not exist because a model
might "get confused"; it exists so `run()` always terminates in bounded
time, including the case where the *content* of a tool result tries to
induce the model into requesting tools indefinitely. That is a real abuse
surface, not just a cosmetic safeguard.

Implementation:

  * A minimal `LLMPort` written directly in this script (not imported
    from `tests/`), scripted so `chat_with_tools` always returns a turn
    with a `tool_calls` entry, never a final turn. Nothing else needs to
    be simulated - one tool call per turn is enough to exercise the loop.
    This is also the first example in the series to implement `LLMPort`
    itself, rather than construct `LiteLLMAdapter`. It is a minimal fake
    with scripted methods, not a formal adapter against a real library -
    but structurally the same move a real adapter makes: subclass the
    `ABC`, satisfy its abstract methods. Here that license to leave
    unfinished what this example does not need is used on purpose:
    `generate` / `structured` / `stream` / `capabilities` are stubs, since
    only `chat_with_tools` is ever called by this run.
  * A trivial `tool_executor` that returns a fixed result; its content
    does not matter for what is being tested here.
  * A permissive catch-all `ToolGuardrail` (`covers=None`) - required
    only by `_validate_tool_coverage`'s fail-fast (Task 12), not the
    focus of this example.
  * `max_tool_iterations` set low and explicit (3), so the output is
    short and easy to read.

Expected output: `llm_call` fires 3 times, then an `error` event with the
reason, and a final `WorkflowResult` with `finish_reason=OTHER`, text
`"stopped: exceeded max_tool_iterations (3)"`, and `tool_calls_made == 3`.

On verification: unlike `06`-`10`, this does not need multiple runs or
paraphrase variation. The script has no randomness anywhere - the fake
`LLMPort` always returns the same scripted turn - so the result is
deterministic by construction, and repeating the run would show nothing
new.

No API keys, no Ollama daemon. Run:

    uv run python examples/11_ai_core_max_iterations.py
"""

import asyncio
from datetime import UTC, datetime
from typing import Any

from odwi_llm.context.types import ContextBundle
from odwi_llm.core.port import LLMPort
from odwi_llm.core.requirements import LLMCapabilities
from odwi_llm.core.types import (
    ChatTurnResponse,
    FinishReason,
    LLMRequest,
    LLMResponse,
    Message,
    StreamChunk,
    StructuredResponse,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)
from odwi_llm.guardrails.port import GuardrailSet
from odwi_llm.guardrails.types import Allow, Decision, PolicyContext
from odwi_llm.orchestration.composition import Composer
from odwi_llm.orchestration.types import WorkflowTask
from odwi_llm.orchestration.workflow import Workflow

PROVIDER = "scripted"
MODEL = "always-calls-a-tool"

KEEP_LOOKING = ToolSpec(
    name="keep_looking",
    description="A tool the scripted model always decides to call again.",
    parameters_schema={"type": "object", "properties": {}},
)


class NeverSatisfiedLLM(LLMPort):
    """A minimal `LLMPort`, scripted for this one property: every call to
    `chat_with_tools` returns a turn that requests `keep_looking` again,
    never a tool-free final turn. `generate` / `structured` / `stream`
    are not exercised by this example (the Workflow only calls them when
    there are no tools) and are stubbed just enough to satisfy the ABC.
    """

    @property
    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            structured_output=False,
            tool_calling=True,
            streaming=False,
            vision=False,
            context_tokens=None,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError("not exercised by this example")

    async def structured(
        self, request: LLMRequest, schema: type[Any]
    ) -> StructuredResponse[Any]:
        raise NotImplementedError("not exercised by this example")

    def stream(self, request: LLMRequest) -> Any:
        raise NotImplementedError("not exercised by this example")

    async def chat_with_tools(
        self, request: LLMRequest, tools: list[ToolSpec]
    ) -> ChatTurnResponse:
        return ChatTurnResponse(
            text="",
            model=MODEL,
            provider=PROVIDER,
            finish_reason=FinishReason.TOOL_CALLS,
            usage=Usage(input_tokens=0, output_tokens=0),
            tool_calls=[
                ToolCall(id=f"call-{id(request)}", name="keep_looking", arguments={})
            ],
        )


class AllowInput:
    async def check(
        self, message: Message, ctx: PolicyContext
    ) -> Decision[str]:
        return Allow()


class AllowEverythingTool:
    """Catch-all (`covers=None`), needed only to satisfy
    `_validate_tool_coverage`'s fail-fast. Not the point of this example.
    """

    @property
    def covers(self) -> frozenset[str] | None:
        return None

    async def before(
        self, call: ToolCall, ctx: PolicyContext
    ) -> Decision[dict[str, Any]]:
        return Allow()

    async def after(
        self, call: ToolCall, result: ToolResult, ctx: PolicyContext
    ) -> Decision[str]:
        return Allow()


class KeepLookingExecutor:
    """The tool's own result content is irrelevant here - the scripted
    model never stops asking regardless of what comes back.
    """

    async def execute(
        self, call: ToolCall, ctx: PolicyContext
    ) -> ToolResult:
        return ToolResult(tool_call_id=call.id, content="nothing found yet")


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
    llm = NeverSatisfiedLLM()
    context = FixedContext()
    guardrails = GuardrailSet(
        input_guardrails=[AllowInput()],
        tool_guardrails=[AllowEverythingTool()],
    )
    observability = PrintObservability()

    # Composer does not expose max_tool_iterations (it is not part of the
    # composition-root surface, §11), so the Workflow is built directly
    # here, with the cap set low, and handed in via orchestrator= - the
    # same path a custom Orchestrator would take.
    comp = Composer(
        llm=llm,
        context=context,
        guardrails=guardrails,
        tools=[KEEP_LOOKING],
        tool_executor=KeepLookingExecutor(),
        observability=observability,
        orchestrator=Workflow(
            llm=llm,
            context=context,
            guardrails=guardrails,
            tools=[KEEP_LOOKING],
            tool_executor=KeepLookingExecutor(),
            max_tool_iterations=3,
            observability=observability,
        ),
    )

    ctx = PolicyContext(
        session_id="ex-11",
        provider=PROVIDER,
        model=MODEL,
        resolved_at=datetime.now(UTC),
    )
    task = WorkflowTask(
        session_id="ex-11", input={"message": "Find the answer, whatever it takes."}
    )

    print("events:")
    result = await comp.orchestrator.run(task, ctx)

    print(f"\ntext: {result.text!r}")
    print(
        f"[finish={result.finish_reason.value} "
        f"tool_calls={result.tool_calls_made}]"
    )


if __name__ == "__main__":
    asyncio.run(main())
