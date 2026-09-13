"""Reference orchestrator — design §10 (Stage 2, v0.7).

A thin hand-rolled loop, the default `Orchestrator` implementation. It
consults Policy at defined points; it never holds security rules, it only
knows it has to ask. `run()` is async; a guardrail, `context.select` or a
tool executor that needs an LLM does so through the async `LLMPort`, never
`SyncLLM` (which would block this event loop).

When `tools` are registered, `run()` drives a bounded tool-calling loop:
`chat_with_tools` -> for each `ToolCall`, the covering `ToolGuardrail`s'
`before` (may `Deny` or `Redact` the args), then `tool_executor.execute`
(the app runs the tool, never this Workflow), then the covering
guardrails' `after` (may `Deny` or `Redact` the result), feed results
back, repeat until no more tool calls or `max_tool_iterations`.

`schema` (v0.7) is orthogonal to `tools` — four dispatch paths, not two:
neither -> `generate()`; only `schema` -> `structured()` directly, no
loop; only `tools` -> the loop above, unchanged; both -> the loop until a
clean close (no more `tool_calls`), then one extra `structured()` call
with the accumulated message history. If the loop instead stops on
`max_tool_iterations`, `structured()` is never called — the loop did not
close cleanly, and forcing a validated call over a state the model never
declared finished would be the orchestrator deciding on its behalf, the
same principle behind the `tool_before`/`tool_after` `Deny` asymmetry
(v0.6). A `Redact` from an output guardrail against a structured final
response is treated as `Deny` (fail-closed): `Redact` only ever rewrites
`.text`, never `.data`, and a consumer reading `.data` — the whole point
of this path — would silently bypass the redaction otherwise.
"""

from pydantic import BaseModel

from odwi_llm.context.port import ContextPort
from odwi_llm.core.port import LLMPort
from odwi_llm.core.types import (
    ChatTurnResponse,
    FinishReason,
    LLMRequest,
    LLMResponse,
    Message,
    Role,
    StructuredResponse,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from odwi_llm.guardrails.errors import PolicyConfigError
from odwi_llm.guardrails.port import GuardrailSet, ToolGuardrail
from odwi_llm.guardrails.types import Allow, Deny, PolicyContext, Redact
from odwi_llm.observability.null import NullObservability
from odwi_llm.observability.port import ObservabilityPort
from odwi_llm.orchestration.port import ToolExecutor
from odwi_llm.orchestration.types import WorkflowResult, WorkflowTask


def _validate_tool_coverage(
    tools: list[ToolSpec], guardrails: list[ToolGuardrail]
) -> None:
    """Fail-fast at construction, not at runtime — same pattern as
    `CapabilityError` in `core/` (Stage 1). Coarse check by default: is
    there AT LEAST one guardrail covering each tool? `covers=None` on a
    catch-all guardrail already satisfies this for all of them. Fine
    granularity (a dedicated guardrail per tool) is the SAME validation,
    just with more specific guardrails — not a second mechanism.

    Sync on purpose while the rest of `Workflow` is async: it runs in
    `__init__`, does no I/O, only inspects each guardrail's `covers`.
    """
    if not tools:
        return
    if not guardrails:
        raise PolicyConfigError("tools registered but no ToolGuardrail provided")
    uncovered = {
        t.name
        for t in tools
        if not any(g.covers is None or t.name in g.covers for g in guardrails)
    }
    if uncovered:
        raise PolicyConfigError(
            f"tools without covering guardrail: {sorted(uncovered)}"
        )


class Workflow:
    """The default `Orchestrator` (Specify §4): a thin, hand-rolled loop.
    Consults Policy at the defined points; never holds security rules.
    """

    def __init__(
        self,
        *,
        llm: LLMPort,
        context: ContextPort,
        guardrails: GuardrailSet,
        tools: list[ToolSpec] | None = None,
        tool_executor: ToolExecutor | None = None,
        max_tool_iterations: int = 6,
        schema: type[BaseModel] | None = None,
        observability: ObservabilityPort | None = None,
    ) -> None:
        tools = tools or []
        _validate_tool_coverage(tools, guardrails.tool_guardrails)
        if tools and tool_executor is None:
            raise PolicyConfigError(
                "tools registered but no tool_executor provided"
            )
        self._llm = llm
        self._context = context
        self._guardrails = guardrails
        self._tools = tools
        self._tool_executor = tool_executor
        self._max_tool_iterations = max_tool_iterations
        self._schema = schema
        self._obs = observability or NullObservability()

    async def run(
        self, task: WorkflowTask, ctx: PolicyContext
    ) -> WorkflowResult:
        self._obs.emit("execution_started", session_id=task.session_id)
        message = Message(
            role=Role.USER, content=str(task.input.get("message", ""))
        )

        for in_guard in self._guardrails.input_guardrails:
            in_decision = await in_guard.check(message, ctx)
            self._obs.emit(
                "policy_decision", phase="input", decision=in_decision.kind
            )
            if isinstance(in_decision, Deny):
                self._obs.emit("error", reason=in_decision.reason)
                return WorkflowResult(
                    text=in_decision.reason,
                    finish_reason=FinishReason.CONTENT_FILTER,
                )

        await self._context.select(message=message.content or None, ctx=ctx)
        # Assembling the real prompt + context is the app's job, not this
        # reference Workflow's.

        response: LLMResponse
        messages = [message]
        if self._tools:
            outcome = await self._tool_loop(messages, ctx)
            if isinstance(outcome, WorkflowResult):
                return outcome  # hit max_tool_iterations — schema not forced
            turn, tool_calls_made = outcome
            if self._schema is not None:
                # Clean close: the loop stopped because the model made no
                # more tool_calls, so the accumulated history (including
                # every Role.TOOL result message) is ready for a schema-
                # validated final answer.
                self._obs.emit(
                    "llm_call", provider=ctx.provider, model=ctx.model
                )
                response = await self._llm.structured(
                    LLMRequest(messages=list(messages)), self._schema
                )
            else:
                response = turn
        else:
            tool_calls_made = 0
            request = LLMRequest(messages=messages)
            self._obs.emit("llm_call", provider=ctx.provider, model=ctx.model)
            if self._schema is not None:
                response = await self._llm.structured(request, self._schema)
            else:
                response = await self._llm.generate(request)

        grounded = True
        for out_guard in self._guardrails.output_guardrails:
            out_decision = await out_guard.check(response, ctx)
            self._obs.emit(
                "policy_decision", phase="output", decision=out_decision.kind
            )
            if isinstance(out_decision, Deny):
                self._obs.emit("error", reason=out_decision.reason)
                return WorkflowResult(
                    text=out_decision.reason,
                    finish_reason=FinishReason.CONTENT_FILTER,
                    tool_calls_made=tool_calls_made,
                )
            if isinstance(out_decision, Redact):
                if isinstance(response, StructuredResponse):
                    # (v0.7) Redact rewrites .text only, never .data — a
                    # consumer reading .data would silently bypass it.
                    # Fail-closed instead of pretending the redaction held.
                    reason = (
                        "redact not supported on structured output "
                        f"(guardrail requested: {out_decision.reason})"
                    )
                    self._obs.emit("error", reason=reason)
                    return WorkflowResult(
                        text=reason,
                        finish_reason=FinishReason.CONTENT_FILTER,
                        tool_calls_made=tool_calls_made,
                    )
                response = response.model_copy(
                    update={"text": out_decision.redacted}
                )
            if isinstance(out_decision, Allow) and not out_decision.grounded:
                # the LAST non-grounded Allow wins; several OutputGuardrails
                # could disagree
                grounded = False

        self._obs.emit("result", finish_reason=response.finish_reason.value)
        return WorkflowResult(
            text=response.text,
            finish_reason=response.finish_reason,
            grounded=grounded,
            tool_calls_made=tool_calls_made,
            data=response.data if isinstance(response, StructuredResponse) else None,
        )

    # --- tool-calling loop ------------------------------------------

    async def _tool_loop(
        self, messages: list[Message], ctx: PolicyContext
    ) -> tuple[ChatTurnResponse, int] | WorkflowResult:
        """Returns the final tool-free turn and the number of tool calls
        made, or a `WorkflowResult` if `max_tool_iterations` was hit.
        """
        assert self._tool_executor is not None  # guaranteed by __init__
        tool_calls_made = 0
        for _ in range(self._max_tool_iterations):
            self._obs.emit(
                "llm_call", provider=ctx.provider, model=ctx.model
            )
            turn = await self._llm.chat_with_tools(
                LLMRequest(messages=list(messages)), self._tools
            )
            if not turn.tool_calls:
                return turn, tool_calls_made

            messages.append(
                Message(role=Role.ASSISTANT, content=turn.text)
            )
            for call in turn.tool_calls:
                tool_calls_made += 1
                self._obs.emit("tool_call", name=call.name, id=call.id)
                result = await self._run_tool_call(call, ctx)
                messages.append(
                    Message(
                        role=Role.TOOL,
                        content=result.content,
                        tool_call_id=result.tool_call_id,
                        name=call.name,
                    )
                )

        self._obs.emit("error", reason="exceeded max_tool_iterations")
        return WorkflowResult(
            text=(
                f"stopped: exceeded max_tool_iterations "
                f"({self._max_tool_iterations})"
            ),
            finish_reason=FinishReason.OTHER,
            tool_calls_made=tool_calls_made,
        )

    async def _run_tool_call(
        self, call: ToolCall, ctx: PolicyContext
    ) -> ToolResult:
        assert self._tool_executor is not None  # guaranteed by __init__
        covering = [
            tg
            for tg in self._guardrails.tool_guardrails
            if tg.covers is None or call.name in tg.covers
        ]

        args = call.arguments
        for tg in covering:
            before = await tg.before(call, ctx)
            self._obs.emit(
                "policy_decision",
                phase="tool_before",
                tool=call.name,
                decision=before.kind,
            )
            if isinstance(before, Deny):
                return ToolResult(
                    tool_call_id=call.id, content=before.reason, is_error=True
                )
            if isinstance(before, Redact):
                args = before.redacted

        effective = (
            call
            if args is call.arguments
            else call.model_copy(update={"arguments": args})
        )
        result = await self._tool_executor.execute(effective, ctx)

        for tg in covering:
            after = await tg.after(effective, result, ctx)
            self._obs.emit(
                "policy_decision",
                phase="tool_after",
                tool=call.name,
                decision=after.kind,
            )
            if isinstance(after, Deny):
                return ToolResult(
                    tool_call_id=result.tool_call_id,
                    content=after.reason,
                    is_error=True,
                )
            if isinstance(after, Redact):
                result = ToolResult(
                    tool_call_id=result.tool_call_id, content=after.redacted
                )

        return result
