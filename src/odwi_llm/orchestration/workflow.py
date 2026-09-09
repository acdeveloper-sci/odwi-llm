"""Reference orchestrator — design §10 (Stage 2, v0.5).

A thin hand-rolled loop, the default `Orchestrator` implementation. It
consults Policy at defined points; it never holds security rules, it only
knows it has to ask. `run()` is async; a guardrail or `context.select`
that needs an LLM does so through the async `LLMPort`, never `SyncLLM`
(which would block this event loop).

Tool-calling is NOT here yet — that is Task 13, where `ToolGuardrail`
`before`/`after` come in. This module fixes the constructor shape and the
coverage fail-fast so Task 13 does not have to touch them.
"""

from odwi_llm.context.port import ContextPort
from odwi_llm.core.port import LLMPort
from odwi_llm.core.types import (
    FinishReason,
    LLMRequest,
    Message,
    Role,
    ToolSpec,
)
from odwi_llm.guardrails.errors import PolicyConfigError
from odwi_llm.guardrails.port import GuardrailSet, ToolGuardrail
from odwi_llm.guardrails.types import Allow, Deny, PolicyContext, Redact
from odwi_llm.observability.null import NullObservability
from odwi_llm.observability.port import ObservabilityPort
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
        max_tool_iterations: int = 6,
        observability: ObservabilityPort | None = None,
    ) -> None:
        tools = tools or []
        _validate_tool_coverage(tools, guardrails.tool_guardrails)
        self._llm = llm
        self._context = context
        self._guardrails = guardrails
        self._tools = tools
        self._max_tool_iterations = max_tool_iterations
        self._obs = observability or NullObservability()
        # The real tool-calling loop (call, run the tool via
        # ToolGuardrail.before/after — dispatching only the guardrails
        # whose `covers` includes that tool —, repeat until
        # max_tool_iterations or no more tool_calls) is Task 13. Here we
        # only fix the constructor shape and the coverage fail-fast.

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

        await self._context.select(
            message=message.content or None, ctx=ctx
        )
        # Assembling the real prompt + context is the app's job, not this
        # reference Workflow's.
        request = LLMRequest(messages=[message])
        self._obs.emit("llm_call", provider=ctx.provider, model=ctx.model)
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
                )
            if isinstance(out_decision, Redact):
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
        )
