"""12 - AI Core composition seams: two ways to plug into `Composer`.

Unlike `05`-`11`, this example introduces no new domain and no new
guardrail pattern - the focus is `Composer`'s own composition seam. Two
independent parts in the same script.

Part 1: `Composer(orchestrator=...)` with a custom `Orchestrator`,
not `Workflow`. `RawGenerateOrchestrator` skips every Stage 2 phase - no
guardrails, no context, no tool loop - and calls `llm.generate()`
directly from `task.input["message"]`. See its own docstring for why
this is a real, narrow case rather than a contrived one, and why it is
not a general recommendation.

Part 2: `Composer(llm=FallbackLLM(...))` with the default `Workflow`
(no `orchestrator=` passed). Same primary/fallback pair as
`04_fallback.py` - the point here is not to re-test `FallbackLLM` (Stage
1 already does), it is to show that Stage 2 accepts any composed
`LLMPort` from Stage 1 without friction: `Composer` only ever sees the
`LLMPort` interface, never `FallbackLLM` itself.

No API keys. Part 1 and Part 2 both only need a running Ollama daemon
with the model pulled - LM Studio is not required unless the primary in
Part 2 actually fails (same happy-path-only note as `04`):

    ollama pull qwen3:0.6b

Run:

    uv run python examples/12_ai_core_composition_seams.py
"""

import asyncio
from datetime import UTC, datetime
from typing import Any

from odwi_llm.adapters.litellm_adapter import LiteLLMAdapter
from odwi_llm.context.types import ContextBundle
from odwi_llm.core.composition import FallbackLLM
from odwi_llm.core.port import LLMPort
from odwi_llm.core.requirements import LLMRequirements
from odwi_llm.core.types import ChatTurnResponse, LLMRequest, LLMResponse, Message, Role
from odwi_llm.guardrails.port import GuardrailSet
from odwi_llm.guardrails.types import Allow, Decision, PolicyContext
from odwi_llm.orchestration.composition import Composer
from odwi_llm.orchestration.types import WorkflowResult, WorkflowTask

PROVIDER = "ollama"
MODEL = "qwen3:0.6b"


class AllowEverythingInput:
    async def check(
        self, message: Message, ctx: PolicyContext
    ) -> Decision[str]:
        return Allow()


class AllowEverythingOutput:
    async def check(
        self, response: LLMResponse | ChatTurnResponse, ctx: PolicyContext
    ) -> Decision[str]:
        return Allow()


class FixedContext:
    async def select(
        self, *, message: str | None = None, ctx: PolicyContext
    ) -> ContextBundle:
        return ContextBundle()


class PrintObservability:
    def emit(self, event: str, **fields: Any) -> None:
        detail = f"  {fields}" if fields else ""
        print(f"  * {event}{detail}")


class RawGenerateOrchestrator:
    """Skips input policy, context, the tool loop and output policy
    entirely - calls `llm.generate()` straight from the task input.

    Real motivating case, not a contrived one: an app that standardized
    all its code on `Composer` / `WorkflowTask` / `WorkflowResult` for
    consistency, but has one narrow endpoint - a health check, an
    internal debug utility - that genuinely needs no policy phase at all.
    Implementing `Orchestrator` here keeps that one endpoint on the same
    call shape as the rest of the app, instead of reaching for
    `LiteLLMAdapter` directly and breaking the pattern.

    This is NOT a general recommendation. It skips every policy phase on
    purpose, and a real custom `Orchestrator` would normally still want
    to enforce something - not necessarily through `GuardrailSet`, but
    something. Reach for this shape only when the case for "no policy at
    all" is as deliberate and narrow as the one above.
    """

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    async def run(
        self, task: WorkflowTask, ctx: PolicyContext
    ) -> WorkflowResult:
        message = str(task.input.get("message", ""))
        response = await self._llm.generate(
            LLMRequest(messages=[Message(role=Role.USER, content=message)])
        )
        return WorkflowResult(
            text=response.text, finish_reason=response.finish_reason
        )


async def part1_custom_orchestrator() -> None:
    print("=== part 1: Composer(orchestrator=RawGenerateOrchestrator(...)) ===")
    llm = LiteLLMAdapter(PROVIDER, MODEL, LLMRequirements())

    comp = Composer(
        llm=llm,
        # context= and guardrails= are still required by Composer's
        # signature (Task 14) even though RawGenerateOrchestrator never
        # touches them - they are unused here, not optional to omit.
        context=FixedContext(),
        guardrails=GuardrailSet(),
        orchestrator=RawGenerateOrchestrator(llm),
    )

    ctx = PolicyContext(
        session_id="ex-12-1",
        provider=PROVIDER,
        model=MODEL,
        resolved_at=datetime.now(UTC),
    )
    task = WorkflowTask(
        session_id="ex-12-1", input={"message": "Name three primary colors."}
    )
    result = await comp.orchestrator.run(task, ctx)
    print(f"text: {result.text!r}")
    print(f"[finish={result.finish_reason.value}]")


async def part2_default_workflow_over_fallback() -> None:
    print("\n=== part 2: Composer(llm=FallbackLLM(...)), default Workflow ===")
    primary = LiteLLMAdapter(PROVIDER, MODEL, LLMRequirements())
    fallback = LiteLLMAdapter(
        "lmstudio", "llama-3.2-3b-instruct", LLMRequirements()
    )
    # Composer only ever sees this as an LLMPort - it has no idea a
    # fallback exists behind it. Happy path only, same as 04: the
    # fallback is never contacted here, so LM Studio does not need to be
    # running for this to work.
    llm = FallbackLLM(primary=primary, fallback=fallback)

    comp = Composer(
        llm=llm,
        context=FixedContext(),
        guardrails=GuardrailSet(
            input_guardrails=[AllowEverythingInput()],
            output_guardrails=[AllowEverythingOutput()],
        ),
        observability=PrintObservability(),
    )

    ctx = PolicyContext(
        session_id="ex-12-2",
        provider=PROVIDER,
        model=MODEL,
        resolved_at=datetime.now(UTC),
    )
    task = WorkflowTask(
        session_id="ex-12-2", input={"message": "Name three primary colors."}
    )
    print("events:")
    result = await comp.orchestrator.run(task, ctx)
    print(f"\ntext: {result.text!r}")
    print(f"[finish={result.finish_reason.value} grounded={result.grounded}]")


async def main() -> None:
    await part1_custom_orchestrator()
    await part2_default_workflow_over_fallback()


if __name__ == "__main__":
    asyncio.run(main())
