"""15 - AI Core structured output: the "schema only" path.

Design v0.7 (Fase F, just closed) let `Workflow`/`Composer` accept a
`schema=` orthogonal to `tools` — four dispatch paths depending on
whether `tools`/`schema` are configured (see `orchestration/workflow.py`
and `CHANGELOG.md` 0.3.0). This example shows the simplest of those four:
`schema` alone, no `tools`. Same role `05`/`06` played for their pieces of
the Core — one new piece at a time, before combining it with anything
else. Combining `schema` with the tool-calling loop is a later example,
not this one.

Domain: extracting a small, fixed structure out of a free-text meeting
request - date, location, reason. No guardrails with real rules, no
tools - the only thing this example checks is that `result.data` comes
back populated and validated against `EventRequest`, via `Composer`, not
by calling `structured()` directly (that already has Stage 1 coverage in
`03_structured_output.py` - this example is about the Stage 2 dispatch
path, not `structured()` itself).

With `schema=EventRequest` and no `tools`, `Workflow.run()` calls
`llm.structured(request, schema)` directly - no tool-calling loop at all
(the "only tools" and "both" paths are not exercised here). Output
guardrails still run against the response the normal way; a `Redact` from
one of them would be treated as `Deny` fail-closed once `.data` is
populated (Design v0.7) - not exercised here since guardrails are
permissive, but see `08_ai_core_context_patterns.py`'s docstring for the
general lesson about guardrail policy being the app's job, not this
library's.

No API keys. It only needs a running Ollama daemon with the model pulled:

    ollama pull qwen3:0.6b

Run:

    uv run python examples/15_structured_output.py
"""

import asyncio
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from odwi_llm.adapters.litellm_adapter import LiteLLMAdapter
from odwi_llm.context.types import ContextBundle
from odwi_llm.core.requirements import LLMRequirements
from odwi_llm.core.types import ChatTurnResponse, LLMResponse, Message
from odwi_llm.guardrails.port import GuardrailSet
from odwi_llm.guardrails.types import Allow, Decision, PolicyContext
from odwi_llm.orchestration.composition import Composer
from odwi_llm.orchestration.types import WorkflowTask

PROVIDER = "ollama"
MODEL = "qwen3:0.6b"


class EventRequest(BaseModel):
    date: str
    location: str
    reason: str


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
    llm = LiteLLMAdapter(PROVIDER, MODEL, LLMRequirements(structured_output=True))

    comp = Composer(
        llm=llm,
        context=FixedContext(),
        guardrails=GuardrailSet(
            input_guardrails=[AllowInput()],
            output_guardrails=[AllowOutput()],
        ),
        # No tools here on purpose - the "schema only" path, the simplest
        # of the four the dispatch table covers (Design v0.7 §10).
        schema=EventRequest,
        observability=PrintObservability(),
    )

    ctx = PolicyContext(
        session_id="ex-15",
        provider=PROVIDER,
        model=MODEL,
        resolved_at=datetime.now(UTC),
    )
    task = WorkflowTask(
        session_id="ex-15",
        input={
            "message": (
                "We need to meet on Tuesday at the office to review the "
                "budget."
            )
        },
    )

    print("events:")
    result = await comp.orchestrator.run(task, ctx)

    print(f"\ntext: {result.text!r}")
    print(f"[finish={result.finish_reason.value}]")
    print(f"data: {result.data!r}")
    assert isinstance(result.data, EventRequest), (
        "schema-only path must populate WorkflowResult.data"
    )
    print(
        f"  date={result.data.date!r} location={result.data.location!r} "
        f"reason={result.data.reason!r}"
    )


if __name__ == "__main__":
    asyncio.run(main())
