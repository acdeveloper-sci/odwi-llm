"""05 - AI Core, the smallest path through Composer.

Stage 2 wires policy, context and observability around an LLMPort.
This is the minimum: guardrails that always `Allow`, a context source
that returns a fixed bundle, and an observability sink that prints every
event so the lifecycle is visible:

    execution_started -> policy_decision(input) -> llm_call
                      -> policy_decision(output) -> result

No tools. One `run()` call.

No API keys. It only needs a running Ollama daemon with the model pulled:

    ollama pull qwen3:0.6b

Run:

    uv run python examples/05_ai_core_minimal.py
"""

import asyncio
from datetime import UTC, datetime
from typing import Any

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


class AllowEverythingInput:
    # A guardrail is just an object with the right async method — there is
    # no base class to inherit. This one never blocks anything; a real one
    # would do topic-scope / data-access / injection checks here.
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
    # The reference Workflow does not fold this bundle into the prompt
    # (assembling the real prompt is the app's job), but it does call
    # select() every turn — this is where retrieval / RAG would live.
    async def select(
        self, *, message: str | None = None, ctx: PolicyContext
    ) -> ContextBundle:
        return ContextBundle(knowledge={"assistant_name": "odwi"})


class PrintObservability:
    # Not NullObservability: printing each event is the whole point of
    # this example. A real sink would forward to OpenTelemetry / Langfuse
    # / structured logging instead.
    def emit(self, event: str, **fields: Any) -> None:
        detail = f"  {fields}" if fields else ""
        print(f"  * {event}{detail}")


async def main() -> None:
    llm = LiteLLMAdapter(PROVIDER, MODEL, LLMRequirements())

    # Composer assembles already-built pieces; without orchestrator= it
    # builds the default Workflow and forwards everything to it.
    comp = Composer(
        llm=llm,
        context=FixedContext(),
        guardrails=GuardrailSet(
            input_guardrails=[AllowEverythingInput()],
            output_guardrails=[AllowEverythingOutput()],
        ),
        observability=PrintObservability(),
    )

    # PolicyContext is frozen for the run: it records which provider/model
    # produced the result and is never re-read mid-conversation.
    ctx = PolicyContext(
        session_id="ex-05",
        provider=PROVIDER,
        model=MODEL,
        resolved_at=datetime.now(UTC),
    )
    task = WorkflowTask(
        session_id="ex-05", input={"message": "Name three primary colors."}
    )

    print("events:")
    result = await comp.orchestrator.run(task, ctx)

    print(f"\ntext: {result.text}")
    print(
        f"[finish={result.finish_reason.value} "
        f"grounded={result.grounded} tool_calls={result.tool_calls_made}]"
    )


if __name__ == "__main__":
    asyncio.run(main())
