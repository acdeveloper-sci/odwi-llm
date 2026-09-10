"""06 - AI Core guardrails: Allow, Deny, Redact.

Same skeleton as `05`, but the guardrails actually decide something:

  * the InputGuardrail returns `Deny` when the message contains a
    keyword  -> the run stops with WorkflowResult(finish_reason=
    CONTENT_FILTER) and the deny reason as the text;
  * the OutputGuardrail returns `Redact` when the response contains a
    token -> the Workflow swaps the redacted text into the result.

The script sends three messages so all three `Decision` variants show:
one benign (Allow / Allow), one that trips the input Deny, one crafted so
the model echoes a token the output guardrail redacts.

No API keys. It only needs a running Ollama daemon with the model pulled:

    ollama pull qwen3:0.6b

Run:

    uv run python examples/06_ai_core_guardrails.py
"""

import asyncio
from datetime import UTC, datetime
from typing import Any

from odwi_llm.adapters.litellm_adapter import LiteLLMAdapter
from odwi_llm.context.types import ContextBundle
from odwi_llm.core.requirements import LLMRequirements
from odwi_llm.core.types import ChatTurnResponse, LLMResponse, Message
from odwi_llm.guardrails.port import GuardrailSet
from odwi_llm.guardrails.types import Allow, Decision, Deny, PolicyContext, Redact
from odwi_llm.orchestration.composition import Composer
from odwi_llm.orchestration.types import WorkflowTask

PROVIDER = "ollama"
MODEL = "qwen3:0.6b"

DENY_KEYWORD = "password"
REDACT_TOKEN = "ZEBRA"


class KeywordDenyInput:
    async def check(
        self, message: Message, ctx: PolicyContext
    ) -> Decision[str]:
        # Keyed off the message, so this branch is fully deterministic.
        # A real input guardrail would run topic-scope / data-access /
        # injection checks and return the same Decision type.
        if DENY_KEYWORD in message.content.lower():
            return Deny(reason=f"input mentions {DENY_KEYWORD!r}; refused")
        return Allow()


class TokenRedactOutput:
    async def check(
        self, response: LLMResponse | ChatTurnResponse, ctx: PolicyContext
    ) -> Decision[str]:
        if REDACT_TOKEN in response.text:
            # Redact carries the replacement text; the Workflow swaps it
            # into response.text before building the WorkflowResult. For
            # tool arguments this would instead be Decision[dict[str, Any]].
            return Redact(
                reason=f"response contained {REDACT_TOKEN!r}",
                redacted=response.text.replace(REDACT_TOKEN, "[redacted]"),
            )
        return Allow()


class FixedContext:
    async def select(
        self, *, message: str | None = None, ctx: PolicyContext
    ) -> ContextBundle:
        return ContextBundle()


class PrintObservability:
    def emit(self, event: str, **fields: Any) -> None:
        detail = f"  {fields}" if fields else ""
        print(f"    * {event}{detail}")


async def main() -> None:
    llm = LiteLLMAdapter(PROVIDER, MODEL, LLMRequirements())

    comp = Composer(
        llm=llm,
        context=FixedContext(),
        guardrails=GuardrailSet(
            input_guardrails=[KeywordDenyInput()],
            output_guardrails=[TokenRedactOutput()],
        ),
        observability=PrintObservability(),
    )

    ctx = PolicyContext(
        session_id="ex-06",
        provider=PROVIDER,
        model=MODEL,
        resolved_at=datetime.now(UTC),
    )

    messages = [
        # 1. benign: both guardrails Allow, normal answer.
        "What is 2 + 2?",
        # 2. trips KeywordDenyInput -> Deny -> CONTENT_FILTER, no LLM call.
        f"My {DENY_KEYWORD} is hunter2, is that a strong one?",
        # 3. crafted so the reply reliably contains REDACT_TOKEN (a 0.6b
        #    model repeats a short string on request); TokenRedactOutput
        #    then rewrites the text. This is a demo contrivance — a real
        #    output guardrail would key off something meaningful.
        f"Repeat this sentence exactly, nothing else: 'My token is {REDACT_TOKEN}.'",
    ]

    for i, msg in enumerate(messages, 1):
        print(f"\n--- message {i}: {msg}")
        result = await comp.orchestrator.run(
            WorkflowTask(session_id=f"ex-06-{i}", input={"message": msg}), ctx
        )
        print(f"  => {result.text!r}")
        print(
            f"     [finish={result.finish_reason.value} "
            f"grounded={result.grounded}]"
        )


if __name__ == "__main__":
    asyncio.run(main())
