"""Guardrail protocols — design §3 (Stage 2, v0.5).

Three protocols, one per execution phase (input / tool / output), not one
protocol with three methods: each phase sees genuinely different data (an
incoming message vs. a tool call with its result vs. an already-generated
response), and a single interface would force `Any` or artificial unions.
Validated against the real OpenAI Agents SDK (Specify §3).

The I/O methods are `async def` (`InputGuardrail.check`,
`ToolGuardrail.before`/`after`, `OutputGuardrail.check`). `Workflow.run()`
(§10) is async and runs on an event loop; a guardrail that needs an LLM
(jailbreak detection via `any-guardrail`, Specify §3) calls it through the
async `LLMPort`, like any other Core piece — never through `SyncLLM`,
which would block that loop. A guardrail that does no I/O simply never
`await`s; the only cost is the `async` keyword. `ToolGuardrail.covers`
stays sync: it is configuration metadata, not I/O.
"""

from typing import Any, Protocol

from odwi_llm.core.types import (
    ChatTurnResponse,
    LLMResponse,
    Message,
    ToolCall,
    ToolResult,
)
from odwi_llm.guardrails.types import Decision, PolicyContext


class InputGuardrail(Protocol):
    async def check(
        self, message: Message, ctx: PolicyContext
    ) -> Decision[str]: ...


class ToolGuardrail(Protocol):
    @property
    def covers(self) -> frozenset[str] | None:
        """Tool names this guardrail governs. `None` = covers ALL tools
        (catch-all). Several `ToolGuardrail`s may apply to the same tool
        at once — every one that declares coverage over it runs, not just
        one. Sync on purpose: configuration metadata, does no I/O.
        """
        ...

    async def before(
        self, call: ToolCall, ctx: PolicyContext
    ) -> Decision[dict[str, Any]]: ...

    async def after(
        self, call: ToolCall, result: ToolResult, ctx: PolicyContext
    ) -> Decision[str]: ...


class OutputGuardrail(Protocol):
    async def check(
        self, response: LLMResponse | ChatTurnResponse, ctx: PolicyContext
    ) -> Decision[str]: ...


class GuardrailSet:
    """Not Pydantic — it carries protocols/callables, not data. Same
    reason as `ProviderConfig` in `adapters/config.py`.

    Parameters and attributes use the full `*_guardrails` suffix (matching
    the OpenAI Agents SDK, and so `input`/`output` do not shadow builtins).
    """

    def __init__(
        self,
        *,
        input_guardrails: list[InputGuardrail] | None = None,
        tool_guardrails: list[ToolGuardrail] | None = None,
        output_guardrails: list[OutputGuardrail] | None = None,
    ) -> None:
        self.input_guardrails = input_guardrails or []
        self.tool_guardrails = tool_guardrails or []
        self.output_guardrails = output_guardrails or []
