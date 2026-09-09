"""Task 4 — deterministic, in-memory guardrail fakes for the AI Core suite.

Same idea as `tests/contract/_fake_adapter.py`: one knob per behaviour,
no real policy logic. Each fake also records its calls, so a test can
assert which phases ran (e.g. a `Deny` in input means the output
guardrails were never called).

Not collected by pytest (leading underscore); imported by the `test_*.py`
files in this package.
"""

from typing import Any

from odwi_llm.core.types import (
    ChatTurnResponse,
    LLMResponse,
    Message,
    ToolCall,
    ToolResult,
)
from odwi_llm.guardrails.types import Allow, Decision, PolicyContext


class FakeInputGuardrail:
    """`check` returns `decision` every time (default `Allow()`)."""

    def __init__(self, decision: Decision[str] | None = None) -> None:
        self.decision: Decision[str] = Allow() if decision is None else decision
        self.calls: list[tuple[Message, PolicyContext]] = []

    async def check(
        self, message: Message, ctx: PolicyContext
    ) -> Decision[str]:
        self.calls.append((message, ctx))
        return self.decision


class FakeOutputGuardrail:
    """`check` returns `decision` every time (default `Allow()`)."""

    def __init__(self, decision: Decision[str] | None = None) -> None:
        self.decision: Decision[str] = Allow() if decision is None else decision
        self.calls: list[
            tuple[LLMResponse | ChatTurnResponse, PolicyContext]
        ] = []

    async def check(
        self, response: LLMResponse | ChatTurnResponse, ctx: PolicyContext
    ) -> Decision[str]:
        self.calls.append((response, ctx))
        return self.decision


class FakeToolGuardrail:
    """`covers` is configurable, including `None` (catch-all, the default).
    `before` / `after` return their configured decision every time.
    """

    def __init__(
        self,
        *,
        covers: frozenset[str] | None = None,
        before: Decision[dict[str, Any]] | None = None,
        after: Decision[str] | None = None,
    ) -> None:
        self._covers = covers
        self.before_decision: Decision[dict[str, Any]] = (
            Allow() if before is None else before
        )
        self.after_decision: Decision[str] = Allow() if after is None else after
        self.before_calls: list[tuple[ToolCall, PolicyContext]] = []
        self.after_calls: list[tuple[ToolCall, ToolResult, PolicyContext]] = []

    @property
    def covers(self) -> frozenset[str] | None:
        return self._covers

    async def before(
        self, call: ToolCall, ctx: PolicyContext
    ) -> Decision[dict[str, Any]]:
        self.before_calls.append((call, ctx))
        return self.before_decision

    async def after(
        self, call: ToolCall, result: ToolResult, ctx: PolicyContext
    ) -> Decision[str]:
        self.after_calls.append((call, result, ctx))
        return self.after_decision
