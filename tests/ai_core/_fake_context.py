"""Task 9 — a deterministic ContextPort for the AI Core suite.

Returns a fixed (or configured) `ContextBundle`, ignoring `message` — no
real selection logic, that belongs to each app. Records each call so a
test can assert `select` ran and with what.

Not collected by pytest (leading underscore); imported by the `test_*.py`
files in this package.
"""

from odwi_llm.context.types import ContextBundle
from odwi_llm.guardrails.types import PolicyContext


class FakeContext:
    """`select` returns `bundle` every time (default: an empty bundle)."""

    def __init__(self, bundle: ContextBundle | None = None) -> None:
        self.bundle = ContextBundle() if bundle is None else bundle
        self.calls: list[tuple[str | None, PolicyContext]] = []

    async def select(
        self, *, message: str | None = None, ctx: PolicyContext
    ) -> ContextBundle:
        self.calls.append((message, ctx))
        return self.bundle
