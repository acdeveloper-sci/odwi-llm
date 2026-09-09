"""Smoke test for the context fake itself (Task 9)."""

from datetime import datetime

from odwi_llm.context.port import ContextPort
from odwi_llm.context.types import ContextBundle
from odwi_llm.guardrails.types import PolicyContext

from ._fake_context import FakeContext

_CTX = PolicyContext(
    session_id="s1",
    provider="fake",
    model="fake-model",
    resolved_at=datetime(2026, 1, 1),
)


def test_satisfies_the_port() -> None:
    c: ContextPort = FakeContext()
    assert isinstance(c, FakeContext)


async def test_returns_the_configured_bundle_ignoring_message() -> None:
    bundle = ContextBundle(knowledge={"k": 1}, data={"run": {"score": 9}})
    c = FakeContext(bundle)

    assert await c.select(ctx=_CTX) is bundle
    assert await c.select(message="anything", ctx=_CTX) is bundle


async def test_defaults_to_an_empty_bundle_and_records_calls() -> None:
    c = FakeContext()
    out = await c.select(message="hi", ctx=_CTX)

    assert out == ContextBundle()
    assert c.calls == [("hi", _CTX)]
