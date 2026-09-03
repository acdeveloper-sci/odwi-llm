"""Contract: generate — plain text, and Intent is accepted (§4.1, §5.2)."""

import pytest

from odwi_llm.core.types import FinishReason, Intent, LLMRequest, LLMResponse, Message, Role

from conftest import AdapterFactory

_MSGS = [Message(role=Role.USER, content="say hi")]


async def test_generate_returns_text(adapter_factory: AdapterFactory) -> None:
    adapter = adapter_factory(text="hello there")
    resp = await adapter.generate(LLMRequest(messages=_MSGS))
    assert isinstance(resp, LLMResponse)
    assert resp.text == "hello there"
    assert resp.finish_reason is FinishReason.STOP
    assert resp.usage.input_tokens >= 0
    assert resp.model and resp.provider


@pytest.mark.parametrize("intent", list(Intent), ids=lambda i: i.value)
async def test_intent_is_accepted(
    adapter_factory: AdapterFactory, intent: Intent
) -> None:
    adapter = adapter_factory()
    resp = await adapter.generate(LLMRequest(messages=_MSGS, intent=intent))
    assert isinstance(resp, LLMResponse)
