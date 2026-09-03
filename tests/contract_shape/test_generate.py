"""Shape: generate + capabilities."""

from odwi_llm.core.port import LLMPort
from odwi_llm.core.requirements import LLMCapabilities
from odwi_llm.core.types import FinishReason, LLMRequest, LLMResponse, Message, Role

_REQ = LLMRequest(
    messages=[Message(role=Role.USER, content="Reply with a short greeting.")],
    # no max_output_tokens: a reasoning model (qwen3) would spend a tight
    # budget entirely on <think> and leave content empty.
)


async def test_generate_returns_shaped_response(adapter: LLMPort) -> None:
    resp = await adapter.generate(_REQ)
    assert isinstance(resp, LLMResponse)
    assert isinstance(resp.text, str) and resp.text.strip() != ""
    assert isinstance(resp.finish_reason, FinishReason)
    assert resp.usage.input_tokens >= 0 and resp.usage.output_tokens >= 0
    assert resp.model and resp.provider


def test_capabilities_is_an_llmcapabilities(adapter: LLMPort) -> None:
    assert isinstance(adapter.capabilities, LLMCapabilities)
