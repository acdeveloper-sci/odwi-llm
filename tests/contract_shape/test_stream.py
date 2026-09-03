"""Shape: the stream is a StreamChunk sequence that ends in kind="done",
and reasoning never leaks into a text_delta (FINDINGS §1.1 / §1.5).
"""

from odwi_llm.core.port import LLMPort
from odwi_llm.core.types import FinishReason, LLMRequest, Message, Role, StreamChunk

_KINDS = {"text_delta", "tool_call_delta", "usage", "done"}
_REQ = LLMRequest(
    messages=[Message(role=Role.USER, content="Say hello in one short sentence.")],
    # no max_output_tokens (see test_generate.py)
)


async def test_stream_shape_and_no_reasoning_leak(adapter: LLMPort) -> None:
    chunks = [c async for c in adapter.stream(_REQ)]
    assert chunks, "stream produced no chunks"
    assert all(isinstance(c, StreamChunk) for c in chunks)
    assert all(c.kind in _KINDS for c in chunks)

    assert chunks[-1].kind == "done"
    assert isinstance(chunks[-1].finish_reason, FinishReason)
    # only the terminal chunk carries a finish_reason
    assert all(c.finish_reason is None for c in chunks[:-1])

    text = "".join(c.text or "" for c in chunks if c.kind == "text_delta")
    assert text.strip() != ""
    # reasoning is discarded in the adapter, never a text_delta
    assert "INTERNAL REASONING" not in text
    assert "<think>" not in text and "</think>" not in text
