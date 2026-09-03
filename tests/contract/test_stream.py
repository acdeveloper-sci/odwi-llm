"""Contract: stream — a StreamChunk sequence that ends with kind="done"
carrying a finish_reason, and reasoning never leaks as a text_delta
(§4.3, FINDINGS §1.1 / §1.5).
"""

from odwi_llm.core.types import (
    FinishReason,
    LLMRequest,
    Message,
    Role,
    StreamChunk,
)

from conftest import AdapterFactory

_MSGS = [Message(role=Role.USER, content="stream something")]


async def test_stream_sequence_ends_with_done(
    adapter_factory: AdapterFactory,
) -> None:
    adapter = adapter_factory(text="abcd")
    chunks = [c async for c in adapter.stream(LLMRequest(messages=_MSGS))]
    assert all(isinstance(c, StreamChunk) for c in chunks)
    assert chunks[-1].kind == "done"
    assert chunks[-1].finish_reason is FinishReason.STOP
    # only the terminal chunk carries a finish_reason
    assert all(c.finish_reason is None for c in chunks[:-1])


async def test_text_deltas_reassemble_and_carry_no_reasoning(
    adapter_factory: AdapterFactory,
) -> None:
    adapter = adapter_factory(text="hello world", hidden_reasoning="INTERNAL THOUGHTS")
    chunks = [c async for c in adapter.stream(LLMRequest(messages=_MSGS))]
    text_deltas = [c.text or "" for c in chunks if c.kind == "text_delta"]
    assert "".join(text_deltas) == "hello world"
    assert all("INTERNAL" not in t for t in text_deltas)
    # reasoning is not a StreamChunk kind at all
    assert {c.kind for c in chunks} <= {"text_delta", "tool_call_delta", "usage", "done"}


async def test_custom_chunk_sequence_is_passed_through(
    adapter_factory: AdapterFactory,
) -> None:
    seq = [
        StreamChunk(kind="text_delta", text="one"),
        StreamChunk(kind="done", finish_reason=FinishReason.LENGTH),
    ]
    adapter = adapter_factory(stream_chunks=seq)
    got = [c async for c in adapter.stream(LLMRequest(messages=_MSGS))]
    assert got == seq
    assert got[-1].finish_reason is FinishReason.LENGTH
