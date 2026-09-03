"""Task 13b — bounded technical retry (design §5.3)."""

from types import SimpleNamespace
from typing import Any

import pytest

from odwi_llm.adapters._shared import call_with_retry
from odwi_llm.adapters.litellm_adapter import LiteLLMAdapter
from odwi_llm.core.errors import LLMAuthError, LLMProviderError, LLMRateLimitError
from odwi_llm.core.requirements import LLMRequirements
from odwi_llm.core.types import LLMRequest, Message, Role

_REQ = LLMRequest(messages=[Message(role=Role.USER, content="hi")])


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []

    async def fake(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr("odwi_llm.adapters._shared.asyncio.sleep", fake)
    return recorded


# --- call_with_retry -----------------------------------------------
async def test_retries_then_succeeds(no_sleep: list[float]) -> None:
    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise LLMProviderError("transient 503")
        return "ok"

    assert await call_with_retry(flaky) == "ok"
    assert calls["n"] == 3
    assert no_sleep == [0.5, 1.0]  # backoff between the three attempts


async def test_gives_up_after_max_attempts(no_sleep: list[float]) -> None:
    calls = {"n": 0}

    async def always_fails() -> str:
        calls["n"] += 1
        raise LLMProviderError("still 503")

    with pytest.raises(LLMProviderError):
        await call_with_retry(always_fails)
    assert calls["n"] == 3  # not infinite


async def test_does_not_retry_non_transient(no_sleep: list[float]) -> None:
    calls = {"n": 0}

    async def auth_fails() -> str:
        calls["n"] += 1
        raise LLMAuthError("bad key")

    with pytest.raises(LLMAuthError):
        await call_with_retry(auth_fails)
    assert calls["n"] == 1
    assert no_sleep == []


async def test_rate_limit_retry_after_is_honoured(no_sleep: list[float]) -> None:
    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMRateLimitError("429", retry_after_seconds=2.0)
        return "ok"

    assert await call_with_retry(flaky) == "ok"
    assert no_sleep == [2.0]


# --- stream: retry only before the first chunk --------------------
def _chunk(
    content: str | None = None, finish: str | None = None, usage: Any = None
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish)],
        usage=usage,
        model="fake-model",
    )


async def _gen(chunks: list[Any]) -> Any:
    for chunk in chunks:
        yield chunk


async def test_stream_retries_before_first_chunk(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:

    calls = {"n": 0}

    async def fake_acompletion(**_: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("503 service unavailable")  # -> LLMProviderError
        return _gen([_chunk(content="hi"), _chunk(finish="stop")])

    monkeypatch.setattr("odwi_llm.adapters.litellm_adapter.litellm.acompletion", fake_acompletion)
    adapter = LiteLLMAdapter("groq", "openai/gpt-oss-120b", LLMRequirements())

    out = [c async for c in adapter.stream(_REQ)]
    assert calls["n"] == 2  # retried once
    assert out[-1].kind == "done"
    assert [c.text for c in out if c.kind == "text_delta"] == ["hi"]


async def test_stream_does_not_retry_after_first_chunk(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:

    calls = {"n": 0}

    async def one_then_boom() -> Any:
        yield _chunk(content="partial")
        raise RuntimeError("503 mid-stream")

    async def fake_acompletion(**_: Any) -> Any:
        calls["n"] += 1
        return one_then_boom()

    monkeypatch.setattr("odwi_llm.adapters.litellm_adapter.litellm.acompletion", fake_acompletion)
    adapter = LiteLLMAdapter("groq", "openai/gpt-oss-120b", LLMRequirements())

    seen: list[Any] = []
    with pytest.raises(LLMProviderError):
        async for chunk in adapter.stream(_REQ):
            seen.append(chunk)
    assert calls["n"] == 1  # no retry once output started
    assert [c.text for c in seen if c.kind == "text_delta"] == ["partial"]
