"""Task 15 — FallbackLLM (§5.4) wired to the REAL adapter classes
(LiteLLMAdapter primary, AnyLLMAdapter fallback — decision #4), not the
Task 7 stubs.

Offline: real adapter classes, faked transport, a simulated
LLMProviderError / LLMAuthError. `--live`: fully real against Groq,
including one case where the primary is unreachable so the fallback
actually serves.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from odwi_llm.adapters.anyllm_adapter import AnyLLMAdapter
from odwi_llm.adapters.config import ProviderConfig
from odwi_llm.adapters.litellm_adapter import LiteLLMAdapter
from odwi_llm.core.composition import FallbackLLM
from odwi_llm.core.errors import LLMAuthError, LLMProviderError
from odwi_llm.core.requirements import LLMRequirements
from odwi_llm.core.types import LLMRequest, Message, Role

_GROQ = ("groq", "openai/gpt-oss-120b")
_REQ = LLMRequest(messages=[Message(role=Role.USER, content="Say hi.")])


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake(_seconds: float) -> None:
        return None

    monkeypatch.setattr("odwi_llm.adapters._shared.asyncio.sleep", fake)


def _ok_completion(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
        model="openai/gpt-oss-120b",
    )


# --- offline: real adapter classes, faked transport --------------
def _patch(monkeypatch: pytest.MonkeyPatch, *, primary: AsyncMock, fallback: AsyncMock) -> None:
    # `primary` mocks litellm.acompletion, which runs INSIDE
    # LiteLLMAdapter._acompletion -> call_with_retry(_call). So a
    # side_effect that raises is retried up to 3 times before the mapped
    # LLMProviderError propagates to FallbackLLM. `fallback` mocks
    # any_llm.acompletion the same way.
    monkeypatch.setattr(
        "odwi_llm.adapters.litellm_adapter.litellm.acompletion", primary
    )
    monkeypatch.setattr("odwi_llm.adapters.anyllm_adapter.acompletion", fallback)


async def test_delegates_to_fallback_after_retry_exhausts(
    monkeypatch: pytest.MonkeyPatch, no_sleep: None
) -> None:
    primary = AsyncMock(side_effect=RuntimeError("503 service unavailable"))
    fallback = AsyncMock(return_value=_ok_completion("from the fallback"))
    _patch(monkeypatch, primary=primary, fallback=fallback)

    llm = FallbackLLM(
        primary=LiteLLMAdapter(*_GROQ, LLMRequirements()),
        fallback=AnyLLMAdapter(*_GROQ, LLMRequirements()),
    )
    resp = await llm.generate(_REQ)

    assert resp.text == "from the fallback"
    assert resp.provider == "groq"
    # the retry ran to exhaustion on the primary before delegating
    assert primary.call_count == 3
    assert fallback.call_count == 1


async def test_retry_absorbs_transient_and_fallback_never_activates(
    monkeypatch: pytest.MonkeyPatch, no_sleep: None
) -> None:
    # primary fails once, recovers on the 2nd retry attempt.
    primary = AsyncMock(
        side_effect=[RuntimeError("503 once"), _ok_completion("primary recovered")]
    )
    fallback = AsyncMock(return_value=_ok_completion("fallback (must stay unused)"))
    _patch(monkeypatch, primary=primary, fallback=fallback)

    llm = FallbackLLM(
        primary=LiteLLMAdapter(*_GROQ, LLMRequirements()),
        fallback=AnyLLMAdapter(*_GROQ, LLMRequirements()),
    )
    resp = await llm.generate(_REQ)

    assert resp.text == "primary recovered"
    assert resp.provider == "groq"
    assert primary.call_count == 2  # failed once, succeeded on the retry
    # a transient 503 is absorbed by the retry; the fallback is not touched,
    # so it does not also spend the second provider's quota.
    assert fallback.call_count == 0


async def test_does_not_delegate_on_auth_error(
    monkeypatch: pytest.MonkeyPatch, no_sleep: None
) -> None:
    class AuthenticationError(Exception):  # name matches litellm's real class
        pass

    primary = AsyncMock(side_effect=AuthenticationError("invalid api key"))
    fallback = AsyncMock(return_value=_ok_completion("should never be used"))
    _patch(monkeypatch, primary=primary, fallback=fallback)

    llm = FallbackLLM(
        primary=LiteLLMAdapter(*_GROQ, LLMRequirements()),
        fallback=AnyLLMAdapter(*_GROQ, LLMRequirements()),
    )
    with pytest.raises(LLMAuthError):
        await llm.generate(_REQ)

    assert primary.call_count == 1  # auth is not a retryable error
    assert fallback.call_count == 0  # §5.4: auth is not retried against the fallback


async def test_no_fallback_reraises_provider_error(
    monkeypatch: pytest.MonkeyPatch, no_sleep: None
) -> None:
    primary = AsyncMock(side_effect=RuntimeError("503"))
    _patch(monkeypatch, primary=primary, fallback=AsyncMock())

    llm = FallbackLLM(primary=LiteLLMAdapter(*_GROQ, LLMRequirements()))  # fallback=None
    with pytest.raises(LLMProviderError):
        await llm.generate(_REQ)
    assert primary.call_count == 3  # retry still ran fully


# --- live: fully real against Groq --------------------------------
@pytest.fixture
def _live(request: pytest.FixtureRequest) -> None:
    if not request.config.getoption("--live"):
        pytest.skip("needs --live")


async def test_live_primary_serves_when_healthy(_live: None) -> None:
    llm = FallbackLLM(
        primary=LiteLLMAdapter(*_GROQ, LLMRequirements()),
        fallback=AnyLLMAdapter(*_GROQ, LLMRequirements()),
    )
    resp = await llm.generate(_REQ)
    assert resp.text.strip()
    assert resp.provider == "groq"


async def test_live_fallback_serves_when_primary_unreachable(_live: None) -> None:
    unreachable = LiteLLMAdapter(
        *_GROQ,
        LLMRequirements(),
        ProviderConfig(api_key="unused", base_url="http://127.0.0.1:1"),
    )
    llm = FallbackLLM(
        primary=unreachable,
        fallback=AnyLLMAdapter(*_GROQ, LLMRequirements()),
    )
    resp = await llm.generate(_REQ)  # primary -> LLMProviderError -> fallback
    assert resp.text.strip()
    assert resp.provider == "groq"
