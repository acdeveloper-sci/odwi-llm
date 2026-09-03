"""The documented exception (plan Task 13): Any-LLM cannot do LM Studio
tool calling — an app that requires it must get a CapabilityError at
construction, not a NotImplementedError mid-call.

Only meaningful for `--live --adapter anyllm`; skipped otherwise (it
asserts codified capability, no network needed, but it is Any-LLM-specific).
"""

import pytest

from odwi_llm.adapters.anyllm_adapter import AnyLLMAdapter
from odwi_llm.core.requirements import CapabilityError, LLMRequirements

_LMSTUDIO = ("lmstudio", "llama-3.2-3b-instruct")


@pytest.fixture(autouse=True)
def _only_anyllm(request: pytest.FixtureRequest) -> None:
    if request.config.getoption("--adapter") != "anyllm":
        pytest.skip("Any-LLM-specific")


def test_lmstudio_tool_requirement_fails_at_construction() -> None:
    with pytest.raises(CapabilityError):
        AnyLLMAdapter(*_LMSTUDIO, LLMRequirements(tool_calling=True))


def test_lmstudio_adapter_reports_no_tool_calling() -> None:
    adapter = AnyLLMAdapter(*_LMSTUDIO, LLMRequirements())
    assert adapter.capabilities.tool_calling is False


async def test_lmstudio_chat_with_tools_raises_capability_error_not_notimplemented() -> None:
    adapter = AnyLLMAdapter(*_LMSTUDIO, LLMRequirements())
    from odwi_llm.core.types import LLMRequest, Message, Role, ToolSpec

    with pytest.raises(CapabilityError):
        await adapter.chat_with_tools(
            LLMRequest(messages=[Message(role=Role.USER, content="weather?")]),
            [ToolSpec(name="get_weather", description="d", parameters_schema={})],
        )


async def test_uncatalogued_lmstudio_model_tools_also_gives_capability_error() -> None:
    """A model NOT in _KNOWN_CAPS inherits the permissive default
    (tool_calling=True), so the construction gate does not fire — but the
    NotImplementedError Any-LLM raises at call time is still mapped to
    CapabilityError, not LLMProviderError.
    """
    from odwi_llm.core.types import LLMRequest, Message, Role, ToolSpec

    adapter = AnyLLMAdapter("lmstudio", "some-uncatalogued-model", LLMRequirements())
    assert adapter.capabilities.tool_calling is True  # default, gate did not fire

    with pytest.raises(CapabilityError):
        await adapter.chat_with_tools(
            LLMRequest(messages=[Message(role=Role.USER, content="weather?")]),
            [ToolSpec(name="get_weather", description="d", parameters_schema={})],
        )
