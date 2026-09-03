"""Structural contract suite — shape, not exact content.

Runs against `FakeAdapter` by default (offline) and against the real
`LiteLLMAdapter` per lab provider with `--live`. Task 14 will swap the
default from `FakeAdapter` to cassette replay; the tests do not change.

`tests/contract/` (the exact fake suite) is separate and never touches
the network.
"""

import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from odwi_llm.adapters.litellm_adapter import LiteLLMAdapter
from odwi_llm.core.port import LLMPort
from odwi_llm.core.requirements import LLMRequirements
from odwi_llm.core.types import ToolCall

# reuse the FakeAdapter from the sibling suite without touching tests/contract/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "contract"))
from _fake_adapter import FakeAdapter  # noqa: E402

# provider id -> model id, the four lab providers (§9.3 table B).
LIVE_TARGETS: list[tuple[str, str]] = [
    ("gemini", "gemini-3.5-flash-lite"),
    ("groq", "openai/gpt-oss-120b"),
    ("ollama", "qwen3:0.6b"),
    ("lmstudio", "llama-3.2-3b-instruct"),
]

_FAKE_TOOL_CALL = ToolCall(
    id="call_fake", name="get_weather", arguments={"city": "Paris"}
)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "adapter" not in metafunc.fixturenames:
        return
    if metafunc.config.getoption("--live"):
        metafunc.parametrize(
            "adapter", LIVE_TARGETS, ids=[p for p, _ in LIVE_TARGETS], indirect=True
        )
    else:
        metafunc.parametrize("adapter", [None], ids=["fake"], indirect=True)


@pytest.fixture
def adapter(request: pytest.FixtureRequest) -> LLMPort:
    target: tuple[str, str] | None = request.param
    if target is None:
        return FakeAdapter(
            text="A short reply.",
            tool_calls=[_FAKE_TOOL_CALL],
            hidden_reasoning="INTERNAL REASONING THAT MUST NOT LEAK",
        )
    provider, model = target
    return LiteLLMAdapter(provider, model, LLMRequirements())


AdapterFactory = Callable[..., LLMPort]
