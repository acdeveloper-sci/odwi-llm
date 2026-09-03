"""Structural contract suite — shape, not exact content.

Runs against `FakeAdapter` by default (offline). With `--live` it runs
against a real adapter per lab provider; `--adapter {litellm,anyllm}`
picks which (default litellm). Task 14 will swap the default from
`FakeAdapter` to cassette replay; the tests do not change.

`tests/contract/` (the exact fake suite) is separate and never networks.
"""

import sys
from pathlib import Path

import pytest

from odwi_llm.adapters.anyllm_adapter import AnyLLMAdapter
from odwi_llm.adapters.litellm_adapter import LiteLLMAdapter
from odwi_llm.core.port import LLMPort
from odwi_llm.core.requirements import LLMRequirements
from odwi_llm.core.types import ToolCall

# reuse the FakeAdapter from the sibling suite without touching tests/contract/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "contract"))
from _fake_adapter import FakeAdapter  # noqa: E402

# provider id -> model id, the four lab providers (§9.3).
LIVE_TARGETS: list[tuple[str, str]] = [
    ("gemini", "gemini-3.5-flash-lite"),
    ("groq", "openai/gpt-oss-120b"),
    ("ollama", "qwen3:0.6b"),
    ("lmstudio", "llama-3.2-3b-instruct"),
]

_ADAPTERS = {"litellm": LiteLLMAdapter, "anyllm": AnyLLMAdapter}

_FAKE_TOOL_CALL = ToolCall(
    id="call_fake", name="get_weather", arguments={"city": "Paris"}
)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "adapter" not in metafunc.fixturenames:
        return
    if not metafunc.config.getoption("--live"):
        metafunc.parametrize("adapter", [None], ids=["fake"], indirect=True)
        return

    targets = list(LIVE_TARGETS)
    adapter_name = metafunc.config.getoption("--adapter")
    # Any-LLM cannot do LM Studio tools at all — that combo is covered by
    # test_anyllm_lmstudio.py (CapabilityError at construction), not here.
    if adapter_name == "anyllm" and metafunc.module.__name__.endswith("test_tools"):
        targets = [t for t in targets if t[0] != "lmstudio"]
    metafunc.parametrize(
        "adapter", targets, ids=[p for p, _ in targets], indirect=True
    )


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
    adapter_cls = _ADAPTERS[request.config.getoption("--adapter")]
    return adapter_cls(provider, model, LLMRequirements())
