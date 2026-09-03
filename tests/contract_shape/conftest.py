"""Structural contract suite — shape, not exact content.

Default (`uv run pytest`): replays `tests/cassettes/` — recorded real
outputs, offline, both adapters. `--live` runs against real providers
(`--adapter {litellm,anyllm}` picks which); add `--record` to (re)write
the cassettes for that adapter.

`tests/contract/` (the exact fake suite) is separate and never networks.
"""

import sys
from pathlib import Path

import pytest

from odwi_llm.adapters.anyllm_adapter import AnyLLMAdapter
from odwi_llm.adapters.litellm_adapter import LiteLLMAdapter
from odwi_llm.core.port import LLMPort
from odwi_llm.core.requirements import LLMRequirements

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cassettes"))
from _replay import CassetteReplay, RecordingAdapter  # noqa: E402

# provider id -> model id, the four lab providers (§9.3).
LIVE_TARGETS: list[tuple[str, str]] = [
    ("gemini", "gemini-3.5-flash-lite"),
    ("groq", "openai/gpt-oss-120b"),
    ("ollama", "qwen3:0.6b"),
    ("lmstudio", "llama-3.2-3b-instruct"),
]

_ADAPTERS = {"litellm": LiteLLMAdapter, "anyllm": AnyLLMAdapter}
# (adapter, provider, model) — replay covers both adapters.
_ALL_TARGETS = [(a, p, m) for a in _ADAPTERS for (p, m) in LIVE_TARGETS]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "adapter" not in metafunc.fixturenames:
        return
    if metafunc.config.getoption("--live"):
        name = metafunc.config.getoption("--adapter")
        targets = [(name, p, m) for (p, m) in LIVE_TARGETS]
    else:
        targets = list(_ALL_TARGETS)
    # Any-LLM cannot do LM Studio tools — that combo is covered by
    # test_anyllm_lmstudio.py (CapabilityError at construction).
    if metafunc.module.__name__.endswith("test_tools"):
        targets = [t for t in targets if not (t[0] == "anyllm" and t[1] == "lmstudio")]
    metafunc.parametrize(
        "adapter",
        targets,
        ids=[f"{a}-{p}" for a, p, _ in targets],
        indirect=True,
    )


@pytest.fixture
def adapter(request: pytest.FixtureRequest) -> LLMPort:
    name, provider, model = request.param
    config = request.config
    if not config.getoption("--live"):
        return CassetteReplay(name, provider, model)
    real = _ADAPTERS[name](provider, model, LLMRequirements())
    if config.getoption("--record"):
        return RecordingAdapter(real, name, provider, model)
    return real
