"""Contract-suite fixtures.

`adapter_factory` builds an `LLMPort` for a scenario. Against `FakeAdapter`
the keyword arguments are forwarded straight through; a later conftest
(plan Task 12/13) can point the same fixture at a real, cassette-backed
adapter without any test changing — the kwargs (`error`, `capabilities`,
`requirements`, `structured_payload`, `tool_calls`, `stream_chunks`) are
all §4 contract concepts, not library-specific.
"""

from collections.abc import Callable

import pytest

from odwi_llm.core.port import LLMPort

from _fake_adapter import FakeAdapter

AdapterFactory = Callable[..., LLMPort]


@pytest.fixture
def adapter_factory() -> AdapterFactory:
    def _make(**kwargs: object) -> LLMPort:
        return FakeAdapter(**kwargs)  # type: ignore[arg-type]

    return _make
