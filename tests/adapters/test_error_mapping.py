"""An operation a model/provider cannot do maps to CapabilityError from
`_map_error` — the same type the construction fail-fast raises — whether or
not the (provider, model) is catalogued in the adapter's _KNOWN_CAPS.
"""

import pytest

from odwi_llm.adapters.anyllm_adapter import AnyLLMAdapter
from odwi_llm.adapters.litellm_adapter import LiteLLMAdapter
from odwi_llm.core.errors import LLMError, LLMProviderError
from odwi_llm.core.requirements import CapabilityError, LLMRequirements

_Adapter = LiteLLMAdapter | AnyLLMAdapter


def _both() -> list[_Adapter]:
    reqs = LLMRequirements()
    return [
        LiteLLMAdapter("groq", "openai/gpt-oss-120b", reqs),
        AnyLLMAdapter("groq", "openai/gpt-oss-120b", reqs),
    ]


@pytest.mark.parametrize("adapter", _both(), ids=["litellm", "anyllm"])
def test_notimplemented_maps_to_capability_error(adapter: _Adapter) -> None:
    with pytest.raises(CapabilityError):
        adapter._map_error(NotImplementedError("tool calling not supported here"))


@pytest.mark.parametrize("adapter", _both(), ids=["litellm", "anyllm"])
def test_generic_error_still_maps_to_llmerror(adapter: _Adapter) -> None:
    mapped = adapter._map_error(RuntimeError("odd provider hiccup"))
    assert isinstance(mapped, LLMError)
    assert isinstance(mapped, LLMProviderError)


def test_litellm_unsupported_params_capability_message_maps_to_capability_error() -> None:
    adapter = LiteLLMAdapter("groq", "openai/gpt-oss-120b", LLMRequirements())

    # litellm's real class is `UnsupportedParamsError`; match by name.
    class UnsupportedParamsError(Exception):
        pass

    with pytest.raises(CapabilityError):
        adapter._map_error(
            UnsupportedParamsError("Function calling is not supported by this model")
        )
    # a sampling-param message stays a provider error
    mapped = adapter._map_error(UnsupportedParamsError("top_k is not supported"))
    assert not isinstance(mapped, CapabilityError)
