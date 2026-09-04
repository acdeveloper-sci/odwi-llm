"""04 - fallback composition.

`FallbackLLM(primary, fallback)` wraps two ports. A call goes to the
primary; only if the primary raises a transient provider-side error
(`LLMRateLimitError` or `LLMProviderError`) is the same call retried on
the fallback. Auth errors, content-filter blocks and schema failures are
*not* retried - they would just repeat on the second provider.

Here the primary is a local Ollama model and the fallback is a local
LM Studio model. In the happy path below the primary answers and the
fallback is never contacted, so this example runs with only Ollama up -
LM Studio does not need to be running unless the primary actually fails.

No API keys. Needs a running Ollama daemon with the model pulled:

    ollama pull qwen3:0.6b

Run:

    uv run python examples/04_fallback.py
"""

import asyncio

from odwi_llm.adapters.litellm_adapter import LiteLLMAdapter
from odwi_llm.core.composition import FallbackLLM
from odwi_llm.core.requirements import LLMRequirements
from odwi_llm.core.types import LLMRequest, Message, Role


async def main() -> None:
    primary = LiteLLMAdapter("ollama", "qwen3:0.6b", LLMRequirements())
    fallback = LiteLLMAdapter(
        "lmstudio", "llama-3.2-3b-instruct", LLMRequirements()
    )

    # An app depends only on this object - it is itself an LLMPort. Stage 1
    # apps pass fallback=None and this behaves exactly like the primary; a
    # later stage flips the fallback on when availability justifies it.
    llm = FallbackLLM(primary=primary, fallback=fallback)

    request = LLMRequest(
        messages=[
            Message(role=Role.SYSTEM, content="You are concise."),
            Message(role=Role.USER, content="Name three primary colors."),
        ]
    )

    # If `primary.generate` raised LLMRateLimitError / LLMProviderError,
    # FallbackLLM would transparently re-issue this same call against the
    # LM Studio adapter and return its answer instead. Any other error
    # (auth, content filter, bad schema) propagates unchanged.
    response = await llm.generate(request)

    print(response.text)
    print(
        f"\n[{response.provider}/{response.model}] "
        f"finish={response.finish_reason.value} "
        f"tokens={response.usage.input_tokens}+{response.usage.output_tokens}"
    )


if __name__ == "__main__":
    asyncio.run(main())
