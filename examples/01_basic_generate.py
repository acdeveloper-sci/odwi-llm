"""01 - basic text generation.

The smallest possible use of the package: one LiteLLMAdapter pointed at a
local Ollama model, one `generate()` call, async.

No API keys. It only needs a running Ollama daemon with the model pulled:

    ollama pull qwen3:0.6b

Run:

    uv run python examples/01_basic_generate.py
"""

import asyncio

from odwi_llm.adapters.litellm_adapter import LiteLLMAdapter
from odwi_llm.core.requirements import LLMRequirements
from odwi_llm.core.types import LLMRequest, Message, Role

PROVIDER = "ollama"  # base_url defaults to http://localhost:11434
MODEL = "qwen3:0.6b"


async def main() -> None:
    # Requirements are checked once, here, at construction. Plain text
    # generation needs nothing special, so the defaults (all False) are
    # enough; an unmet requirement would raise CapabilityError right here
    # instead of failing later on a call.
    llm = LiteLLMAdapter(PROVIDER, MODEL, LLMRequirements())

    request = LLMRequest(
        messages=[
            Message(role=Role.SYSTEM, content="You are concise."),
            Message(role=Role.USER, content="Name three primary colors."),
        ]
    )

    response = await llm.generate(request)

    print(response.text)
    print(
        f"\n[{response.provider}/{response.model}] "
        f"finish={response.finish_reason.value} "
        f"tokens={response.usage.input_tokens}+{response.usage.output_tokens}"
    )


if __name__ == "__main__":
    asyncio.run(main())
