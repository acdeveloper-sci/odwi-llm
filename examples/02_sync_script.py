"""02 - synchronous script.

The same call as 01, for code that does not want to be async. `SyncLLM`
wraps any LLMPort in a blocking facade: no event loop plumbing in the
caller, and it behaves the same from a plain script, a Django view, a
Celery task, a Streamlit app or a Jupyter cell (it runs its own loop on a
worker thread).

No API keys. It only needs a running Ollama daemon with the model pulled:

    ollama pull qwen3:0.6b

Run:

    uv run python examples/02_sync_script.py
"""

from odwi_llm.adapters.litellm_adapter import LiteLLMAdapter
from odwi_llm.core.requirements import LLMRequirements
from odwi_llm.core.sync import SyncLLM
from odwi_llm.core.types import LLMRequest, Message, Role

PROVIDER = "ollama"
MODEL = "qwen3:0.6b"


def main() -> None:
    adapter = LiteLLMAdapter(PROVIDER, MODEL, LLMRequirements())

    request = LLMRequest(
        messages=[
            Message(role=Role.SYSTEM, content="You are concise."),
            Message(role=Role.USER, content="Name three primary colors."),
        ]
    )

    # The `with` block stops the worker loop on exit. Build one SyncLLM per
    # process and share it; this example keeps it local for clarity.
    with SyncLLM(adapter) as llm:
        response = llm.generate(request)

    print(response.text)
    print(
        f"\n[{response.provider}/{response.model}] "
        f"finish={response.finish_reason.value} "
        f"tokens={response.usage.input_tokens}+{response.usage.output_tokens}"
    )


if __name__ == "__main__":
    main()
