"""03 - structured output.

`structured()` takes a Pydantic model and returns a StructuredResponse
whose `.data` is an *already validated* instance of that model, not a
string you have to parse. The adapter asks the provider for JSON matching
the schema and retries a bounded number of times if it does not validate.

No API keys. It only needs a running Ollama daemon with the model pulled:

    ollama pull qwen3:0.6b

Run:

    uv run python examples/03_structured_output.py
"""

import asyncio

from pydantic import BaseModel

from odwi_llm.adapters.litellm_adapter import LiteLLMAdapter
from odwi_llm.core.requirements import LLMRequirements
from odwi_llm.core.types import LLMRequest, Message, Role

PROVIDER = "ollama"
MODEL = "qwen3:0.6b"


class Book(BaseModel):
    title: str
    author: str
    year: int


async def main() -> None:
    # structured_output=True is a real requirement: if the resolved
    # (provider, model) could not do it, this line would raise
    # CapabilityError instead of failing on the call below.
    llm = LiteLLMAdapter(
        PROVIDER, MODEL, LLMRequirements(structured_output=True)
    )

    request = LLMRequest(
        messages=[
            Message(role=Role.SYSTEM, content="Extract the book into the schema."),
            Message(role=Role.USER, content="Nineteen Eighty-Four, by George Orwell, published 1949."),
        ]
    )

    response = await llm.structured(request, Book)

    book = response.data  # a Book instance, validated
    print(f"title : {book.title}")
    print(f"author: {book.author}")
    print(f"year  : {book.year}")
    print(f"\ntype(response.data) = {type(book).__name__}")
    print(
        f"[{response.provider}/{response.model}] "
        f"finish={response.finish_reason.value} "
        f"tokens={response.usage.input_tokens}+{response.usage.output_tokens}"
    )


if __name__ == "__main__":
    asyncio.run(main())
