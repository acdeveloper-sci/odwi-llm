"""Static price table — design decision #3 / FINDINGS §4.

Neither LiteLLM nor Any-LLM is a reliable cost source: LiteLLM's bundled
table lags new models (and hides `response_cost` off the response
object), Any-LLM reports nothing. So the package owns this table. An
adapter fills `Usage.estimated_cost_usd` from `estimate_cost_usd(...)`,
or leaves it `None` for a model that is not listed.

Values are USD per 1,000,000 tokens. Local models (Ollama, LM Studio)
are 0.0 — self-hosted. Cloud rates are the providers' published prices;
re-check them when providers change pricing. Adding a model is a
one-line edit here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    provider: str
    model: str
    input_usd_per_1m: float
    output_usd_per_1m: float


# Keyed by the resolved model id (exact match, as the provider returns it).
_TABLE: dict[str, ModelPrice] = {
    price.model: price
    for price in (
        # Cloud — official published rates, verified 2026-09-03.
        # gemini-3.5-flash-lite: ai.google.dev/gemini-api/docs/pricing
        ModelPrice("gemini", "gemini-3.5-flash-lite", 0.30, 2.50),
        # openai/gpt-oss-120b on Groq: groq.com/pricing
        ModelPrice("groq", "openai/gpt-oss-120b", 0.15, 0.60),
        # Local — self-hosted, no per-token cost.
        ModelPrice("ollama", "qwen3:0.6b", 0.0, 0.0),
        ModelPrice("lmstudio", "llama-3.2-3b-instruct", 0.0, 0.0),
    )
}


def lookup(model: str) -> ModelPrice | None:
    """Exact match on the resolved model id. `None` if not in the table."""
    return _TABLE.get(model)


def estimate_cost_usd(
    model: str, input_tokens: int, output_tokens: int
) -> float | None:
    """`input_tokens·pin + output_tokens·pout`, or `None` for an unlisted model."""
    price = _TABLE.get(model)
    if price is None:
        return None
    return (
        input_tokens * price.input_usd_per_1m
        + output_tokens * price.output_usd_per_1m
    ) / 1_000_000
