"""Port error hierarchy — design §4.5.

The adapter translates any library/provider exception into one of these.
Domain code never catches `openai.RateLimitError`, `litellm.exceptions.*`
or `any_llm.exceptions.*`.

`CapabilityError` is intentionally NOT here — it lives in
`requirements.py`, next to `LLMRequirements`/`LLMCapabilities` (design §4.4).
"""


class LLMError(Exception):
    """Base for every error surfaced by an adapter."""


class LLMAuthError(LLMError):
    """Bad or missing credentials for the provider."""


class LLMRateLimitError(LLMError):
    """Provider throttled the request (HTTP 429)."""

    def __init__(
        self, *args: object, retry_after_seconds: float | None = None
    ) -> None:
        super().__init__(*args)
        self.retry_after_seconds = retry_after_seconds


class LLMContextLengthError(LLMError):
    """The request exceeds the model's context window."""


class LLMContentFilteredError(LLMError):
    """The provider blocked the request or response on content grounds."""


class LLMSchemaError(LLMError):
    """Structured output did not validate after adapter-level retries."""


class LLMProviderError(LLMError):
    """Any other provider-side failure (5xx, timeouts)."""
