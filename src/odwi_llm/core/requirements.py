"""Requirements / capabilities and the fail-fast error — design §4.4.

`CapabilityError` lives here (next to the models it belongs with), not in
`errors.py`, exactly as the §4.4 code block shows. This module depends
only on pydantic — nothing from `errors.py` or `adapters/`.
"""

from pydantic import BaseModel


class LLMRequirements(BaseModel):
    """What the application needs. Checked once at adapter construction."""

    structured_output: bool = False
    tool_calling: bool = False
    streaming: bool = False
    vision: bool = False
    min_context_tokens: int | None = None


class LLMCapabilities(BaseModel):
    """What the resolved model/provider actually supports, as known by the adapter."""

    structured_output: bool
    tool_calling: bool
    streaming: bool
    vision: bool
    context_tokens: int | None


class CapabilityError(Exception):
    """Raised at adapter construction when `LLMRequirements` are not met."""
