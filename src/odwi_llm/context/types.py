"""Context types — design §6 (Stage 2, v0.5).

The two source speeds (Specify §4) are explicit in the type, not blurred
into one flat dict: `knowledge` is near-static (indexed once), `data`
changes per run. `prior_output` carries a frozen snapshot of an earlier
structured result so a multi-turn flow can be told "never contradict what
the user was already shown" as data, not just as a prompt instruction.
"""

from typing import Any

from pydantic import BaseModel, Field


class ContextBundle(BaseModel):
    knowledge: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    prior_output: dict[str, Any] | None = None
