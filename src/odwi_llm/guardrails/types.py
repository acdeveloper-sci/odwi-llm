"""Policy types — design §2 (Stage 2, v0.5).

Implemented verbatim from the design. `Decision` is a discriminated union
of three concrete classes, not one class with optional fields: each
variant carries only the fields that apply to it, and `mypy` forces an
exhaustive check in any `match`/`isinstance` chain (the same problem a
real system hit once with nullable `result`/`error`, resolved here in the
type instead of a dedicated test).
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class PolicyContext(BaseModel):
    """Frozen when the composition is built (Specify §3/§4) — never re-read
    silently in the middle of a conversation.

    `resolved_at` plus `provider`/`model` record which model produced each
    artifact, so a config change mid-conversation cannot go unnoticed.
    """

    session_id: str
    provider: str
    model: str
    resolved_at: datetime

    # Opaque bag: the Core transports it, never interprets it. Only the
    # app's own guardrails / Context know which keys to expect here (role,
    # allowed_data_classes, whatever) — this is how the Core stays
    # uncoupled from any application domain (Specify §5).
    scope: dict[str, Any] = Field(default_factory=dict)


class Allow(BaseModel):
    kind: Literal["allow"] = "allow"
    # False = general knowledge within topic, with no specific datum
    # backing it (Specify §3, "allowed, with different provenance").
    # Not a fourth Decision variant — still an Allow, just with provenance
    # metadata an OutputGuardrail or the app can use to mark the answer.
    grounded: bool = True


class Deny(BaseModel):
    kind: Literal["deny"] = "deny"
    reason: str


class Redact[T](BaseModel):
    """Generic (`Redact[T]`, PEP 695): the replacement content is not
    always text. `InputGuardrail` / `OutputGuardrail` use `Decision[str]`;
    `ToolGuardrail.before` uses `Decision[dict[str, Any]]` (sanitized tool
    arguments — run the tool with a cleaned arg instead of blocking it).
    One type for all three cases, no parallel "redact of dict" variant.
    """

    kind: Literal["redact"] = "redact"
    reason: str
    redacted: T


type Decision[T] = Allow | Deny | Redact[T]
