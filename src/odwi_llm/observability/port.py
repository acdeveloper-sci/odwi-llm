"""Observability port — design §5 (Stage 2, v0.5).

Minimal, with no real implementation yet (Specify §5, "minimal
observability, from the start"). One event for each point Specify names:
execution, request, event, error, policy decision, LLM call, tool call,
result — none of them rigidly typed yet, so we do not commit to an event
shape before there is a real backend to validate it against.

`emit` is sync: `Workflow` (§10) calls it inline, and a real backend
enqueues rather than blocks. It stays sync even though the guardrail
protocols went async in v0.5 — those may need an LLM; emitting an event
never does.
"""

from typing import Any, Protocol


class ObservabilityPort(Protocol):
    def emit(self, event: str, **fields: Any) -> None: ...
