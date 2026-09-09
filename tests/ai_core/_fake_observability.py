"""Task 6 — an ObservabilityPort that records every emit() for assertions.

Not collected by pytest (leading underscore); imported by the `test_*.py`
files in this package.
"""

from typing import Any


class FakeObservability:
    """Records each `emit()` as an `(event, fields)` pair, in call order."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event: str, **fields: Any) -> None:
        self.events.append((event, fields))

    @property
    def names(self) -> list[str]:
        """Just the event names, in order — for `== [...]` assertions."""
        return [name for name, _ in self.events]

    def fields_for(self, event: str) -> list[dict[str, Any]]:
        """The `fields` dict of every emit of `event`, in order."""
        return [fields for name, fields in self.events if name == event]
