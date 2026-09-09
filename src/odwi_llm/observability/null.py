"""Default no-op observability — design §5 (Stage 2, v0.5).

So no app has to wire a real backend just to build a `Composer`.
"""

from typing import Any


class NullObservability:
    """Does nothing. `Workflow` runs to completion with this in place."""

    def emit(self, event: str, **fields: Any) -> None:
        pass
