"""Per-provider adapter configuration — design §5.1.

An adapter's constructor takes `provider`, `model`, `requirements` and one
of these. Never sampling parameters (those come from `Intent`).
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderConfig:
    """Where and how to reach a provider.

    `api_key` wins if set; otherwise `api_key_env` is read from the
    environment at call time. Local providers (Ollama, LM Studio) need no
    key — leave both unset.
    """

    api_key: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    timeout_s: float = 60.0

    def resolve_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None
