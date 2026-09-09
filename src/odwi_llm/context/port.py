"""Context port — design §7 (Stage 2, v0.5).

`Context` answers "what is relevant"; the `Orchestrator` decides "when it
is asked". The interface supports both cadences without assuming which is
"the" one:

  * `message=None`  -> single invocation (AI Insights: select once,
    reuse for the whole session). Proven in production.
  * `message=<text>` -> per-turn invocation (chat, anticipated mode,
    Specify §4). No real case has exercised this yet; the minimal
    implementation may ignore `message` and always return the same
    bundle, which already covers the proven mode.

`select` is `async` (v0.5): Specify §4 anticipates it using an LLM
internally to pick relevance — that call goes through the async `LLMPort`,
inheriting retries / cost tracking / provider limits. An implementation
that just returns a fixed bundle never `await`s and pays nothing for it.
"""

from typing import Protocol

from odwi_llm.context.types import ContextBundle
from odwi_llm.guardrails.types import PolicyContext


class ContextPort(Protocol):
    async def select(
        self, *, message: str | None = None, ctx: PolicyContext
    ) -> ContextBundle: ...
