"""Orchestrator port — design §9 (Stage 2, v0.5).

The substitutable orchestration port (Specify §4). `Workflow` (§10, the
thin reference implementation) and a future agentic adapter (LangGraph /
Agents SDK / whatever) are implementations of THIS SAME protocol, not
separate concepts — the same pattern as `LiteLLMAdapter` / `AnyLLMAdapter`
over `LLMPort`, one layer up.
"""

from typing import Protocol

from odwi_llm.core.types import ToolCall, ToolResult
from odwi_llm.guardrails.types import PolicyContext
from odwi_llm.orchestration.types import WorkflowResult, WorkflowTask


class Orchestrator(Protocol):
    async def run(
        self, task: WorkflowTask, ctx: PolicyContext
    ) -> WorkflowResult: ...


class ToolExecutor(Protocol):
    """The tool-execution registry the app provides (Specify §5).
    `Workflow` only invokes it — it never builds or runs a tool itself.
    `ctx` reaches this far so real tool policy can be enforced here later
    (Specify §3).
    """

    async def execute(
        self, call: ToolCall, ctx: PolicyContext
    ) -> ToolResult: ...
