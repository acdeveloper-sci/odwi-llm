"""Task 11 — a minimal inline stub satisfies the Orchestrator protocol,
without Workflow (Task 12) existing yet. Conformance is checked by mypy
via the typed assignment; the runtime call just confirms it works.
"""

from datetime import datetime

from odwi_llm.core.types import FinishReason
from odwi_llm.guardrails.types import PolicyContext
from odwi_llm.orchestration.port import Orchestrator
from odwi_llm.orchestration.types import WorkflowResult, WorkflowTask

_CTX = PolicyContext(
    session_id="s1",
    provider="fake",
    model="fake-model",
    resolved_at=datetime(2026, 1, 1),
)


class _StubOrchestrator:
    async def run(
        self, task: WorkflowTask, ctx: PolicyContext
    ) -> WorkflowResult:
        return WorkflowResult(text="stub", finish_reason=FinishReason.STOP)


async def test_stub_satisfies_the_protocol() -> None:
    orch: Orchestrator = _StubOrchestrator()
    result = await orch.run(
        WorkflowTask(session_id="s1", input={"message": "hi"}), _CTX
    )
    assert isinstance(result, WorkflowResult)
    assert result.text == "stub"
    assert result.grounded is True
