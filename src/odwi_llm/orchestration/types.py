"""Orchestration types — design §8 (Stage 2, v0.7)."""

from typing import Any

from pydantic import BaseModel

from odwi_llm.core.types import FinishReason


class WorkflowTask(BaseModel):
    """What the app asks the Workflow to do — one turn of execution."""

    session_id: str
    # Free shape; each concrete Workflow defines its own.
    input: dict[str, Any]


class WorkflowResult(BaseModel):
    text: str
    finish_reason: FinishReason
    tool_calls_made: int = 0
    # From the last Allow decision that went through output policy
    # (§2, Allow.grounded). Lost if not carried explicitly this far.
    grounded: bool = True
    # (v0.7) The validated instance when Workflow was built with schema=
    # (§10) and the final response went through structured(). Deliberately
    # NOT generic (WorkflowResult[T] would force Orchestrator.run(), §9, to
    # become generic too, rippling into every existing implementation) -
    # whoever built the Workflow with that schema already knows what to
    # expect and narrows it themselves (see §8/§13).
    data: BaseModel | None = None
