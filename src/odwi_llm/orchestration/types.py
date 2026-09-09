"""Orchestration types — design §8 (Stage 2, v0.5)."""

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
