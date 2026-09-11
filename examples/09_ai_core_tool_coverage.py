"""09 - AI Core tool coverage: a catch-all guardrail and a specific one.

An analysis assistant over a fixed experiment dataset (a dict in this
script, no external file). It shows `ToolGuardrail.covers` with a
catch-all (`covers=None`) and a specific guardrail
(`covers={"query_dataset"}`) BOTH running on the same tool — the
mechanism has existed since the port was written, but no earlier example
exercised it.

Two tools (via `tool_executor`):
  * compute_stat(column, op)   - mean / std over a column. No restriction.
  * query_dataset(column, limit) - raw values from a column. Real data
                                   access.

Two tool guardrails:
  * ToolPermissionPolicy  - covers=None, runs on every tool. Broad and
                            lightweight; here it only passes through, so
                            the output shows a tool_before / tool_after
                            for it on BOTH tools.
  * DataAccessPolicy      - covers={"query_dataset"}, runs on that tool
                            *in addition to* the catch-all. Its `before`
                            inspects the STRUCTURED argument
                            call.arguments["column"] against a restricted
                            set -> Deny for a restricted column.

So `query_dataset` gets two `tool_before` events (catch-all, then
specific); `compute_stat` gets one (catch-all only).

Two questions: one drives compute_stat on an open column (allowed); one
drives query_dataset on the restricted column (denied by the specific
guardrail — the loop then continues with the error ToolResult handed
back to the model, exactly as in `07`).

Note: the `covers` dispatch (catch-all + specific) is plain, deterministic
Python — when a tool call happens, the events above are exactly what you
get. The stochastic part is whether `qwen3:0.6b` calls a tool at all
(`tool_choice="auto"`, same limitation as `07`). The questions name the
tool and column verbatim to push it toward calling; `temperature=0` also
makes repeated identical runs largely the same generation, so consistency
across reruns is not strong evidence by itself. If a run shows
`tool_calls=0`, the model just did not call the tool — run it again.

No API keys. It only needs a running Ollama daemon with the model pulled:

    ollama pull qwen3:0.6b

Run:

    uv run python examples/09_ai_core_tool_coverage.py
"""

import asyncio
import json
import statistics
from datetime import UTC, datetime
from typing import Any

from odwi_llm.adapters.litellm_adapter import LiteLLMAdapter
from odwi_llm.context.types import ContextBundle
from odwi_llm.core.requirements import LLMRequirements
from odwi_llm.core.types import (
    ChatTurnResponse,
    LLMResponse,
    Message,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from odwi_llm.guardrails.port import GuardrailSet
from odwi_llm.guardrails.types import Allow, Decision, Deny, PolicyContext
from odwi_llm.orchestration.composition import Composer
from odwi_llm.orchestration.types import WorkflowTask

PROVIDER = "ollama"
MODEL = "qwen3:0.6b"

# --- the "already loaded" experiment dataset, columns as lists ---------

DATASET: dict[str, list[Any]] = {
    "reaction_time_ms": [412, 388, 455, 401, 470, 399, 433, 420],
    "accuracy": [0.94, 0.97, 0.91, 0.96, 0.89, 0.98, 0.93, 0.95],
    "trial_number": [1, 2, 3, 4, 5, 6, 7, 8],
    "subject_id": ["S-014", "S-014", "S-022", "S-022", "S-031", "S-031", "S-045", "S-045"],
}
_RESTRICTED_COLUMNS = frozenset({"subject_id"})

COMPUTE_STAT = ToolSpec(
    name="compute_stat",
    description="Compute a statistic over a dataset column. op is 'mean' or 'std'.",
    parameters_schema={
        "type": "object",
        "properties": {
            "column": {"type": "string"},
            "op": {"type": "string", "enum": ["mean", "std"]},
        },
        "required": ["column", "op"],
    },
)
QUERY_DATASET = ToolSpec(
    name="query_dataset",
    description="Return raw values from a dataset column, up to `limit` rows.",
    parameters_schema={
        "type": "object",
        "properties": {
            "column": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["column"],
    },
)


class AllowInput:
    async def check(
        self, message: Message, ctx: PolicyContext
    ) -> Decision[str]:
        return Allow()


class AllowOutput:
    async def check(
        self, response: LLMResponse | ChatTurnResponse, ctx: PolicyContext
    ) -> Decision[str]:
        return Allow()


class ToolPermissionPolicy:
    """Broad, lightweight role: `covers=None` -> runs before/after EVERY
    tool call. It blocks nothing here — in a real app this is where you
    would gate which tools are callable at all, apply rate limits, or
    check a caller role. Because it covers every tool, the output shows a
    `tool_before` / `tool_after` for it on both tools, alongside the
    specific `DataAccessPolicy` events on `query_dataset`.
    """

    @property
    def covers(self) -> frozenset[str] | None:
        return None  # catch-all

    async def before(
        self, call: ToolCall, ctx: PolicyContext
    ) -> Decision[dict[str, Any]]:
        return Allow()

    async def after(
        self, call: ToolCall, result: ToolResult, ctx: PolicyContext
    ) -> Decision[str]:
        return Allow()


class DataAccessPolicy:
    """Governs data access, and only on `query_dataset`
    (`covers={"query_dataset"}`). It inspects a STRUCTURED argument of the
    tool call — `call.arguments["column"]` — not free-form text.

    That is the lesson: 08's grounding guardrail read the model's prose
    and was fragile. A tool-call argument has a schema and a fixed shape,
    so checking it is structurally reliable — not luck of this one case.
    """

    @property
    def covers(self) -> frozenset[str] | None:
        return frozenset({"query_dataset"})

    async def before(
        self, call: ToolCall, ctx: PolicyContext
    ) -> Decision[dict[str, Any]]:
        column = str(call.arguments.get("column", ""))
        if column in _RESTRICTED_COLUMNS:
            return Deny(reason=f"column {column!r} is restricted")
        return Allow()

    async def after(
        self, call: ToolCall, result: ToolResult, ctx: PolicyContext
    ) -> Decision[str]:
        return Allow()


class DatasetExecutor:
    """Runs the two tools against DATASET. Defensive: a misrouted or
    malformed call returns an error ToolResult instead of raising, so the
    Workflow loop keeps going.
    """

    async def execute(
        self, call: ToolCall, ctx: PolicyContext
    ) -> ToolResult:
        args = call.arguments
        try:
            column = DATASET[str(args["column"])]
            if call.name == "compute_stat":
                nums = [float(v) for v in column]
                op = str(args.get("op", "mean"))
                value = (
                    statistics.mean(nums)
                    if op == "mean"
                    else statistics.pstdev(nums)
                )
                return ToolResult(
                    tool_call_id=call.id,
                    content=f"{op}({args['column']}) = {value:.2f}",
                )
            if call.name == "query_dataset":
                limit = int(args.get("limit", 5))
                return ToolResult(
                    tool_call_id=call.id, content=json.dumps(column[:limit])
                )
            return ToolResult(
                tool_call_id=call.id,
                content=f"unknown tool {call.name!r}",
                is_error=True,
            )
        except (KeyError, ValueError, TypeError) as exc:
            return ToolResult(
                tool_call_id=call.id, content=f"tool error: {exc}", is_error=True
            )


class FixedContext:
    async def select(
        self, *, message: str | None = None, ctx: PolicyContext
    ) -> ContextBundle:
        return ContextBundle(knowledge={"columns": sorted(DATASET)})


class PrintObservability:
    def emit(self, event: str, **fields: Any) -> None:
        detail = f"  {fields}" if fields else ""
        print(f"    * {event}{detail}")


async def main() -> None:
    llm = LiteLLMAdapter(PROVIDER, MODEL, LLMRequirements())

    comp = Composer(
        llm=llm,
        context=FixedContext(),
        guardrails=GuardrailSet(
            input_guardrails=[AllowInput()],
            # catch-all first, then the specific one — a covered tool runs
            # through both, in this order.
            tool_guardrails=[ToolPermissionPolicy(), DataAccessPolicy()],
            output_guardrails=[AllowOutput()],
        ),
        tools=[COMPUTE_STAT, QUERY_DATASET],
        tool_executor=DatasetExecutor(),
        observability=PrintObservability(),
    )

    ctx = PolicyContext(
        session_id="ex-09",
        provider=PROVIDER,
        model=MODEL,
        resolved_at=datetime.now(UTC),
    )

    questions = [
        # 1. compute_stat on an open column -> only the catch-all runs ->
        #    Allow -> the tool executes.
        "Use the compute_stat tool to answer. What is the mean of the "
        "reaction_time_ms column?",
        # 2. query_dataset on the restricted column -> catch-all Allow,
        #    then DataAccessPolicy Deny -> the tool never runs, the error
        #    ToolResult goes back to the model (as in 07), and the final
        #    answer says it cannot share that column.
        "Use the query_dataset tool to answer. List the raw values of the "
        "subject_id column.",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n--- question {i}: {question}")
        task = WorkflowTask(
            session_id=f"ex-09-{i}", input={"message": question}
        )
        result = await comp.orchestrator.run(task, ctx)
        print(f"  => {result.text!r}")
        print(
            f"     [finish={result.finish_reason.value} "
            f"tool_calls={result.tool_calls_made}]"
        )


if __name__ == "__main__":
    asyncio.run(main())
