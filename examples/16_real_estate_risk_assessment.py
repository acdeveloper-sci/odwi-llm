"""16 - AI Core: tools + schema in the same turn (Design v0.7, Fase F).

This is the "both" path of the `Composer`/`Workflow` dispatch table
(design §10): `tools` AND `schema` configured together. The loop runs
until a clean close (no more `tool_calls`), and only then does `Workflow`
make one extra `structured()` call with the accumulated message history.
This example is not just another illustration - it is the real case that
motivated Fase F (a real-estate investment-analysis domain needing
`structured()` + tools + schema in one turn was, until that delta,
impossible to express with this package's surface). Running it against a
real model is meant to validate that Fase F's code actually does what
Design v0.7 says, not just exercise it as one more demo.

It is also the first example in the series to COMBINE things `07`/`09`/
`10` already demonstrated separately (a tool-calling loop; a catch-all
`ToolGuardrail` and a specific one coexisting via `covers`; a real
`after` `Redact` over a tool's own result) with what `15` just showed on
its own (`schema=`). Nothing new is introduced at the guardrail level -
this is where those pieces meet.

Domain: real-estate investment risk assessment. Two tools:

  * `get_comparable_sales(location, radius_km)` - simulated: returns a
    short list of comparable sales, each one INCLUDING `owner_name` - a
    real piece of PII that should never reach the model. This is the
    same category of leak the whole guardrail-reliability ladder (`08`-
    `10`) has been building toward: content the app fully controls
    (fixed dataset shape returned by the executor), the most reliable
    place to catch it.
  * `compute_investment_metrics(purchase_price, rental_income, expenses)`
    - pure calculation (cap rate), no PII at all.

Tool guardrails - catch-all + specific, same pattern as `09`:

  * `ToolPermissionPolicy` (`covers=None`) - permissive catch-all, runs on
    both tools. Covers `compute_investment_metrics` with no restriction.
  * `ComparableSalesPrivacyPolicy` (`covers={"get_comparable_sales"}`) -
    its `after` does a REAL `Redact`: strips `owner_name` from every
    comparable in the tool's result before it re-enters the conversation.
    This is the guardrail with real content that motivated case C in the
    design notes - not a decorative pass-through like `07`'s.

`Workflow` does not expose its internal message history to the caller by
design (see `orchestration/workflow.py` - assembling/reading the prompt
is the app's concern, this reference `Workflow` only builds it
internally for the `structured()` call). So the redaction is checked here
the same way any consuming app actually could: the owner names must not
appear in `result.text` (the model's final free-text framing before the
structured call) nor leak into `result.data` - not by inspecting private
internals.

Finding, checked empirically (3 runs, not just the happy path once): the
tools+schema loop closed cleanly and `structured()` fired every time,
with zero owner-name leaks into `result.text` or `result.data`.
`qwen3:0.6b` called `get_comparable_sales` more than once per run (3-4
times observed) before settling - the same small-model quirk `10` and
`09` already documented - and `compute_investment_metrics` exactly once;
`tool_calls_made` varied (4-5) across runs as a result. Every single
`get_comparable_sales` call was redacted independently and consistently,
so the repetition never became a repetition of the leak.

No API keys. It only needs a running Ollama daemon with the model pulled:

    ollama pull qwen3:0.6b

Run:

    uv run python examples/16_real_estate_risk_assessment.py
"""

import asyncio
import json
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel

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
from odwi_llm.guardrails.types import Allow, Decision, PolicyContext, Redact
from odwi_llm.orchestration.composition import Composer
from odwi_llm.orchestration.types import WorkflowTask

PROVIDER = "ollama"
MODEL = "qwen3:0.6b"

# Fixed, deterministic PII - real names on purpose, so the demonstration
# at the bottom can check for their literal absence, not just the field
# name "owner_name".
_COMPARABLES: list[dict[str, Any]] = [
    {"address": "12 Oak St", "sale_price": 410_000, "owner_name": "Maria Gonzalez"},
    {"address": "48 Pine Ave", "sale_price": 395_000, "owner_name": "David Chen"},
    {"address": "7 Cedar Ln", "sale_price": 430_000, "owner_name": "Priya Patel"},
]
_OWNER_NAMES: list[str] = [str(c["owner_name"]) for c in _COMPARABLES]

GET_COMPARABLE_SALES = ToolSpec(
    name="get_comparable_sales",
    description=(
        "Return recent comparable property sales near a location, within "
        "a radius in kilometers."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "radius_km": {"type": "number"},
        },
        "required": ["location"],
    },
)
COMPUTE_INVESTMENT_METRICS = ToolSpec(
    name="compute_investment_metrics",
    description=(
        "Compute investment metrics (cap rate) from purchase price, "
        "monthly rental income and monthly expenses."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "purchase_price": {"type": "number"},
            "rental_income": {"type": "number"},
            "expenses": {"type": "number"},
        },
        "required": ["purchase_price", "rental_income", "expenses"],
    },
)


class RiskAssessment(BaseModel):
    property_address: str
    risk_level: Literal["low", "medium", "high"]
    estimated_value: float
    key_risk_factors: list[str]
    recommendation: str


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
    """Catch-all (`covers=None`) - same broad, lightweight role `09` gave
    it: covers every tool, blocks nothing here. `compute_investment_metrics`
    has no PII concern, so the catch-all alone is enough for it.
    """

    @property
    def covers(self) -> frozenset[str] | None:
        return None

    async def before(
        self, call: ToolCall, ctx: PolicyContext
    ) -> Decision[dict[str, Any]]:
        return Allow()

    async def after(
        self, call: ToolCall, result: ToolResult, ctx: PolicyContext
    ) -> Decision[str]:
        return Allow()


class ComparableSalesPrivacyPolicy:
    """Only covers `get_comparable_sales`. Its `after` is a REAL `Redact`
    - not a pass-through: it strips `owner_name` from every comparable in
    the tool's own result before that result re-enters the conversation.
    This is content the app fully controls (a fixed field on data the
    executor built), the most reliable guardrail input in the series'
    reliability ladder (see `10`'s docstring) - not something depending
    on the model's phrasing or choices.
    """

    @property
    def covers(self) -> frozenset[str] | None:
        return frozenset({"get_comparable_sales"})

    async def before(
        self, call: ToolCall, ctx: PolicyContext
    ) -> Decision[dict[str, Any]]:
        return Allow()

    async def after(
        self, call: ToolCall, result: ToolResult, ctx: PolicyContext
    ) -> Decision[str]:
        comparables = json.loads(result.content)
        cleaned = [
            {k: v for k, v in comp.items() if k != "owner_name"}
            for comp in comparables
        ]
        return Redact(
            reason="comparable sales results include owner_name (PII)",
            redacted=json.dumps(cleaned),
        )


class RealEstateExecutor:
    """Runs both tools. `get_comparable_sales` prints its raw output
    (owner names included) directly to the console - labeled clearly as
    the pre-redaction content - so the demonstration below has something
    concrete to contrast against the guardrail-cleaned version and the
    model's final answer.
    """

    async def execute(
        self, call: ToolCall, ctx: PolicyContext
    ) -> ToolResult:
        args = call.arguments
        if call.name == "get_comparable_sales":
            print(
                "  [executor] raw get_comparable_sales output "
                "(owner_name is real PII, never meant to reach the model):"
            )
            for comp in _COMPARABLES:
                print(f"    {comp}")
            return ToolResult(
                tool_call_id=call.id, content=json.dumps(_COMPARABLES)
            )
        if call.name == "compute_investment_metrics":
            try:
                price = float(args["purchase_price"])
                rental = float(args["rental_income"])
                expenses = float(args["expenses"])
            except (KeyError, TypeError, ValueError) as exc:
                return ToolResult(
                    tool_call_id=call.id,
                    content=f"tool error: {exc}",
                    is_error=True,
                )
            noi_annual = (rental - expenses) * 12
            cap_rate_pct = round(noi_annual / price * 100, 2) if price else 0.0
            return ToolResult(
                tool_call_id=call.id,
                content=json.dumps(
                    {"noi_annual": noi_annual, "cap_rate_pct": cap_rate_pct}
                ),
            )
        return ToolResult(
            tool_call_id=call.id,
            content=f"unknown tool {call.name!r}",
            is_error=True,
        )


class FixedContext:
    async def select(
        self, *, message: str | None = None, ctx: PolicyContext
    ) -> ContextBundle:
        return ContextBundle()


class PrintObservability:
    def emit(self, event: str, **fields: Any) -> None:
        detail = f"  {fields}" if fields else ""
        print(f"    * {event}{detail}")


async def main() -> None:
    llm = LiteLLMAdapter(
        PROVIDER,
        MODEL,
        LLMRequirements(tool_calling=True, structured_output=True),
    )

    comp = Composer(
        llm=llm,
        context=FixedContext(),
        guardrails=GuardrailSet(
            input_guardrails=[AllowInput()],
            tool_guardrails=[
                ToolPermissionPolicy(),
                ComparableSalesPrivacyPolicy(),
            ],
            output_guardrails=[AllowOutput()],
        ),
        tools=[GET_COMPARABLE_SALES, COMPUTE_INVESTMENT_METRICS],
        tool_executor=RealEstateExecutor(),
        schema=RiskAssessment,
        observability=PrintObservability(),
    )

    ctx = PolicyContext(
        session_id="ex-16",
        provider=PROVIDER,
        model=MODEL,
        resolved_at=datetime.now(UTC),
    )
    task = WorkflowTask(
        session_id="ex-16",
        input={
            "message": (
                "I'm considering buying a property at 48 Pine Ave, "
                "Springfield, for $400,000. Expected rental income is "
                "$3,200/month and expenses are $900/month. Look up "
                "comparable sales within 2km and compute the investment "
                "metrics, then assess the investment risk."
            )
        },
    )

    print("events:")
    result = await comp.orchestrator.run(task, ctx)

    print(f"\ntext: {result.text!r}")
    print(
        f"[finish={result.finish_reason.value} "
        f"tool_calls={result.tool_calls_made}]"
    )
    print(f"data: {result.data!r}")

    assert isinstance(result.data, RiskAssessment), (
        "the tools+schema path must populate WorkflowResult.data"
    )

    # The actual proof the after() Redact held, checked the same way a
    # consuming app could: no owner name literal anywhere the app can
    # see - not just in the field name, in the free text and the
    # structured fields too.
    visible_text = result.text + " ".join(result.data.key_risk_factors) + (
        result.data.recommendation
    )
    leaked = [name for name in _OWNER_NAMES if name in visible_text]
    print(f"\nowner names that existed in the raw tool output: {_OWNER_NAMES}")
    print(f"owner names leaked into the visible result: {leaked or 'none'}")
    assert not leaked, "owner_name leaked past the after() Redact"


if __name__ == "__main__":
    asyncio.run(main())
