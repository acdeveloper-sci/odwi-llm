"""08 - AI Core context patterns: a follow-up chat about a prior report.

Domain: a follow-up chat over an already-generated support-ticket
analytics report (the domain is interchangeable — the point is the
pattern, not the numbers). Three things `05`-`07` do not show:

1. `ContextBundle.prior_output` populated ONCE. `SessionContext` builds
   the bundle at construction and returns the same object every turn —
   no `select()` re-invoked per message. This is the pattern proven in
   production (fixed context for the whole session); dynamic per-turn
   selection is anticipated but has no real case yet.

2. Manual prompt assembly in the script. The reference `Workflow` calls
   `context.select()` and discards the result — folding context into the
   prompt is the app's job (see `orchestration/workflow.py`, the
   "assembling the real prompt + context is the app's job" comment, and
   `ARCHITECTURE.md` section 2). So this script calls `select()` itself,
   builds the prompt by hand, and only then hands it to `run()`.

3. Content-based grounding + two chained output guardrails.
   `GroundingClassifierOutput` sets `Allow(grounded=True/False)` from
   whether the answer cites concrete values from the report — not from a
   self-tag the model cannot be trusted to produce honestly (see its
   docstring). `LengthCapOutput` is a second, unrelated real output
   policy: it `Redact`s an over-long answer. Order matters — classify
   first, cap second, so a truncation cannot hide the markers the
   classifier reads.

The input guardrail is a real topic-eligibility check against an allowed
vocabulary — not a single banned keyword like in `06`.

Note: the default intent maps to `temperature=0`, so repeated runs are
largely the same greedy generation — consistency across runs here is not
strong evidence of robustness against model variability. `qwen3:0.6b` is
also small: it can append a stray report figure to an otherwise general
answer, which would flip `grounded` to True. Re-run if a result looks
off.

Also worth naming: `GroundingClassifierOutput` went through a real bug
during this example's own development (it first trusted a self-tag the
model was supposed to add to its own answer, which does not hold up).
The `odwi-llm` mechanism — `Decision`, the two guardrails composing in
order, the fail-fast on construction — never changed across that
diagnosis; the bug was entirely an application-layer choice, which
policy to use to decide grounding. That is the boundary this package
draws in practice, not just in the docs: `odwi-llm` gives you the
mechanism, the app still has to get its own policy right and verify it
empirically, the same as any other application code.

No API keys. It only needs a running Ollama daemon with the model pulled:

    ollama pull qwen3:0.6b

Run:

    uv run python examples/08_ai_core_context_patterns.py
"""

import asyncio
import json
import re
from datetime import UTC, datetime
from typing import Any

from odwi_llm.adapters.litellm_adapter import LiteLLMAdapter
from odwi_llm.context.types import ContextBundle
from odwi_llm.core.requirements import LLMRequirements
from odwi_llm.core.types import ChatTurnResponse, LLMResponse, Message
from odwi_llm.guardrails.port import GuardrailSet
from odwi_llm.guardrails.types import (
    Allow,
    Decision,
    Deny,
    PolicyContext,
    Redact,
)
from odwi_llm.orchestration.composition import Composer
from odwi_llm.orchestration.types import WorkflowTask

PROVIDER = "ollama"
MODEL = "qwen3:0.6b"

# --- the "already generated" report, a fixed dict (no LLM needed) -------

DOMAIN_NOTES = {
    "resolution_time_unit": "business hours",
    "sla_target_pct": 90,
    "kpi_note": "sla_met_pct at or above sla_target_pct is healthy",
}
REPORT_DATA = {
    "period": "2026-W01 to 2026-W04",
    "total_tickets": 1240,
    "avg_resolution_business_hours": 6.4,
    "sla_met_pct": 92,
    "busiest_week": "2026-W03",
    "busiest_week_tickets": 415,
}
PRIOR_REPORT = {
    "summary": (
        "Ticket volume peaked in 2026-W03 (415 tickets, ~34% of the "
        "month). Average resolution held at 6.4 business hours and SLA "
        "attainment was 92%, above the 90% target."
    ),
    "findings": [
        "The W03 spike aligns with a product release; no SLA regression "
        "despite the load.",
        "Resolution time is stable week over week.",
    ],
}


def _report_markers(*sources: dict[str, Any]) -> frozenset[str]:
    """Distinctive concrete values pulled straight from the report dicts,
    so the grounding check cannot silently drift if the report changes.
    Numbers become their string form; week codes like `2026-W03` are
    pulled out of string values. Tokens shorter than 3 chars are dropped
    — a bare two-digit number is too common to be a reliable marker.
    """
    markers: set[str] = set()
    week = re.compile(r"\b\d{4}-W\d{2}\b")

    def walk(value: object) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            markers.add(str(value))
        elif isinstance(value, str):
            markers.update(week.findall(value))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for src in sources:
        walk(src)
    return frozenset(m for m in markers if len(m) >= 3)


_REPORT_MARKERS = _report_markers(REPORT_DATA, PRIOR_REPORT)

_MAX_ANSWER_CHARS = 200  # a demo threshold, not a tuned limit — same
# honesty caveat as the contrivances in 06 / 07.

# Allowed-topic vocabulary for the eligibility check.
_ALLOWED_TERMS = frozenset(
    {
        "ticket",
        "sla",
        "resolution",
        "volume",
        "backlog",
        "queue",
        "support",
        "response time",
        "kpi",
        "week",
    }
)


class SessionContext:
    """Builds the bundle ONCE and returns the same object on every
    `select()`, ignoring `message`. Pattern 1 above.
    """

    def __init__(
        self,
        *,
        knowledge: dict[str, Any],
        data: dict[str, Any],
        prior_output: dict[str, Any],
    ) -> None:
        self._bundle = ContextBundle(
            knowledge=knowledge, data=data, prior_output=prior_output
        )

    async def select(
        self, *, message: str | None = None, ctx: PolicyContext
    ) -> ContextBundle:
        # `message` ignored on purpose. The Workflow also calls this once
        # per turn and discards it; returning the same object is free.
        return self._bundle


class TopicEligibilityInput:
    """Denies questions outside the report's topic — a real check against
    `_ALLOWED_TERMS`, not one banned keyword. It reads only the QUESTION
    segment the assembler wrote (the app owns both, so the format is
    known); the rest of the message is embedded report context.
    """

    async def check(
        self, message: Message, ctx: PolicyContext
    ) -> Decision[str]:
        question = message.content.rsplit("QUESTION:", 1)[-1].lower()
        if any(term in question for term in _ALLOWED_TERMS):
            return Allow()
        return Deny(
            reason="question is outside the support-ticket report's topic"
        )


class GroundingClassifierOutput:
    """Sets `Allow(grounded=...)` from whether the answer cites concrete
    values from the report data.

    It does NOT ask the model to self-tag its provenance. Relying on a
    model to report its own grounding is a bad pattern in general, not
    just unreliable on a 0.6b model: the same model that can hallucinate
    a figure would be the one certifying whether it hallucinated.
    Checking the response for real overlap with the report is more honest
    as a pattern — and here it is also deterministic where the self-tag
    was not.

    "Grounded" here just means "the answer restates data from this
    report" — a coarse but checkable signal. `Allow` cannot rewrite text;
    this guardrail MUST run before `LengthCapOutput` so a truncation
    cannot drop the markers it reads.
    """

    async def check(
        self, response: LLMResponse | ChatTurnResponse, ctx: PolicyContext
    ) -> Decision[str]:
        cites_report = any(m in response.text for m in _REPORT_MARKERS)
        return Allow(grounded=cites_report)


class LengthCapOutput:
    """A second, unrelated output policy: cap the answer length. `qwen3:
    0.6b` often returns bloated or duplicated text, so this `Redact`
    fires often enough to see in a run. Runs AFTER the grounding
    classifier so a truncation can never hide the report markers that
    classifier reads. A real policy would set the limit from the
    product's needs, not a round demo number.
    """

    async def check(
        self, response: LLMResponse | ChatTurnResponse, ctx: PolicyContext
    ) -> Decision[str]:
        if len(response.text) > _MAX_ANSWER_CHARS:
            return Redact(
                reason=f"answer exceeds {_MAX_ANSWER_CHARS} chars",
                redacted=response.text[:_MAX_ANSWER_CHARS].rstrip() + " ...",
            )
        return Allow()


class PrintObservability:
    def emit(self, event: str, **fields: Any) -> None:
        detail = f"  {fields}" if fields else ""
        print(f"    * {event}{detail}")


def _assemble(bundle: ContextBundle, question: str) -> str:
    # Pattern 2: the reference Workflow does not do this — it calls
    # select() and drops the result. Folding context into the prompt is
    # app-specific, so the app does it here, before run(). The model is
    # NOT asked to tag its answer — grounding is judged afterwards from
    # the answer's content (see GroundingClassifierOutput).
    return (
        "You are answering follow-up questions about a support-ticket "
        "analytics report. Use the data below when it answers the "
        "question; otherwise answer from general knowledge.\n\n"
        f"DOMAIN NOTES: {json.dumps(bundle.knowledge)}\n"
        f"REPORT DATA: {json.dumps(bundle.data)}\n"
        f"PRIOR REPORT: {json.dumps(bundle.prior_output)}\n\n"
        f"QUESTION: {question}"
    )


async def main() -> None:
    llm = LiteLLMAdapter(PROVIDER, MODEL, LLMRequirements())

    # Built once, here — reused for every turn below.
    context = SessionContext(
        knowledge=DOMAIN_NOTES, data=REPORT_DATA, prior_output=PRIOR_REPORT
    )

    comp = Composer(
        llm=llm,
        context=context,
        guardrails=GuardrailSet(
            input_guardrails=[TopicEligibilityInput()],
            # order matters: classify grounding first, cap length second —
            # a truncation must not drop the markers the classifier reads.
            output_guardrails=[
                GroundingClassifierOutput(),
                LengthCapOutput(),
            ],
        ),
        observability=PrintObservability(),
    )

    ctx = PolicyContext(
        session_id="ex-08",
        provider=PROVIDER,
        model=MODEL,
        resolved_at=datetime.now(UTC),
    )

    questions = [
        # 1. answerable from the report -> the answer cites report values
        #    -> grounded=True
        "Which week had the highest ticket volume, and did SLA hold that week?",
        # 2. on-topic but a definition the report cannot answer -> no
        #    report values in the answer -> grounded=False
        "In general, what does the term SLA stand for and why does it "
        "matter for a support team?",
        # 3. outside the report's topic -> denied at input -> CONTENT_FILTER
        "What is the capital of France?",
    ]

    for i, question in enumerate(questions, 1):
        # select() ONCE per turn, here, for the manual assembly — the same
        # fixed bundle every time.
        bundle = await context.select(message=question, ctx=ctx)
        assembled = _assemble(bundle, question)
        task = WorkflowTask(
            session_id=f"ex-08-{i}", input={"message": assembled}
        )

        print(f"\n--- question {i}: {question}")
        result = await comp.orchestrator.run(task, ctx)
        print(f"  => {result.text!r}")
        print(
            f"     [finish={result.finish_reason.value} "
            f"grounded={result.grounded}]"
        )


if __name__ == "__main__":
    asyncio.run(main())
