"""10 - AI Core tool Redact: one guardrail, before AND after.

Domain: search over a fixed set of internal support notes, all about the
same recurring issue (so any reasonable query finds them). One note has
a phone number embedded in it, the way a real note sometimes ends up with
a contact detail pasted in by accident.

One tool: `search_notes(query, max_results)` - substring search over the
fixed notes, returns up to `max_results` matches. One `ToolGuardrail`,
`NotesAccessPolicy`, implements BOTH `before` and `after` - a variety on
`09`, which used two separate guardrails.

  * `before` clamps `max_results` to a cap via `Redact` of the
    arguments, if the model asked for more. Deterministic on a number,
    not on the model getting a phrase right.
  * `after` scans the tool's returned `content` - fixed dataset text,
    never model-generated - for a phone-like pattern, and `Redact`s it
    out before the result re-enters the conversation.

This closes the reliability ladder the series has been building, from
least to most reliable:

  1. Free text the LLM itself generates (`08`'s OutputGuardrail read the
     model's prose and had a real bug from trusting it).
  2. A structured argument the model chooses to fill in (`09`'s
     ToolGuardrail read `call.arguments["column"]` - more reliable, but
     still depends on the model deciding something).
  3. Content the app fully controls (`after`, here: the fixed dataset
     text an executor returns, never anything the model wrote or asked
     for). This is the most reliable of the three, because nothing about
     it depends on what the model decides to write or request. `before`
     here is still category 2 - it inspects an argument the model chose.

Because `after` runs before the tool result re-enters the conversation,
a redaction here keeps the model from ever seeing the raw phone number,
not just from repeating it — a stronger place to catch it than an
OutputGuardrail on the model's own final answer (`08`).

Two questions - run with real paraphrase variation, not the same prompt
repeated (`temperature=0` makes identical reruns weak evidence on their
own; see `08`'s note):

  1. Ask for a small number of results (within the cap) -> `before`
     Allows, `after` finds and redacts the phone number - the `after`
     phase alone.
  2. Ask for more results than the cap -> `before` Redacts (clamps) the
     arguments. Whether `after` ALSO fires in that same run depends on
     whether the clamped result window still contains the note with the
     phone number - this example does not assume the answer; see the
     comment on NOTES below and the finding reported next.

Finding, checked empirically (2 committed runs plus 8 distinct
paraphrases - not reruns of the same prompt, since `temperature=0` makes
identical reruns weak evidence on their own; see `08`'s note): `after`
fired on every single run of BOTH questions, with zero leaks of the raw
phone number into the final text. Clamping `max_results` to the cap in
question 2 never excluded the escalation note, because it is listed
first in NOTES on purpose. A dataset where the sensitive note ranked
lower could clamp it out of the window before `after` ever saw it - that
would be a real gap (the cap protects volume, not which rows survive it),
and this example does not paper over that; it just is not the scenario
built here. A separate, also-observed small-model quirk: `qwen3:0.6b`
sometimes calls `search_notes` more than once for the same question
(1-4 times, across the runs). Every call is gated independently and
consistently, so this did not change any guardrail's decision or cause a
leak - just extra `tool_call` / `tool_before` / `tool_after` events.

No API keys. It only needs a running Ollama daemon with the model pulled:

    ollama pull qwen3:0.6b

Run:

    uv run python examples/10_ai_core_tool_redact.py
"""

import asyncio
import json
import re
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
from odwi_llm.guardrails.types import Allow, Decision, PolicyContext, Redact
from odwi_llm.orchestration.composition import Composer
from odwi_llm.orchestration.types import WorkflowTask

PROVIDER = "ollama"
MODEL = "qwen3:0.6b"

_MAX_RESULTS_CAP = 3
_CONTACT_PATTERN = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")

# The escalation note (with the phone number) is listed FIRST on purpose:
# any query that matches at all is very likely to include it even at a
# small `max_results`. This is a deliberate demo simplification (it is
# not testing search relevance) so question 1 can reliably exercise
# `after` alone. See the module docstring for what that means for
# question 2, where `before` also clamps `max_results`.
NOTES = [
    "2026-02-06: Escalated the recurring VPN disconnect issue to network "
    "vendor support at 555-201-4477 if it happens again after the "
    "firmware patch.",
    "2026-02-03: User reports an intermittent VPN disconnect on office "
    "wifi, roughly every 20 minutes. Reboot did not help.",
    "2026-02-05: Second VPN disconnect report this week, different "
    "building. Client version is up to date on both.",
    "2026-02-10: VPN disconnect issue stopped after the router firmware "
    "update on 2026-02-08. Marking as likely root cause.",
    "2026-02-11: One more VPN disconnect reported today, same symptom as "
    "before the firmware update. Reopening the investigation.",
]

SEARCH_NOTES = ToolSpec(
    name="search_notes",
    description=(
        "Search internal support notes by substring. Returns up to "
        "max_results matching notes."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        "required": ["query"],
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


class NotesAccessPolicy:
    """One guardrail, two phases — see the module docstring for the
    reliability lesson this closes out.

    `before`: if the model asked for more than `_MAX_RESULTS_CAP`
    results, `Redact` the arguments to clamp it. Deterministic on a
    number the model filled in, regardless of how it phrased the request.

    `after`: scan the tool's own returned text for a phone-like pattern
    and `Redact` it out. The content here is fixed dataset text the
    executor returns — never anything the model generated or chose — so
    this check cannot be fooled by model phrasing the way `08`'s could.
    Only one guardrail is registered (`covers=None`), since there is only
    one tool; `09` already covers the catch-all-vs-specific dimension.
    """

    @property
    def covers(self) -> frozenset[str] | None:
        return None

    async def before(
        self, call: ToolCall, ctx: PolicyContext
    ) -> Decision[dict[str, Any]]:
        requested = int(call.arguments.get("max_results", _MAX_RESULTS_CAP))
        if requested > _MAX_RESULTS_CAP:
            clamped = dict(call.arguments)
            clamped["max_results"] = _MAX_RESULTS_CAP
            return Redact(
                reason=(
                    f"max_results {requested} exceeds the cap of "
                    f"{_MAX_RESULTS_CAP}"
                ),
                redacted=clamped,
            )
        return Allow()

    async def after(
        self, call: ToolCall, result: ToolResult, ctx: PolicyContext
    ) -> Decision[str]:
        if _CONTACT_PATTERN.search(result.content):
            cleaned = _CONTACT_PATTERN.sub("[redacted]", result.content)
            return Redact(
                reason="tool result contains a phone-like pattern",
                redacted=cleaned,
            )
        return Allow()


class NotesExecutor:
    async def execute(
        self, call: ToolCall, ctx: PolicyContext
    ) -> ToolResult:
        args = call.arguments
        try:
            query = str(args.get("query", "")).lower()
            limit = int(args.get("max_results", _MAX_RESULTS_CAP))
            matches = [n for n in NOTES if query in n.lower()]
            return ToolResult(
                tool_call_id=call.id, content=json.dumps(matches[:limit])
            )
        except (ValueError, TypeError) as exc:
            return ToolResult(
                tool_call_id=call.id, content=f"tool error: {exc}", is_error=True
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
    llm = LiteLLMAdapter(PROVIDER, MODEL, LLMRequirements())

    comp = Composer(
        llm=llm,
        context=FixedContext(),
        guardrails=GuardrailSet(
            input_guardrails=[AllowInput()],
            tool_guardrails=[NotesAccessPolicy()],
            output_guardrails=[AllowOutput()],
        ),
        tools=[SEARCH_NOTES],
        tool_executor=NotesExecutor(),
        observability=PrintObservability(),
    )

    ctx = PolicyContext(
        session_id="ex-10",
        provider=PROVIDER,
        model=MODEL,
        resolved_at=datetime.now(UTC),
    )

    # The query value is given verbatim, exactly as it appears in the
    # notes (substring search, not semantic) — same reason `09` named
    # columns verbatim: it is the part a small model has to copy exactly,
    # not the part the guardrails are actually demonstrating.
    questions = [
        # 1. within the cap -> before Allows; after should find and
        #    redact the phone number (it is in the first-listed note).
        "Call search_notes with query set to 'VPN disconnect' and "
        "max_results set to 2.",
        # 2. over the cap -> before clamps via Redact. Whether after
        #    also fires is checked empirically, not assumed.
        "Call search_notes with query set to 'VPN disconnect' and "
        "max_results set to 10.",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n--- question {i}: {question}")
        task = WorkflowTask(
            session_id=f"ex-10-{i}", input={"message": question}
        )
        result = await comp.orchestrator.run(task, ctx)
        print(f"  => {result.text!r}")
        print(
            f"     [finish={result.finish_reason.value} "
            f"tool_calls={result.tool_calls_made}]"
        )


if __name__ == "__main__":
    asyncio.run(main())
