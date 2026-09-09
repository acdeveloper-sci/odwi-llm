"""Composition root — design §11 (Stage 2, v0.6)."""

from odwi_llm.context.port import ContextPort
from odwi_llm.core.port import LLMPort
from odwi_llm.core.types import ToolSpec
from odwi_llm.guardrails.port import GuardrailSet
from odwi_llm.observability.null import NullObservability
from odwi_llm.observability.port import ObservabilityPort
from odwi_llm.orchestration.port import Orchestrator, ToolExecutor
from odwi_llm.orchestration.workflow import Workflow


class Composer:
    """A `Composer` instance IS the concrete realization of AI Core
    (Specify §1) for a given app — not a component inside AI Core, nor
    something next to it. Specify uses "AI Core" for the architectural
    concept (the box in the diagram: Policy + Workflow + Context + LLM
    Abstraction + Tools); `Composer` is what assembles it in real code.
    The name deliberately does not repeat "Core".

    It assembles already-built pieces — it does not resolve config or wire
    guardrails by phase implicitly (Specify §5, "no class that does
    everything").
    """

    def __init__(
        self,
        *,
        llm: LLMPort,
        context: ContextPort,
        guardrails: GuardrailSet,
        tools: list[ToolSpec] | None = None,
        tool_executor: ToolExecutor | None = None,
        orchestrator: Orchestrator | None = None,
        observability: ObservabilityPort | None = None,
    ) -> None:
        self.llm = llm
        self.context = context
        self.guardrails = guardrails
        self.observability: ObservabilityPort = (
            observability or NullObservability()
        )
        # A custom orchestrator ignores tools / tool_executor — running
        # tools is its own concern (§11).
        self.orchestrator: Orchestrator = orchestrator or Workflow(
            llm=llm,
            context=context,
            guardrails=guardrails,
            tools=tools,
            tool_executor=tool_executor,
            observability=self.observability,
        )
