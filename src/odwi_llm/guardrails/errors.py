"""Policy error hierarchy — design §4 (Stage 2, v0.5).

Separate from `LLMError` (`core/errors.py`): the same "the domain never
sees a raw error" principle, one level up (Specify §3). A policy failure
is not an LLM failure and code that handles one should not accidentally
catch the other.
"""


class PolicyError(Exception):
    """Base for every policy / guardrail failure."""


class PolicyConfigError(PolicyError):
    """A GuardrailSet or composition was built wrong — a required phase is
    missing, a list holds the wrong type, tools have no covering guardrail.
    Raised while assembling the composition, never at runtime.
    """


class PolicyDeniedError(PolicyError):
    """A `Deny` decision reached the orchestrator with nobody handling it
    explicitly. It should never escape silently; an orchestrator that does
    not turn a `Deny` into a result raises this instead.
    """
