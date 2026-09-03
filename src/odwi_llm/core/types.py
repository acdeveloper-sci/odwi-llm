"""Port types — design §4.1–§4.3.

Implemented verbatim from the contract. Any change here is a contract
change and must be discussed against the design doc first, not improvised.
"""

from enum import Enum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

# --- §4.1 Base types -------------------------------------------------


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    role: Role
    content: str
    # Only for role=TOOL: which tool call this message answers
    tool_call_id: str | None = None
    name: str | None = None


class Intent(str, Enum):
    """Provider-agnostic expression of sampling intent.

    Adapters map this to temperature/top_p or ignore it if unsupported.
    """

    DETERMINISTIC = "deterministic"
    BALANCED = "balanced"
    CREATIVE = "creative"


class LLMRequest(BaseModel):
    messages: list[Message]
    intent: Intent = Intent.DETERMINISTIC
    max_output_tokens: int | None = None
    # Opaque metadata for logging/tracing only. Never sent to the provider.
    metadata: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    # Optional; some providers/libraries report it, some do not
    estimated_cost_usd: float | None = None


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    OTHER = "other"


class LLMResponse(BaseModel):
    text: str
    model: str  # Resolved model id actually used
    provider: str  # Resolved provider actually used
    finish_reason: FinishReason
    usage: Usage
    # Raw provider payload for debugging. Never used by domain code.
    raw: dict[str, Any] | None = None


# --- §4.2 Structured output (Stage 1) ------------------------------

T = TypeVar("T", bound=BaseModel)


class StructuredResponse(LLMResponse, Generic[T]):
    data: T  # Parsed and validated against the requested schema


# --- §4.3 Tool calling and streaming (declared in Stage 1) --------


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters_schema: dict[str, Any]  # JSON Schema, generated from a Pydantic model


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]  # Already parsed; never a raw JSON string


class ToolResult(BaseModel):
    tool_call_id: str
    content: str  # Serialized result handed back to the model
    is_error: bool = False


class ChatTurnResponse(LLMResponse):
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # If tool_calls is non-empty, the orchestrator must execute them and call again.


class StreamChunk(BaseModel):
    kind: Literal["text_delta", "tool_call_delta", "usage", "done"]
    text: str | None = None
    tool_call: ToolCall | None = None  # Complete tool call when kind == tool_call_delta
    usage: Usage | None = None
    finish_reason: FinishReason | None = None
