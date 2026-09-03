"""Helpers shared by the adapters. No provider library is imported here."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from odwi_llm.core.errors import LLMError, LLMProviderError, LLMRateLimitError
from odwi_llm.core.requirements import LLMCapabilities, LLMRequirements
from odwi_llm.core.types import FinishReason, Message

_R = TypeVar("_R")

_FINISH = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "max_tokens": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "function_call": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.CONTENT_FILTER,
}


def map_finish_reason(raw: str | None) -> FinishReason:
    if raw is None:
        return FinishReason.STOP
    return _FINISH.get(raw.lower(), FinishReason.OTHER)


def retry_after_seconds(exc: BaseException) -> float | None:
    headers = getattr(exc, "headers", None)
    if not isinstance(headers, dict):
        headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def message_to_dict(message: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": message.role.value, "content": message.content}
    if message.tool_call_id is not None:
        out["tool_call_id"] = message.tool_call_id
    if message.name is not None:
        out["name"] = message.name
    return out


def unmet_requirements(req: LLMRequirements, caps: LLMCapabilities) -> list[str]:
    missing: list[str] = []
    if req.structured_output and not caps.structured_output:
        missing.append("structured_output")
    if req.tool_calling and not caps.tool_calling:
        missing.append("tool_calling")
    if req.streaming and not caps.streaming:
        missing.append("streaming")
    if req.vision and not caps.vision:
        missing.append("vision")
    if req.min_context_tokens is not None and (
        caps.context_tokens is None or caps.context_tokens < req.min_context_tokens
    ):
        missing.append("min_context_tokens")
    return missing


# --- bounded technical retry (design §5.3) ---------------------------

MAX_RETRY_ATTEMPTS = 3
# Only the transient / technical ones. Auth, context length, content
# filter, schema and CapabilityError are not retried — retrying does not
# help and wastes quota.
RETRYABLE_ERRORS: tuple[type[LLMError], ...] = (LLMRateLimitError, LLMProviderError)
_BACKOFF_SECONDS = (0.5, 1.0)


async def retry_sleep(exc: LLMError, attempt: int) -> None:
    """Wait before the next attempt: honour `retry_after_seconds` if the
    error carries one, otherwise a small fixed backoff."""
    if isinstance(exc, LLMRateLimitError) and exc.retry_after_seconds:
        await asyncio.sleep(exc.retry_after_seconds)
        return
    await asyncio.sleep(_BACKOFF_SECONDS[min(attempt - 1, len(_BACKOFF_SECONDS) - 1)])


async def call_with_retry(
    fn: Callable[[], Awaitable[_R]], *, max_attempts: int = MAX_RETRY_ATTEMPTS
) -> _R:
    """Bounded technical retry for a non-streaming call (design §5.3).

    `fn` must already translate provider errors to the §4.5 hierarchy;
    this retries only `RETRYABLE_ERRORS`. `max_attempts` total (3), not
    configurable by callers — §5.3 fixes it.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except RETRYABLE_ERRORS as exc:
            if attempt == max_attempts:
                raise
            await retry_sleep(exc, attempt)
    raise AssertionError("unreachable")
