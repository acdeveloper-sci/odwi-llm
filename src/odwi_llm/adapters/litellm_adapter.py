"""LiteLLM adapter — design §5, applying everything FINDINGS.md §1 says an
adapter must absorb.

What is handled here so the domain never sees it:
  - reasoning: `message.reasoning` / `message.reasoning_content` /
    `provider_specific_fields.thought_signatures`, and their streaming
    form `delta.reasoning` / `delta.reasoning_content` — never surfaced.
  - tool-call `arguments`: LiteLLM always hands back a JSON string ->
    parsed to a dict here (`ToolCall.arguments` is a dict, §4.3).
  - tool-call `id`: LiteLLM smuggles a Gemini thought signature into it
    as `<id>__thought__<sig>` -> the suffix is cut.
  - `finish_reason`: `"stop"` / `"length"` / `"tool_calls"` /
    `"content_filter"` -> `FinishReason`. Empty text + LENGTH is a valid
    result, not an error.
  - errors: any LiteLLM / provider exception -> the §4.5 hierarchy.
  - streaming: `stream_options={"include_usage": True}`, and the trailing
    `choices: []` usage chunk is consumed (not indexed into).
  - cost: from `adapters.pricing`, never `litellm.completion_cost()`.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

import litellm
from pydantic import ValidationError

from odwi_llm.adapters.config import ProviderConfig
from odwi_llm.adapters.pricing import estimate_cost_usd
from odwi_llm.core.errors import (
    LLMAuthError,
    LLMContentFilteredError,
    LLMContextLengthError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMSchemaError,
)
from odwi_llm.core.port import LLMPort
from odwi_llm.core.requirements import (
    CapabilityError,
    LLMCapabilities,
    LLMRequirements,
)
from odwi_llm.core.types import (
    ChatTurnResponse,
    FinishReason,
    Intent,
    LLMRequest,
    LLMResponse,
    Message,
    StreamChunk,
    StructuredResponse,
    T,
    ToolCall,
    ToolSpec,
    Usage,
)

_SCHEMA_RETRIES = 2  # total attempts = 1 + retries (§5.3, bounded)

_litellm_configured = False


def _ensure_litellm_configured() -> None:
    """Apply our litellm settings on first adapter use, not at import time.

    Importing this module must not mutate global litellm state. Idempotent:
    re-running it just re-sets the same flags.
      - drop_params: silently drop a param a provider rejects (e.g.
        temperature=0 on an adaptive-thinking model) instead of erroring (§5.2).
      - suppress_debug_info: quieter console output.
    """
    global _litellm_configured
    if _litellm_configured:
        return
    litellm.drop_params = True
    litellm.suppress_debug_info = True
    _litellm_configured = True

# LiteLLM model-string prefix per provider.
_PREFIX = {
    "gemini": "gemini",
    "groq": "groq",
    "ollama": "ollama_chat",
    "lmstudio": "lm_studio",
}
_DEFAULT_KEY_ENV = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}
_DEFAULT_BASE_URL = {
    "ollama": "http://localhost:11434",
    "lmstudio": "http://localhost:1234/v1",
}

# DETERMINISTIC -> 0, CREATIVE -> 1, BALANCED -> omit (§5.2, FINDINGS §2).
_INTENT_TEMPERATURE = {Intent.DETERMINISTIC: 0.0, Intent.CREATIVE: 1.0}

# capabilities, codified from §9.3 table B (LiteLLM covers all four ops for
# every lab provider) + public model context windows. vision stays False:
# the lab did not exercise it, so an app may not require what we did not
# verify. Keyed by (provider, model) so a future provider serving a model
# of the same name but different capabilities does not silently inherit
# this entry. Unknown (provider, model) -> the conservative default below.
_OPS = {"structured_output": True, "tool_calling": True, "streaming": True, "vision": False}
_KNOWN_CAPS: dict[tuple[str, str], LLMCapabilities] = {
    ("gemini", "gemini-3.5-flash-lite"): LLMCapabilities(**_OPS, context_tokens=1_000_000),
    ("groq", "openai/gpt-oss-120b"): LLMCapabilities(**_OPS, context_tokens=128_000),
    ("ollama", "qwen3:0.6b"): LLMCapabilities(**_OPS, context_tokens=32_768),
    ("lmstudio", "llama-3.2-3b-instruct"): LLMCapabilities(**_OPS, context_tokens=131_072),
}
_DEFAULT_CAPS = LLMCapabilities(**_OPS, context_tokens=None)


def _map_finish(raw: str | None) -> FinishReason:
    return {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        "max_tokens": FinishReason.LENGTH,
        "tool_calls": FinishReason.TOOL_CALLS,
        "function_call": FinishReason.TOOL_CALLS,
        "content_filter": FinishReason.CONTENT_FILTER,
    }.get((raw or "").lower(), FinishReason.OTHER if raw else FinishReason.STOP)


def _retry_after_seconds(exc: BaseException) -> float | None:
    headers = getattr(exc, "headers", None)
    if not isinstance(headers, dict):
        resp = getattr(exc, "response", None)
        headers = getattr(resp, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _message_dict(message: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": message.role.value, "content": message.content}
    if message.tool_call_id is not None:
        out["tool_call_id"] = message.tool_call_id
    if message.name is not None:
        out["name"] = message.name
    return out


def _unmet(req: LLMRequirements, caps: LLMCapabilities) -> list[str]:
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


class LiteLLMAdapter(LLMPort):
    def __init__(
        self,
        provider: str,
        model: str,
        requirements: LLMRequirements,
        config: ProviderConfig | None = None,
    ) -> None:
        if provider not in _PREFIX:
            raise ValueError(f"unknown provider {provider!r}; expected one of {sorted(_PREFIX)}")
        _ensure_litellm_configured()
        self._provider = provider
        self._model = model
        self._litellm_model = f"{_PREFIX[provider]}/{model}"
        self._caps = _KNOWN_CAPS.get((provider, model), _DEFAULT_CAPS)

        cfg = config or ProviderConfig()
        if cfg.api_key is None and cfg.api_key_env is None and provider in _DEFAULT_KEY_ENV:
            cfg = ProviderConfig(
                api_key_env=_DEFAULT_KEY_ENV[provider],
                base_url=cfg.base_url,
                timeout_s=cfg.timeout_s,
            )
        if cfg.base_url is None and provider in _DEFAULT_BASE_URL:
            cfg = ProviderConfig(
                api_key=cfg.api_key,
                api_key_env=cfg.api_key_env,
                base_url=_DEFAULT_BASE_URL[provider],
                timeout_s=cfg.timeout_s,
            )
        self._config = cfg

        unmet = _unmet(requirements, self._caps)
        if unmet:
            raise CapabilityError(
                f"{provider}/{model} does not meet requirements: {', '.join(unmet)}"
            )

    @property
    def capabilities(self) -> LLMCapabilities:
        return self._caps

    # --- request / response plumbing --------------------------------
    def _kwargs(self, request: LLMRequest, **extra: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._litellm_model,
            "messages": [_message_dict(m) for m in request.messages],
            "timeout": self._config.timeout_s,
        }
        temperature = _INTENT_TEMPERATURE.get(request.intent)
        if temperature is not None:
            kwargs["temperature"] = temperature
        if request.max_output_tokens is not None:
            kwargs["max_tokens"] = request.max_output_tokens
        api_key = self._config.resolve_api_key()
        if api_key:
            kwargs["api_key"] = api_key
        if self._config.base_url:
            kwargs["api_base"] = self._config.base_url
        kwargs.update(extra)
        return kwargs

    async def _acompletion(self, request: LLMRequest, **extra: Any) -> Any:
        try:
            return await litellm.acompletion(**self._kwargs(request, **extra))
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - the domain must not see raw provider errors
            raise self._map_error(exc) from exc

    def _usage(self, resp: Any) -> Usage:
        raw = getattr(resp, "usage", None)
        inp = int(getattr(raw, "prompt_tokens", 0) or 0)
        out = int(getattr(raw, "completion_tokens", 0) or 0)
        model = getattr(resp, "model", None) or self._model
        return Usage(
            input_tokens=inp,
            output_tokens=out,
            estimated_cost_usd=estimate_cost_usd(model, inp, out),
        )

    def _llm_response(self, resp: Any) -> LLMResponse:
        choice = resp.choices[0]
        # reasoning fields on choice.message are deliberately not read.
        return LLMResponse(
            text=choice.message.content or "",
            model=getattr(resp, "model", None) or self._model,
            provider=self._provider,
            finish_reason=_map_finish(choice.finish_reason),
            usage=self._usage(resp),
        )

    def _tool_call(self, raw: Any) -> ToolCall:
        arguments = raw.function.arguments
        if isinstance(arguments, str):
            try:
                parsed: dict[str, Any] = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                parsed = {}
        elif isinstance(arguments, dict):
            parsed = arguments
        else:
            parsed = {}
        call_id = str(raw.id or "").split("__thought__", 1)[0]
        return ToolCall(id=call_id, name=raw.function.name, arguments=parsed)

    # --- LLMPort ---------------------------------------------------
    async def generate(self, request: LLMRequest) -> LLMResponse:
        return self._llm_response(await self._acompletion(request))

    async def structured(
        self, request: LLMRequest, schema: type[T]
    ) -> StructuredResponse[T]:
        last: Exception | None = None
        for _ in range(_SCHEMA_RETRIES + 1):
            resp = await self._acompletion(request, response_format=schema)
            text = resp.choices[0].message.content or ""
            try:
                data = schema.model_validate_json(text)
            except (ValidationError, ValueError) as exc:
                last = exc
                continue
            return StructuredResponse[T](
                text=text,
                model=getattr(resp, "model", None) or self._model,
                provider=self._provider,
                finish_reason=_map_finish(resp.choices[0].finish_reason),
                usage=self._usage(resp),
                data=data,
            )
        raise LLMSchemaError(
            f"structured output did not validate after {_SCHEMA_RETRIES + 1} attempts: {last}"
        )

    async def chat_with_tools(
        self, request: LLMRequest, tools: list[ToolSpec]
    ) -> ChatTurnResponse:
        litellm_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters_schema,
                },
            }
            for tool in tools
        ]
        resp = await self._acompletion(
            request, tools=litellm_tools, tool_choice="auto"
        )
        choice = resp.choices[0]
        calls = [self._tool_call(tc) for tc in (choice.message.tool_calls or [])]
        return ChatTurnResponse(
            text=choice.message.content or "",
            model=getattr(resp, "model", None) or self._model,
            provider=self._provider,
            finish_reason=_map_finish(choice.finish_reason),
            usage=self._usage(resp),
            tool_calls=calls,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        kwargs = self._kwargs(
            request, stream=True, stream_options={"include_usage": True}
        )
        final_finish: FinishReason | None = None
        final_usage: Usage | None = None
        try:
            response = await litellm.acompletion(**kwargs)
            async for chunk in response:
                choices = getattr(chunk, "choices", None) or []
                if choices:
                    delta = choices[0].delta
                    # delta.reasoning / delta.reasoning_content are ignored.
                    content = getattr(delta, "content", None)
                    if content:
                        yield StreamChunk(kind="text_delta", text=content)
                    raw_finish = choices[0].finish_reason
                    if raw_finish is not None:
                        final_finish = _map_finish(raw_finish)
                raw_usage = getattr(chunk, "usage", None)
                if raw_usage is not None:
                    inp = int(getattr(raw_usage, "prompt_tokens", 0) or 0)
                    out = int(getattr(raw_usage, "completion_tokens", 0) or 0)
                    model = getattr(chunk, "model", None) or self._model
                    final_usage = Usage(
                        input_tokens=inp,
                        output_tokens=out,
                        estimated_cost_usd=estimate_cost_usd(model, inp, out),
                    )
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise self._map_error(exc) from exc

        if final_usage is not None:
            yield StreamChunk(kind="usage", usage=final_usage)
        yield StreamChunk(
            kind="done", finish_reason=final_finish or FinishReason.STOP
        )

    # --- error mapping -------------------------------------------
    def _map_error(self, exc: BaseException) -> LLMError:
        name = type(exc).__name__
        msg = str(exc)
        low = msg.lower()

        if name == "AuthenticationError":
            return LLMAuthError(msg)
        if name == "RateLimitError":
            return LLMRateLimitError(msg, retry_after_seconds=_retry_after_seconds(exc))
        if name == "ContextWindowExceededError":
            return LLMContextLengthError(msg)
        if name == "ContentPolicyViolationError":
            return LLMContentFilteredError(msg)
        if name in {
            "Timeout",
            "APIConnectionError",
            "InternalServerError",
            "ServiceUnavailableError",
            "BadGatewayError",
            "APIError",
        }:
            return LLMProviderError(msg)
        if name in {"BadRequestError", "InvalidRequestError", "UnprocessableEntityError"}:
            if any(
                marker in low
                for marker in ("context", "maximum context", "too many tokens", "exceed_context_size")
            ):
                return LLMContextLengthError(msg)
            if any(marker in low for marker in ("content_policy", "content filter", "safety")):
                return LLMContentFilteredError(msg)
            return LLMProviderError(msg)
        # Unknown: still must not reach the domain as a raw provider error.
        return LLMProviderError(f"{name}: {msg}")
