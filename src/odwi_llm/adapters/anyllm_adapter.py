"""Any-LLM adapter — design §5 / FINDINGS §1, §3.

Differences from the LiteLLM adapter, all from FINDINGS:
  - `structured()` uses `message.parsed` directly — Any-LLM returns a
    `ParsedChatCompletion` with an already-validated Pydantic instance.
  - LM Studio + tools is impossible here (Any-LLM's LM Studio provider
    wraps the native `lmstudio-python` SDK, tools only via `.act()`), so
    that capability is codified as False and an app that requires it gets
    a `CapabilityError` at construction (§4.4), not a `NotImplementedError`
    mid-call.
  - Errors come from Any-LLM's own typed hierarchy and map ~1:1 to §4.5.
  - Any-LLM reports no cost; `estimated_cost_usd` is the package's own
    estimate from `adapters.pricing` (or None for an unlisted model),
    kept consistent with the LiteLLM adapter so `FallbackLLM` behaves the
    same whichever one served.
"""

import json
from collections.abc import AsyncIterator
from typing import Any, cast

from any_llm import acompletion
from any_llm import exceptions as any_exc
from pydantic import ValidationError

from odwi_llm.adapters._shared import (
    MAX_RETRY_ATTEMPTS,
    RETRYABLE_ERRORS,
    call_with_retry,
    map_finish_reason,
    message_to_dict,
    retry_after_seconds,
    retry_sleep,
    unmet_requirements,
)
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
    StreamChunk,
    StructuredResponse,
    T,
    ToolCall,
    ToolSpec,
    Usage,
)

_SCHEMA_RETRIES = 2

_PROVIDERS = {"gemini", "groq", "ollama", "lmstudio"}
_DEFAULT_KEY_ENV = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}
_DEFAULT_BASE_URL = {
    "ollama": "http://localhost:11434",
    "lmstudio": "http://localhost:1234/v1",
}

_INTENT_TEMPERATURE = {Intent.DETERMINISTIC: 0.0, Intent.CREATIVE: 1.0}

# Codified per §9.3 table C (Any-LLM). Same as table B EXCEPT LM Studio
# tool calling, which Any-LLM cannot do (FINDINGS §1). Keyed by
# (provider, model); unknown -> conservative default.
_OPS = {"structured_output": True, "tool_calling": True, "streaming": True, "vision": False}
_KNOWN_CAPS: dict[tuple[str, str], LLMCapabilities] = {
    ("gemini", "gemini-3.5-flash-lite"): LLMCapabilities(**_OPS, context_tokens=1_000_000),
    ("groq", "openai/gpt-oss-120b"): LLMCapabilities(**_OPS, context_tokens=128_000),
    ("ollama", "qwen3:0.6b"): LLMCapabilities(**_OPS, context_tokens=32_768),
    ("lmstudio", "llama-3.2-3b-instruct"): LLMCapabilities(
        structured_output=True,
        tool_calling=False,  # Any-LLM's LM Studio provider has no OpenAI-style tools
        streaming=True,
        vision=False,
        context_tokens=131_072,
    ),
}
_DEFAULT_CAPS = LLMCapabilities(**_OPS, context_tokens=None)


class AnyLLMAdapter(LLMPort):
    def __init__(
        self,
        provider: str,
        model: str,
        requirements: LLMRequirements,
        config: ProviderConfig | None = None,
    ) -> None:
        if provider not in _PROVIDERS:
            raise ValueError(
                f"unknown provider {provider!r}; expected one of {sorted(_PROVIDERS)}"
            )
        self._provider = provider
        self._model = model
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

        unmet = unmet_requirements(requirements, self._caps)
        if unmet:
            raise CapabilityError(
                f"{provider}/{model} does not meet requirements: {', '.join(unmet)}"
            )

    @property
    def capabilities(self) -> LLMCapabilities:
        return self._caps

    # --- plumbing ------------------------------------------------
    def _kwargs(self, request: LLMRequest, **extra: Any) -> dict[str, Any]:
        # No `timeout`: several Any-LLM providers reject it ("set it via
        # client_args"). ProviderConfig.timeout_s is honoured by the
        # LiteLLM adapter; Any-LLM uses its provider SDK's default.
        kwargs: dict[str, Any] = {
            "model": self._model,
            "provider": self._provider,
            "messages": [message_to_dict(m) for m in request.messages],
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
        async def _call() -> Any:
            try:
                return await acompletion(**self._kwargs(request, **extra))
            except LLMError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise self._map_error(exc) from exc

        return await call_with_retry(_call)

    def _usage(self, resp: Any) -> Usage:
        raw = getattr(resp, "usage", None)
        inp = int(getattr(raw, "prompt_tokens", 0) or 0)
        out = int(getattr(raw, "completion_tokens", 0) or 0)
        return Usage(
            input_tokens=inp,
            output_tokens=out,
            estimated_cost_usd=estimate_cost_usd(self._model, inp, out),
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
        return ToolCall(id=str(raw.id or ""), name=raw.function.name, arguments=parsed)

    # --- LLMPort -----------------------------------------------
    async def generate(self, request: LLMRequest) -> LLMResponse:
        resp = await self._acompletion(request)
        choice = resp.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            model=self._model,
            provider=self._provider,
            finish_reason=map_finish_reason(choice.finish_reason),
            usage=self._usage(resp),
        )

    async def structured(
        self, request: LLMRequest, schema: type[T]
    ) -> StructuredResponse[T]:
        last: Exception | None = None
        for _ in range(_SCHEMA_RETRIES + 1):
            resp = await self._acompletion(request, response_format=schema)
            message = resp.choices[0].message
            parsed = getattr(message, "parsed", None)
            data: T
            if isinstance(parsed, schema):
                data = parsed
            else:
                try:
                    data = schema.model_validate_json(message.content or "")
                except (ValidationError, ValueError) as exc:
                    last = exc
                    continue
            return StructuredResponse[T](
                text=message.content or "",
                model=self._model,
                provider=self._provider,
                finish_reason=map_finish_reason(resp.choices[0].finish_reason),
                usage=self._usage(resp),
                data=data,
            )
        raise LLMSchemaError(
            f"structured output did not validate after {_SCHEMA_RETRIES + 1} attempts: {last}"
        )

    async def chat_with_tools(
        self, request: LLMRequest, tools: list[ToolSpec]
    ) -> ChatTurnResponse:
        if not self._caps.tool_calling:
            raise CapabilityError(
                f"{self._provider}/{self._model} does not support tool calling via Any-LLM"
            )
        any_tools = [
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
        resp = await self._acompletion(request, tools=any_tools, tool_choice="auto")
        choice = resp.choices[0]
        calls = [self._tool_call(tc) for tc in (choice.message.tool_calls or [])]
        return ChatTurnResponse(
            text=choice.message.content or "",
            model=self._model,
            provider=self._provider,
            finish_reason=map_finish_reason(choice.finish_reason),
            usage=self._usage(resp),
            tool_calls=calls,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        kwargs = self._kwargs(
            request, stream=True, stream_options={"include_usage": True}
        )
        emitted = False
        final_finish: FinishReason | None = None
        final_usage: Usage | None = None
        # retry only while nothing has been yielded yet (§5.3 + §5.4 principle).
        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            final_finish = None
            final_usage = None
            try:
                response = cast(AsyncIterator[Any], await acompletion(**kwargs))
                async for chunk in response:
                    choices = getattr(chunk, "choices", None) or []
                    if choices:
                        delta = choices[0].delta
                        # delta.reasoning (an object here) is ignored.
                        content = getattr(delta, "content", None)
                        if content:
                            emitted = True
                            yield StreamChunk(kind="text_delta", text=content)
                        raw_finish = choices[0].finish_reason
                        if raw_finish is not None:
                            final_finish = map_finish_reason(raw_finish)
                    raw_usage = getattr(chunk, "usage", None)
                    if raw_usage is not None:
                        inp = int(getattr(raw_usage, "prompt_tokens", 0) or 0)
                        out = int(getattr(raw_usage, "completion_tokens", 0) or 0)
                        final_usage = Usage(
                            input_tokens=inp,
                            output_tokens=out,
                            estimated_cost_usd=estimate_cost_usd(self._model, inp, out),
                        )
                break
            except LLMError as exc:
                if (
                    emitted
                    or not isinstance(exc, RETRYABLE_ERRORS)
                    or attempt == MAX_RETRY_ATTEMPTS
                ):
                    raise
                await retry_sleep(exc, attempt)
            except Exception as exc:  # noqa: BLE001
                mapped = self._map_error(exc)
                if (
                    emitted
                    or not isinstance(mapped, RETRYABLE_ERRORS)
                    or attempt == MAX_RETRY_ATTEMPTS
                ):
                    raise mapped from exc
                await retry_sleep(mapped, attempt)

        if final_usage is not None:
            yield StreamChunk(kind="usage", usage=final_usage)
        yield StreamChunk(kind="done", finish_reason=final_finish or FinishReason.STOP)

    # --- error mapping ---------------------------------------
    def _map_error(self, exc: BaseException) -> LLMError:
        msg = str(exc)
        if isinstance(exc, (any_exc.AuthenticationError, any_exc.MissingApiKeyError)):
            return LLMAuthError(msg)
        if isinstance(exc, any_exc.RateLimitError):
            return LLMRateLimitError(msg, retry_after_seconds=retry_after_seconds(exc))
        if isinstance(exc, any_exc.ContextLengthExceededError):
            return LLMContextLengthError(msg)
        if isinstance(
            exc,
            (any_exc.ContentFilterError, any_exc.ContentFilterFinishReasonError),
        ):
            return LLMContentFilteredError(msg)
        if isinstance(exc, any_exc.LengthFinishReasonError):
            # only raised when response_format was requested: the structured
            # output was truncated -> a schema failure, not a provider one.
            return LLMSchemaError(msg)
        if isinstance(
            exc,
            (
                any_exc.GatewayTimeoutError,
                any_exc.UpstreamProviderError,
                any_exc.ProviderError,
            ),
        ):
            return LLMProviderError(msg)
        if isinstance(
            exc,
            (
                any_exc.InvalidRequestError,
                any_exc.ModelNotFoundError,
                any_exc.UnsupportedParameterError,
                any_exc.UnsupportedProviderError,
            ),
        ):
            low = msg.lower()
            if any(m in low for m in ("context", "token", "exceed_context_size")):
                return LLMContextLengthError(msg)
            return LLMProviderError(msg)
        if isinstance(exc, NotImplementedError):
            # Any-LLM's LM Studio + tools path, if reached despite the
            # construction gate (e.g. an uncatalogued model that inherited
            # the permissive default capabilities). Same cause as the
            # fail-fast -> same error type, catalogued or not.
            raise CapabilityError(
                f"{self._provider}/{self._model}: operation not supported: {msg}"
            ) from exc
        return LLMProviderError(f"{type(exc).__name__}: {msg}")
