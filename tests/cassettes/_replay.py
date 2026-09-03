"""Cassette record & replay for tests/contract_shape/ (design §9.5).

Recording happens at the LLMPort boundary — never at the HTTP layer (see
README.md). A cassette is the adapter's final, retry-resolved,
reasoning-stripped output object; retry itself is verified only in
tests/adapters/test_retry.py, never against a cassette.

Not collected by pytest (leading underscore).
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import AsyncIterator
from importlib.metadata import version
from pathlib import Path
from typing import Any

from odwi_llm.adapters import anyllm_adapter as _any
from odwi_llm.adapters import litellm_adapter as _lite
from odwi_llm.core.port import LLMPort
from odwi_llm.core.requirements import LLMCapabilities
from odwi_llm.core.types import (
    ChatTurnResponse,
    FinishReason,
    LLMRequest,
    LLMResponse,
    StreamChunk,
    StructuredResponse,
    T,
    ToolSpec,
    Usage,
)

CASSETTE_DIR = Path(__file__).resolve().parent

_ADAPTER_LIB = {"litellm": "litellm", "anyllm": "any-llm-sdk"}
# reuse the real adapters' capability tables — never a new one here.
_CAPS_TABLE: dict[str, tuple[dict[tuple[str, str], LLMCapabilities], LLMCapabilities]] = {
    "litellm": (_lite._KNOWN_CAPS, _lite._DEFAULT_CAPS),
    "anyllm": (_any._KNOWN_CAPS, _any._DEFAULT_CAPS),
}


class MissingCassette(RuntimeError):
    pass


def _caps_for(adapter: str, provider: str, model: str) -> LLMCapabilities:
    known, default = _CAPS_TABLE[adapter]
    return known.get((provider, model), default)


def _path(adapter: str, provider: str, scenario: str) -> Path:
    return CASSETTE_DIR / f"{adapter}__{provider}__{scenario}.json"


# --- serialization -----------------------------------------------
def _meta(adapter: str, provider: str, model: str, scenario: str) -> dict[str, Any]:
    lib = _ADAPTER_LIB[adapter]
    return {
        "recorded_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "adapter": adapter,
        "adapter_lib": lib,
        "adapter_lib_version": version(lib),
        "provider": provider,
        "model": model,
        "scenario": scenario,
    }


def _dump_response(obj: LLMResponse) -> dict[str, Any]:
    # exclude `raw`: adapters don't fill it today, but if they ever do it
    # could carry provider account data — it must never be committed.
    return obj.model_dump(mode="json", exclude={"raw"})


def _write_cassette(
    adapter: str, provider: str, model: str, scenario: str, kind: str, payload: Any
) -> None:
    doc = {"meta": _meta(adapter, provider, model, scenario), "kind": kind, "payload": payload}
    _path(adapter, provider, scenario).write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _read_payload(adapter: str, provider: str, scenario: str) -> Any:
    path = _path(adapter, provider, scenario)
    if not path.is_file():
        raise MissingCassette(
            f"no cassette {path.name}; record it with "
            f"`uv run pytest tests/contract_shape/ --live --record --adapter {adapter}`"
        )
    return json.loads(path.read_text(encoding="utf-8"))["payload"]


# --- replay ----------------------------------------------------
class CassetteReplay(LLMPort):
    """An LLMPort that returns recorded real outputs. No network."""

    def __init__(self, adapter: str, provider: str, model: str) -> None:
        self._adapter = adapter
        self._provider = provider
        self._model = model
        self._caps = _caps_for(adapter, provider, model)

    @property
    def capabilities(self) -> LLMCapabilities:
        return self._caps

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse.model_validate(
            _read_payload(self._adapter, self._provider, "generate")
        )

    async def structured(
        self, request: LLMRequest, schema: type[T]
    ) -> StructuredResponse[T]:
        payload = _read_payload(self._adapter, self._provider, "structured")
        return StructuredResponse[T](
            text=payload["text"],
            model=payload["model"],
            provider=payload["provider"],
            finish_reason=FinishReason(payload["finish_reason"]),
            usage=Usage.model_validate(payload["usage"]),
            data=schema.model_validate(payload["data"]),
        )

    async def chat_with_tools(
        self, request: LLMRequest, tools: list[ToolSpec]
    ) -> ChatTurnResponse:
        return ChatTurnResponse.model_validate(
            _read_payload(self._adapter, self._provider, "tools")
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        for chunk in _read_payload(self._adapter, self._provider, "stream"):
            yield StreamChunk.model_validate(chunk)


# --- recording proxy -----------------------------------------
class RecordingAdapter(LLMPort):
    """Wraps a real adapter; tees each result to a cassette on the way out."""

    def __init__(self, inner: LLMPort, adapter: str, provider: str, model: str) -> None:
        self._inner = inner
        self._adapter = adapter
        self._provider = provider
        self._model = model

    @property
    def capabilities(self) -> LLMCapabilities:
        return self._inner.capabilities

    async def generate(self, request: LLMRequest) -> LLMResponse:
        resp = await self._inner.generate(request)
        _write_cassette(self._adapter, self._provider, self._model, "generate",
                        "LLMResponse", _dump_response(resp))
        return resp

    async def structured(
        self, request: LLMRequest, schema: type[T]
    ) -> StructuredResponse[T]:
        resp = await self._inner.structured(request, schema)
        _write_cassette(self._adapter, self._provider, self._model, "structured",
                        "StructuredResponse", _dump_response(resp))
        return resp

    async def chat_with_tools(
        self, request: LLMRequest, tools: list[ToolSpec]
    ) -> ChatTurnResponse:
        resp = await self._inner.chat_with_tools(request, tools)
        _write_cassette(self._adapter, self._provider, self._model, "tools",
                        "ChatTurnResponse", _dump_response(resp))
        return resp

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        chunks: list[StreamChunk] = []
        async for chunk in self._inner.stream(request):
            chunks.append(chunk)
            yield chunk
        _write_cassette(self._adapter, self._provider, self._model, "stream",
                        "stream", [c.model_dump(mode="json") for c in chunks])
