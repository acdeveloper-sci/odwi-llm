"""Synchronous facade over any LLMPort — design §4.7.

The core is async; this wraps it for scripts, classic Django/Flask views,
Celery workers, Streamlit, Jupyter, etc.

`_LoopRunner` runs a private event loop on a dedicated daemon thread. It
never calls `asyncio.run()` and never touches the caller's loop, so
`SyncLLM` methods behave the same whether or not an event loop is already
running in the calling thread — which is the whole reason this class
exists instead of a naive `asyncio.run(coro)` facade.
"""

import asyncio
import atexit
import threading
from collections.abc import AsyncIterator, Coroutine, Iterator
from typing import Any, Self, TypeVar

from odwi_llm.core.port import LLMPort
from odwi_llm.core.types import (
    ChatTurnResponse,
    LLMRequest,
    LLMResponse,
    StreamChunk,
    StructuredResponse,
    T,
    ToolSpec,
)

_R = TypeVar("_R")


async def _anext(ait: AsyncIterator[_R]) -> _R:
    """`__anext__` as a real coroutine, so it can be submitted cross-thread."""
    return await ait.__anext__()


class _LoopRunner:
    """A private asyncio loop on a dedicated daemon thread.

    Coroutines are submitted with `run_coroutine_threadsafe` and awaited
    from the caller, which merely blocks on a `concurrent.futures.Future`.
    This works even when the caller is itself inside a running event loop.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._closed = False
        ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(ready,), name="odwi-llm-loop", daemon=True
        )
        self._thread.start()
        ready.wait()
        atexit.register(self.close)

    def _run(self, ready: threading.Event) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(ready.set)
        self._loop.run_forever()
        self._loop.close()

    def run(self, coro: Coroutine[Any, Any, _R]) -> _R:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def iterate(self, async_iter: AsyncIterator[_R]) -> Iterator[_R]:
        try:
            while True:
                try:
                    yield asyncio.run_coroutine_threadsafe(
                        _anext(async_iter), self._loop
                    ).result()
                except StopAsyncIteration:
                    return
        finally:
            aclose = getattr(async_iter, "aclose", None)
            if aclose is not None:
                try:
                    asyncio.run_coroutine_threadsafe(aclose(), self._loop).result()
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


class SyncLLM:
    """Blocking facade over any LLMPort. Framework-agnostic."""

    def __init__(self, port: LLMPort) -> None:
        self._port = port
        self._runner = _LoopRunner()

    def generate(self, request: LLMRequest) -> LLMResponse:
        return self._runner.run(self._port.generate(request))

    def structured(
        self, request: LLMRequest, schema: type[T]
    ) -> StructuredResponse[T]:
        return self._runner.run(self._port.structured(request, schema))

    def stream(self, request: LLMRequest) -> Iterator[StreamChunk]:
        yield from self._runner.iterate(self._port.stream(request))

    def chat_with_tools(
        self, request: LLMRequest, tools: list[ToolSpec]
    ) -> ChatTurnResponse:
        return self._runner.run(self._port.chat_with_tools(request, tools))

    def close(self) -> None:
        """Stop the worker loop and join its thread. Idempotent."""
        self._runner.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
