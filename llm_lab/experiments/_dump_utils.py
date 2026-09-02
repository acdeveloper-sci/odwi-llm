"""Shared helpers for the Fase B raw-dump scripts (Tasks 6-9b).

Every script asks the same thing of each provider and writes the *full*
response object to experiments/dumps/, so we can look at the real shape
instead of just the final text.
"""

from __future__ import annotations

import datetime as _dt
import json
import time
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, TypeVar

DUMPS_DIR = Path(__file__).resolve().parent / "dumps"

_T = TypeVar("_T")


def pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def call_with_retry(
    fn: Callable[[], _T],
    *,
    retry_on: tuple[type[BaseException], ...],
    delays: tuple[int, ...] = (3, 8, 20, 45),
    label: str = "",
) -> _T:
    """Call fn() and retry on the given exception types with fixed backoff.

    len(delays) + 1 attempts total; the exception from the last attempt is
    re-raised. This is the design 5.3 "technical retry" (5xx / transient),
    kept deliberately small - the free Gemini tier throws 503 UNAVAILABLE
    under load and every Fase B script needs to ride through it.
    """
    tag = f"{label} " if label else ""
    for attempt, delay in enumerate((*delays, None), start=1):
        try:
            return fn()
        except retry_on as exc:  # noqa: PERF203
            if delay is None:
                raise
            code = getattr(exc, "code", None) or getattr(exc, "status_code", "")
            print(
                f"  {tag}{type(exc).__name__} {code} on attempt {attempt}, "
                f"retrying in {delay}s..."
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def _to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of an SDK response object to something json can
    render. Pydantic models expose model_dump; everything else falls back to
    repr via json's default hook."""
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            return obj.model_dump()
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    return obj


def render_object(obj: Any) -> str:
    """Human-readable multi-section view of one response object."""
    cls = type(obj)
    sections = [f"type: {cls.__module__}.{cls.__qualname__}", ""]

    jsonable = _to_jsonable(obj)
    if jsonable is not obj:
        sections.append("# structured view (model_dump / to_dict):")
        sections.append(
            json.dumps(jsonable, indent=2, ensure_ascii=False, default=repr)
        )
        sections.append("")

    sections.append("# repr():")
    sections.append(repr(obj))
    return "\n".join(sections)


def write_dump(
    filename: str,
    *,
    provider: str,
    model: str,
    sdk: str,
    prompt: str,
    obj: Any,
    extra_header: dict[str, str] | None = None,
) -> Path:
    DUMPS_DIR.mkdir(parents=True, exist_ok=True)
    header = {
        "provider": provider,
        "model": model,
        "sdk": sdk,
        "generated_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "prompt": prompt,
        **(extra_header or {}),
    }
    head_lines = [f"{k}: {v}" for k, v in header.items()]
    body = "\n".join(head_lines) + "\n" + ("=" * 72) + "\n" + render_object(obj) + "\n"

    path = DUMPS_DIR / filename
    path.write_text(body, encoding="utf-8")
    return path


def write_sections_dump(
    filename: str,
    *,
    provider: str,
    model: str,
    sdk: str,
    sections: list[dict[str, Any]],
    extra_header: dict[str, str] | None = None,
) -> Path:
    """Like write_dump but for several scenarios in one file.

    Each section is a dict with keys: label, prompt (optional), note
    (optional), obj.
    """
    DUMPS_DIR.mkdir(parents=True, exist_ok=True)
    header = {
        "provider": provider,
        "model": model,
        "sdk": sdk,
        "generated_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        **(extra_header or {}),
    }
    parts = ["\n".join(f"{k}: {v}" for k, v in header.items())]
    for sec in sections:
        parts.append("=" * 72)
        parts.append(f"### {sec['label']}")
        if sec.get("prompt"):
            parts.append(f"prompt: {sec['prompt']}")
        if sec.get("note"):
            parts.append(f"note: {sec['note']}")
        parts.append("-" * 72)
        parts.append(render_object(sec["obj"]))
        parts.append("")

    path = DUMPS_DIR / filename
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return path


def write_stream_dump(
    filename: str,
    *,
    provider: str,
    model: str,
    sdk: str,
    prompt: str,
    chunks: list[Any],
    line: Any,  # Callable[[chunk], str]
    note: str = "",
    full_head: int = 2,
    full_tail: int = 3,
    extra_header: dict[str, str] | None = None,
) -> Path:
    """Dump a streamed response: one compact `line(chunk)` per chunk, then
    the full object for the first `full_head` and last `full_tail` chunks."""
    DUMPS_DIR.mkdir(parents=True, exist_ok=True)
    head = {
        "provider": provider,
        "model": model,
        "sdk": sdk,
        "generated_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "prompt": prompt,
        "chunk_count": len(chunks),
        **(extra_header or {}),
    }
    if note:
        head["note"] = note
    parts = ["\n".join(f"{k}: {v}" for k, v in head.items())]
    parts.append("=" * 72)
    parts.append("# every chunk (compact):")
    for i, ch in enumerate(chunks):
        try:
            parts.append(f"[{i:3}] {line(ch)}")
        except Exception as exc:  # noqa: BLE001
            parts.append(f"[{i:3}] <line() failed: {type(exc).__name__}: {exc}>")

    n = len(chunks)
    idxs = sorted(set(range(min(full_head, n))) | set(range(max(0, n - full_tail), n)))
    for i in idxs:
        parts.append("")
        parts.append("=" * 72)
        tag = "first" if i < full_head else "tail"
        parts.append(f"### full chunk [{i}]  ({tag})")
        parts.append(render_object(chunks[i]))

    path = DUMPS_DIR / filename
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return path
