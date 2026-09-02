"""Shared helpers for the Fase B raw-dump scripts (Tasks 6-9b).

Every script asks the same thing of each provider and writes the *full*
response object to experiments/dumps/, so we can look at the real shape
instead of just the final text.
"""

from __future__ import annotations

import datetime as _dt
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

DUMPS_DIR = Path(__file__).resolve().parent / "dumps"


def pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


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
