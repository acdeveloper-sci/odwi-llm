"""Task 7 - raw response objects for structured output, per native mechanism.

One Pydantic schema (nested model + enum) requested from each provider
through its native structured-output path:
  Gemini    -> GenerateContentConfig(response_schema=<pydantic class>)
  Groq      -> response_format json_schema, falling back to json_object
  Ollama    -> response_format json_schema, falling back to json_object
  LM Studio -> response_format json_schema, falling back to json_object

Each dump records: which mechanism actually worked, whether the payload
validated against the schema with no retry, and the full response object.

Run: uv run python experiments/02_raw_structured.py

FIELD MAP / OUTCOMES comment at the bottom is filled in after reading the
dumps (Task 7 "Hecho cuando").
"""

from __future__ import annotations

import json
import traceback
from enum import Enum

from pydantic import BaseModel, ValidationError

import _config as cfg
from _dump_utils import call_with_retry, pkg_version, write_dump


# --- test schema: 4 fields, one nested model, one enum ------------------
class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Assignee(BaseModel):
    name: str
    email: str


class TaskItem(BaseModel):
    title: str
    priority: Priority
    assignee: Assignee
    done: bool


SCHEMA = TaskItem.model_json_schema()

PROMPT = (
    "Create a task item for reviewing the Q3 budget, assigned to Dana Lee "
    "(dana@example.com), high priority, not done yet. Respond with JSON only."
)


def _validate(text: str | None) -> str:
    if not text:
        return "no text to validate"
    try:
        TaskItem.model_validate_json(text)
        return "OK (validated against TaskItem, no retry)"
    except ValidationError as exc:
        return f"ValidationError: {exc.error_count()} error(s) -> {exc.errors()!r}"
    except json.JSONDecodeError as exc:
        return f"JSONDecodeError: {exc}"


# --- Gemini -----------------------------------------------------------
def run_gemini() -> dict:
    import logging

    logging.getLogger("google_genai").setLevel(logging.ERROR)
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types

    client = genai.Client(api_key=cfg.GEMINI_API_KEY)
    cfg_obj = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=TaskItem,
    )
    # Free tier throws transient 503 UNAVAILABLE ("high demand").
    resp = call_with_retry(
        lambda: client.models.generate_content(
            model=cfg.GEMINI_MODEL, contents=PROMPT, config=cfg_obj
        ),
        retry_on=(genai_errors.ServerError,),
        label="gemini",
    )
    parsed = getattr(resp, "parsed", None)
    return {
        "sdk": f"google-genai ({pkg_version('google-genai')})",
        "mechanism": "GenerateContentConfig(response_schema=TaskItem)",
        "obj": resp,
        "validation": (
            f"resp.parsed -> {type(parsed).__name__}: {parsed!r}"
            if parsed is not None
            else _validate(resp.text)
        ),
    }


# --- OpenAI-shaped SDKs (native groq + openai against local endpoints) --
def _request_structured(create, sdk_label: str) -> dict:
    """`create` is a callable: create(**response_format_kwargs) -> chat
    completion. Any OpenAI-shaped SDK works (groq, openai). Tries the
    json_schema response_format first, records and falls back to
    json_object if the provider rejects it."""
    json_schema_fmt = {
        "type": "json_schema",
        "json_schema": {"name": "task_item", "schema": SCHEMA},
    }
    try:
        resp = create(response_format=json_schema_fmt)
        mechanism = 'response_format={"type":"json_schema", ...}'
    except Exception as exc:  # noqa: BLE001 - record and try the weaker mode
        resp = create(response_format={"type": "json_object"})
        mechanism = (
            f"json_schema REJECTED ({type(exc).__name__}: {exc}) "
            '-> fell back to response_format={"type":"json_object"}'
        )
    return {
        "sdk": sdk_label,
        "mechanism": mechanism,
        "obj": resp,
        "validation": _validate(resp.choices[0].message.content),
    }


def run_groq() -> dict:
    from groq import Groq

    client = Groq(api_key=cfg.GROQ_API_KEY, timeout=120.0)
    messages = [{"role": "user", "content": PROMPT}]
    return _request_structured(
        lambda **rf: client.chat.completions.create(
            model=cfg.GROQ_MODEL, messages=messages, **rf
        ),
        f"groq ({pkg_version('groq')})",
    )


def _run_openai_compat(base_url: str, model: str, api_key: str) -> dict:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
    messages = [{"role": "user", "content": PROMPT}]
    return _request_structured(
        lambda **rf: client.chat.completions.create(
            model=model, messages=messages, **rf
        ),
        f"openai ({pkg_version('openai')})",
    )


def run_ollama() -> dict:
    return _run_openai_compat(cfg.OLLAMA_BASE_URL, cfg.OLLAMA_MODEL, "ollama")


def run_lmstudio() -> dict:
    return _run_openai_compat(cfg.LMSTUDIO_BASE_URL, cfg.LMSTUDIO_MODEL, "lm-studio")


PROVIDERS = (
    ("gemini", cfg.GEMINI_MODEL, run_gemini),
    ("groq", cfg.GROQ_MODEL, run_groq),
    ("ollama", cfg.OLLAMA_MODEL, run_ollama),
    ("lmstudio", cfg.LMSTUDIO_MODEL, run_lmstudio),
)


def main() -> int:
    failures = 0
    for name, model, fn in PROVIDERS:
        try:
            r = fn()
            path = write_dump(
                f"{name}_structured.txt",
                provider=name,
                model=model,
                sdk=r["sdk"],
                prompt=PROMPT,
                obj=r["obj"],
                extra_header={"mechanism": r["mechanism"], "validation": r["validation"]},
            )
            print(f"[{name:9}] OK   -> {path.name}  | {r['validation'][:60]}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[{name:9}] FAIL -> {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print()
    print(f"{len(PROVIDERS) - failures}/{len(PROVIDERS)} dumps written to experiments/dumps/")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


# =====================================================================
# OUTCOMES  (from the four dumps, run 2026-09-02 - Task 7 "Hecho cuando")
# =====================================================================
#
# Test schema: TaskItem { title: str, priority: enum(low|medium|high),
#              assignee: { name, email }, done: bool }
#   -> model_json_schema() emits the nested model as $defs + $ref.
#
# Headline: ALL FOUR honored the schema natively. No prompt fallback and
# no validation retry was needed anywhere. Every payload validated against
# TaskItem on the first try. Enum and nested object came back correct in
# all four. The $ref/$defs schema was accepted verbatim by the three
# OpenAI-shaped paths (native groq SDK, and openai SDK -> Ollama / LM
# Studio) - no need to inline refs or add additionalProperties:false /
# strict.
#
#   provider   mechanism that worked                       parsed result lives in
#   --------   ----------------------------------------     ----------------------
#   gemini     GenerateContentConfig(                       resp.parsed  -> a real
#                response_mime_type="application/json",       TaskItem INSTANCE
#                response_schema=TaskItem)                    (SDK parses + coerces
#              (pass the pydantic CLASS, not a dict)          enum to Priority.HIGH);
#                                                             resp.text = JSON string
#   groq       response_format={"type":"json_schema",       resp.choices[0].message
#                "json_schema":{"name":..,"schema":..}}       .content = JSON STRING;
#              accepted as-is (no fallback)                   caller must json.loads +
#   ollama     same json_schema response_format;             validate. Same for
#              Ollama maps it to native format=<schema>       ollama and lmstudio.
#   lmstudio   same json_schema response_format;
#              enforced via llama.cpp grammar
#
# Things the adapter (design 5.1 steps 4-5) must handle:
#   - Two return shapes: Gemini hands back a parsed object (resp.parsed);
#     the OpenAI-shaped three hand back a STRING in message.content that
#     still needs json.loads + schema validation in the adapter.
#   - Reasoning is NOT suppressed by structured mode:
#       groq   -> message.reasoning present (134 reasoning_tokens broken out)
#       ollama -> message.reasoning present (no token breakout; details=None)
#       lmstudio -> message.reasoning_content = "" (non-reasoning model)
#       gemini -> part.thought_signature bytes still attached
#     The adapter strips all of these and parses only content / resp.parsed.
#   - Prompt token count grows when the schema is sent (Gemini 17->35,
#     Groq 87->302) - expected, the schema rides in the request.
#   - JSON whitespace varies (compact vs 2-space pretty); irrelevant post-parse.
#   - Ollama completion_tokens (52) looks too low next to the reasoning text
#     it emitted - token accounting under thinking is unreliable there.
#
# Operational note: gemini-3.5-flash-lite free tier intermittently returns
# 503 UNAVAILABLE ("high demand"); run_gemini() wraps the call in
# _dump_utils.call_with_retry (backoff 3/8/20/45s). That is design 5.3
# technical retry, not a schema issue.
