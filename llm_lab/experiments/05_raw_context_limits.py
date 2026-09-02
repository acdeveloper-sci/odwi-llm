"""Task 9b - two different limits, do not confuse them.

(a) OUTPUT truncation (FinishReason.LENGTH): set max_output_tokens
    deliberately low (10) on a prompt that wants a long answer, for ALL
    four providers. Cheap. Observe how each one signals "I was cut off
    by the output cap".

(b) INPUT / context-window overflow (LLMContextLengthError): LOCAL ONLY.
    Ollama via native /api/chat with a small options.num_ctx; LM Studio
    with a prompt far larger than the loaded context. Dump the raw
    error / response. For Gemini and Groq this is NOT triggered (would
    burn free quota) - see the DOC NOTES block near the bottom for what
    their docs say the error looks like.

Dumps: experiments/dumps/{provider}_context_limits.txt

Run: uv run python experiments/05_raw_context_limits.py
"""

from __future__ import annotations

import traceback

import httpx

import _config as cfg
from _dump_utils import call_with_retry, pkg_version, write_sections_dump

LONG_PROMPT = (
    "Write a detailed 500-word essay on the history of the printing press, "
    "covering Gutenberg, movable type, and the spread of literacy in Europe."
)
MAX_TOKENS_LOW = 10

# (b) Ollama: filler prompt a few hundred tokens long, run under num_ctx=256.
OLLAMA_NUM_CTX = 256
_CTX_FILLER = "The quick brown fox jumps over the lazy dog. " * 300  # ~2.6k tokens

# (b) LM Studio: prompt much larger than a typical loaded context window.
_OVERFLOW_PROMPT = "The quick brown fox jumps over the lazy dog. " * 4000  # ~35k tokens


# --- (a) summaries --------------------------------------------------
def _summ_gemini_a(resp: object) -> str:
    cand = resp.candidates[0] if resp.candidates else None
    fr = getattr(cand, "finish_reason", None)
    um = resp.usage_metadata
    return (
        f"candidates[0].finish_reason={fr!r} ; "
        f"text={ (resp.text or '')[:60]!r } ; "
        f"usage_metadata.candidates_token_count="
        f"{getattr(um, 'candidates_token_count', None)}"
    )


def _summ_openai_a(resp: object) -> str:
    ch0 = resp.choices[0] if resp.choices else None
    return (
        f"choices[0].finish_reason={getattr(ch0, 'finish_reason', None)!r} ; "
        f"content={(ch0.message.content or '')[:60]!r} ; "
        f"usage.completion_tokens={getattr(resp.usage, 'completion_tokens', None)}"
    )


# --- (a) callers ---------------------------------------------------
def gemini_a() -> object:
    import logging

    logging.getLogger("google_genai").setLevel(logging.ERROR)
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types

    client = genai.Client(api_key=cfg.GEMINI_API_KEY)
    return call_with_retry(
        lambda: client.models.generate_content(
            model=cfg.GEMINI_MODEL,
            contents=LONG_PROMPT,
            config=types.GenerateContentConfig(max_output_tokens=MAX_TOKENS_LOW),
        ),
        retry_on=(genai_errors.ServerError,),
        label="gemini",
    )


def groq_a() -> object:
    from groq import Groq

    client = Groq(api_key=cfg.GROQ_API_KEY, timeout=60.0)
    return client.chat.completions.create(
        model=cfg.GROQ_MODEL,
        messages=[{"role": "user", "content": LONG_PROMPT}],
        max_tokens=MAX_TOKENS_LOW,
    )


def _openai_a(base_url: str, model: str, api_key: str) -> object:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0)
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": LONG_PROMPT}],
        max_tokens=MAX_TOKENS_LOW,
    )


# --- (b) context overflow, local only -----------------------------
def ollama_b() -> dict:
    """Native /api/chat so we can pass options.num_ctx. Ollama's OpenAI
    layer does not expose it."""
    url = f"{cfg.OLLAMA_NATIVE_URL}/api/chat"
    body = {
        "model": cfg.OLLAMA_MODEL,
        "messages": [{"role": "user", "content": _CTX_FILLER}],
        "stream": False,
        # cap generation so a tiny-context degenerate loop does not bloat
        # the dump; we only care about prompt_eval_count / done_reason / error
        "options": {"num_ctx": OLLAMA_NUM_CTX, "num_predict": 40},
    }
    r = httpx.post(url, json=body, timeout=120.0)
    return {
        "request": f"POST {url}  options.num_ctx={OLLAMA_NUM_CTX}, prompt~2.6k tokens",
        "status_code": r.status_code,
        "json": r.json(),
    }


def lmstudio_b() -> object:
    from openai import OpenAI

    client = OpenAI(base_url=cfg.LMSTUDIO_BASE_URL, api_key="lm-studio", timeout=120.0)
    return client.chat.completions.create(
        model=cfg.LMSTUDIO_MODEL,
        messages=[{"role": "user", "content": _OVERFLOW_PROMPT}],
        max_tokens=32,
    )


# --- run ---------------------------------------------------------
def _safe(fn):
    try:
        return fn(), None
    except Exception as exc:  # noqa: BLE001 - the error IS the finding
        return None, exc


def _err_section(label: str, exc: BaseException) -> dict:
    return {
        "label": label,
        "note": f"RAISED {type(exc).__module__}.{type(exc).__name__}: {exc}",
        "obj": {
            "exception_type": f"{type(exc).__module__}.{type(exc).__name__}",
            "str": str(exc),
            "status_code": getattr(exc, "status_code", None),
            "code": getattr(exc, "code", None),
            "body": getattr(exc, "body", None),
        },
    }


def run_gemini() -> dict:
    obj, exc = _safe(gemini_a)
    sec_a = (
        _err_section("(a) max_output_tokens=10", exc)
        if exc
        else {"label": "(a) max_output_tokens=10", "prompt": LONG_PROMPT,
              "note": _summ_gemini_a(obj), "obj": obj}
    )
    return {"sdk": f"google-genai ({pkg_version('google-genai')})", "sections": [sec_a]}


def run_groq() -> dict:
    obj, exc = _safe(groq_a)
    sec_a = (
        _err_section("(a) max_tokens=10", exc)
        if exc
        else {"label": "(a) max_tokens=10", "prompt": LONG_PROMPT,
              "note": _summ_openai_a(obj), "obj": obj}
    )
    return {"sdk": f"groq ({pkg_version('groq')})", "sections": [sec_a]}


def run_ollama() -> dict:
    a_obj, a_exc = _safe(lambda: _openai_a(cfg.OLLAMA_BASE_URL, cfg.OLLAMA_MODEL, "ollama"))
    sec_a = (
        _err_section("(a) max_tokens=10", a_exc)
        if a_exc
        else {"label": "(a) max_tokens=10", "prompt": LONG_PROMPT,
              "note": _summ_openai_a(a_obj), "obj": a_obj}
    )

    b_obj, b_exc = _safe(ollama_b)
    if b_exc:
        sec_b = _err_section(f"(b) context overflow (num_ctx={OLLAMA_NUM_CTX})", b_exc)
    else:
        j = b_obj["json"]
        note = (
            f"HTTP {b_obj['status_code']}; "
            f"keys={sorted(j)}; "
            f"prompt_eval_count={j.get('prompt_eval_count')}; "
            f"error={j.get('error')!r}; "
            f"done_reason={j.get('done_reason')!r}"
        )
        sec_b = {
            "label": f"(b) context overflow (num_ctx={OLLAMA_NUM_CTX})",
            "prompt": b_obj["request"],
            "note": note,
            "obj": b_obj,
        }

    return {"sdk": f"openai + httpx ({pkg_version('openai')})", "sections": [sec_a, sec_b]}


def run_lmstudio() -> dict:
    a_obj, a_exc = _safe(
        lambda: _openai_a(cfg.LMSTUDIO_BASE_URL, cfg.LMSTUDIO_MODEL, "lm-studio")
    )
    sec_a = (
        _err_section("(a) max_tokens=10", a_exc)
        if a_exc
        else {"label": "(a) max_tokens=10", "prompt": LONG_PROMPT,
              "note": _summ_openai_a(a_obj), "obj": a_obj}
    )

    b_obj, b_exc = _safe(lmstudio_b)
    if b_exc:
        sec_b = _err_section("(b) context overflow (~35k-token prompt)", b_exc)
    else:
        sec_b = {
            "label": "(b) context overflow (~35k-token prompt)",
            "note": (
                "NO overflow error - the loaded context window is larger than "
                f"the prompt. finish_reason="
                f"{b_obj.choices[0].finish_reason!r}. Reload the model in "
                "LM Studio with a smaller context to reproduce the real error."
            ),
            "obj": b_obj,
        }

    return {"sdk": f"openai ({pkg_version('openai')})", "sections": [sec_a, sec_b]}


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
            path = write_sections_dump(
                f"{name}_context_limits.txt",
                provider=name,
                model=model,
                sdk=r["sdk"],
                sections=r["sections"],
            )
            print(f"[{name:9}] OK   -> {path.name}")
            for s in r["sections"]:
                print(f"            {s['label']}: {s['note']}")
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
# DOC NOTES for (b) on Gemini / Groq  - NOT triggered (would burn quota)
# =====================================================================
# Gemini (google-genai): sending more input tokens than the model's
#   input limit returns HTTP 400 INVALID_ARGUMENT, raised as
#   google.genai.errors.ClientError, with a message of the form
#   "The input token count (N) exceeds the maximum number of tokens
#   allowed (M)."  (ai.google.dev - Gemini API, error codes: 400
#   INVALID_ARGUMENT covers "request body is malformed" / token limits.)
#   It is a request-time 400, not a streamed finish_reason.
#
# Groq (OpenAI-compatible): exceeding the context window returns HTTP 400
#   raised as groq.BadRequestError, body
#     {"error": {"message": "Please reduce the length of the messages or
#      completion.", "type": "invalid_request_error",
#      "code": "context_length_exceeded"}}
#   (console.groq.com/docs - OpenAI compatibility; same shape and
#   "context_length_exceeded" code as OpenAI.)
#
# =====================================================================
# OUTCOMES  (from the four dumps, run 2026-09-02 - Task 9b "Hecho cuando")
# =====================================================================
#
# (a) OUTPUT truncation, max tokens = 10, all four providers
#
#   provider    field                          value            content
#   --------    -----------------------------  ---------------  -----------------
#   gemini      candidates[0].finish_reason    MAX_TOKENS       6 tokens of real
#                                              (enum, UPPER)    text
#   groq        choices[0].finish_reason       "length"         ""  (10 tokens
#               gpt-oss-120b                                     spent on reasoning;
#                                                                reasoning_tokens=8)
#   ollama      choices[0].finish_reason       "length"         ""  (10 tokens on
#               qwen3:0.6b                                       partial reasoning)
#   lmstudio    choices[0].finish_reason       "length"         real text
#               llama-3.2-3b                                     (non-reasoning)
#
#   -> Gemini says MAX_TOKENS; the OpenAI-shaped three say "length".
#      On a reasoning model a low cap gives EMPTY content but still
#      finish_reason="length" - not an error.
#
# (b) CONTEXT-WINDOW overflow
#
#   ollama  : num_ctx=256 + ~2.6k-token prompt -> HTTP 200, NO error.
#             Native /api/chat shows prompt_eval_count=130: the prompt was
#             SILENTLY TRUNCATED (2600 -> 130 tokens) to fit the window.
#             done_reason="length"; no "error" key. Ollama does not signal
#             input overflow at all.
#   lmstudio: ~40035-token prompt vs loaded n_ctx=8192 -> HTTP 400,
#             openai.BadRequestError. Body is DOUBLE-wrapped:
#               outer {"error": "Engine protocol predict request returned
#                       400: {...}"}
#               inner {"error":{"code":400,"message":"request (40035
#                       tokens) exceeds the available context size (8192
#                       tokens)...","type":"exceed_context_size_error",
#                       "n_prompt_tokens":40035,"n_ctx":8192}}
#             exc.status_code=400, exc.code=None (not OpenAI-standard shape),
#             exc.body = the outer string.
#   gemini / groq : NOT triggered - see the DOC NOTES block above.
#
# Consequences for the adapter (design 4.5 error hierarchy):
#   - finish_reason: map Gemini MAX_TOKENS and OpenAI "length" both to
#     FinishReason.LENGTH. Empty text + LENGTH is a valid result, not an
#     error to raise on.
#   - LLMContextLengthError should be raised from: LM Studio's
#     exceed_context_size_error 400, Gemini's INVALID_ARGUMENT token-limit
#     400, Groq's context_length_exceeded 400. There is NO reliable signal
#     from Ollama (silent truncation) - document this as a known gap; the
#     adapter cannot turn an Ollama context overflow into an error.
#   - LM Studio's error JSON is double-wrapped and non-standard: parse
#     exc.body / message text, do not rely on exc.code.
