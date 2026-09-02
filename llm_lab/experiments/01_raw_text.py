"""Task 6 - raw response objects for a simple text call, per native SDK.

Same prompt to the four providers (Gemini + Groq via native SDK, Ollama +
LM Studio via the OpenAI SDK against their local endpoint), dumping the
full response object to experiments/dumps/{provider}_text.txt.

Run: uv run python experiments/01_raw_text.py

The FIELD MAP comment at the bottom of this file is filled in by hand
after reading the four dumps (that is the "Hecho cuando" of Task 6).
"""

from __future__ import annotations

import logging
import traceback

import _config as cfg
from _dump_utils import pkg_version, write_dump

PROMPT = "In one sentence, list the three primary colors of light in the additive model."


def run_gemini() -> tuple[str, object]:
    # google-genai logs an AFC advisory on plain-string contents; silence it.
    logging.getLogger("google_genai").setLevel(logging.ERROR)
    from google import genai

    client = genai.Client(api_key=cfg.GEMINI_API_KEY)
    resp = client.models.generate_content(model=cfg.GEMINI_MODEL, contents=PROMPT)
    return "google-genai", resp


def run_groq() -> tuple[str, object]:
    from groq import Groq

    client = Groq(api_key=cfg.GROQ_API_KEY, timeout=60.0)
    resp = client.chat.completions.create(
        model=cfg.GROQ_MODEL,
        messages=[{"role": "user", "content": PROMPT}],
    )
    return "groq", resp


def _openai_call(base_url: str, model: str, api_key: str) -> object:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
    )


def run_ollama() -> tuple[str, object]:
    return "openai-sdk", _openai_call(cfg.OLLAMA_BASE_URL, cfg.OLLAMA_MODEL, "ollama")


def run_lmstudio() -> tuple[str, object]:
    return "openai-sdk", _openai_call(
        cfg.LMSTUDIO_BASE_URL, cfg.LMSTUDIO_MODEL, "lm-studio"
    )


PROVIDERS = (
    ("gemini", cfg.GEMINI_MODEL, "google-genai", run_gemini),
    ("groq", cfg.GROQ_MODEL, "groq", run_groq),
    ("ollama", cfg.OLLAMA_MODEL, "openai", run_ollama),
    ("lmstudio", cfg.LMSTUDIO_MODEL, "openai", run_lmstudio),
)


def main() -> int:
    failures = 0
    for name, model, sdk_pkg, fn in PROVIDERS:
        try:
            sdk_label, resp = fn()
            path = write_dump(
                f"{name}_text.txt",
                provider=name,
                model=model,
                sdk=f"{sdk_label} ({sdk_pkg} {pkg_version(sdk_pkg)})",
                prompt=PROMPT,
                obj=resp,
            )
            print(f"[{name:9}] OK   -> {path.name}")
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
# FIELD MAP  (from the four dumps, run 2026-09-02 - Task 6 "Hecho cuando")
# =====================================================================
#
# Per response object, where each thing lives:
#
#                     GEMINI (google-genai)          GROQ / OLLAMA / LM STUDIO (openai-shaped)
#   response type      GenerateContentResponse        ChatCompletion (groq.* / openai.* types)
#   assistant text     .text  (helper) OR             .choices[0].message.content
#                      .candidates[0].content
#                        .parts[0].text
#   finish reason      .candidates[0].finish_reason   .choices[0].finish_reason
#                      -> enum "STOP" (UPPERCASE)     -> "stop" (lowercase)
#   resolved model     .model_version                 .model
#   token usage        .usage_metadata:               .usage:
#                        prompt_token_count             prompt_tokens
#                        candidates_token_count         completion_tokens
#                        total_token_count              total_tokens
#                        thoughts_token_count (None     completion_tokens_details
#                          unless thinking)               .reasoning_tokens  (Groq: 46;
#                        cached_content_token_count       LM Studio: 0; Ollama: details=None
#                        prompt_tokens_details[]           even with a 418-token reasoning blob)
#                          (per-modality)
#   reasoning text     NOT returned. Each part has     Groq  : .choices[0].message.reasoning
#                      thought_signature (opaque       Ollama: .choices[0].message.reasoning
#                      bytes) + thought=None;          LM Stu: .choices[0].message.reasoning_content
#                      only the token count leaks      (!! different field name; "" when the
#                      via thoughts_token_count.        model is non-reasoning, e.g. llama-3.2)
#
# Provider-specific extras:
#   Gemini  : .response_id, .sdk_http_response (HTTP headers incl. server-timing),
#             .automatic_function_calling_history, .parsed (used by structured),
#             .prompt_feedback, candidates[].safety_ratings / .avg_logprobs (None here).
#             model_dump(mode="json") renders enums as UPPERCASE strings;
#             repr() renders them as <FinishReason.STOP: 'STOP'>.
#   Groq    : .x_groq (request id, seed), .service_tier="on_demand",
#             .system_fingerprint, .usage.{prompt,completion,queue,total}_time,
#             .usage_breakdown (None), message.executed_tools, message.annotations.
#   Ollama  : .system_fingerprint="fp_ollama" (constant sentinel), short id
#             ("chatcmpl-676"), usage.*_details = None, no stats block,
#             tool_calls = None.
#   LM Studio: .stats = {} (LM-Studio-only; can carry tok/s), system_fingerprint =
#             the model id, tool_calls = []  (empty list, not None),
#             message.reasoning_content instead of message.reasoning.
#
# Consequences for the adapters (design 5.1 steps 4-5):
#   - Normalize finish_reason casing: Gemini UPPERCASE enum -> lowercase.
#   - Drop reasoning from THREE different places: message.reasoning (Groq/Ollama),
#     message.reasoning_content (LM Studio), part.thought_signature (Gemini).
#   - Text extraction path differs (Gemini parts[] vs OpenAI content str).
#   - Usage key names differ (*_token_count vs *_tokens); map to one Usage type.
#   - No provider returns a cost field (relevant to design decision #3):
#     Groq gives timings, not price; the rest give nothing.
