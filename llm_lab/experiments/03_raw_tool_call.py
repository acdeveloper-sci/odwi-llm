"""Task 8 - raw response objects for tool calling, per native SDK.

One trivial tool, get_weather(city). Two scenarios per provider:
  1. single  - ask for one city, expect one tool call
  2. parallel - ask for two cities in one turn, see if the provider emits
     two tool calls at once

Both scenarios go into experiments/dumps/{provider}_tools.txt. What we are
here to observe (Task 8 "Hecho cuando"):
  - is `arguments` a JSON string or an already-parsed dict?
  - are multiple tool calls in a single turn supported?

Run: uv run python experiments/03_raw_tool_call.py
"""

from __future__ import annotations

import traceback

import _config as cfg
from _dump_utils import call_with_retry, pkg_version, write_sections_dump

TOOL_NAME = "get_weather"
TOOL_DESCRIPTION = "Get the current weather for a city."

# JSON Schema for the tool parameters (OpenAI-shaped providers).
TOOL_PARAMS = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "description": "City name, e.g. 'Paris'"},
    },
    "required": ["city"],
    "additionalProperties": False,
}

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": TOOL_DESCRIPTION,
            "parameters": TOOL_PARAMS,
        },
    }
]

SINGLE_PROMPT = "What is the current weather in Paris? You must call the get_weather tool."
PARALLEL_PROMPT = (
    "Get the current weather in Paris and in Tokyo. "
    "Call get_weather once for each city, in this same turn."
)


# --- tool-call summaries (go into each section's `note`) ---------------
def _summ_openai(resp: object) -> str:
    msg = resp.choices[0].message
    calls = msg.tool_calls or []
    if not calls:
        return f"NO tool_calls (finish_reason={resp.choices[0].finish_reason}, content={msg.content!r})"
    rows = []
    for tc in calls:
        args = tc.function.arguments
        rows.append(
            f"{tc.function.name}(args={args!r}, type={type(args).__name__})"
        )
    return f"{len(calls)} tool_call(s): " + " | ".join(rows)


def _summ_gemini(resp: object) -> str:
    calls = resp.function_calls or []
    if not calls:
        return f"NO function_calls (finish_reason={resp.candidates[0].finish_reason})"
    rows = [f"{fc.name}(args={fc.args!r}, type={type(fc.args).__name__})" for fc in calls]
    return f"{len(calls)} function_call(s): " + " | ".join(rows)


def _section(label: str, prompt: str, call, summarize) -> dict:
    """Run one scenario. If the provider raises (e.g. LM Studio's grammar
    parser rejecting a malformed tool call), capture that instead of losing
    the whole provider's dump."""
    try:
        obj = call(prompt)
        note = summarize(obj)
    except Exception as exc:  # noqa: BLE001 - the error IS the finding here
        obj = {"error": f"{type(exc).__name__}: {exc}"}
        note = f"ERROR raised: {type(exc).__name__}: {exc}"
    return {"label": label, "prompt": prompt, "note": note, "obj": obj}


# --- Gemini ----------------------------------------------------------
def run_gemini() -> dict:
    import logging

    logging.getLogger("google_genai").setLevel(logging.ERROR)
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types

    client = genai.Client(api_key=cfg.GEMINI_API_KEY)
    tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=TOOL_NAME,
                description=TOOL_DESCRIPTION,
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "city": types.Schema(type="STRING", description="City name")
                    },
                    required=["city"],
                ),
            )
        ]
    )

    def ask(prompt: str) -> object:
        config = types.GenerateContentConfig(
            tools=[tool],
            # do NOT let the SDK execute the function; we want the raw request
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            ),
        )
        return call_with_retry(
            lambda: client.models.generate_content(
                model=cfg.GEMINI_MODEL, contents=prompt, config=config
            ),
            retry_on=(genai_errors.ServerError,),
            label="gemini",
        )

    return {
        "sdk": f"google-genai ({pkg_version('google-genai')})",
        "sections": [
            _section("single", SINGLE_PROMPT, ask, _summ_gemini),
            _section("parallel", PARALLEL_PROMPT, ask, _summ_gemini),
        ],
    }


# --- OpenAI-shaped (native groq SDK + openai SDK to local endpoints) --
def _run_openai_shaped(create, sdk_label: str) -> dict:
    """create(prompt, tool_choice) -> chat completion. Scenario 1 forces a
    call with tool_choice="required"; scenario 2 leaves it to the model."""
    return {
        "sdk": sdk_label,
        "sections": [
            _section("single", SINGLE_PROMPT, lambda p: create(p, "required"), _summ_openai),
            _section("parallel", PARALLEL_PROMPT, lambda p: create(p, "auto"), _summ_openai),
        ],
    }


def run_groq() -> dict:
    from groq import Groq

    client = Groq(api_key=cfg.GROQ_API_KEY, timeout=120.0)

    def create(prompt: str, tool_choice: str) -> object:
        return client.chat.completions.create(
            model=cfg.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            tools=OPENAI_TOOLS,
            tool_choice=tool_choice,
            parallel_tool_calls=True,
        )

    return _run_openai_shaped(create, f"groq ({pkg_version('groq')})")


def _local_openai(base_url: str, model: str, api_key: str) -> dict:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)

    def create(prompt: str, tool_choice: str) -> object:
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            tools=OPENAI_TOOLS,
            tool_choice=tool_choice,
        )

    return _run_openai_shaped(create, f"openai ({pkg_version('openai')})")


def run_ollama() -> dict:
    return _local_openai(cfg.OLLAMA_BASE_URL, cfg.OLLAMA_MODEL, "ollama")


def run_lmstudio() -> dict:
    return _local_openai(cfg.LMSTUDIO_BASE_URL, cfg.LMSTUDIO_MODEL, "lm-studio")


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
                f"{name}_tools.txt",
                provider=name,
                model=model,
                sdk=r["sdk"],
                sections=r["sections"],
                extra_header={"tool": f"{TOOL_NAME}(city: str)"},
            )
            notes = " || ".join(f"{s['label']}: {s['note']}" for s in r["sections"])
            print(f"[{name:9}] OK   -> {path.name}")
            print(f"            {notes}")
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
# OUTCOMES  (from the four dumps, run 2026-09-02 - Task 8 "Hecho cuando")
# =====================================================================
#
#                    GEMINI                 GROQ gpt-oss-120b   OLLAMA qwen3:0.6b   LM STUDIO llama-3.2-3b
#  calls live in     candidates[0]          choices[0]          same as Groq        same as Groq
#                     .content.parts[i]      .message
#                     .function_call          .tool_calls[]
#                    (also resp.
#                     function_calls)
#  arguments type    dict  {'city':'Paris'}  JSON str           JSON str            JSON str
#                                            '{"city":"Paris"}'
#  finish signal     finish_reason="STOP"   finish_reason=      finish_reason=      finish_reason=
#                    (NO dedicated value -    "tool_calls"       "tool_calls"        "tool_calls"
#                     detect by presence
#                     of function_call
#                     parts)
#  call id           "call_1022777"         "fc_<uuid>"         "call_<8 alnum>"    "<32 alnum>", no
#                    (numeric)                                  (+ has .index field)  prefix
#  content on the    part.text = null       message.content    message.content     message.content = ""
#   tool turn                                 = null             = ""  (empty str)
#  reasoning on it   part.thought_signature message.reasoning  message.reasoning   message.reasoning_content
#                    (bytes)                 (text)             (text)              = ""
#  python type of    -                      ChatCompletion     ChatCompletion      ChatCompletion
#   a tool call                              MessageToolCall    MessageFunction     MessageFunctionToolCall
#                                                              ToolCall (+index)
#
#  PARALLEL (two get_weather calls in ONE turn):
#    gemini    -> YES. Two function_call parts in one message (2nd part
#                 drops thought_signature).
#    groq      -> NO. Emitted ONE tool_call (Paris) even with
#                 parallel_tool_calls=True and an explicit "call twice"
#                 prompt; its own reasoning says "call the function twice".
#                 gpt-oss-120b on Groq is a sequential tool-caller.
#    ollama    -> YES. tool_calls[0..1] with .index 0 and 1.
#    lmstudio  -> CRASHES. HTTP 400, 'The model produced output that does
#                 not match the expected peg-native format' - LM Studio's
#                 llama.cpp PEG grammar parser rejects llama-3.2-3b's
#                 malformed multi-tool output. Single call is fine.
#
# Consequences for the adapter (design 5.1 step 5 "parse tool args"):
#   - Normalize arguments to a dict: Gemini already gives one; the three
#     OpenAI-shaped give a JSON string -> adapter does json.loads. Matches
#     design 4.3: ToolCall.arguments is "already parsed; never a raw JSON
#     string".
#   - Tool-turn detection cannot rely on finish_reason alone: Gemini says
#     "STOP". Check for function_call parts (Gemini) / "tool_calls"
#     (others) and map both to FinishReason.TOOL_CALLS.
#   - "No text" on a tool turn is either null or "" depending on provider.
#   - Reasoning is still attached on tool turns - strip it (as in Task 6).
#   - Parallel tool calls are NOT portable: 2 of 4 fail (gpt-oss-120b
#     sequential; llama-3.2-3b 400s LM Studio). The Etapa 2 orchestration
#     loop must assume one-call-per-turn and must not batch. Feeds design
#     decision #2.
#   - Map LM Studio's grammar 400 to LLMProviderError, not LLMSchemaError:
#     it is server-side generation failure, not our schema.
#
# Reading note: genai's repr() prints FunctionCall as <... Max depth ...>;
# the full args/name/id are in the "structured view" (model_dump) section.
