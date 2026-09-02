"""Task 10 - repeat Tasks 6-9 through the abstraction libraries.

Same four providers, same four operations (text / structured / tools /
stream), but called via litellm.completion(...) and any_llm.completion(...)
instead of the native SDK. Dumps go to
experiments/dumps/{lib}_{provider}_{op}.txt so they sit next to the Fase B
dumps for comparison.

Run: uv run python experiments/10_lib_matrix.py
     uv run python experiments/10_lib_matrix.py litellm gemini text   # one cell

Unsupported (lib, provider) combos are expected - they are recorded in the
dump and printed as ERR, not treated as a crash.
"""

from __future__ import annotations

import sys
import traceback

import _config as cfg
import _fixtures as fx
from _dump_utils import pkg_version, write_dump, write_stream_dump

OPS = ("text", "structured", "tools", "stream")

# (model string, extra kwargs) per library and provider.
MODELS: dict[str, dict[str, tuple[str, dict]]] = {
    "litellm": {
        "gemini": (f"gemini/{cfg.GEMINI_MODEL}", {"api_key": cfg.GEMINI_API_KEY}),
        "groq": (f"groq/{cfg.GROQ_MODEL}", {"api_key": cfg.GROQ_API_KEY}),
        "ollama": (f"ollama_chat/{cfg.OLLAMA_MODEL}", {"api_base": cfg.OLLAMA_NATIVE_URL}),
        "lmstudio": (
            f"lm_studio/{cfg.LMSTUDIO_MODEL}",
            {"api_base": cfg.LMSTUDIO_BASE_URL, "api_key": "lm-studio"},
        ),
    },
    "anyllm": {  # any-llm wants "provider:model" (the "/" form is deprecated)
        "gemini": (f"gemini:{cfg.GEMINI_MODEL}", {"api_key": cfg.GEMINI_API_KEY}),
        "groq": (f"groq:{cfg.GROQ_MODEL}", {"api_key": cfg.GROQ_API_KEY}),
        "ollama": (f"ollama:{cfg.OLLAMA_MODEL}", {"api_base": cfg.OLLAMA_NATIVE_URL}),
        "lmstudio": (
            f"lmstudio:{cfg.LMSTUDIO_MODEL}",
            {"api_base": cfg.LMSTUDIO_BASE_URL},
        ),
    },
}

_MSG = {
    "text": [{"role": "user", "content": fx.TEXT_PROMPT}],
    "structured": [{"role": "user", "content": fx.STRUCTURED_PROMPT}],
    "tools": [{"role": "user", "content": fx.TOOL_PROMPT_SINGLE}],
    "stream": [{"role": "user", "content": fx.STREAM_PROMPT}],
}


# --- the two libraries -----------------------------------------------
def call_litellm(model: str, kwargs: dict, op: str):
    import litellm

    litellm.drop_params = True  # ignore params a provider doesn't accept
    if op == "text":
        return litellm.completion(model=model, messages=_MSG[op], **kwargs)
    if op == "structured":
        return litellm.completion(
            model=model, messages=_MSG[op], response_format=fx.TaskItem, **kwargs
        )
    if op == "tools":
        return litellm.completion(
            model=model,
            messages=_MSG[op],
            tools=fx.OPENAI_TOOLS,
            tool_choice="required",
            **kwargs,
        )
    if op == "stream":
        return list(
            litellm.completion(model=model, messages=_MSG[op], stream=True, **kwargs)
        )
    raise ValueError(op)


def call_anyllm(model: str, kwargs: dict, op: str):
    from any_llm import completion

    if op == "text":
        return completion(model=model, messages=_MSG[op], **kwargs)
    if op == "structured":
        return completion(
            model=model, messages=_MSG[op], response_format=fx.TaskItem, **kwargs
        )
    if op == "tools":
        return completion(
            model=model,
            messages=_MSG[op],
            tools=fx.OPENAI_TOOLS,
            tool_choice="required",
            **kwargs,
        )
    if op == "stream":
        return list(completion(model=model, messages=_MSG[op], stream=True, **kwargs))
    raise ValueError(op)


CALLERS = {"litellm": call_litellm, "anyllm": call_anyllm}
SDK_LABEL = {
    "litellm": lambda: f"litellm ({pkg_version('litellm')})",
    "anyllm": lambda: f"any-llm-sdk ({pkg_version('any-llm-sdk')})",
}


# --- compact per-chunk line for stream dumps ------------------------
def _chunk_line(ch) -> str:
    # works for litellm ModelResponse chunks and any-llm ChatCompletionChunk
    choices = getattr(ch, "choices", None)
    c0 = choices[0] if choices else None
    delta = getattr(c0, "delta", None) if c0 else None
    content = getattr(delta, "content", None) if delta else None
    reasoning = (
        getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        if delta
        else None
    )
    finish = getattr(c0, "finish_reason", None) if c0 else None
    usage = getattr(ch, "usage", None)
    return (
        f"content={content!r} reasoning={reasoning!r} finish={finish!r} "
        f"choices={'[]' if not choices else len(choices)} "
        f"usage={'yes' if usage else 'no'}"
    )


# --- run one cell --------------------------------------------------
def run_cell(lib: str, provider: str, op: str) -> str:
    model, kwargs = MODELS[lib][provider]
    fname = f"{lib}_{provider}_{op}.txt"
    header = {"lib_model": model, "op": op}
    try:
        result = CALLERS[lib](model, dict(kwargs), op)
    except Exception as exc:  # noqa: BLE001 - unsupported combo is data
        write_dump(
            fname,
            provider=provider,
            model=model,
            sdk=SDK_LABEL[lib](),
            prompt=str(_MSG[op]),
            obj={
                "ERROR": f"{type(exc).__module__}.{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
            extra_header=header,
        )
        return f"ERR  {type(exc).__name__}: {str(exc)[:90]}"

    if op == "stream":
        write_stream_dump(
            fname,
            provider=provider,
            model=model,
            sdk=SDK_LABEL[lib](),
            prompt=fx.STREAM_PROMPT,
            chunks=result,
            line=_chunk_line,
            extra_header=header,
        )
        return f"OK   {len(result)} chunks -> {fname}"

    # litellm hides cost / call metadata off the pydantic model (not in
    # model_dump); surface it - relevant to design decision #3.
    hp = getattr(result, "_hidden_params", None)
    if isinstance(hp, dict):
        header["litellm_response_cost"] = repr(hp.get("response_cost"))
        header["litellm_hidden_params_keys"] = ",".join(sorted(hp))

    write_dump(
        fname,
        provider=provider,
        model=model,
        sdk=SDK_LABEL[lib](),
        prompt=str(_MSG[op]),
        obj=result,
        extra_header=header,
    )
    return f"OK   {type(result).__module__}.{type(result).__name__} -> {fname}"


def main(argv: list[str]) -> int:
    libs = [argv[0]] if len(argv) > 0 else list(MODELS)
    provs = [argv[1]] if len(argv) > 1 else list(MODELS["litellm"])
    ops = [argv[2]] if len(argv) > 2 else list(OPS)

    for lib in libs:
        for provider in provs:
            for op in ops:
                try:
                    msg = run_cell(lib, provider, op)
                except Exception as exc:  # noqa: BLE001
                    msg = f"FAIL (dump step) {type(exc).__name__}: {exc}"
                print(f"[{lib:8} {provider:9} {op:10}] {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


# =====================================================================
# COMPARISON  (from the 32 dumps, run 2026-09-02 - Task 10 "Hecho cuando")
# =====================================================================
#
# Cells: 4 providers x 4 ops x 2 libs = 32.
#   litellm : 16/16 OK.
#   any-llm : 15/16 OK; 1 UNSUPPORTED (anyllm lmstudio tools).
#
# RETURN TYPES
#   litellm  -> always litellm.types.utils.ModelResponse (OpenAI-shaped),
#               even for structured (no parsed object). Stream -> iterator
#               of ModelResponse chunks.
#   any-llm  -> any_llm.types.completion.ChatCompletion; structured ->
#               ParsedChatCompletion with message.parsed = a validated
#               Pydantic INSTANCE. Stream -> ChatCompletionChunk iterator.
#
# NORMALIZATION both libs do (vs the Fase B native dumps)
#   - Gemini GenerateContentResponse is fully flattened to OpenAI shape
#     (choices[0].message...). finish_reason "STOP" -> "stop".
#   - litellm ALSO flips Gemini finish_reason to "tool_calls" when there
#     are function_call parts (native Gemini says "STOP" - see Task 8).
#     any-llm: not re-verified here, but same OpenAI shaping.
#   - Tool-call arguments: BOTH hand back a JSON STRING for every
#     provider, incl. Gemini (native Gemini gives a dict) -> litellm
#     json.dumps'es it.
#   - model id: litellm's own "groq/"/"ollama_chat/" prefix is stripped
#     in the response (.model = "openai/gpt-oss-120b" / "ollama_chat/..").
#
# REASONING - where it lands (NOT unified across routes!)
#   native (Task 6): message.reasoning (Groq/Ollama), message.reasoning_content
#                    (LM Studio), part.thought_signature (Gemini).
#   litellm : groq   -> message.reasoning        (str)
#             ollama -> message.reasoning_content (str)   <-- different key
#             gemini -> message.provider_specific_fields.thought_signatures
#                       (list[str]); in tools ALSO smuggled into the tool
#                       call id as  "call_<n>__thought__<sig>"  (!!)
#   any-llm : message.reasoning is a Reasoning(content=...) OBJECT, not a
#             string, on every provider (stream: Reasoning deltas too).
#
# WHAT EACH LIB ADDS
#   litellm : message.images, message.thinking_blocks, message.
#             provider_specific_fields; usage.completion_tokens_details
#             (CompletionTokensDetailsWrapper, keeps reasoning_tokens for
#             Groq), usage.prompt_tokens_details (modality breakdown for
#             Gemini); vertex_ai_grounding/url_context/safety/citation
#             _metadata = [] on Gemini; _hidden_params (OFF the model_dump)
#             with response_cost - see COST below.
#   any-llm : message.extra_content, tool_call.extra_content,
#             ParsedChatCompletion/ParsedChoice/ParsedChatCompletionMessage
#             wrapper types for structured.
#
# WHAT EACH LIB LOSES / MANGLES
#   litellm : Gemini response_id/http headers; drops the trailing
#             stream usage chunk (no per-stream usage unless you pass
#             stream_options include_usage yourself); Groq x_groq kept but
#             trimmed to {id, seed}.
#   any-llm : Groq x_groq DROPPED entirely; system_fingerprint -> null;
#             usage.completion_tokens_details DROPPED (loses Groq's
#             reasoning_tokens breakdown); Gemini id -> "google_genai_
#             response", created -> 0, thought_signature GONE.
#
# PER-OP
#   text       : both fine on all 4 providers.
#   structured : any-llm gives message.parsed (validated TaskItem) on
#                every provider - nicer. litellm gives only a JSON string
#                in message.content even with response_format=<model>.
#   tools      : arguments = JSON string in BOTH. finish_reason
#                "tool_calls" in both. any-llm CANNOT do LM Studio tools
#                (NotImplementedError: LM Studio provider wraps the native
#                lmstudio-python SDK, only its agentic .act() supports
#                tools). litellm CAN (uses LM Studio's OpenAI endpoint).
#                Parallel calls not retested here (see Task 8: not portable).
#   stream     : litellm -> no usage chunk by default; any-llm -> usage
#                present (Groq: on the stop chunk; Gemini: cumulative on
#                every chunk, like native). Reasoning streams before
#                content in both. Text-only, as in Task 9.
#
# COST  (design decision #3)
#   litellm : response._hidden_params["response_cost"] -> a float
#             (e.g. 5.6e-5 USD; 0.0 for local). BUT it is NOT in
#             model_dump/repr, and litellm.completion_cost() RAISED
#             "model isn't mapped yet" for openai/gpt-oss-120b - the
#             bundled price table lags new models.
#   any-llm : NO cost anywhere (no field, model_extra empty).
#   => neither library is a reliable cost source; the package must own a
#      price table. Feeds decision #3.
#
# UNSUPPORTED / NEEDS EXTRAS
#   any-llm ollama & lmstudio need extras: any-llm-sdk[ollama,lmstudio]
#     (pull the `ollama` and `lmstudio` python SDKs). Added to pyproject.
#   any-llm lmstudio TOOLS: not supported at all (see PER-OP/tools).
#   litellm: no extras needed; all 16 cells work out of the box.
