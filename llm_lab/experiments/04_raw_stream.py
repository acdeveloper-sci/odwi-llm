"""Task 9 - raw chunk sequence for a streamed text response, per native SDK.

Streams one short prompt from each provider and dumps the WHOLE chunk
sequence (not just the concatenated text) to
experiments/dumps/{provider}_stream.txt:
  - one compact line per chunk (index, delta text, finish_reason, usage)
  - the full object for the first 2 and last 3 chunks, where the
    structural detail (finish_reason, usage) shows up

What we are here to observe (Task 9 "Hecho cuando"):
  - does the provider send a final `usage` chunk?
  - how is "this chunk carries text" told apart from "stream is done"?

Run: uv run python experiments/04_raw_stream.py
"""

from __future__ import annotations

import datetime as _dt
import traceback

import _config as cfg
from _dump_utils import DUMPS_DIR, call_with_retry, pkg_version, render_object

PROMPT = "Write three short sentences about why the sky is blue."

FULL_HEAD = 2
FULL_TAIL = 3


def _trunc(value: object, limit: int = 60) -> str:
    s = repr(value)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _write_stream_dump(
    filename: str,
    *,
    provider: str,
    model: str,
    sdk: str,
    chunks: list,
    line,  # Callable[[chunk], str]
    note: str,
) -> str:
    DUMPS_DIR.mkdir(parents=True, exist_ok=True)
    head = {
        "provider": provider,
        "model": model,
        "sdk": sdk,
        "generated_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "prompt": PROMPT,
        "chunk_count": len(chunks),
        "note": note,
    }
    parts = ["\n".join(f"{k}: {v}" for k, v in head.items())]
    parts.append("=" * 72)
    parts.append("# every chunk (compact):")
    for i, ch in enumerate(chunks):
        try:
            parts.append(f"[{i:3}] {line(ch)}")
        except Exception as exc:  # noqa: BLE001
            parts.append(f"[{i:3}] <line() failed: {type(exc).__name__}: {exc}>")

    n = len(chunks)
    idxs = sorted(
        set(range(min(FULL_HEAD, n))) | set(range(max(0, n - FULL_TAIL), n))
    )
    for i in idxs:
        parts.append("")
        parts.append("=" * 72)
        tag = "first" if i < FULL_HEAD else "tail"
        parts.append(f"### full chunk [{i}]  ({tag})")
        parts.append(render_object(chunks[i]))

    path = DUMPS_DIR / filename
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return path.name


def _auto_note(chunks: list, *, text_of, finish_of, usage_of) -> str:
    n = len(chunks)
    first_finish = next(
        (i for i, c in enumerate(chunks) if finish_of(c) is not None), None
    )
    usage_idxs = [i for i, c in enumerate(chunks) if usage_of(c) is not None]
    text_chunks = sum(1 for c in chunks if text_of(c))
    return (
        f"{n} chunks; {text_chunks} carry text; "
        f"first finish_reason at chunk {first_finish}; "
        f"usage present on chunk(s) {usage_idxs or 'none'}"
    )


# --- Gemini ----------------------------------------------------------
def run_gemini() -> dict:
    import logging

    logging.getLogger("google_genai").setLevel(logging.ERROR)
    from google import genai
    from google.genai import errors as genai_errors

    client = genai.Client(api_key=cfg.GEMINI_API_KEY)
    chunks = call_with_retry(
        lambda: list(
            client.models.generate_content_stream(
                model=cfg.GEMINI_MODEL, contents=PROMPT
            )
        ),
        retry_on=(genai_errors.ServerError,),
        label="gemini",
    )

    def text_of(c):
        try:
            return c.text
        except Exception:
            return None

    def finish_of(c):
        return c.candidates[0].finish_reason if c.candidates else None

    def usage_of(c):
        return c.usage_metadata

    def line(c):
        return (
            f"text={_trunc(text_of(c))} "
            f"finish={finish_of(c)} "
            f"usage={'yes' if usage_of(c) else 'no'}"
        )

    return {
        "sdk": f"google-genai ({pkg_version('google-genai')})",
        "chunks": chunks,
        "line": line,
        "note": _auto_note(
            chunks, text_of=text_of, finish_of=finish_of, usage_of=usage_of
        ),
    }


# --- OpenAI-shaped (native groq SDK + openai SDK to local endpoints) --
def _run_openai_shaped(make_stream, sdk_label: str) -> dict:
    chunks = list(make_stream())

    def _choice0(c):
        return c.choices[0] if getattr(c, "choices", None) else None

    def text_of(c):
        ch0 = _choice0(c)
        return getattr(ch0.delta, "content", None) if ch0 else None

    def reasoning_of(c):
        ch0 = _choice0(c)
        if not ch0:
            return None
        return getattr(ch0.delta, "reasoning", None) or getattr(
            ch0.delta, "reasoning_content", None
        )

    def finish_of(c):
        ch0 = _choice0(c)
        return ch0.finish_reason if ch0 else None

    def usage_of(c):
        return getattr(c, "usage", None)

    def line(c):
        return (
            f"delta.content={_trunc(text_of(c))} "
            f"delta.reasoning={_trunc(reasoning_of(c))} "
            f"finish={finish_of(c)} "
            f"choices={'[]' if not getattr(c, 'choices', None) else len(c.choices)} "
            f"usage={'yes' if usage_of(c) else 'no'}"
        )

    reasoning_chunks = sum(1 for c in chunks if reasoning_of(c))
    note = _auto_note(
        chunks, text_of=text_of, finish_of=finish_of, usage_of=usage_of
    )
    if reasoning_chunks:
        note += f"; {reasoning_chunks} carry reasoning deltas"
    return {"sdk": sdk_label, "chunks": chunks, "line": line, "note": note}


def run_groq() -> dict:
    from groq import Groq

    client = Groq(api_key=cfg.GROQ_API_KEY, timeout=120.0)
    # the groq SDK has no stream_options kwarg; pass it through extra_body
    return _run_openai_shaped(
        lambda: client.chat.completions.create(
            model=cfg.GROQ_MODEL,
            messages=[{"role": "user", "content": PROMPT}],
            stream=True,
            extra_body={"stream_options": {"include_usage": True}},
        ),
        f"groq ({pkg_version('groq')})",
    )


def _local_openai(base_url: str, model: str, api_key: str) -> dict:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
    return _run_openai_shaped(
        lambda: client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            stream=True,
            stream_options={"include_usage": True},
        ),
        f"openai ({pkg_version('openai')})",
    )


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
            fname = _write_stream_dump(
                f"{name}_stream.txt",
                provider=name,
                model=model,
                sdk=r["sdk"],
                chunks=r["chunks"],
                line=r["line"],
                note=r["note"],
            )
            print(f"[{name:9}] OK   -> {fname}")
            print(f"            {r['note']}")
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
# OUTCOMES  (from the four dumps, run 2026-09-02 - Task 9 "Hecho cuando")
# =====================================================================
#
# Prompt: "Write three short sentences about why the sky is blue."
# Chunk counts: gemini 5, groq 104, ollama 219, lmstudio 91.
#
#                     GEMINI                    GROQ / OLLAMA / LM STUDIO
#  chunk type         GenerateContentResponse   ChatCompletionChunk
#                     (SAME class as non-       (groq.* / openai.* .chat
#                      streaming)                .chat_completion_chunk)
#  text delta at      chunk.candidates[0]       chunk.choices[0].delta.content
#                      .content.parts[0].text    (str; "" or None when no text)
#                     (or chunk.text helper)
#  granularity        COARSE - ~1 chunk per     token-ish (many small chunks)
#                     phrase/sentence (5 total)
#  role               not per chunk            delta.role="assistant" on chunk[0]
#                                              only
#
#  END-OF-STREAM signal
#  gemini    : final chunk has text="" , finish_reason="STOP" (enum,
#              UPPERCASE), and a thought_signature. finish_reason is None
#              on every earlier chunk. No [DONE] sentinel (SDK).
#  groq/ollama/lmstudio: a chunk with finish_reason="stop" (lowercase)
#              and delta.content=None. Then usually ONE MORE chunk with
#              choices=[] (see usage below). No [DONE] surfaced by the SDK.
#
#  USAGE
#  gemini    : NO separate usage chunk. usage_metadata is on EVERY chunk,
#              cumulative (candidates_token_count grows 1->19->43->59->59).
#              Read final usage from the last chunk.
#  groq      : usage on the finish_reason="stop" chunk AND again on a
#              trailing choices=[] chunk (so twice); also mirrored in
#              chunk.x_groq.usage. Needs stream_options include_usage,
#              which the groq SDK only accepts via extra_body=.
#  ollama    : finish chunk has usage=None; a following choices=[] chunk
#              carries usage (completion_tokens_details=None).
#  lmstudio  : same as ollama - trailing choices=[] chunk carries usage
#              (with a populated completion_tokens_details struct). The
#              non-stream `stats` block does NOT appear while streaming.
#
#  REASONING DELTAS (must be dropped by the adapter, design principle 3)
#  gemini    : none here (thoughts_token_count=None). Would arrive as
#              part.thought / thought_signature if the model thought.
#  groq      : chunk.choices[0].delta.reasoning, 30 chunks, ALL BEFORE
#              the content deltas - never interleaved.
#  ollama    : same field, 160 chunks, all before content.
#  lmstudio  : none (llama-3.2-3b; the reasoning field is simply absent).
#
# Consequences for the adapter (mapping to design 4.3 StreamChunk:
# text_delta / tool_call_delta / usage / done):
#   - "text delta" = non-empty parts[].text (Gemini) / delta.content
#     (others). Skip empty-string and None deltas; the empty final delta
#     is the done marker, not text.
#   - Emit kind="done" on finish_reason; normalize "STOP" -> "stop".
#   - Emit kind="usage" from: Gemini's last chunk usage_metadata; the
#     others' trailing choices=[] chunk (Groq also on the stop chunk -
#     de-dupe).
#   - GUARD chunk.choices: the trailing usage chunk has choices=[], so
#     chunk.choices[0] blows up (this script's _choice0 does the guard).
#   - Drop delta.reasoning / part.thought entirely - otherwise 30-160
#     reasoning chunks leak into the text stream.
#
# Scope note: Task 9 is text-only by design. Streaming + tool_call deltas
# (the hard, provider-divergent case) is NOT covered here - it is exactly
# what design decision #2 leaves open, to be decided from this matrix.
