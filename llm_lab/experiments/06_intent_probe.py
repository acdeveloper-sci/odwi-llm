"""Supporting probe for Task 12 - does `temperature=0` behave the way the
design 5.2 Intent map assumes?

For each provider (native SDK) we send the same slightly-entropic prompt
twice at temperature=0 and once at temperature=1, and record:
  - was temperature=0 accepted (no error / no forced override)?
  - were the two temperature=0 replies byte-identical (deterministic)?
  - did temperature=1 differ from temperature=0?

Run: uv run python experiments/06_intent_probe.py
"""

from __future__ import annotations

import traceback

import _config as cfg
from _dump_utils import call_with_retry, write_sections_dump

PROMPT = "Invent one unusual 8-word sentence about the sea. Sentence only."


def _norm(s: str | None) -> str:
    return " ".join((s or "").split())


# --- Gemini --------------------------------------------------------
def gemini(temp: float) -> str:
    import logging

    logging.getLogger("google_genai").setLevel(logging.ERROR)
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types

    client = genai.Client(api_key=cfg.GEMINI_API_KEY)
    resp = call_with_retry(
        lambda: client.models.generate_content(
            model=cfg.GEMINI_MODEL,
            contents=PROMPT,
            config=types.GenerateContentConfig(temperature=temp),
        ),
        retry_on=(genai_errors.ServerError,),
        label="gemini",
    )
    return _norm(resp.text)


def groq(temp: float) -> str:
    from groq import Groq

    client = Groq(api_key=cfg.GROQ_API_KEY, timeout=60.0)
    resp = client.chat.completions.create(
        model=cfg.GROQ_MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=temp,
    )
    return _norm(resp.choices[0].message.content)


def _openai_compat(base_url: str, model: str, key: str, temp: float) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=key, timeout=60.0)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=temp,
    )
    return _norm(resp.choices[0].message.content)


def ollama(temp: float) -> str:
    return _openai_compat(cfg.OLLAMA_BASE_URL, cfg.OLLAMA_MODEL, "ollama", temp)


def lmstudio(temp: float) -> str:
    return _openai_compat(cfg.LMSTUDIO_BASE_URL, cfg.LMSTUDIO_MODEL, "lm-studio", temp)


PROVIDERS = (
    ("gemini", cfg.GEMINI_MODEL, gemini),
    ("groq", cfg.GROQ_MODEL, groq),
    ("ollama", cfg.OLLAMA_MODEL, ollama),
    ("lmstudio", cfg.LMSTUDIO_MODEL, lmstudio),
)


def main() -> int:
    rows = []
    for name, model, fn in PROVIDERS:
        rec: dict[str, object] = {"provider": name, "model": model}
        try:
            a = fn(0.0)
            b = fn(0.0)
            rec["temp0_accepted"] = True
            rec["temp0_deterministic"] = a == b
            rec["temp0_reply_1"] = a
            rec["temp0_reply_2"] = b
        except Exception as exc:  # noqa: BLE001
            rec["temp0_accepted"] = False
            rec["temp0_error"] = f"{type(exc).__name__}: {exc}"
            rec["_tb"] = traceback.format_exc()
        try:
            rec["temp1_reply"] = fn(1.0)
            rec["temp1_differs_from_temp0"] = rec.get("temp1_reply") != rec.get(
                "temp0_reply_1"
            )
        except Exception as exc:  # noqa: BLE001
            rec["temp1_error"] = f"{type(exc).__name__}: {exc}"

        det = rec.get("temp0_deterministic")
        acc = rec.get("temp0_accepted")
        print(
            f"[{name:9}] temp0 accepted={acc} deterministic={det} "
            f"temp1!=temp0={rec.get('temp1_differs_from_temp0')}"
        )
        rows.append(rec)

    write_sections_dump(
        "intent_probe.txt",
        provider="ALL",
        model="see rows",
        sdk="native SDKs",
        sections=[
            {"label": r["provider"], "note": f"model={r['model']}", "obj": r}
            for r in rows
        ],
    )
    print("\n-> experiments/dumps/intent_probe.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
