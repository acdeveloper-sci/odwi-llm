"""Task 5 - connectivity smoke test.

One minimal text call ("say 'hola'") to each of the four providers, via their
native SDK (Gemini, Groq) or the OpenAI-compatible local endpoint (Ollama,
LM Studio). Only checks that a call returns text without raising. No object
dumping yet - that starts in Task 6.

Run: uv run python experiments/00_smoke_test.py
Done when: all four calls print text without an exception.
"""

from __future__ import annotations

import sys

import _config as cfg

PROMPT = "Reply with exactly one word: hola"


def _one_line(text: str | None) -> str:
    return " ".join((text or "").split()) or "<empty>"


def gemini() -> str:
    from google import genai

    client = genai.Client(api_key=cfg.GEMINI_API_KEY)
    resp = client.models.generate_content(model=cfg.GEMINI_MODEL, contents=PROMPT)
    return _one_line(resp.text)


def groq() -> str:
    from groq import Groq

    client = Groq(api_key=cfg.GROQ_API_KEY, timeout=60.0)
    resp = client.chat.completions.create(
        model=cfg.GROQ_MODEL,
        messages=[{"role": "user", "content": PROMPT}],
    )
    return _one_line(resp.choices[0].message.content)


def _openai_compat(base_url: str, model: str, api_key: str) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
    )
    return _one_line(resp.choices[0].message.content)


def ollama() -> str:
    return _openai_compat(cfg.OLLAMA_BASE_URL, cfg.OLLAMA_MODEL, "ollama")


def lmstudio() -> str:
    return _openai_compat(cfg.LMSTUDIO_BASE_URL, cfg.LMSTUDIO_MODEL, "lm-studio")


PROVIDERS = (
    ("gemini", cfg.GEMINI_MODEL, gemini),
    ("groq", cfg.GROQ_MODEL, groq),
    ("ollama", cfg.OLLAMA_MODEL, ollama),
    ("lmstudio", cfg.LMSTUDIO_MODEL, lmstudio),
)


def main() -> int:
    failures = 0
    for name, model, fn in PROVIDERS:
        try:
            text = fn()
            print(f"[{name:9}] OK   {model} -> {text}")
        except Exception as exc:  # noqa: BLE001 - smoke test, report and continue
            failures += 1
            print(f"[{name:9}] FAIL {model} -> {type(exc).__name__}: {exc}")
    print()
    if failures:
        print(f"{failures}/{len(PROVIDERS)} provider(s) failed.")
        return 1
    print("All providers responded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
