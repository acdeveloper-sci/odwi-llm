"""Central config for the Fase B / C experiment scripts.

Loads .env and exposes the confirmed provider/model list from
odwi_llm_experiments_task_plan.md (proveedores table). Override any model
via an env var of the same name if needed.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# --- API keys -------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# --- Local OpenAI-compatible endpoints ----------------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
# Ollama's native REST API (format=json, num_ctx) lives without the /v1 suffix.
OLLAMA_NATIVE_URL = OLLAMA_BASE_URL.removesuffix("/v1")

# --- Models (plan proveedores table) ----------------------------------
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_PREVIEW_MODEL = os.environ.get("GROQ_PREVIEW_MODEL", "qwen/qwen3.6-27b")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:0.6b")
LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_MODEL", "llama-3.2-3b-instruct")
LMSTUDIO_CODER_MODEL = os.environ.get("LMSTUDIO_CODER_MODEL", "qwen2.5-coder-7b-instruct")
LMSTUDIO_R1_MODEL = os.environ.get("LMSTUDIO_R1_MODEL", "deepseek-r1-distill-llama-8b")
