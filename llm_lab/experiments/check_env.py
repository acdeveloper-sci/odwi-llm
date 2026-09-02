"""Check that the credentials the lab needs are present.

Prints only presence (yes / MISSING) and the value's length, never the value
itself. Exits non-zero if any required variable is missing or empty.

Run: uv run python experiments/check_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
import os

REQUIRED = ("GEMINI_API_KEY", "GROQ_API_KEY")

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

print(f".env: {env_path} ({'found' if env_path.is_file() else 'NOT FOUND'})")

missing: list[str] = []
for name in REQUIRED:
    value = os.environ.get(name, "")
    if value:
        print(f"  {name}: present (len={len(value)})")
    else:
        print(f"  {name}: MISSING")
        missing.append(name)

if missing:
    print(f"\n{len(missing)} variable(s) missing: {', '.join(missing)}")
    sys.exit(1)

print("\nAll required variables present.")
