"""Task 8 — architecture boundaries (design §3.2).

Two boundaries, scanned over `src/odwi_llm/` ONLY (never the whole repo):

1. Provider libraries (`litellm`, `any_llm`, `openai`, `anthropic`,
   `google.genai`) may appear only under `odwi_llm/adapters/`.
2. Application frameworks (`streamlit`, `reflex`, `flet`, `django`,
   `flask`, `fastapi`) may not appear anywhere under `odwi_llm/`.

`llm_lab/experiments/` is deliberately out of scope: it imports those
libraries directly, by design (exploratory, not part of the package).
`test_scanner_detects_violations_in_experiments` proves the scanner
really works — pointed at `experiments/` with no allow-list it DOES
fire — so a green result on `src/odwi_llm/` is a real pass, not a
mis-built no-op.
"""

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PKG = _REPO_ROOT / "src" / "odwi_llm"
_ADAPTERS = _PKG / "adapters"
_EXPERIMENTS = _REPO_ROOT / "llm_lab" / "experiments"

PROVIDER_LIBS = frozenset(
    {"litellm", "any_llm", "openai", "anthropic", "google.genai"}
)
FRAMEWORK_LIBS = frozenset(
    {"streamlit", "reflex", "flet", "django", "flask", "fastapi"}
)

_SKIP_DIRS = {".venv", "__pycache__", ".git", ".mypy_cache", ".pytest_cache"}


def _python_files(root: Path) -> list[Path]:
    return [
        p
        for p in root.rglob("*.py")
        if not _SKIP_DIRS.intersection(p.parts)
    ]


def _imported_paths(path: Path) -> set[str]:
    """Dotted module paths a file imports.

    `from google import genai` -> {"google", "google.genai"}, so a
    forbidden entry like "google.genai" matches by exact/prefix.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or node.module is None:  # relative import
                continue
            out.add(node.module)
            for alias in node.names:
                out.add(f"{node.module}.{alias.name}")
    return out


def _hits(imported: set[str], forbidden: frozenset[str]) -> set[str]:
    found: set[str] = set()
    for mod in imported:
        for bad in forbidden:
            if mod == bad or mod.startswith(bad + "."):
                found.add(bad)
    return found


def _violations(
    root: Path, forbidden: frozenset[str], allowed_dir: Path | None
) -> list[tuple[Path, set[str]]]:
    result: list[tuple[Path, set[str]]] = []
    for file in _python_files(root):
        if allowed_dir is not None and allowed_dir in file.parents:
            continue
        hits = _hits(_imported_paths(file), forbidden)
        if hits:
            result.append((file.relative_to(_REPO_ROOT), hits))
    return result


def test_package_source_exists() -> None:
    assert _PKG.is_dir(), f"expected package at {_PKG}"


def test_provider_libs_only_under_adapters() -> None:
    bad = _violations(_PKG, PROVIDER_LIBS, allowed_dir=_ADAPTERS)
    assert bad == [], (
        "provider libraries imported outside odwi_llm/adapters/:\n"
        + "\n".join(f"  {p}: {sorted(h)}" for p, h in bad)
    )


def test_no_framework_libs_anywhere_in_package() -> None:
    bad = _violations(_PKG, FRAMEWORK_LIBS, allowed_dir=None)
    assert bad == [], (
        "application frameworks imported inside odwi_llm/:\n"
        + "\n".join(f"  {p}: {sorted(h)}" for p, h in bad)
    )


def test_scanner_detects_violations_in_experiments() -> None:
    """Evidence the scanner is real: experiments/ imports provider libs."""
    if not _EXPERIMENTS.is_dir():
        pytest.skip("llm_lab/experiments/ not present")
    bad = _violations(_EXPERIMENTS, PROVIDER_LIBS, allowed_dir=None)
    assert bad, (
        "scanner found no provider-lib imports under experiments/ — either "
        "the scanner is broken or experiments/ changed shape"
    )
