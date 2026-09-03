"""Repo-root conftest — only the pieces pytest requires at the top level.

`pytest_addoption` must live in the rootdir conftest (pytest reads
command-line options during startup, before nested conftests load).
Everything else stays in the per-directory conftests.
"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="run tests/contract_shape/ against real providers instead of the fake",
    )
    parser.addoption(
        "--adapter",
        action="store",
        default="litellm",
        choices=("litellm", "anyllm"),
        help="which real adapter tests/contract_shape/ --live exercises",
    )
