"""Temporary: deliberately failing test to confirm CI turns red.

This file is reverted immediately after the red run is observed. It must
not exist on main for more than the verification window.
"""


def test_ci_redcheck_intentionally_fails() -> None:
    assert False, "intentional failure to verify the CI workflow reports red"
