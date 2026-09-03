"""Task 11 — the price table (decision #3)."""

import pytest

from odwi_llm.adapters.pricing import ModelPrice, estimate_cost_usd, lookup

_LAB_MODELS = [
    "gemini-3.5-flash-lite",
    "openai/gpt-oss-120b",
    "qwen3:0.6b",
    "llama-3.2-3b-instruct",
]


@pytest.mark.parametrize("model", _LAB_MODELS)
def test_lab_models_are_in_the_table(model: str) -> None:
    price = lookup(model)
    assert isinstance(price, ModelPrice)
    assert price.model == model
    assert price.input_usd_per_1m >= 0.0
    assert price.output_usd_per_1m >= 0.0


def test_unknown_model_returns_none_not_exception() -> None:
    assert lookup("gpt-4o") is None
    assert lookup("") is None
    assert estimate_cost_usd("gpt-4o", 1000, 1000) is None


def test_local_models_are_free() -> None:
    for model in ("qwen3:0.6b", "llama-3.2-3b-instruct"):
        assert estimate_cost_usd(model, 10_000, 10_000) == 0.0


def test_cost_arithmetic() -> None:
    # groq/openai/gpt-oss-120b: 0.15 in, 0.60 out per 1M (groq.com/pricing)
    got = estimate_cost_usd("openai/gpt-oss-120b", 1_000_000, 2_000_000)
    assert got == pytest.approx(0.15 + 2 * 0.60)


def test_gemini_flash_lite_rate() -> None:
    # 0.30 in, 2.50 out per 1M (ai.google.dev/gemini-api/docs/pricing)
    got = estimate_cost_usd("gemini-3.5-flash-lite", 2_000_000, 1_000_000)
    assert got == pytest.approx(2 * 0.30 + 2.50)


def test_lookup_is_exact_match() -> None:
    assert lookup("GEMINI-3.5-FLASH-LITE") is None  # case-sensitive
    assert lookup("gpt-oss-120b") is None  # provider prefix is part of the id
