"""Integration tests for M1.3 — real OpenRouter calls.

Each test burns a small amount of free-tier quota. Limited to ~8 calls
total across the suite so you can run it without wrecking your daily
limit.

Enable:
    export OPENROUTER_INTEGRATION_OK=1
    export OPENROUTER_API_KEY=<your-key>
    uv run pytest -v -m openrouter_integration
"""

from __future__ import annotations

from datetime import datetime

import pytest

from firefly_agent.openrouter import OpenRouterClient


pytestmark = pytest.mark.openrouter_integration


# Match config.toml values but hard-coded here so tests are
# self-contained.
ALLOWED_CATEGORIES = [
    "Entertainment",
    "Fees & Taxes",
    "Food & Beverages",
    "Gifts",
    "Groceries",
    "Health",
    "Housing",
    "Income",
    "Shopping",
    "Tech",
    "Transportation",
    "Travel",
    "Utilities",
]
ALLOWED_TAGS = [
    "coffee", "dining-out", "alcohol", "delivery",
    "llm-api", "cloud-hosting", "domain", "dev-tool",
    "clothes", "hobby", "electronics", "home-goods",
    "medical", "pharmacy", "dental", "vision", "skincare", "fitness",
    "reimbursable", "reimbursed", "subscription", "annual",
]
ALLOWED_CURRENCIES = ["IDR", "USD", "EUR", "GBP", "JPY", "SGD", "MYR", "THB"]
TAG_GROUPS = {
    "food": ["coffee", "dining-out", "alcohol", "delivery"],
    "tech": ["llm-api", "cloud-hosting", "domain", "dev-tool"],
    "shopping": ["clothes", "hobby", "electronics", "home-goods"],
    "health": ["medical", "pharmacy", "dental", "vision", "skincare", "fitness"],
    "cadence": ["subscription", "annual"],
    "money_movement": ["reimbursable", "reimbursed"],
}


MODELS = ["openai/gpt-5.4-nano", "openai/gpt-4o-mini"]


@pytest.fixture
async def llm(openrouter_api_key: str):
    """Construct client with the real chain."""
    client = OpenRouterClient(
        api_key=openrouter_api_key,
        models=MODELS,
        allowed_categories=ALLOWED_CATEGORIES,
        allowed_tags=ALLOWED_TAGS,
        allowed_currencies=ALLOWED_CURRENCIES,
        default_currency="IDR",
        tag_groups=TAG_GROUPS,
    )
    async with client as c:
        yield c


# ============================================================
# Core smoke tests
# ============================================================


async def test_parse_simple_english(llm: OpenRouterClient) -> None:
    """'coffee at starbucks 45k' → Food & Beverages + coffee tag, IDR 45000."""
    p = await llm.parse_transaction_text("coffee at starbucks 45k")
    assert p.type == "withdrawal"
    assert p.category == "Food & Beverages"
    assert p.currency == "IDR"
    assert p.amount == 45000
    assert "coffee" in p.tags


async def test_parse_indonesian_shorthand(llm: OpenRouterClient) -> None:
    """'gojek ke kantor 25rb' → Transportation, IDR 25000."""
    p = await llm.parse_transaction_text("gojek ke kantor 25rb")
    assert p.type == "withdrawal"
    assert p.category == "Transportation"
    assert p.currency == "IDR"
    assert p.amount == 25000


async def test_parse_usd_llm_api(llm: OpenRouterClient) -> None:
    """'$20 claude subscription' → Tech + USD.

    Tag completeness is a nice-to-have but subjective. The prompt
    encourages both 'llm-api' and 'subscription'; the model may emit
    either or both. We only assert one or the other is present (not
    an unrelated tag) plus the category and currency.
    """
    p = await llm.parse_transaction_text("$20 claude subscription this month")
    assert p.type == "withdrawal"
    assert p.category == "Tech"
    assert p.currency == "USD"
    # Accept any reasonable tagging — at least one of the expected tags
    # must be present. If neither, the model missed the tagging entirely.
    assert "llm-api" in p.tags or "subscription" in p.tags, (
        f"expected llm-api or subscription in tags, got {p.tags}"
    )


async def test_parse_deposit(llm: OpenRouterClient) -> None:
    """Income recognition: 'salary 15jt' → deposit, Income, IDR 15000000."""
    p = await llm.parse_transaction_text("salary paid 15jt")
    assert p.type == "deposit"
    assert p.category == "Income"
    assert p.amount == 15000000


async def test_parse_groceries_vs_food(llm: OpenRouterClient) -> None:
    """Disambiguation: 'alfamart 50k' should be Groceries, not Food & Beverages."""
    p = await llm.parse_transaction_text("alfamart 50k")
    assert p.category == "Groceries"


async def test_parse_travel(llm: OpenRouterClient) -> None:
    """'flight to bali 2.5jt' → Travel or Transportation, IDR 2500000.

    Flights are arguably either category (Travel = trip category,
    Transportation = mode of movement). Both are defensible; either
    passes. The important part is amount + currency parsing.
    """
    p = await llm.parse_transaction_text("flight to bali 2.5jt")
    assert p.category in {"Travel", "Transportation"}
    assert p.currency == "IDR"
    assert p.amount == 2500000


async def test_confidence_signal_on_vague_input(llm: OpenRouterClient) -> None:
    """Low-information input should produce confidence != high."""
    p = await llm.parse_transaction_text("paid something 10k")
    assert p.confidence in {"low", "medium"}


# ============================================================
# Stats tracking sanity
# ============================================================


async def test_stats_incremented_after_calls(llm: OpenRouterClient) -> None:
    """Stats should increment per model *attempted*, not per parse call.

    If the primary model 503s and we fall over to the secondary, that's
    TWO stats increments for one parse_transaction_text call. So we assert
    "went up" rather than "by exactly 1".
    """
    before = sum(s.calls for s in llm.stats.per_model.values())
    await llm.parse_transaction_text("coffee 30k")
    after = sum(s.calls for s in llm.stats.per_model.values())
    assert after > before, "stats didn't increment at all"
    # At least one model succeeded (otherwise parse would have raised)
    assert sum(s.successes for s in llm.stats.per_model.values()) >= 1


async def test_parse_transaction_new_signature(llm: OpenRouterClient) -> None:
    """Sanity: the unified parse_transaction method works with text= kwarg.

    Phase 2 will add image_bytes= to this same method.
    """
    p = await llm.parse_transaction(text="coffee 30k")
    assert p.type == "withdrawal"
    assert p.currency == "IDR"


async def test_parse_transaction_rejects_both_empty(llm: OpenRouterClient) -> None:
    """parse_transaction with neither text nor image should raise."""
    with pytest.raises(ValueError, match="text, image"):
        await llm.parse_transaction()


# ============================================================
# Failure path — bad key
# ============================================================


async def test_bad_api_key_raises_auth_error() -> None:
    from firefly_agent.errors import OpenRouterAuthError

    client = OpenRouterClient(
        api_key="sk-or-v1-obviously-wrong-key-for-testing",
        models=MODELS,
        allowed_categories=ALLOWED_CATEGORIES,
        allowed_tags=ALLOWED_TAGS,
        allowed_currencies=ALLOWED_CURRENCIES,
        default_currency="IDR",
        tag_groups=TAG_GROUPS,
    )
    async with client as c:
        with pytest.raises(OpenRouterAuthError):
            await c.parse_transaction(text="coffee 30k")
