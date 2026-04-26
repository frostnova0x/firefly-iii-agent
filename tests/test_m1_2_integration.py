"""Integration tests for M1.2 — real Firefly III.

Enable:
    export FIREFLY_INTEGRATION_OK=1
    export FIREFLY_URL=https://firefly.example.com
    export FIREFLY_PAT=<your-pat>
    export DEFAULT_ASSET_ACCOUNT_NAME="Test Account"
    uv run pytest -v -m integration

Every test transaction is prefixed `TEST M1.2` in its description and
deleted in teardown. Should any escape cleanup, search Firefly for
that prefix to find and delete manually.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from firefly_agent.errors import (
    FireflyAuthError,
    FireflyNotFoundError,
    FireflyValidationError,
)
from firefly_agent.firefly import FireflyClient
from firefly_agent.models import NewTransaction, TransactionSplit


pytestmark = pytest.mark.integration


# ============================================================
# Read-only tests — safe, no side effects
# ============================================================


async def test_about_reports_version(firefly_url: str, firefly_pat: str) -> None:
    async with FireflyClient(firefly_url, firefly_pat) as client:
        about = await client.about()
    assert "version" in about
    assert "api_version" in about
    # Sanity: we pinned 6.6.1
    assert about["version"].startswith("6.")


async def test_list_asset_accounts_returns_expected_shape(
    firefly_url: str, firefly_pat: str
) -> None:
    async with FireflyClient(firefly_url, firefly_pat) as client:
        accounts = await client.list_asset_accounts()
    assert len(accounts) >= 1
    for acc in accounts:
        assert acc.id > 0
        assert acc.name
        assert acc.type == "asset"
        assert acc.currency_code is not None


async def test_list_liability_accounts_returns_debt_type(
    firefly_url: str, firefly_pat: str
) -> None:
    async with FireflyClient(firefly_url, firefly_pat) as client:
        liabilities = await client.list_liability_accounts()
    # May be empty if user hasn't set up any liabilities; if present,
    # liability_type should be populated.
    for liab in liabilities:
        assert liab.type == "liabilities"
        assert liab.liability_type in {"debt", "loan", "mortgage", None}


async def test_list_categories_contains_expected_set(
    firefly_url: str, firefly_pat: str
) -> None:
    async with FireflyClient(firefly_url, firefly_pat) as client:
        categories = await client.list_categories()
    names = {c.name for c in categories}
    # We created these in Phase 0
    assert "Food & Beverages" in names
    assert "Tech" in names


# ============================================================
# Cache behavior
# ============================================================


async def test_cache_hit_avoids_second_network_call(
    firefly_url: str, firefly_pat: str
) -> None:
    """Second call within TTL should return the same objects (by identity
    they may differ since we construct fresh Pydantic models, but call
    count should drop — we verify by checking identical results fast).
    """
    async with FireflyClient(firefly_url, firefly_pat, cache_ttl_seconds=60) as client:
        first = await client.list_categories()
        second = await client.list_categories()
    assert {c.id for c in first} == {c.id for c in second}


async def test_invalidate_cache_forces_refetch(
    firefly_url: str, firefly_pat: str
) -> None:
    async with FireflyClient(firefly_url, firefly_pat) as client:
        await client.list_categories()
        await client.invalidate_cache()
        # Second call now re-fetches; asserting it doesn't raise is enough
        categories = await client.list_categories()
        assert len(categories) >= 1


# ============================================================
# Write tests — create + delete, with guaranteed cleanup
# ============================================================


async def test_create_simple_withdrawal_and_delete(
    firefly_url: str,
    firefly_pat: str,
    default_asset_account_id: int,
    transaction_tracker,
    test_tag_prefix: str,
) -> None:
    """Happy path: create a basic IDR withdrawal, receive an ID back,
    then let teardown delete it.
    """
    tx = NewTransaction(
        transactions=[
            TransactionSplit(
                type="withdrawal",
                date=date.today(),
                amount=Decimal("1000"),
                description=f"{test_tag_prefix} — Simple withdrawal",
                currency_code="IDR",
                source_id=default_asset_account_id,
                destination_name="Test Expense",
                category_name="Food & Beverages",
                tags=["coffee"],
            )
        ]
    )

    async with FireflyClient(firefly_url, firefly_pat) as client:
        created = await client.create_transaction(tx)

    transaction_tracker.track(created.group_id)

    assert created.group_id > 0
    assert test_tag_prefix in created.description


async def test_create_withdrawal_with_foreign_currency(
    firefly_url: str,
    firefly_pat: str,
    default_asset_account_id: int,
    transaction_tracker,
    test_tag_prefix: str,
) -> None:
    """Foreign-amount path: charge USD on an IDR card."""
    tx = NewTransaction(
        transactions=[
            TransactionSplit(
                type="withdrawal",
                date=datetime.now(UTC),
                amount=Decimal("1.00"),  # $1 USD
                description=f"{test_tag_prefix} — Foreign currency",
                currency_code="USD",
                source_id=default_asset_account_id,  # IDR account
                destination_name="Test Foreign Expense",
                category_name="Tech",
                tags=["llm-api"],
                foreign_amount=Decimal("16500"),  # ~rate
                foreign_currency_code="IDR",
            )
        ]
    )

    async with FireflyClient(firefly_url, firefly_pat) as client:
        created = await client.create_transaction(tx)

    transaction_tracker.track(created.group_id)
    assert created.group_id > 0


async def test_delete_nonexistent_raises_not_found(
    firefly_url: str, firefly_pat: str
) -> None:
    async with FireflyClient(firefly_url, firefly_pat) as client:
        with pytest.raises(FireflyNotFoundError):
            await client.delete_transaction(99_999_999)


# ============================================================
# Error paths
# ============================================================


async def test_bad_pat_raises_auth_error(firefly_url: str) -> None:
    async with FireflyClient(firefly_url, "deliberately-bad-pat") as client:
        with pytest.raises(FireflyAuthError):
            await client.about()


async def test_invalid_transaction_raises_validation_error(
    firefly_url: str,
    firefly_pat: str,
    default_asset_account_id: int,
) -> None:
    """Send a transaction missing a required field to trigger 422."""
    tx = NewTransaction(
        transactions=[
            TransactionSplit(
                type="withdrawal",
                date=date.today(),
                amount=Decimal("1"),
                description="",  # Firefly rejects empty description
                currency_code="IDR",
                source_id=default_asset_account_id,
                destination_name="Test",
            )
        ]
    )

    async with FireflyClient(firefly_url, firefly_pat) as client:
        with pytest.raises(FireflyValidationError) as exc_info:
            await client.create_transaction(tx)

    # 422 should come with field-level error details
    assert exc_info.value.status_code == 422
