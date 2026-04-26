"""Unit tests for M1.2 — no network required.

These validate:
- Model round-tripping (JSON:API in → Python object → Firefly POST JSON out)
- TTL cache behavior
- Error type hierarchy

Integration tests live in test_m1_2_integration.py.
"""

from __future__ import annotations

import asyncio
from datetime import date
from decimal import Decimal

import pytest

from firefly_agent.errors import (
    FireflyAuthError,
    FireflyError,
    FireflyNotFoundError,
    FireflyUnavailableError,
    FireflyValidationError,
)
from firefly_agent.firefly import FireflyClient, _TTLCache
from firefly_agent.models import (
    Account,
    Category,
    NewTransaction,
    TransactionSplit,
)


# ============================================================
# Model: Account
# ============================================================


class TestAccountFromJsonApi:
    def test_parses_asset_account(self) -> None:
        payload = {
            "type": "accounts",
            "id": "6",
            "attributes": {
                "name": "BCA savings account",
                "type": "asset",
                "currency_code": "IDR",
                "current_balance": "1000000",
            },
        }
        a = Account.from_jsonapi(payload)
        assert a.id == 6
        assert a.name == "BCA savings account"
        assert a.type == "asset"
        assert a.currency_code == "IDR"
        assert a.current_balance == Decimal("1000000")

    def test_parses_liability_account(self) -> None:
        payload = {
            "id": "12",
            "attributes": {
                "name": "Account Payable",
                "type": "liabilities",
                "currency_code": "IDR",
                "liability_type": "debt",
            },
        }
        a = Account.from_jsonapi(payload)
        assert a.id == 12
        assert a.type == "liabilities"
        assert a.liability_type == "debt"


# ============================================================
# Model: Category
# ============================================================


class TestCategoryFromJsonApi:
    def test_parses_minimal(self) -> None:
        payload = {"id": "3", "attributes": {"name": "Food & Beverages"}}
        c = Category.from_jsonapi(payload)
        assert c.id == 3
        assert c.name == "Food & Beverages"


# ============================================================
# Model: TransactionSplit → Firefly JSON
# ============================================================


class TestTransactionSplitSerialization:
    def _base_split(self, **overrides: object) -> TransactionSplit:
        defaults: dict[str, object] = {
            "type": "withdrawal",
            "date": date(2026, 4, 24),
            "amount": Decimal("45000"),
            "description": "Coffee at Starbucks",
            "currency_code": "IDR",
            "source_id": 6,
            "destination_name": "Starbucks",
            "category_name": "Food & Beverages",
            "tags": ["coffee"],
        }
        defaults.update(overrides)
        return TransactionSplit(**defaults)  # type: ignore[arg-type]

    def test_basic_withdrawal_shape(self) -> None:
        split = self._base_split()
        payload = split.to_firefly_json()

        assert payload["type"] == "withdrawal"
        assert payload["date"] == "2026-04-24"
        assert payload["amount"] == "45000"
        assert payload["description"] == "Coffee at Starbucks"
        assert payload["currency_code"] == "IDR"
        assert payload["source_id"] == "6"  # stringified for Firefly
        assert payload["destination_name"] == "Starbucks"
        assert payload["category_name"] == "Food & Beverages"
        assert payload["tags"] == ["coffee"]

    def test_foreign_amount_fields_included_when_set(self) -> None:
        split = self._base_split(
            currency_code="USD",
            amount=Decimal("1.00"),
            foreign_amount=Decimal("16500"),
            foreign_currency_code="IDR",
        )
        payload = split.to_firefly_json()
        assert payload["currency_code"] == "USD"
        assert payload["amount"] == "1.00"
        assert payload["foreign_amount"] == "16500"
        assert payload["foreign_currency_code"] == "IDR"

    def test_foreign_amount_omitted_when_absent(self) -> None:
        split = self._base_split()
        payload = split.to_firefly_json()
        assert "foreign_amount" not in payload
        assert "foreign_currency_code" not in payload

    def test_empty_tags_omitted(self) -> None:
        split = self._base_split(tags=[])
        payload = split.to_firefly_json()
        assert "tags" not in payload

    def test_optional_fields_omitted(self) -> None:
        split = self._base_split(category_name=None, tags=[])
        payload = split.to_firefly_json()
        assert "category_name" not in payload
        assert "tags" not in payload

    def test_new_transaction_envelope(self) -> None:
        split = self._base_split()
        tx = NewTransaction(transactions=[split])
        payload = tx.to_firefly_json()
        assert payload["apply_rules"] is True
        assert payload["error_if_duplicate_hash"] is False
        assert len(payload["transactions"]) == 1


# ============================================================
# TTL cache
# ============================================================


class TestTTLCache:
    @pytest.mark.asyncio
    async def test_hit_returns_cached(self) -> None:
        c = _TTLCache(ttl_seconds=60)
        await c.set("key", [1, 2, 3])
        assert await c.get("key") == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_miss_returns_none(self) -> None:
        c = _TTLCache(ttl_seconds=60)
        assert await c.get("never-set") is None

    @pytest.mark.asyncio
    async def test_expiry_returns_none(self) -> None:
        c = _TTLCache(ttl_seconds=0.05)
        await c.set("key", "value")
        await asyncio.sleep(0.1)
        assert await c.get("key") is None

    @pytest.mark.asyncio
    async def test_invalidate_all(self) -> None:
        c = _TTLCache(ttl_seconds=60)
        await c.set("a", 1)
        await c.set("b", 2)
        await c.invalidate()
        assert await c.get("a") is None
        assert await c.get("b") is None

    @pytest.mark.asyncio
    async def test_invalidate_single_key(self) -> None:
        c = _TTLCache(ttl_seconds=60)
        await c.set("a", 1)
        await c.set("b", 2)
        await c.invalidate("a")
        assert await c.get("a") is None
        assert await c.get("b") == 2


# ============================================================
# Client construction
# ============================================================


class TestClientConstruction:
    def test_rejects_url_without_scheme(self) -> None:
        with pytest.raises(ValueError, match="http"):
            FireflyClient("firefly.example.com", "pat")

    def test_rejects_empty_pat(self) -> None:
        with pytest.raises(ValueError, match="personal_access_token"):
            FireflyClient("https://firefly.example.com", "")

    def test_strips_trailing_slash(self) -> None:
        c = FireflyClient("https://firefly.example.com/", "pat")
        assert c._base_url == "https://firefly.example.com"


# ============================================================
# Exception hierarchy
# ============================================================


class TestExceptionHierarchy:
    def test_all_inherit_base(self) -> None:
        for exc_cls in (
            FireflyAuthError,
            FireflyValidationError,
            FireflyNotFoundError,
            FireflyUnavailableError,
        ):
            assert issubclass(exc_cls, FireflyError)

    def test_validation_error_carries_field_errors(self) -> None:
        err = FireflyValidationError(
            "bad", field_errors={"description": ["must not be empty"]}
        )
        assert err.field_errors == {"description": ["must not be empty"]}

    def test_status_code_attached_when_provided(self) -> None:
        err = FireflyAuthError("denied", status_code=401)
        assert err.status_code == 401
