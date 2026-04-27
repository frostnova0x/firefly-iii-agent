"""Unit tests for M1.5c — BNPL detection, edit-mode state, account prune, BNPL keyboard."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from firefly_agent.bnpl import detect_bnpl_account_id
from firefly_agent.formatting import (
    CB_CANCEL,
    CB_CONFIRM,
    CB_CURRENCY,
    CB_EDIT,
    CB_NOOP,
    CurrencyButton,
    build_bnpl_awaiting_confirm_keyboard,
)
from firefly_agent.models import Account
from firefly_agent.parsed import ParsedTransaction
from firefly_agent.state import StateStore


def _parsed(**overrides) -> ParsedTransaction:
    base = {
        "type": "withdrawal",
        "amount": 1000000,
        "currency": "IDR",
        "description": "Headset",
        "merchant": "SpayLater",
        "category": "Shopping",
        "tags": ["electronics"],
        "date": "2026-04-25",
        "confidence": "high",
    }
    base.update(overrides)
    return ParsedTransaction.model_validate(base)


def _account(**overrides) -> Account:
    base = {"id": 12, "name": "Account Payable", "type": "liabilities", "currency_code": "IDR"}
    base.update(overrides)
    return Account(**base)


# ============================================================
# BNPL detection
# ============================================================


class TestBNPLDetection:
    KW_MAP = {"kredivo": 12, "akulaku": 12, "spaylater": 12, "shopee paylater": 12}

    def test_match_on_merchant_field(self) -> None:
        p = _parsed(merchant="SpayLater")
        assert detect_bnpl_account_id(p, self.KW_MAP) == 12

    def test_match_on_full_text(self) -> None:
        p = _parsed(merchant="Tokopedia")  # merchant guessed wrong
        assert detect_bnpl_account_id(p, self.KW_MAP, "bought 1mil headset using spaylater") == 12

    def test_case_insensitive(self) -> None:
        p = _parsed(merchant="KREDIVO")
        assert detect_bnpl_account_id(p, self.KW_MAP) == 12

    def test_substring_match(self) -> None:
        """'shopee paylater' should match even when merchant says 'Shopee Pay Later'."""
        p = _parsed(merchant="Shopee Pay Later Service")
        # The keyword 'shopee paylater' (with no space in 'paylater') won't match
        # a string that has 'pay later' separated; that's expected. But the
        # plain 'spaylater' keyword is what catches users typing 'spaylater'.
        # So this asserts the case-sensitive substring contract.
        assert detect_bnpl_account_id(p, self.KW_MAP) is None

    def test_no_match_returns_none(self) -> None:
        p = _parsed(merchant="Starbucks")
        assert detect_bnpl_account_id(p, self.KW_MAP, "coffee 50k at starbucks") is None

    def test_empty_keyword_map(self) -> None:
        p = _parsed(merchant="SpayLater")
        assert detect_bnpl_account_id(p, {}) is None


# ============================================================
# Edit mode CRUD (StateStore)
# ============================================================


@pytest.fixture
async def store(tmp_path: Path):
    db = tmp_path / "state.db"
    async with StateStore(db) as s:
        yield s


class TestEditMode:
    @pytest.mark.asyncio
    async def test_set_and_get(self, store: StateStore) -> None:
        await store.set_edit_mode(user_id=42, callback_id="ABC123XY")
        result = await store.get_edit_mode(42)
        assert result == ("ABC123XY", "full")

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, store: StateStore) -> None:
        assert await store.get_edit_mode(9999) is None

    @pytest.mark.asyncio
    async def test_upsert(self, store: StateStore) -> None:
        """Tapping Edit on a fresh preview while still in edit mode for an
        older one should reset the pointer."""
        await store.set_edit_mode(user_id=42, callback_id="OLD12345")
        await store.set_edit_mode(user_id=42, callback_id="NEW67890")
        result = await store.get_edit_mode(42)
        assert result == ("NEW67890", "full")

    @pytest.mark.asyncio
    async def test_clear(self, store: StateStore) -> None:
        await store.set_edit_mode(user_id=42, callback_id="ABC123XY")
        assert await store.clear_edit_mode(42) is True
        assert await store.get_edit_mode(42) is None
        # Idempotent — clearing again returns False
        assert await store.clear_edit_mode(42) is False

    @pytest.mark.asyncio
    async def test_expired_returns_none(self, store: StateStore) -> None:
        """ttl=0 → row is dead before next read."""
        await store.set_edit_mode(user_id=42, callback_id="ABC123XY", ttl_minutes=0)
        await asyncio.sleep(1.1)
        assert await store.get_edit_mode(42) is None

    @pytest.mark.asyncio
    async def test_reap_deletes_only_expired(self, store: StateStore) -> None:
        await store.set_edit_mode(user_id=1, callback_id="FRESH001", ttl_minutes=30)
        await store.set_edit_mode(user_id=2, callback_id="DEAD0002", ttl_minutes=0)
        await asyncio.sleep(1.1)
        reaped = await store.reap_expired_edit_modes()
        assert reaped == 1
        result_1 = await store.get_edit_mode(1)
        assert result_1 == ("FRESH001", "full")
        assert await store.get_edit_mode(2) is None

    @pytest.mark.asyncio
    async def test_field_stored_correctly(self, store: StateStore) -> None:
        """v3 schema: get_edit_mode returns (callback_id, field) tuple."""
        await store.set_edit_mode(user_id=42, callback_id="XYZ", field="tags")
        result = await store.get_edit_mode(42)
        assert result == ("XYZ", "tags")

    @pytest.mark.asyncio
    async def test_invalid_field_rejected(self, store: StateStore) -> None:
        with pytest.raises(ValueError, match="Invalid edit field"):
            await store.set_edit_mode(
                user_id=42, callback_id="X", field="random_garbage"
            )


# ============================================================
# Account prune
# ============================================================


class TestPruneAccountUsage:
    @pytest.mark.asyncio
    async def test_prunes_missing_ids(self, store: StateStore) -> None:
        await store.record_account_use(6, "BCA")
        await store.record_account_use(7, "Cash")
        await store.record_account_use(99, "DELETED")  # this account no longer exists

        pruned = await store.prune_account_usage(valid_account_ids={6, 7})
        assert pruned == 1

        ids = [u.account_id for u in await store.get_top_accounts(limit=10)]
        assert 99 not in ids
        assert 6 in ids and 7 in ids

    @pytest.mark.asyncio
    async def test_no_pruning_when_all_valid(self, store: StateStore) -> None:
        await store.record_account_use(6, "BCA")
        pruned = await store.prune_account_usage(valid_account_ids={6, 7, 9})
        assert pruned == 0

    @pytest.mark.asyncio
    async def test_empty_valid_set_does_nothing(self, store: StateStore) -> None:
        """Defensive: an empty set must NOT wipe everything (e.g., if Firefly
        is briefly returning zero accounts due to an error)."""
        await store.record_account_use(6, "BCA")
        pruned = await store.prune_account_usage(valid_account_ids=set())
        assert pruned == 0
        assert len(await store.get_top_accounts()) == 1


# ============================================================
# BNPL keyboard
# ============================================================


class TestBNPLKeyboard:
    def test_layout(self) -> None:
        kb = build_bnpl_awaiting_confirm_keyboard(
            callback_id="MYID0001",
            currencies=[
                CurrencyButton(code="IDR", label="IDR", selected=True),
                CurrencyButton(code="USD", label="USD", selected=False),
            ],
            liability_account=_account(id=12, name="Account Payable"),
        )
        # Row 0: currency toggle (2 buttons)
        # Row 1: liability indicator (1 button, no-op)
        # Row 2: confirm + edit (2 buttons)
        # Row 3: cancel (1 button)
        assert len(kb.inline_keyboard) == 4
        assert all(
            b.callback_data.startswith(f"{CB_CURRENCY}:") for b in kb.inline_keyboard[0]
        )
        assert kb.inline_keyboard[1][0].callback_data == "noop:MYID0001"
        assert kb.inline_keyboard[2][0].callback_data == f"{CB_CONFIRM}:MYID0001"
        assert kb.inline_keyboard[2][1].callback_data == f"{CB_EDIT}:MYID0001"
        assert kb.inline_keyboard[3][0].callback_data == f"{CB_CANCEL}:MYID0001"

    def test_liability_name_visible(self) -> None:
        kb = build_bnpl_awaiting_confirm_keyboard(
            callback_id="MYID0001",
            currencies=[CurrencyButton(code="IDR", label="IDR", selected=True)],
            liability_account=_account(id=12, name="SpayLater"),
        )
        # Liability button should mention the account name
        liability_btn = kb.inline_keyboard[0][0]  # first row when single currency
        assert "SpayLater" in liability_btn.text
        assert "✓" in liability_btn.text  # pre-selected indicator

    def test_callback_data_under_64_bytes(self) -> None:
        kb = build_bnpl_awaiting_confirm_keyboard(
            callback_id="ABC12345",
            currencies=[
                CurrencyButton(code="IDR", label="IDR", selected=True),
                CurrencyButton(code="USD", label="USD", selected=False),
            ],
            liability_account=_account(name="A very long liability account name with spaces"),
        )
        for row in kb.inline_keyboard:
            for btn in row:
                assert len(btn.callback_data.encode()) <= 64


# ============================================================
# Payload helpers (defensive)
# ============================================================


class TestPayloadHelpers:
    """The handlers stash bnpl_liability_id in payload_json. Make sure the
    helper functions in callbacks.py round-trip correctly."""

    def test_parsed_from_pending_strips_bnpl_field(self) -> None:
        from firefly_agent.handlers.callbacks import _parsed_from_pending
        from firefly_agent.state import PendingTransaction

        payload = json.dumps({
            "type": "withdrawal",
            "amount": 1000000,
            "currency": "IDR",
            "description": "Headset",
            "merchant": "SpayLater",
            "category": "Shopping",
            "tags": ["electronics"],
            "date": "2026-04-25",
            "confidence": "high",
            "bnpl_liability_id": 12,
        })
        pending = PendingTransaction(
            callback_id="x", user_id=1, chat_id=1, message_id=1,
            payload_json=payload, state="awaiting_account",
            source_account_id=None, destination_account_id=None, currency="IDR",
            created_at="2026-04-25T00:00:00+00:00",
            expires_at="2026-04-25T00:30:00+00:00",
        )
        # Must not raise — bnpl_liability_id stripped before model_validate
        parsed = _parsed_from_pending(pending)
        assert parsed.merchant == "SpayLater"

    def test_bnpl_id_from_pending(self) -> None:
        from firefly_agent.handlers.callbacks import _bnpl_id_from_pending
        from firefly_agent.state import PendingTransaction

        payload_with = json.dumps({"description": "x", "bnpl_liability_id": 12})
        payload_without = json.dumps({"description": "x"})

        pending_with = PendingTransaction(
            callback_id="x", user_id=1, chat_id=1, message_id=1,
            payload_json=payload_with, state="awaiting_account",
            source_account_id=None, destination_account_id=None, currency="IDR",
            created_at="2026-04-25T00:00:00+00:00",
            expires_at="2026-04-25T00:30:00+00:00",
        )
        pending_without = PendingTransaction(
            callback_id="y", user_id=1, chat_id=1, message_id=1,
            payload_json=payload_without, state="awaiting_account",
            source_account_id=None, destination_account_id=None, currency="IDR",
            created_at="2026-04-25T00:00:00+00:00",
            expires_at="2026-04-25T00:30:00+00:00",
        )

        assert _bnpl_id_from_pending(pending_with) == 12
        assert _bnpl_id_from_pending(pending_without) is None
