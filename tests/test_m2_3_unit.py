"""Unit tests for M2.3 — same-currency account transfers."""

from __future__ import annotations

from pathlib import Path

import pytest

from firefly_agent.formatting import (
    CB_BACK,
    CB_CANCEL,
    CB_DESTINATION,
    build_transfer_destination_keyboard,
    format_preview_message,
    format_transfer_confirm_message,
)
from firefly_agent.models import Account
from firefly_agent.parsed import ParsedTransaction
from firefly_agent.state import StateStore


def _parsed(**overrides) -> ParsedTransaction:
    base = {
        "type": "withdrawal",
        "amount": 500000,
        "currency": "IDR",
        "description": "Transfer",
        "merchant": "Cash wallet",
        "category": "Other",
        "tags": [],
        "date": "2026-04-25",
        "time": "",
        "confidence": "high",
        "intent": "transfer",
        "notes": "",
    }
    base.update(overrides)
    return ParsedTransaction.model_validate(base)


def _account(**overrides) -> Account:
    base = {"id": 6, "name": "BCA savings", "type": "asset", "currency_code": "IDR"}
    base.update(overrides)
    return Account(**base)


# ============================================================
# Schema / intent
# ============================================================


class TestTransferIntent:
    def test_transfer_in_intent_enum(self) -> None:
        from firefly_agent.parsed import parsed_transaction_json_schema
        s = parsed_transaction_json_schema(
            allowed_categories=["Other"],
            allowed_tags=["x"],
            allowed_currencies=["IDR"],
        )
        assert "transfer" in s["schema"]["properties"]["intent"]["enum"]

    def test_parses_transfer_intent(self) -> None:
        p = _parsed(intent="transfer")
        assert p.intent == "transfer"


# ============================================================
# Transfer destination keyboard
# ============================================================


class TestTransferDestinationKeyboard:
    def test_layout(self) -> None:
        kb = build_transfer_destination_keyboard(
            callback_id="ABC12345",
            destinations=[
                _account(id=7, name="Cash wallet"),
                _account(id=10, name="Other savings"),
            ],
        )
        # 2 destination rows + 1 back/cancel row
        assert len(kb.inline_keyboard) == 3
        assert kb.inline_keyboard[0][0].callback_data == f"{CB_DESTINATION}:ABC12345:7"
        assert kb.inline_keyboard[1][0].callback_data == f"{CB_DESTINATION}:ABC12345:10"
        assert kb.inline_keyboard[2][0].callback_data == f"{CB_BACK}:ABC12345"
        assert kb.inline_keyboard[2][1].callback_data == f"{CB_CANCEL}:ABC12345"

    def test_arrow_prefix_in_label(self) -> None:
        kb = build_transfer_destination_keyboard(
            callback_id="X",
            destinations=[_account(id=7, name="Cash wallet")],
        )
        assert "→" in kb.inline_keyboard[0][0].text
        assert "Cash wallet" in kb.inline_keyboard[0][0].text

    def test_callback_under_64_bytes(self) -> None:
        kb = build_transfer_destination_keyboard(
            callback_id="ABCDEFGH",
            destinations=[_account(id=999, name="Some long account name with spaces")],
        )
        for row in kb.inline_keyboard:
            for btn in row:
                assert len(btn.callback_data.encode()) <= 64


# ============================================================
# Preview / confirm rendering for transfer
# ============================================================


class TestTransferRendering:
    def test_preview_shows_transfer_label(self) -> None:
        p = _parsed(intent="transfer", description="Top up cash wallet")
        msg = format_preview_message(p)
        assert "🔄 Transfer" in msg
        assert "Withdrawal" not in msg

    def test_confirm_shows_source_to_destination(self) -> None:
        p = _parsed(intent="transfer", description="Top up cash wallet")
        msg = format_transfer_confirm_message(
            p,
            selected_currency="IDR",
            source_account=_account(id=6, name="BCA savings"),
            destination_account=_account(id=7, name="Cash wallet"),
        )
        assert "BCA savings" in msg
        assert "Cash wallet" in msg
        assert "→" in msg

    def test_confirm_excludes_redundant_description(self) -> None:
        """If description is just 'transfer' don't show it as a subtitle."""
        p = _parsed(intent="transfer", description="transfer")
        msg = format_transfer_confirm_message(
            p,
            selected_currency="IDR",
            source_account=_account(id=6, name="BCA savings"),
            destination_account=_account(id=7, name="Cash wallet"),
        )
        # Should NOT have an italic descriptor row showing just "transfer"
        # (case-insensitive check the bare word doesn't appear standalone)
        assert "<i>transfer</i>" not in msg.lower()


# ============================================================
# State transitions
# ============================================================


@pytest.fixture
async def store(tmp_path: Path):
    db = tmp_path / "state.db"
    async with StateStore(db) as s:
        yield s


class TestTransferStateMachine:
    @pytest.mark.asyncio
    async def test_advance_to_awaiting_destination(self, store: StateStore) -> None:
        cid = await store.insert_pending_transaction(
            user_id=1, chat_id=1, message_id=1,
            payload_json='{"x":1}', currency="IDR",
        )
        ok = await store.advance_to_awaiting_destination(cid, source_account_id=6)
        assert ok is True

        pending = await store.get_pending_transaction(cid)
        assert pending is not None
        assert pending.state == "awaiting_destination"
        assert pending.source_account_id == 6
        assert pending.destination_account_id is None

    @pytest.mark.asyncio
    async def test_advance_destination_to_confirm(self, store: StateStore) -> None:
        cid = await store.insert_pending_transaction(
            user_id=1, chat_id=1, message_id=1,
            payload_json='{"x":1}', currency="IDR",
        )
        await store.advance_to_awaiting_destination(cid, source_account_id=6)
        ok = await store.advance_destination_to_confirm(cid, destination_account_id=7)
        assert ok is True

        pending = await store.get_pending_transaction(cid)
        assert pending is not None
        assert pending.state == "awaiting_confirm"
        assert pending.source_account_id == 6
        assert pending.destination_account_id == 7

    @pytest.mark.asyncio
    async def test_back_clears_destination_too(self, store: StateStore) -> None:
        """Back-button from confirm should reset BOTH source and destination
        for transfers, sending the user back to source picker."""
        cid = await store.insert_pending_transaction(
            user_id=1, chat_id=1, message_id=1,
            payload_json='{"x":1}', currency="IDR",
        )
        await store.advance_to_awaiting_destination(cid, source_account_id=6)
        await store.advance_destination_to_confirm(cid, destination_account_id=7)
        await store.update_pending_back_to_awaiting_account(cid)

        pending = await store.get_pending_transaction(cid)
        assert pending is not None
        assert pending.state == "awaiting_account"
        assert pending.source_account_id is None
        assert pending.destination_account_id is None

    @pytest.mark.asyncio
    async def test_back_from_destination_picker(self, store: StateStore) -> None:
        """Back from awaiting_destination also returns to awaiting_account."""
        cid = await store.insert_pending_transaction(
            user_id=1, chat_id=1, message_id=1,
            payload_json='{"x":1}', currency="IDR",
        )
        await store.advance_to_awaiting_destination(cid, source_account_id=6)
        await store.update_pending_back_to_awaiting_account(cid)

        pending = await store.get_pending_transaction(cid)
        assert pending.state == "awaiting_account"
        assert pending.source_account_id is None
