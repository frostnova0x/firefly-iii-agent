"""Unit tests for M1.5a — formatting helpers + keyboard builders.

The async Telegram handlers themselves are best validated by actually
DMing the bot. What we test here is the pure-logic layer (currency
formatting, message rendering, keyboard structure) which has well-defined
inputs and outputs.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from firefly_agent.formatting import (
    CB_ACCOUNT,
    CB_BACK,
    CB_CANCEL,
    CB_CONFIRM,
    CB_CURRENCY,
    CB_EDIT,
    CurrencyButton,
    build_awaiting_account_keyboard,
    build_awaiting_confirm_keyboard,
    build_bnpl_awaiting_confirm_keyboard,
    format_cancelled_message,
    format_confirm_message,
    format_currency,
    format_error_message,
    format_expired_message,
    format_logged_message,
    format_preview_message,
)
from firefly_agent.models import Account
from firefly_agent.parsed import ParsedTransaction


# ============================================================
# Currency formatting
# ============================================================


class TestFormatCurrency:
    def test_idr_no_decimals(self) -> None:
        assert format_currency(45000, "IDR") == "Rp 45,000"

    def test_idr_large_amount(self) -> None:
        assert format_currency(2_500_000, "IDR") == "Rp 2,500,000"

    def test_usd_two_decimals(self) -> None:
        assert format_currency(Decimal("19.99"), "USD") == "$19.99"

    def test_usd_whole_amount_keeps_decimals(self) -> None:
        assert format_currency(20, "USD") == "$20.00"

    def test_jpy_no_decimals(self) -> None:
        assert format_currency(1500, "JPY") == "¥1,500"

    def test_unknown_currency_uses_iso_code(self) -> None:
        out = format_currency(100, "XYZ")
        assert "XYZ" in out
        assert "100" in out

    def test_lowercase_currency_normalized(self) -> None:
        assert format_currency(45000, "idr") == "Rp 45,000"

    def test_thb(self) -> None:
        assert format_currency(500, "THB") == "฿500.00"


# ============================================================
# Preview message
# ============================================================


def _sample_parsed(**overrides) -> ParsedTransaction:
    base = {
        "type": "withdrawal",
        "amount": 45000,
        "currency": "IDR",
        "description": "Coffee",
        "merchant": "Starbucks",
        "category": "Food & Beverages",
        "tags": ["coffee"],
        "date": "2026-04-24",
        "confidence": "high",
    }
    base.update(overrides)
    return ParsedTransaction.model_validate(base)


def _sample_account(**overrides) -> Account:
    base = {
        "id": 6,
        "name": "BCA savings account",
        "type": "asset",
        "currency_code": "IDR",
    }
    base.update(overrides)
    return Account(**base)


class TestPreviewMessage:
    def test_includes_amount_currency_description(self) -> None:
        msg = format_preview_message(_sample_parsed())
        assert "Rp 45,000" in msg
        assert "Coffee" in msg
        assert "Starbucks" in msg
        assert "Food &amp; Beverages" in msg or "Food & Beverages" in msg
        assert "coffee" in msg
        assert "2026-04-24" in msg

    def test_html_escape_in_description(self) -> None:
        msg = format_preview_message(
            _sample_parsed(description='<script>alert("x")</script>')
        )
        assert "<script>" not in msg
        assert "&lt;script&gt;" in msg

    def test_html_escape_in_merchant(self) -> None:
        msg = format_preview_message(
            _sample_parsed(merchant='<b>boom</b>')
        )
        assert "<b>boom</b>" not in msg
        assert "&lt;b&gt;" in msg

    def test_dedups_when_description_equals_merchant(self) -> None:
        """For "alfamart 50k" cases the LLM may set both fields to 'Alfamart'.
        The preview shouldn't render 'Alfamart at Alfamart'."""
        msg = format_preview_message(
            _sample_parsed(description="Alfamart", merchant="Alfamart")
        )
        assert msg.count("Alfamart") == 1
        assert " at " not in msg

    def test_deposit_uses_from_connector(self) -> None:
        msg = format_preview_message(
            _sample_parsed(type="deposit", description="Salary", merchant="ACME Corp")
        )
        assert "Salary" in msg
        assert "ACME Corp" in msg
        assert "from" in msg

    def test_low_confidence_warning_shown(self) -> None:
        msg = format_preview_message(_sample_parsed(confidence="low"))
        assert "low confidence" in msg

    def test_high_confidence_no_warning(self) -> None:
        msg = format_preview_message(_sample_parsed(confidence="high"))
        assert "low confidence" not in msg

    def test_no_tags_no_tag_section(self) -> None:
        msg = format_preview_message(_sample_parsed(tags=[]))
        # No middle-dot tag separator
        assert " · " not in msg

    def test_currency_override(self) -> None:
        """User toggled to USD — preview should re-render with USD formatting."""
        # IDR-stored amount, but user wants to see USD interpretation
        msg = format_preview_message(
            _sample_parsed(amount=20, currency="IDR"),
            selected_currency="USD",
        )
        assert "$20.00" in msg
        assert "Rp" not in msg


# ============================================================
# Confirm message
# ============================================================


class TestConfirmMessage:
    def test_includes_source_account(self) -> None:
        msg = format_confirm_message(
            _sample_parsed(),
            selected_currency="IDR",
            source_account=_sample_account(name="BCA savings account"),
        )
        assert "BCA savings account" in msg

    def test_foreign_amount_shown_when_provided(self) -> None:
        msg = format_confirm_message(
            _sample_parsed(amount=20, currency="USD"),
            selected_currency="USD",
            source_account=_sample_account(currency_code="IDR"),
            foreign_amount=Decimal("330000"),
            foreign_currency="IDR",
        )
        assert "$20.00" in msg
        assert "Rp 330,000" in msg
        assert "Booked" in msg

    def test_foreign_amount_omitted_when_currencies_match(self) -> None:
        msg = format_confirm_message(
            _sample_parsed(),
            selected_currency="IDR",
            source_account=_sample_account(),
        )
        assert "Booked" not in msg


# ============================================================
# Final-state messages
# ============================================================


class TestFinalMessages:
    def test_logged_includes_id_and_description(self) -> None:
        msg = format_logged_message(123, "Coffee at Starbucks")
        assert "123" in msg
        assert "Coffee at Starbucks" in msg
        assert "✅" in msg

    def test_logged_escapes_description(self) -> None:
        msg = format_logged_message(1, "<b>boom</b>")
        assert "<b>boom</b>" not in msg
        assert "&lt;b&gt;" in msg

    def test_cancelled_message(self) -> None:
        assert "Cancel" in format_cancelled_message()

    def test_expired_message(self) -> None:
        assert "Expired" in format_expired_message()

    def test_error_message(self) -> None:
        msg = format_error_message("PAT rejected")
        assert "PAT rejected" in msg
        assert "⚠" in msg


# ============================================================
# Keyboards
# ============================================================


class TestAccountKeyboard:
    def _build(self, currencies=None, accounts=None):
        currencies = currencies or [
            CurrencyButton(code="IDR", label="IDR", selected=True),
            CurrencyButton(code="USD", label="USD", selected=False),
        ]
        accounts = accounts or [
            _sample_account(id=6, name="BCA savings"),
            _sample_account(id=7, name="Cash wallet"),
            _sample_account(id=9, name="Credit Card BCA"),
        ]
        return build_awaiting_account_keyboard(
            callback_id="ABC123XY",
            currencies=currencies,
            accounts=accounts,
        )

    def test_currency_row_present_when_multiple(self) -> None:
        kb = self._build()
        first_row = kb.inline_keyboard[0]
        assert all(b.callback_data.startswith(f"{CB_CURRENCY}:") for b in first_row)
        assert len(first_row) == 2

    def test_currency_row_hidden_when_single(self) -> None:
        kb = self._build(
            currencies=[CurrencyButton(code="IDR", label="IDR", selected=True)]
        )
        first_row = kb.inline_keyboard[0]
        # First row should be an account row, not currency
        assert not first_row[0].callback_data.startswith(f"{CB_CURRENCY}:")

    def test_selected_currency_has_check(self) -> None:
        kb = self._build()
        currency_row = kb.inline_keyboard[0]
        idr_button = currency_row[0]
        usd_button = currency_row[1]
        assert "✓" in idr_button.text
        assert "✓" not in usd_button.text

    def test_account_buttons_have_callback_data(self) -> None:
        kb = self._build()
        # Skip the currency row at index 0 and the cancel row at the end
        account_rows = kb.inline_keyboard[1:-1]
        assert len(account_rows) == 3  # 3 accounts
        for row in account_rows:
            assert row[0].callback_data.startswith(f"{CB_ACCOUNT}:")

    def test_callback_data_under_64_bytes(self) -> None:
        """Telegram limits callback_data to 64 bytes."""
        kb = self._build()
        for row in kb.inline_keyboard:
            for btn in row:
                assert len(btn.callback_data.encode("utf-8")) <= 64

    def test_cancel_button_at_bottom(self) -> None:
        """Bottom row is now [Edit, Cancel]."""
        kb = self._build()
        last_row = kb.inline_keyboard[-1]
        assert len(last_row) == 2
        # First button is Edit, second is Cancel
        assert last_row[0].callback_data.startswith(f"{CB_EDIT}:")
        assert last_row[1].callback_data.startswith(f"{CB_CANCEL}:")


class TestConfirmKeyboard:
    def test_four_buttons(self) -> None:
        """2 rows of 2: [Confirm, Back], [Edit, Cancel]."""
        kb = build_awaiting_confirm_keyboard("ABC123XY")
        assert len(kb.inline_keyboard) == 2
        assert len(kb.inline_keyboard[0]) == 2
        assert len(kb.inline_keyboard[1]) == 2

    def test_confirm_action(self) -> None:
        kb = build_awaiting_confirm_keyboard("ABC123XY")
        assert kb.inline_keyboard[0][0].callback_data == f"{CB_CONFIRM}:ABC123XY"

    def test_back_action(self) -> None:
        kb = build_awaiting_confirm_keyboard("ABC123XY")
        assert kb.inline_keyboard[0][1].callback_data == f"{CB_BACK}:ABC123XY"

    def test_edit_action(self) -> None:
        kb = build_awaiting_confirm_keyboard("ABC123XY")
        assert kb.inline_keyboard[1][0].callback_data == f"{CB_EDIT}:ABC123XY"

    def test_cancel_action(self) -> None:
        kb = build_awaiting_confirm_keyboard("ABC123XY")
        assert kb.inline_keyboard[1][1].callback_data == f"{CB_CANCEL}:ABC123XY"


# ============================================================
# Smoke: callback_data uses callback_id correctly
# ============================================================


class TestCallbackDataFormat:
    """Round-trip: build a keyboard, parse the callback_data the way
    handlers/callbacks.py does (split on ':')."""

    def test_currency_callback_parses_back(self) -> None:
        kb = build_awaiting_account_keyboard(
            callback_id="MYID0001",
            currencies=[CurrencyButton(code="USD", label="USD", selected=False)],
            accounts=[_sample_account()],
        )
        # No currency row when only one currency, so directly look for account
        # rows. Add a multi-currency case explicitly:
        kb = build_awaiting_account_keyboard(
            callback_id="MYID0001",
            currencies=[
                CurrencyButton(code="IDR", label="IDR", selected=True),
                CurrencyButton(code="USD", label="USD", selected=False),
            ],
            accounts=[_sample_account()],
        )
        cd = kb.inline_keyboard[0][1].callback_data  # USD button
        action, cid, code = cd.split(":")
        assert action == CB_CURRENCY
        assert cid == "MYID0001"
        assert code == "USD"

    def test_account_callback_parses_back(self) -> None:
        kb = build_awaiting_account_keyboard(
            callback_id="MYID0002",
            currencies=[CurrencyButton(code="IDR", label="IDR", selected=True)],
            accounts=[_sample_account(id=42)],
        )
        # First row is the account (no currency row since only 1 currency)
        cd = kb.inline_keyboard[0][0].callback_data
        action, cid, account_id = cd.split(":")
        assert action == CB_ACCOUNT
        assert cid == "MYID0002"
        assert int(account_id) == 42
