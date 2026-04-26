"""Unit tests for M1.5d — intent reconcile, edit sub-menu, repayment flow."""

from __future__ import annotations

import pytest

from firefly_agent.bnpl import (
    detect_repayment_intent,
    reconcile_intent,
)
from firefly_agent.formatting import (
    CB_BACK,
    CB_EDIT_FIELD,
    EDIT_PROMPTS,
    build_edit_submenu_keyboard,
    format_confirm_message,
    format_preview_message,
)
from firefly_agent.models import Account
from firefly_agent.parsed import ParsedTransaction


def _parsed(**overrides) -> ParsedTransaction:
    base = {
        "type": "withdrawal",
        "amount": 1000000,
        "currency": "IDR",
        "description": "Headset",
        "merchant": "SpayLater",
        "category": "Shopping",
        "tags": [],
        "date": "2026-04-25",
        "confidence": "high",
        "intent": "regular",
        "notes": "",
    }
    base.update(overrides)
    return ParsedTransaction.model_validate(base)


def _account(**overrides) -> Account:
    base = {"id": 6, "name": "BCA savings", "type": "asset", "currency_code": "IDR"}
    base.update(overrides)
    return Account(**base)


# ============================================================
# Intent reconcile
# ============================================================


class TestIntentReconcile:
    def test_llm_says_repayment_keyword_agrees(self) -> None:
        p = _parsed(intent="repayment", description="repay spaylater")
        assert reconcile_intent(p, full_text="repay spaylater 1mil") == "repayment"

    def test_llm_says_repayment_keyword_disagrees_trust_llm(self) -> None:
        p = _parsed(intent="repayment", description="something")
        assert reconcile_intent(p, full_text="random text") == "repayment"

    def test_llm_says_regular_keyword_says_repayment_override(self) -> None:
        """LLM mis-classifies; keywords catch it. Override to repayment."""
        p = _parsed(intent="regular", description="bayar spaylater")
        assert reconcile_intent(p, full_text="bayar spaylater 1mil") == "repayment"

    def test_llm_says_purchase_keyword_silent_trusts_llm(self) -> None:
        p = _parsed(intent="purchase", description="bought headset")
        assert reconcile_intent(p, full_text="bought headset 1mil spaylater") == "purchase"

    def test_llm_says_regular_keyword_silent_stays_regular(self) -> None:
        p = _parsed(intent="regular", description="Coffee", merchant="Excelso")
        assert reconcile_intent(p, full_text="coffee 50k at excelso") == "regular"

    def test_keyword_repay(self) -> None:
        p = _parsed(description="x")
        assert detect_repayment_intent(p, full_text="repay spaylater 1mil") is True

    def test_keyword_bayar(self) -> None:
        p = _parsed(description="x")
        assert detect_repayment_intent(p, full_text="bayar kredivo 350k") is True

    def test_keyword_lunas(self) -> None:
        p = _parsed(description="x")
        assert detect_repayment_intent(p, full_text="lunas akulaku") is True

    def test_keyword_no_match(self) -> None:
        p = _parsed(description="Coffee", merchant="Excelso")
        assert detect_repayment_intent(p, full_text="coffee 50k") is False


# ============================================================
# Edit sub-menu keyboard
# ============================================================


class TestEditSubmenu:
    def test_six_buttons_in_three_rows(self) -> None:
        kb = build_edit_submenu_keyboard("ABC123XY")
        assert len(kb.inline_keyboard) == 3
        assert all(len(row) == 2 for row in kb.inline_keyboard)

    def test_field_actions_correct(self) -> None:
        kb = build_edit_submenu_keyboard("ABC123XY")
        # Row 0: Description, Merchant
        assert kb.inline_keyboard[0][0].callback_data == f"{CB_EDIT_FIELD}:ABC123XY:description"
        assert kb.inline_keyboard[0][1].callback_data == f"{CB_EDIT_FIELD}:ABC123XY:merchant"
        # Row 1: Tags, Notes
        assert kb.inline_keyboard[1][0].callback_data == f"{CB_EDIT_FIELD}:ABC123XY:tags"
        assert kb.inline_keyboard[1][1].callback_data == f"{CB_EDIT_FIELD}:ABC123XY:notes"
        # Row 2: Redo all (= "full"), Back
        assert kb.inline_keyboard[2][0].callback_data == f"{CB_EDIT_FIELD}:ABC123XY:full"
        assert kb.inline_keyboard[2][1].callback_data == f"{CB_BACK}:ABC123XY"

    def test_callback_data_under_64_bytes(self) -> None:
        kb = build_edit_submenu_keyboard("ABC123XY")
        for row in kb.inline_keyboard:
            for btn in row:
                assert len(btn.callback_data.encode()) <= 64

    def test_prompts_have_current_placeholder_for_per_field(self) -> None:
        for field in ("description", "merchant", "tags", "notes"):
            assert "{current}" in EDIT_PROMPTS[field], field

    def test_prompt_for_full_has_no_placeholder(self) -> None:
        # 'full' edit doesn't need to show "current value"
        assert "{current}" not in EDIT_PROMPTS["full"]


# ============================================================
# Preview/confirm rendering with intent + notes
# ============================================================


class TestPreviewWithIntentAndNotes:
    def test_repayment_label(self) -> None:
        p = _parsed(intent="repayment", description="SpayLater repayment")
        msg = format_preview_message(p)
        assert "BNPL repayment" in msg

    def test_purchase_label(self) -> None:
        p = _parsed(intent="purchase", description="Headset")
        msg = format_preview_message(p)
        assert "BNPL purchase" in msg

    def test_regular_keeps_old_label(self) -> None:
        p = _parsed(intent="regular", type="withdrawal", description="Coffee", merchant="Excelso")
        msg = format_preview_message(p)
        assert "Withdrawal" in msg
        assert "BNPL" not in msg

    def test_notes_shown_when_present(self) -> None:
        p = _parsed(notes="Subtotal Rp 45000 + service Rp 2500")
        msg = format_preview_message(p)
        assert "Subtotal" in msg
        assert "📓" in msg

    def test_notes_hidden_when_empty(self) -> None:
        p = _parsed(notes="")
        msg = format_preview_message(p)
        assert "📓" not in msg

    def test_notes_html_escaped(self) -> None:
        p = _parsed(notes='<b>boom</b>')
        msg = format_preview_message(p)
        assert "<b>boom</b>" not in msg
        assert "&lt;b&gt;" in msg


class TestConfirmWithIntentAndNotes:
    def test_repayment_label_in_confirm(self) -> None:
        p = _parsed(intent="repayment", description="SpayLater repayment")
        msg = format_confirm_message(
            p, selected_currency="IDR",
            source_account=_account(name="BCA savings"),
        )
        assert "BNPL repayment" in msg

    def test_notes_in_confirm(self) -> None:
        p = _parsed(notes="Reimbursable through Q3 budget")
        msg = format_confirm_message(
            p, selected_currency="IDR",
            source_account=_account(name="BCA"),
        )
        assert "Reimbursable" in msg


# ============================================================
# Schema includes intent + notes as required
# ============================================================


class TestSchemaIncludesNewFields:
    def test_intent_required(self) -> None:
        from firefly_agent.parsed import parsed_transaction_json_schema
        schema = parsed_transaction_json_schema(
            allowed_categories=["Food & Beverages"],
            allowed_tags=["coffee"],
            allowed_currencies=["IDR", "USD"],
        )
        assert "intent" in schema["schema"]["required"]
        assert "notes" in schema["schema"]["required"]
        assert schema["schema"]["properties"]["intent"]["enum"] == [
            "purchase", "repayment", "regular"
        ]

    def test_intent_default_is_regular(self) -> None:
        # When not provided, defaults — used by older test fixtures
        p = ParsedTransaction.model_validate({
            "type": "withdrawal", "amount": 1000, "currency": "IDR",
            "description": "x", "merchant": "y", "category": "Food & Beverages",
            "tags": [], "date": "2026-04-25", "confidence": "high",
        })
        assert p.intent == "regular"
        assert p.notes == ""
