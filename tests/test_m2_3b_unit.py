"""Unit tests for transfer reclassification — validates LLM transfer claims
against the user's actual asset-account list.

This is the bot-side guard that prevents over-triggering. The LLM may
classify "transfer 500k to Joko" as transfer (ambiguous), but the bot
checks: is "Joko" in the user's account list? No → downgrade to regular.
"""

from __future__ import annotations

from firefly_agent.bnpl import reconcile_intent
from firefly_agent.parsed import ParsedTransaction


def _parsed(**overrides) -> ParsedTransaction:
    base = {
        "type": "withdrawal",
        "amount": 500000,
        "currency": "IDR",
        "description": "Transfer",
        "merchant": "",
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


class TestTransferReclassification:
    USER_ACCOUNTS = ["BCA savings", "Cash wallet", "GoPay", "USD account"]

    def test_real_transfer_kept(self) -> None:
        """Two of MY accounts mentioned → keep as transfer."""
        p = _parsed(intent="transfer", merchant="Cash wallet")
        result = reconcile_intent(
            p,
            full_text="transfer 500k from bca to cash wallet",
            asset_account_names=self.USER_ACCOUNTS,
        )
        assert result == "transfer"

    def test_transfer_to_person_downgraded(self) -> None:
        """LLM hallucinated transfer for sending money to a person."""
        p = _parsed(intent="transfer", merchant="Joko")
        result = reconcile_intent(
            p,
            full_text="transfer 500k to joko for dinner",
            asset_account_names=self.USER_ACCOUNTS,
        )
        assert result == "regular"

    def test_top_up_my_account_kept(self) -> None:
        """User has GoPay account → top-up is real transfer."""
        p = _parsed(intent="transfer", merchant="GoPay")
        result = reconcile_intent(
            p,
            full_text="top up gopay 200k from bca",
            asset_account_names=self.USER_ACCOUNTS,
        )
        assert result == "transfer"

    def test_top_up_foreign_service_downgraded(self) -> None:
        """User has NO Dana account → 'top up dana' is a withdrawal."""
        p = _parsed(intent="transfer", merchant="Dana")
        result = reconcile_intent(
            p,
            full_text="top up dana 100k",
            asset_account_names=self.USER_ACCOUNTS,
        )
        assert result == "regular"

    def test_substring_match_via_token(self) -> None:
        """Account 'BCA savings' matches user typing just 'bca'."""
        p = _parsed(intent="transfer", merchant="BCA")
        result = reconcile_intent(
            p,
            full_text="transfer 500k from cash wallet to bca",
            asset_account_names=self.USER_ACCOUNTS,
        )
        assert result == "transfer"

    def test_no_account_list_trusts_llm(self) -> None:
        """If we can't fetch accounts, fall back to trusting the LLM."""
        p = _parsed(intent="transfer", merchant="Joko")
        result = reconcile_intent(
            p,
            full_text="transfer 500k to joko",
            asset_account_names=None,
        )
        assert result == "transfer"

    def test_empty_account_list_treats_as_no_match(self) -> None:
        """Empty list (rather than None) means no accounts match → regular."""
        p = _parsed(intent="transfer", merchant="Anything")
        result = reconcile_intent(
            p,
            full_text="transfer 500k",
            asset_account_names=[],
        )
        assert result == "regular"

    def test_non_transfer_intent_unchanged(self) -> None:
        """Reconciliation only affects transfer claims; regular stays regular."""
        p = _parsed(intent="regular", merchant="Excelso")
        result = reconcile_intent(
            p,
            full_text="coffee 50k at excelso",
            asset_account_names=self.USER_ACCOUNTS,
        )
        assert result == "regular"

    def test_repayment_takes_precedence_over_transfer_validation(self) -> None:
        """If keywords say repayment, that wins over a transfer claim."""
        p = _parsed(intent="transfer", merchant="kredivo", description="bayar")
        result = reconcile_intent(
            p,
            full_text="bayar kredivo 1mil",
            asset_account_names=self.USER_ACCOUNTS,
        )
        assert result == "repayment"

    def test_short_account_name_skipped(self) -> None:
        """Account names ≤2 chars are too short to reliably match —
        random text containing those chars shouldn't fake a transfer."""
        p = _parsed(intent="transfer", merchant="random store")
        result = reconcile_intent(
            p,
            full_text="bought stuff at random store",
            asset_account_names=["X", "Yy"],  # all too short
        )
        assert result == "regular"
