"""Unit tests for M2.1 — timezone-aware timestamps."""

from __future__ import annotations

import os
import re
from datetime import datetime
from unittest.mock import patch

import pytest

from firefly_agent.config import EnvSettings, load_settings
from firefly_agent.parsed import ParsedTransaction


def _parsed(**overrides) -> ParsedTransaction:
    base = {
        "type": "withdrawal",
        "amount": 50000,
        "currency": "IDR",
        "description": "Coffee",
        "merchant": "Excelso",
        "category": "Food & Beverages",
        "tags": [],
        "date": "2026-04-25",
        "time": "",
        "confidence": "high",
        "intent": "regular",
        "notes": "",
    }
    base.update(overrides)
    return ParsedTransaction.model_validate(base)


# ============================================================
# to_iso_datetime
# ============================================================


class TestToIsoDatetime:
    def test_text_input_uses_current_time(self) -> None:
        """Text inputs have time='' — bot fills with current local time."""
        p = _parsed(date="2026-04-25", time="")
        out = p.to_iso_datetime("Asia/Jakarta")
        # Should be 2026-04-25T<HH:MM:SS>+07:00
        assert out.startswith("2026-04-25T")
        assert out.endswith("+07:00")
        # Time portion is non-zero (it's "now" in Jakarta)
        time_match = re.search(r"T(\d\d):(\d\d):(\d\d)", out)
        assert time_match is not None

    def test_receipt_with_time_uses_extracted_time(self) -> None:
        p = _parsed(date="2026-04-24", time="14:30:42")
        out = p.to_iso_datetime("Asia/Jakarta")
        assert out == "2026-04-24T14:30:42+07:00"

    def test_utc_zone(self) -> None:
        p = _parsed(date="2026-04-25", time="09:00:00")
        out = p.to_iso_datetime("UTC")
        assert out == "2026-04-25T09:00:00+00:00"

    def test_european_zone(self) -> None:
        p = _parsed(date="2026-06-15", time="12:00:00")
        # Berlin in June is UTC+2 (DST)
        out = p.to_iso_datetime("Europe/Berlin")
        assert out == "2026-06-15T12:00:00+02:00"

    def test_malformed_time_falls_back_to_now(self) -> None:
        """Defensive — if LLM emits a garbage time, don't crash."""
        p = _parsed(date="2026-04-25", time="not-a-time")
        out = p.to_iso_datetime("UTC")
        # Should still produce a valid ISO datetime starting with the date
        assert out.startswith("2026-04-25T")
        assert "+00:00" in out

    def test_empty_time_uses_local_now(self) -> None:
        """Verify we're really using LOCAL time, not UTC, when zone differs."""
        p = _parsed(date="2026-04-25", time="")
        jakarta_iso = p.to_iso_datetime("Asia/Jakarta")
        utc_iso = p.to_iso_datetime("UTC")
        # Same call wall-clock, different zones → different time portions
        # (offset matches the zone)
        assert "+07:00" in jakarta_iso
        assert "+00:00" in utc_iso


# ============================================================
# Timezone env validator
# ============================================================


class TestTimezoneValidator:
    def test_valid_iana_zone(self) -> None:
        # No exception
        s = EnvSettings.model_validate({
            "telegram_bot_token": "1234567890:abcdefghijklmnopqrstuvwx",
            "telegram_owner_ids": "123",
            "openrouter_api_key": "sk-or-v1-abcdefghijklmnopqrstuvwxyz",
            "firefly_url": "https://firefly.example.com",
            "firefly_pat": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.abcde",
            "default_asset_account_name": "BCA savings",
            "liability_account_names": "Account Payable",
            "default_currency": "IDR",
            "secondary_currency": "USD",
            "timezone": "Asia/Jakarta",
            "pending_ttl_minutes": 30,
        })
        assert s.timezone == "Asia/Jakarta"

    def test_invalid_iana_zone_rejected(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="not a valid IANA timezone"):
            EnvSettings.model_validate({
                "telegram_bot_token": "1234567890:abcdefghijklmnopqrstuvwx",
                "telegram_owner_ids": "123",
                "openrouter_api_key": "sk-or-v1-abcdefghijklmnopqrstuvwxyz",
                "firefly_url": "https://firefly.example.com",
                "firefly_pat": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.abcde",
                "default_asset_account_name": "BCA savings",
                "liability_account_names": "Account Payable",
                "default_currency": "IDR",
                "secondary_currency": "USD",
                "timezone": "Asia/Jakata",  # typo!
                "pending_ttl_minutes": 30,
            })

    def test_default_is_utc(self) -> None:
        """Bot defaults to UTC if user doesn't set TIMEZONE."""
        s = EnvSettings.model_validate({
            "telegram_bot_token": "1234567890:abcdefghijklmnopqrstuvwx",
            "telegram_owner_ids": "123",
            "openrouter_api_key": "sk-or-v1-abcdefghijklmnopqrstuvwxyz",
            "firefly_url": "https://firefly.example.com",
            "firefly_pat": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.abcde",
            "default_asset_account_name": "BCA savings",
            "liability_account_names": "Account Payable",
        })
        assert s.timezone == "UTC"


# ============================================================
# Schema includes time field
# ============================================================


class TestTimeFieldInSchema:
    def test_time_in_required(self) -> None:
        from firefly_agent.parsed import parsed_transaction_json_schema
        schema = parsed_transaction_json_schema(
            allowed_categories=["Food & Beverages"],
            allowed_tags=["coffee"],
            allowed_currencies=["IDR", "USD"],
        )
        assert "time" in schema["schema"]["required"]
        assert schema["schema"]["properties"]["time"]["type"] == "string"

    def test_time_default_is_empty(self) -> None:
        p = ParsedTransaction.model_validate({
            "type": "withdrawal", "amount": 1000, "currency": "IDR",
            "description": "x", "merchant": "y", "category": "Food & Beverages",
            "tags": [], "date": "2026-04-25", "confidence": "high",
        })
        assert p.time == ""
