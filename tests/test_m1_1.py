"""Test config + security modules.

Run: `uv run pytest -v`
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from firefly_agent.config import EnvSettings, TomlSettings, load_settings
from firefly_agent.security import OwnerGuard, redact, require_owner


# Find the example config.toml shipped with the repo. The runtime
# config.toml is gitignored (per-user) so we test the example file
# instead — which is the canonical "this is what users start from".
REPO_CONFIG_TOML = Path(__file__).parent.parent / "config.toml.example"


# ============================================================
# Env parsing
# ============================================================


def _valid_env() -> dict[str, str]:
    """Minimal valid env dict — used as base for parametrized tests."""
    return {
        "TELEGRAM_BOT_TOKEN": "1234567890:abcdefghijklmnopqrstuvwx",
        "TELEGRAM_OWNER_IDS": "123456789",
        "OPENROUTER_API_KEY": "sk-or-v1-abcdefghijklmnopqrstuvwxyz",
        "FIREFLY_URL": "https://firefly.example.com",
        "FIREFLY_PAT": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.abcde",
        "DEFAULT_ASSET_ACCOUNT_NAME": "BCA savings",
        "LIABILITY_ACCOUNT_NAMES": "Account Payable",
        "DEFAULT_CURRENCY": "IDR",
        "SECONDARY_CURRENCY": "USD",
        "PENDING_TTL_MINUTES": "30",
    }


class TestEnvSettings:
    def test_valid_env_parses(self) -> None:
        with patch.dict(os.environ, _valid_env(), clear=True):
            s = EnvSettings.from_env()
        assert s.telegram_owner_ids == [123456789]
        assert s.liability_account_names == ["Account Payable"]
        assert s.default_asset_account_name == "BCA savings"
        assert s.default_currency == "IDR"

    def test_multiple_owner_ids_csv(self) -> None:
        env = _valid_env() | {"TELEGRAM_OWNER_IDS": "123456789, 999, 42"}
        with patch.dict(os.environ, env, clear=True):
            s = EnvSettings.from_env()
        assert s.telegram_owner_ids == [123456789, 999, 42]

    def test_empty_liability_list_is_valid(self) -> None:
        env = _valid_env() | {"LIABILITY_ACCOUNT_NAMES": ""}
        with patch.dict(os.environ, env, clear=True):
            s = EnvSettings.from_env()
        assert s.liability_account_names == []

    def test_multiple_liability_names_csv(self) -> None:
        env = _valid_env() | {"LIABILITY_ACCOUNT_NAMES": "Account Payable, BNPL Card"}
        with patch.dict(os.environ, env, clear=True):
            s = EnvSettings.from_env()
        assert s.liability_account_names == ["Account Payable", "BNPL Card"]

    def test_missing_required_exits(self) -> None:
        env = _valid_env()
        del env["FIREFLY_PAT"]
        with patch.dict(os.environ, env, clear=True), pytest.raises(SystemExit):
            EnvSettings.from_env()

    def test_invalid_currency_code_rejected(self) -> None:
        env = _valid_env() | {"DEFAULT_CURRENCY": "idr"}  # lowercase, bad
        with patch.dict(os.environ, env, clear=True), pytest.raises(SystemExit):
            EnvSettings.from_env()

    def test_invalid_url_rejected(self) -> None:
        env = _valid_env() | {"FIREFLY_URL": "not-a-url"}
        with patch.dict(os.environ, env, clear=True), pytest.raises(SystemExit):
            EnvSettings.from_env()


# ============================================================
# TOML loading
# ============================================================


class TestTomlSettings:
    def test_repo_config_toml_is_valid(self) -> None:
        """Ensures the shipped config.toml.example parses cleanly.

        This is what every new user copies to start. If the example
        breaks, every install breaks.
        """
        assert REPO_CONFIG_TOML.is_file(), (
            f"config.toml.example not found at {REPO_CONFIG_TOML}"
        )
        s = TomlSettings.from_toml(REPO_CONFIG_TOML)
        # The example has 12 categories (we removed one personal one);
        # this is just a "non-empty + parses" smoke test.
        assert len(s.allowed_categories) > 0
        assert "Food & Beverages" in s.allowed_categories
        assert "coffee" in s.tag_groups["food"]
        # Example uses 3-letter ISO codes; don't pin specific currencies
        assert len(s.currencies.primary) == 3
        assert len(s.currencies.secondary) == 3
        assert s.currencies.primary in s.currencies.allowed
        assert len(s.llm.models) >= 1

    def test_missing_toml_exits(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            TomlSettings.from_toml(tmp_path / "does-not-exist.toml")


# ============================================================
# Combined settings + validators
# ============================================================


class TestSettings:
    def test_load_settings_end_to_end(self) -> None:
        env = _valid_env() | {"CONFIG_TOML_PATH": str(REPO_CONFIG_TOML)}
        with patch.dict(os.environ, env, clear=True):
            s = load_settings()
        assert s.is_valid_category("Food & Beverages")
        assert not s.is_valid_category("Made Up Category")
        assert s.is_valid_tag("coffee")
        assert s.is_valid_tag("trip:bali-2026")
        assert not s.is_valid_tag("trip:")  # empty suffix rejected
        assert not s.is_valid_tag("nonsense-tag")
        assert s.is_valid_currency(s.toml.currencies.primary)
        assert not s.is_valid_currency("XYZ")


# ============================================================
# Secret redaction
# ============================================================


class TestRedact:
    def test_redacts_obvious_keys(self) -> None:
        data = {
            "username": "alice",
            "password": "hunter2",
            "api_key": "secret",
            "firefly_pat": "ey...",
            "nested": {"bearer_token": "ey..."},
        }
        result = redact(data)
        assert result["username"] == "alice"
        assert result["password"] == "[REDACTED]"
        assert result["api_key"] == "[REDACTED]"
        assert result["firefly_pat"] == "[REDACTED]"
        assert result["nested"]["bearer_token"] == "[REDACTED]"

    def test_handles_non_dict_gracefully(self) -> None:
        assert redact("plain string") == "plain string"
        assert redact(42) == 42
        assert redact([1, 2, 3]) == [1, 2, 3]
        assert redact(None) is None

    def test_recursion_guard(self) -> None:
        """Pathological self-referencing dicts shouldn't crash."""
        # Build a nested dict deeper than the guard
        obj: dict[str, object] = {"level": 0}
        cur = obj
        for i in range(20):
            cur["next"] = {"level": i + 1}
            cur = cur["next"]  # type: ignore[assignment]
        # Should not raise; deep branches marked
        redact(obj)


# ============================================================
# OwnerGuard + require_owner decorator
# ============================================================


class TestOwnerGuard:
    def test_is_owner_checks_membership(self) -> None:
        g = OwnerGuard([1, 2, 3])
        assert g.is_owner(1)
        assert g.is_owner(2)
        assert not g.is_owner(99)
        assert not g.is_owner(None)

    def test_empty_whitelist_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            OwnerGuard([])


class TestRequireOwner:
    @pytest.mark.asyncio
    async def test_blocks_unauthorized(self) -> None:
        guard = OwnerGuard([1])

        class FakeUser:
            id = 99

        class FakeUpdate:
            effective_user = FakeUser()

        calls: list[str] = []

        @require_owner(guard)
        async def handler(update: object, context: object) -> None:
            calls.append("called")

        await handler(FakeUpdate(), None)
        assert calls == []  # handler never ran

    @pytest.mark.asyncio
    async def test_allows_authorized(self) -> None:
        guard = OwnerGuard([1])

        class FakeUser:
            id = 1

        class FakeUpdate:
            effective_user = FakeUser()

        calls: list[str] = []

        @require_owner(guard)
        async def handler(update: object, context: object) -> None:
            calls.append("called")

        await handler(FakeUpdate(), None)
        assert calls == ["called"]

    @pytest.mark.asyncio
    async def test_no_effective_user_is_dropped(self) -> None:
        guard = OwnerGuard([1])

        class FakeUpdate:
            effective_user = None

        calls: list[str] = []

        @require_owner(guard)
        async def handler(update: object, context: object) -> None:
            calls.append("called")

        await handler(FakeUpdate(), None)
        assert calls == []
