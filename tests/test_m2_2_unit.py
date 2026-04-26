"""Unit tests for M2.2 — /balance command formatting + grouping."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from firefly_agent.handlers.commands import _account_emoji, cmd_balance
from firefly_agent.models import Account


def _account(**overrides) -> Account:
    base = {"id": 1, "name": "X", "type": "asset", "currency_code": "IDR",
            "current_balance": Decimal("0")}
    base.update(overrides)
    return Account(**base)


# ============================================================
# Emoji picker
# ============================================================


class TestAccountEmoji:
    def test_credit_card(self) -> None:
        assert _account_emoji("Credit Card BCA") == "💳"

    def test_savings(self) -> None:
        assert _account_emoji("BCA savings") == "🏦"

    def test_cash(self) -> None:
        assert _account_emoji("Cash wallet") == "💵"

    def test_usd(self) -> None:
        assert _account_emoji("USD account") == "💵"

    def test_default(self) -> None:
        assert _account_emoji("Random") == "🏦"


# ============================================================
# /balance flow
# ============================================================


@pytest.mark.asyncio
async def test_balance_renders_assets_and_liabilities() -> None:
    """Smoke test: command produces a reply with all key sections."""
    services = MagicMock()
    services.firefly = MagicMock()
    services.firefly.list_asset_accounts = AsyncMock(return_value=[
        _account(id=6, name="BCA savings", currency_code="IDR",
                 current_balance=Decimal("4500000")),
        _account(id=10, name="USD account", currency_code="USD",
                 current_balance=Decimal("145.30")),
    ])
    services.firefly.list_liability_accounts = AsyncMock(return_value=[
        _account(id=12, name="Account Payable", type="liabilities",
                 currency_code="IDR", current_balance=Decimal("-850000")),
    ])

    services.settings = MagicMock()
    services.settings.toml.currencies.primary = "IDR"

    update = MagicMock()
    update.message.reply_text = AsyncMock()
    update.effective_user = MagicMock(id=1)

    context = MagicMock()
    context.bot_data = {"services": services}

    # Patch Services.get to return our mock
    from firefly_agent.handlers import commands
    real_get = commands.Services.get
    commands.Services.get = staticmethod(lambda c: services)
    try:
        await cmd_balance(update, context)
    finally:
        commands.Services.get = real_get

    assert update.message.reply_text.called
    msg = update.message.reply_text.call_args[0][0]
    # Check grouping
    assert "Assets" in msg
    assert "Liabilities" in msg
    # Check accounts appear
    assert "BCA savings" in msg
    assert "USD account" in msg
    assert "Account Payable" in msg
    # Liability should be shown POSITIVE (owed) not negative
    assert "-850" not in msg.replace("&minus;850", "")  # not literal minus
    # Currency totals
    assert "Total IDR" in msg
    assert "Total USD" in msg
    assert "Total owed IDR" in msg
    # Net section
    assert "Net (per currency)" in msg


@pytest.mark.asyncio
async def test_balance_handles_no_accounts() -> None:
    services = MagicMock()
    services.firefly.list_asset_accounts = AsyncMock(return_value=[])
    services.firefly.list_liability_accounts = AsyncMock(return_value=[])
    services.settings.toml.currencies.primary = "IDR"

    update = MagicMock()
    update.message.reply_text = AsyncMock()
    update.effective_user = MagicMock(id=1)

    context = MagicMock()

    from firefly_agent.handlers import commands
    real_get = commands.Services.get
    commands.Services.get = staticmethod(lambda c: services)
    try:
        await cmd_balance(update, context)
    finally:
        commands.Services.get = real_get

    msg = update.message.reply_text.call_args[0][0]
    assert "No accounts found" in msg
