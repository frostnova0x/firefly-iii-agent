"""Slash-command handlers.

/start, /accounts, /balance, /cancel, /undo, /yes.
"""

from __future__ import annotations

import html
import logging
from decimal import Decimal

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from firefly_agent.firefly import FireflyError
from firefly_agent.formatting import format_currency, format_error_message
from firefly_agent.services import Services

log = logging.getLogger(__name__)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Greeting + authorization confirmation."""
    if update.message is None or update.effective_user is None:
        return
    services = Services.get(context)

    uid = update.effective_user.id
    if not services.guard.is_owner(uid):
        # Whitelist handled at dispatch level; reaching here means it was
        # dropped silently. This branch is a safety net.
        return

    await update.message.reply_text(
        "👋 <b>Firefly III Agent</b>\n\n"
        "Send me a transaction in plain text — in English or Bahasa Indonesia — "
        "and I'll categorize it and log it to Firefly III.\n\n"
        "<b>Examples:</b>\n"
        "• <code>coffee 45k at starbucks</code>\n"
        "• <code>gojek ke kantor 25rb</code>\n"
        "• <code>$20 claude subscription</code>\n"
        "• <code>alfamart 50k</code>\n\n"
        "<b>Commands:</b>\n"
        "/accounts — show asset accounts\n"
        "/balance — show all balances and net worth\n"
        "/cancel — cancel pending transaction\n"
        "/undo — undo the last logged transaction (asks for /yes)\n",
        parse_mode=ParseMode.HTML,
    )


async def cmd_accounts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List asset accounts (debugging aid)."""
    if update.message is None:
        return
    services = Services.get(context)

    try:
        accounts = await services.firefly.list_asset_accounts()
    except FireflyError as e:
        await update.message.reply_text(
            format_error_message(f"Couldn't list accounts: {e}"),
            parse_mode=ParseMode.HTML,
        )
        return

    if not accounts:
        await update.message.reply_text("No asset accounts found.")
        return

    lines = ["<b>Asset accounts</b>"]
    for a in accounts:
        cur = a.currency_code or "?"
        lines.append(f"• <b>{a.name}</b> ({cur}) — id <code>{a.id}</code>")

    # Also show top-N from our usage counter
    top = await services.store.get_top_accounts(limit=5)
    if top:
        lines.append("")
        lines.append("<b>Most used:</b>")
        for u in top:
            lines.append(f"• {u.account_name} — {u.use_count} uses")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Cancel the user's most recent pending transaction.

    We don't have a direct 'most recent by user' query, but the pending
    rows are short-lived and few. Scan and delete.
    """
    if update.message is None or update.effective_user is None:
        return
    services = Services.get(context)

    user_id = update.effective_user.id

    conn = services.store._require_conn()  # noqa: SLF001 — intentional, ad-hoc query
    cur = await conn.execute(
        """
        SELECT callback_id FROM pending_transactions
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (user_id,),
    )
    row = await cur.fetchone()

    if row is None:
        await update.message.reply_text("Nothing to cancel.")
        return

    await services.store.delete_pending_transaction(row[0])
    await update.message.reply_text("❌ Pending transaction cancelled.")


# /undo confirmation lives in PTB's per-user dict. Small enough that
# losing it on bot restart is fine (user just types /undo again).
_UNDO_PENDING_KEY = "undo_pending_group_id"


async def cmd_undo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the most recent transaction and ask for /yes confirmation.

    Doesn't actually delete anything — that happens on /yes.
    """
    if update.message is None or update.effective_user is None:
        return
    services = Services.get(context)
    user_id = update.effective_user.id

    last = await services.store.get_last_transaction(user_id)
    if last is None:
        await update.message.reply_text(
            "Nothing to undo. /undo only works on the most recent transaction "
            "you logged via this bot."
        )
        return

    # Stash the candidate group_id in user_data; /yes reads it.
    if context.user_data is not None:
        context.user_data[_UNDO_PENDING_KEY] = last.firefly_transaction_group_id

    desc = html.escape(last.description)
    await update.message.reply_text(
        "<b>About to undo:</b>\n"
        f"{desc}\n"
        f"<code>Firefly ID: {last.firefly_transaction_group_id}</code>\n"
        f"<code>Logged at: {html.escape(last.logged_at)}</code>\n\n"
        "Reply <b>/yes</b> to confirm, or anything else to abort.",
        parse_mode=ParseMode.HTML,
    )


async def cmd_yes(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Confirm /undo. Deletes the Firefly transaction stashed by cmd_undo."""
    if update.message is None or update.effective_user is None:
        return
    services = Services.get(context)
    user_id = update.effective_user.id

    pending_id = (
        context.user_data.get(_UNDO_PENDING_KEY)
        if context.user_data is not None
        else None
    )
    if pending_id is None:
        await update.message.reply_text(
            "Nothing pending. /yes only confirms an immediately-prior /undo."
        )
        return

    # Single-shot: clear the pending state regardless of what happens next
    if context.user_data is not None:
        context.user_data.pop(_UNDO_PENDING_KEY, None)

    try:
        await services.firefly.delete_transaction(int(pending_id))
    except FireflyError as e:
        log.warning("Firefly delete failed for group %s: %s", pending_id, e)
        await update.message.reply_text(
            format_error_message(f"Couldn't delete from Firefly: {e}"),
            parse_mode=ParseMode.HTML,
        )
        return

    await services.store.clear_last_transaction(user_id)
    await update.message.reply_text(
        f"✅ Undone. <code>Firefly ID: {pending_id}</code> deleted.",
        parse_mode=ParseMode.HTML,
    )
    log.info("User %d undone Firefly transaction group %s", user_id, pending_id)


async def cmd_balance(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show all asset + liability account balances grouped, with totals.

    Asset balances are shown positive. Liability balances are shown as
    "owed" — Firefly stores them as negative numbers; we flip the sign
    for display since "owed: Rp 500k" reads better than "balance: -Rp 500k".

    Cross-currency totals: each currency is summed independently. We don't
    convert here (FX module lands in M2.4); each currency line stands alone.
    """
    if update.message is None or update.effective_user is None:
        return
    services = Services.get(context)

    try:
        asset_accounts = await services.firefly.list_asset_accounts(use_cache=False)
        liability_accounts = await services.firefly.list_liability_accounts(use_cache=False)
    except FireflyError as e:
        log.warning("Balance fetch failed: %s", e)
        await update.message.reply_text(
            format_error_message(f"Couldn't fetch balances: {e}"),
            parse_mode=ParseMode.HTML,
        )
        return

    # Group totals per currency, separately for asset and liability.
    asset_totals: dict[str, Decimal] = {}
    liability_totals: dict[str, Decimal] = {}

    lines: list[str] = ["💰 <b>Balances</b>", ""]

    if asset_accounts:
        lines.append("<b>Assets</b>")
        for acc in sorted(asset_accounts, key=lambda a: a.name.lower()):
            bal = acc.current_balance or Decimal("0")
            cc = (acc.currency_code or services.settings.toml.currencies.primary).upper()
            asset_totals[cc] = asset_totals.get(cc, Decimal("0")) + bal
            emoji = _account_emoji(acc.name)
            lines.append(
                f"  {emoji} {html.escape(acc.name)} — "
                f"<code>{format_currency(bal, cc)}</code>"
            )
        lines.append("")
        for cc, total in asset_totals.items():
            lines.append(f"  <i>Total {cc}: {format_currency(total, cc)}</i>")
        lines.append("")

    if liability_accounts:
        lines.append("<b>Liabilities (owed)</b>")
        for acc in sorted(liability_accounts, key=lambda a: a.name.lower()):
            # Firefly returns liability balances as negative numbers
            # (you "owe" money). Flip sign for friendlier display.
            owed = abs(acc.current_balance or Decimal("0"))
            cc = (acc.currency_code or services.settings.toml.currencies.primary).upper()
            liability_totals[cc] = liability_totals.get(cc, Decimal("0")) + owed
            lines.append(
                f"  💳 {html.escape(acc.name)} — "
                f"<code>{format_currency(owed, cc)}</code>"
            )
        lines.append("")
        for cc, total in liability_totals.items():
            lines.append(f"  <i>Total owed {cc}: {format_currency(total, cc)}</i>")
        lines.append("")

    # Net worth per currency (assets minus liabilities, in matching currency)
    if asset_totals or liability_totals:
        lines.append("<b>Net (per currency)</b>")
        all_currencies = set(asset_totals.keys()) | set(liability_totals.keys())
        for cc in sorted(all_currencies):
            net = asset_totals.get(cc, Decimal("0")) - liability_totals.get(cc, Decimal("0"))
            lines.append(f"  {cc}: <code>{format_currency(net, cc)}</code>")

    if not asset_accounts and not liability_accounts:
        lines = ["No accounts found in Firefly III. Create some first."]

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.HTML,
    )


def _account_emoji(name: str) -> str:
    """Best-guess emoji from account name. Cosmetic only."""
    n = name.lower()
    if "usd" in n or "dollar" in n:
        return "💵"
    if "credit" in n or " cc " in f" {n} ":
        return "💳"
    if "cash" in n or "wallet" in n:
        return "💵"
    if "saving" in n:
        return "🏦"
    return "🏦"
