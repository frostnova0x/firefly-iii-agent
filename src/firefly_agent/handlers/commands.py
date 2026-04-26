"""Slash-command handlers.

/start, /accounts, /cancel, /undo, /yes (the /undo confirm step).
"""

from __future__ import annotations

import html
import logging

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from firefly_agent.firefly import FireflyError
from firefly_agent.formatting import format_error_message
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
        "👋 <b>Firefly-iii Logger bot</b>\n\n"
        "Send me a transaction in plain text in English or Bahasa Indonesia "
        "and I'll categorize it and log it to Firefly III, by : frostnova0x.\n\n"
        "<b>Examples:</b>\n"
        "• <code>coffee 45k at starbucks</code>\n"
        "• <code>gojek ke kantor 25rb</code>\n"
        "• <code>$20 claude subscription</code>\n"
        "• <code>alfamart 50k</code>\n\n"
        "<b>Commands:</b>\n"
        "/accounts — show asset accounts\n"
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
