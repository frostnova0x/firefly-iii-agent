"""Inline-keyboard callback handler — all button actions route here.

Callback_data format: "action:callback_id[:arg]"

Actions:
    cur  — user tapped a currency toggle (edit message, swap highlight)
    acc  — user picked a source account (advance to confirm screen)
    conf — user confirmed (post to Firefly, finalize message)
    back — user went back from confirm to preview
    canc — user cancelled

All actions begin with validation:
    - Verify the pending transaction exists and isn't expired
    - Verify the tapping user matches the row's user_id
    - Answer the callback query immediately (Telegram requires it)
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from decimal import Decimal

from telegram import Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from firefly_agent.errors import (
    FireflyAuthError,
    FireflyError,
    FireflyValidationError,
)
from firefly_agent.formatting import (
    CB_ACCOUNT,
    CB_BACK,
    CB_CANCEL,
    CB_CONFIRM,
    CB_CURRENCY,
    CB_EDIT,
    CB_EDIT_FIELD,
    CB_NOOP,
    EDIT_PROMPTS,
    CurrencyButton,
    build_awaiting_account_keyboard,
    build_awaiting_confirm_keyboard,
    build_edit_submenu_keyboard,
    format_cancelled_message,
    format_confirm_message,
    format_error_message,
    format_expired_message,
    format_logged_message,
    format_preview_message,
)
from firefly_agent.models import NewTransaction, TransactionSplit
from firefly_agent.parsed import ParsedTransaction
from firefly_agent.services import Services
from firefly_agent.state import PendingTransaction

log = logging.getLogger(__name__)


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Dispatch on callback_data prefix."""
    query = update.callback_query
    if query is None or query.data is None or update.effective_user is None:
        return

    # ALWAYS answer immediately. Telegram spins a loading indicator on the
    # client until we do; not answering makes the button feel stuck.
    await query.answer()

    parts = query.data.split(":")
    if len(parts) < 2:
        log.warning("Malformed callback_data: %r", query.data)
        return

    action = parts[0]
    cid = parts[1]
    arg = parts[2] if len(parts) >= 3 else None

    services = Services.get(context)
    user_id = update.effective_user.id

    # Load pending. Expired rows return None.
    pending = await services.store.get_pending_transaction(cid)
    if pending is None:
        await _edit(query, format_expired_message(), reply_markup=None)
        return

    # Cross-user check: the user pressing buttons must be the row's owner.
    if pending.user_id != user_id:
        log.warning(
            "User %d tried to act on pending %s owned by %d",
            user_id, cid, pending.user_id,
        )
        await query.answer(text="Not your transaction.", show_alert=True)
        return

    # Dispatch
    if action == CB_CURRENCY:
        await _on_currency(query, services, pending, new_currency=arg or "")
    elif action == CB_ACCOUNT:
        await _on_account(query, services, pending, account_id_str=arg or "")
    elif action == CB_CONFIRM:
        await _on_confirm(query, services, pending)
    elif action == CB_BACK:
        await _on_back(query, services, pending)
    elif action == CB_CANCEL:
        await _on_cancel(query, services, pending)
    elif action == CB_EDIT:
        await _on_edit(query, services, pending)
    elif action == CB_EDIT_FIELD:
        await _on_edit_field(query, services, pending, field=arg or "full")
    elif action == CB_NOOP:
        # The pre-selected liability button on BNPL — already answered above.
        # No state change.
        pass
    else:
        log.warning("Unknown callback action: %r", action)


# ============================================================
# Per-action handlers
# ============================================================


async def _on_currency(
    query,  # type: ignore[no-untyped-def]
    services: Services,
    pending: PendingTransaction,
    *,
    new_currency: str,
) -> None:
    """User tapped a currency button on the preview."""
    if pending.state != "awaiting_account":
        # User tapped the currency after we already advanced — ignore.
        return
    if not new_currency:
        return

    # Only accept known codes to defend against callback_data tampering
    if not services.settings.is_valid_currency(new_currency):
        await query.answer(text="Unsupported currency.", show_alert=True)
        return

    ok = await services.store.update_pending_currency(pending.callback_id, new_currency)
    if not ok:
        await _edit(query, format_expired_message(), reply_markup=None)
        return

    # Re-render the preview with new currency selected.
    parsed = _parsed_from_pending(pending)
    preview = format_preview_message(parsed, selected_currency=new_currency)

    # Rebuild the keyboard to flip the currency checkmark.
    from firefly_agent.handlers.text import _build_initial_keyboard  # local import to avoid cycle
    kb = await _build_initial_keyboard(services, pending.callback_id, parsed)
    # Adjust the selected flag in the keyboard currencies to reflect new_currency
    kb = _rebuild_keyboard_with_selected_currency(kb, new_currency, pending.callback_id)

    await _edit(query, preview, reply_markup=kb)


def _rebuild_keyboard_with_selected_currency(kb, new_currency: str, cid: str):  # type: ignore[no-untyped-def]
    """Swap the ✓ prefix in the currency row of an existing keyboard.

    Markup is immutable, so we reconstruct it. We assume the first row
    contains the currency buttons (or there's no currency row at all).
    """
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    new_rows: list[list[InlineKeyboardButton]] = []
    for row in kb.inline_keyboard:
        if row and row[0].callback_data.startswith(f"{CB_CURRENCY}:"):
            updated_row: list[InlineKeyboardButton] = []
            for btn in row:
                parts = (btn.callback_data or "").split(":")
                code = parts[-1] if len(parts) == 3 else ""
                label = code
                prefix = "✓ " if code == new_currency else ""
                updated_row.append(
                    InlineKeyboardButton(
                        text=f"{prefix}{label}",
                        callback_data=f"{CB_CURRENCY}:{cid}:{code}",
                    )
                )
            new_rows.append(updated_row)
        else:
            new_rows.append(list(row))
    return InlineKeyboardMarkup(new_rows)


async def _on_account(
    query,  # type: ignore[no-untyped-def]
    services: Services,
    pending: PendingTransaction,
    *,
    account_id_str: str,
) -> None:
    """User picked the source account → advance to confirm screen."""
    if pending.state != "awaiting_account":
        return

    try:
        account_id = int(account_id_str)
    except ValueError:
        log.warning("Non-int account_id in callback: %r", account_id_str)
        return

    # Fetch accounts to validate + display name
    accounts = await services.firefly.list_asset_accounts()
    source = next((a for a in accounts if a.id == account_id), None)
    if source is None:
        await query.answer(text="Unknown account.", show_alert=True)
        return

    ok = await services.store.update_pending_to_awaiting_confirm(
        pending.callback_id, source_account_id=account_id
    )
    if not ok:
        await _edit(query, format_expired_message(), reply_markup=None)
        return

    # Re-read the row to get the (possibly updated) currency.
    updated = await services.store.get_pending_transaction(pending.callback_id)
    if updated is None:
        await _edit(query, format_expired_message(), reply_markup=None)
        return

    parsed = _parsed_from_pending(updated)

    # If transaction currency differs from account currency, we'll book
    # as foreign_amount. The user doesn't enter the native amount — we
    # show a best-effort placeholder so they know it'll be converted by
    # Firefly. Actual conversion rate is Firefly's concern.
    foreign_amount = None
    foreign_currency = None
    if updated.currency != (source.currency_code or updated.currency):
        # We don't have a live FX rate; show the foreign line without an
        # amount so the user knows conversion will happen.
        foreign_currency = source.currency_code
        # Heuristic placeholder only for UX — Firefly does the actual math
        foreign_amount = None

    confirm_text = format_confirm_message(
        parsed,
        selected_currency=updated.currency,
        source_account=source,
        foreign_amount=foreign_amount,
        foreign_currency=foreign_currency,
    )
    kb = build_awaiting_confirm_keyboard(pending.callback_id)

    await _edit(query, confirm_text, reply_markup=kb)


async def _on_confirm(
    query,  # type: ignore[no-untyped-def]
    services: Services,
    pending: PendingTransaction,
) -> None:
    """User pressed ✅ Confirm → post to Firefly → finalize.

    Two paths:
      - Normal: state must be 'awaiting_confirm' with source_account_id set.
      - BNPL:   state may still be 'awaiting_account' (no asset picker).
                We detect via 'bnpl_liability_id' in the payload.
    """
    parsed = _parsed_from_pending(pending)
    bnpl_id = _bnpl_id_from_pending(pending)
    is_repayment = parsed.intent == "repayment"

    if is_repayment:
        # BNPL repayment in Firefly's data model:
        # Per Firefly III docs and maintainer guidance, asset → liability
        # movements are MODELED AS WITHDRAWALS, not transfers. Firefly
        # automatically reduces the liability's "amount due" when you
        # withdraw money from an asset account TO a liability account.
        # This is the only API shape Firefly accepts for paying down debt.
        #
        # Reference: https://docs.firefly-iii.org/explanation/financial-concepts/liabilities/
        # > "Sending money to the liability is a 'withdrawal', because it
        # >  withdraws money from your accounts."
        if pending.state != "awaiting_confirm" or pending.source_account_id is None:
            return
        asset_accounts = await services.firefly.list_asset_accounts()
        liability_accounts = await services.firefly.list_liability_accounts()
        source = next((a for a in asset_accounts if a.id == pending.source_account_id), None)
        if source is None:
            await _edit(
                query,
                format_error_message("Source account no longer exists."),
                reply_markup=None,
            )
            await services.store.delete_pending_transaction(pending.callback_id)
            return
        # Find the destination liability — prefer the bnpl_id we stashed,
        # otherwise look it up by merchant via the keyword map again.
        destination = None
        if bnpl_id is not None:
            destination = next((a for a in liability_accounts if a.id == bnpl_id), None)
        if destination is None:
            from firefly_agent.bnpl import detect_bnpl_account_id as _redetect
            redetect_id = _redetect(
                parsed,
                keyword_map=services.liability_keyword_id_map,
            )
            if redetect_id is not None:
                destination = next(
                    (a for a in liability_accounts if a.id == redetect_id), None
                )
        if destination is None:
            await _edit(
                query,
                format_error_message(
                    "Couldn't identify which liability account to repay. "
                    "Edit the merchant to a recognized BNPL provider name."
                ),
                reply_markup=None,
            )
            await services.store.delete_pending_transaction(pending.callback_id)
            return

        split = TransactionSplit(
            type="withdrawal",
            date=parsed.to_iso_datetime(services.settings.env.timezone),
            amount=parsed.amount,
            description=parsed.description or f"{destination.name} repayment",
            currency_code=pending.currency,
            source_id=source.id,
            destination_id=destination.id,
            tags=list(parsed.tags) + ["bnpl-repayment"],
            notes=parsed.notes or None,
            # No category — repayments aren't a "spending category." Firefly
            # tracks them via the liability balance change, not as expense.
        )
        new_tx = NewTransaction(transactions=[split])

    elif bnpl_id is not None:
        # BNPL purchase: source = liability account, destination = merchant.
        asset_accounts = await services.firefly.list_asset_accounts()
        liability_accounts = await services.firefly.list_liability_accounts()
        source = next(
            (a for a in liability_accounts if a.id == bnpl_id), None
        )
        if source is None:
            await _edit(
                query,
                format_error_message("Liability account no longer exists in Firefly."),
                reply_markup=None,
            )
            await services.store.delete_pending_transaction(pending.callback_id)
            return

        split_kwargs = {
            "type": "withdrawal",
            "date": parsed.to_iso_datetime(services.settings.env.timezone),
            "amount": parsed.amount,
            "description": parsed.description,
            "currency_code": pending.currency,
            "source_id": source.id,
            "category_name": parsed.category,
            "tags": list(parsed.tags),
            "notes": parsed.notes or None,
        }
        if parsed.merchant.lower() in {
            "spaylater", "kredivo", "akulaku", "shopee paylater",
        }:
            split_kwargs["destination_name"] = "Online Purchase"
        else:
            split_kwargs["destination_name"] = parsed.merchant
        split = TransactionSplit(**split_kwargs)
        new_tx = NewTransaction(transactions=[split])

    else:
        # Normal path: withdrawal or deposit, asset account source
        if pending.state != "awaiting_confirm" or pending.source_account_id is None:
            return
        accounts = await services.firefly.list_asset_accounts()
        source = next((a for a in accounts if a.id == pending.source_account_id), None)
        if source is None:
            await _edit(
                query,
                format_error_message("Source account no longer exists."),
                reply_markup=None,
            )
            await services.store.delete_pending_transaction(pending.callback_id)
            return

        split_kwargs = {
            "type": parsed.type,
            "date": parsed.to_iso_datetime(services.settings.env.timezone),
            "amount": parsed.amount,
            "description": parsed.description,
            "currency_code": pending.currency,
            "source_id": source.id,
            "category_name": parsed.category,
            "tags": list(parsed.tags),
            "notes": parsed.notes or None,
        }
        if parsed.type == "withdrawal":
            split_kwargs["destination_name"] = parsed.merchant
        else:  # deposit
            split_kwargs["source_id"] = None
            split_kwargs["source_name"] = parsed.merchant
            split_kwargs["destination_id"] = source.id
        split = TransactionSplit(**split_kwargs)
        new_tx = NewTransaction(transactions=[split])

    try:
        created = await services.firefly.create_transaction(new_tx)
    except FireflyAuthError:
        log.error("Firefly PAT rejected")
        await _edit(
            query,
            format_error_message(
                "Firefly rejected the PAT. Regenerate it and update the env file."
            ),
            reply_markup=None,
        )
        return
    except FireflyValidationError as e:
        log.warning("Firefly 422: %s", e.field_errors)
        field_hint = ""
        if e.field_errors:
            first_field = next(iter(e.field_errors))
            field_hint = f" (field: {first_field})"
        await _edit(
            query,
            format_error_message(f"Firefly rejected the transaction{field_hint}."),
            reply_markup=None,
        )
        return
    except FireflyError as e:
        log.warning("Firefly error: %s", e)
        await _edit(
            query,
            format_error_message(f"Couldn't save to Firefly: {e}"),
            reply_markup=None,
        )
        return

    # Success! Record usage, set last_transaction, delete pending.
    # Skip account_usage for BNPL purchases (source is a liability — not
    # what the asset-keyboard top-N is for). Repayments DO use an asset
    # source, so we record usage normally.
    is_bnpl_purchase = bnpl_id is not None and not is_repayment
    if not is_bnpl_purchase:
        await services.store.record_account_use(source.id, source.name)
    await services.store.set_last_transaction(
        user_id=pending.user_id,
        firefly_transaction_group_id=created.group_id,
        description=parsed.description,
    )
    await services.store.delete_pending_transaction(pending.callback_id)

    await _edit(
        query,
        format_logged_message(created.group_id, parsed.description),
        reply_markup=None,
    )
    log.info(
        "Logged transaction: user=%d, firefly_id=%d, category=%s",
        pending.user_id, created.group_id, parsed.category,
    )


async def _on_back(
    query,  # type: ignore[no-untyped-def]
    services: Services,
    pending: PendingTransaction,
) -> None:
    """User tapped ⬅️ Back on confirm screen — return to preview."""
    ok = await services.store.update_pending_back_to_awaiting_account(pending.callback_id)
    if not ok:
        await _edit(query, format_expired_message(), reply_markup=None)
        return

    updated = await services.store.get_pending_transaction(pending.callback_id)
    if updated is None:
        return

    parsed = _parsed_from_pending(updated)
    preview = format_preview_message(parsed, selected_currency=updated.currency)

    from firefly_agent.handlers.text import _build_initial_keyboard
    kb = await _build_initial_keyboard(services, pending.callback_id, parsed)
    kb = _rebuild_keyboard_with_selected_currency(kb, updated.currency, pending.callback_id)

    await _edit(query, preview, reply_markup=kb)


async def _on_cancel(
    query,  # type: ignore[no-untyped-def]
    services: Services,
    pending: PendingTransaction,
) -> None:
    """User tapped ❌ Cancel from any state."""
    await services.store.delete_pending_transaction(pending.callback_id)
    # Also clear edit mode if they happened to be in it
    await services.store.clear_edit_mode(pending.user_id)
    await _edit(query, format_cancelled_message(), reply_markup=None)


async def _on_edit(
    query,  # type: ignore[no-untyped-def]
    services: Services,
    pending: PendingTransaction,
) -> None:
    """User tapped ✏️ Edit. Show the field-picker sub-menu.

    The sub-menu lets the user pick which field to edit (description,
    merchant, tags, notes) or redo the whole transaction.
    """
    parsed = _parsed_from_pending(pending)
    preview = format_preview_message(parsed, selected_currency=pending.currency)
    submenu = build_edit_submenu_keyboard(pending.callback_id)
    await _edit(query, preview, reply_markup=submenu)


async def _on_edit_field(
    query,  # type: ignore[no-untyped-def]
    services: Services,
    pending: PendingTransaction,
    *,
    field: str,
) -> None:
    """User picked a field from the edit sub-menu. Set edit_mode for that
    field and prompt them to send the new value.
    """
    if field not in EDIT_PROMPTS:
        log.warning("Unknown edit field: %r", field)
        return

    parsed = _parsed_from_pending(pending)

    # Build the prompt with the current value for context
    if field == "tags":
        current = " ".join(parsed.tags) if parsed.tags else "(none)"
    elif field == "notes":
        current = parsed.notes if parsed.notes else "(none)"
    elif field == "full":
        current = ""
    else:
        current = getattr(parsed, field, "") or "(none)"

    prompt_template = EDIT_PROMPTS[field]
    if "{current}" in prompt_template:
        prompt = prompt_template.format(current=html.escape(current))
    else:
        prompt = prompt_template

    await services.store.set_edit_mode(
        user_id=pending.user_id,
        callback_id=pending.callback_id,
        field=field,
        ttl_minutes=5,
    )

    await _edit(query, prompt, reply_markup=None)
    log.info(
        "User %d entered edit mode (field=%s) for callback_id=%s",
        pending.user_id, field, pending.callback_id,
    )


# ============================================================
# Helpers
# ============================================================


def _parsed_from_pending(pending: PendingTransaction) -> ParsedTransaction:
    """Deserialize the payload_json back into the ParsedTransaction model.

    The payload may have an extra `bnpl_liability_id` field that the
    LLM didn't produce — we filter it out for ParsedTransaction.
    """
    data = json.loads(pending.payload_json)
    # Strip non-LLM fields before validating
    data.pop("bnpl_liability_id", None)
    return ParsedTransaction.model_validate(data)


def _bnpl_id_from_pending(pending: PendingTransaction) -> int | None:
    """Read the bnpl_liability_id we stashed in payload_json (if any)."""
    data = json.loads(pending.payload_json)
    bnpl_id = data.get("bnpl_liability_id")
    if isinstance(bnpl_id, int):
        return bnpl_id
    return None


async def _edit(query, text: str, *, reply_markup) -> None:  # type: ignore[no-untyped-def]
    """Edit the message this callback_query is attached to.

    Swallows 'message is not modified' errors (happens if user double-taps
    and we try to apply the same update twice) but re-raises others.
    """
    try:
        await query.edit_message_text(
            text=text,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup,
        )
    except BadRequest as e:
        if "not modified" in str(e).lower():
            return
        log.warning("edit_message_text failed: %s", e)
