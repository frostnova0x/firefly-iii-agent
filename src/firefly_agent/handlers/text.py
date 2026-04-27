"""Text message handler — parse a transaction and show preview keyboard.

Flow:
    1. Owner whitelist (decorator in bot.py registers this properly)
    2. LLM parse
    3. Persist pending transaction
    4. Reply with preview + inline keyboard

All the clients come from context.bot_data["services"].
"""

from __future__ import annotations

import logging

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from firefly_agent.bnpl import detect_bnpl_account_id, reconcile_intent
from firefly_agent.errors import (
    OpenRouterAllModelsFailedError,
    OpenRouterAuthError,
    OpenRouterError,
)
from firefly_agent.formatting import (
    CurrencyButton,
    build_awaiting_account_keyboard,
    build_bnpl_awaiting_confirm_keyboard,
    format_error_message,
    format_preview_message,
)
from firefly_agent.parsed import ParsedTransaction
from firefly_agent.services import Services

log = logging.getLogger(__name__)


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Main text-message handler. Assumes whitelist already checked."""
    if update.message is None or update.effective_user is None:
        return

    text = update.message.text or ""
    if not text.strip():
        return

    # Ignore slash commands — they have their own handlers
    if text.startswith("/"):
        return

    services = Services.get(context)
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id if update.effective_chat else user_id

    # ============================================================
    # Edit-mode branch — different paths depending on which field
    # the user is editing.
    # ============================================================
    edit_state = await services.store.get_edit_mode(user_id)
    if edit_state is not None:
        edit_cid, field = edit_state
        # Always clear edit mode now that we've consumed the message
        await services.store.clear_edit_mode(user_id)

        if field == "full":
            # Re-parse the whole transaction. Falls through to normal flow
            # below, but first delete the old pending.
            log.info("User %d in full-edit mode for %s; re-parsing", user_id, edit_cid)
            await services.store.delete_pending_transaction(edit_cid)
            # Fall through to the normal parse flow below
        else:
            # Per-field edit: update the named field on the existing pending,
            # then re-render the preview WITHOUT calling the LLM.
            log.info(
                "User %d edited %s on pending %s",
                user_id, field, edit_cid,
            )
            await _apply_field_edit(
                services=services,
                update=update,
                callback_id=edit_cid,
                field=field,
                new_value=text.strip(),
            )
            return  # don't fall through to fresh-parse path

    # ============================================================
    # Normal flow — parse a fresh transaction
    # ============================================================
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    # Fetch the user's asset accounts and pass their names into the LLM.
    # The LLM uses this to distinguish "transfer to MY GoPay" (a real transfer)
    # from "transfer to Joko" (a person — withdrawal to expense).
    try:
        asset_accts_for_prompt = await services.firefly.list_asset_accounts()
        asset_names = [a.name for a in asset_accts_for_prompt]
    except Exception as e:  # noqa: BLE001
        log.warning("Couldn't fetch asset accounts for prompt context: %s", e)
        asset_names = []

    try:
        parsed = await services.llm.parse_transaction(
            text=text, asset_account_names=asset_names
        )
    except OpenRouterAuthError:
        log.error("OpenRouter auth failed — API key invalid")
        await update.message.reply_text(
            format_error_message(
                "LLM authentication failed. The operator needs to rotate the OpenRouter API key."
            ),
            parse_mode=ParseMode.HTML,
        )
        return
    except OpenRouterAllModelsFailedError as e:
        log.warning("All LLM models failed: %d failures", len(e.failures))
        await update.message.reply_text(
            format_error_message(
                "All models are temporarily unavailable. Please try again in a minute."
            ),
            parse_mode=ParseMode.HTML,
        )
        return
    except OpenRouterError as e:
        log.warning("LLM parse failed: %s", e)
        await update.message.reply_text(
            format_error_message(f"Couldn't parse that. {e}"),
            parse_mode=ParseMode.HTML,
        )
        return

    # 2. Detect BNPL — does the merchant match a liability account?
    import json as _json
    bnpl_account_id = detect_bnpl_account_id(
        parsed,
        keyword_map=services.liability_keyword_id_map,
        full_text=text,
    )

    # Cross-check LLM-emitted intent against keywords. The reconciled
    # intent is what we trust going forward.
    final_intent = reconcile_intent(
        parsed, full_text=text, asset_account_names=asset_names
    )
    if final_intent != parsed.intent:
        log.info(
            "Intent reconciled: LLM=%s → final=%s (keyword cross-check)",
            parsed.intent, final_intent,
        )
        # Mutate the model copy so downstream rendering reflects the
        # corrected intent.
        parsed = parsed.model_copy(update={"intent": final_intent})

    # 3. Persist pending transaction. We embed the BNPL flag in the
    #    payload so it survives across the user's button taps.
    payload_dict = parsed.model_dump(mode="json")
    if bnpl_account_id is not None:
        payload_dict["bnpl_liability_id"] = bnpl_account_id
        log.info(
            "BNPL detected: user=%d, merchant=%r → liability account %d (intent=%s)",
            user_id, parsed.merchant, bnpl_account_id, final_intent,
        )
    payload_json = _json.dumps(payload_dict)

    cid = await services.store.insert_pending_transaction(
        user_id=user_id,
        chat_id=chat_id,
        message_id=0,  # placeholder, updated after send
        payload_json=payload_json,
        currency=parsed.currency,
        ttl_minutes=services.settings.env.pending_ttl_minutes,
    )

    # 4. Build and send the preview.
    # Three flows:
    #   transfer → source picker, then destination picker (M2.3, same-currency)
    #   BNPL purchase → liability is pre-selected, single tap
    #   everything else → normal asset picker
    if final_intent == "transfer":
        # Pre-flight: need ≥2 asset accounts in this currency, otherwise
        # there's nothing to transfer between.
        all_assets = await services.firefly.list_asset_accounts()
        compatible = [
            a for a in all_assets
            if (a.currency_code or "").upper() == parsed.currency.upper()
        ]
        if len(compatible) < 2:
            await services.store.delete_pending_transaction(cid)
            await update.message.reply_text(
                format_error_message(
                    f"Need at least 2 asset accounts in {parsed.currency} to "
                    f"do a transfer. You have {len(compatible)}."
                ),
                parse_mode=ParseMode.HTML,
            )
            return
        keyboard = await _build_transfer_source_keyboard(services, cid, parsed)
    elif bnpl_account_id is not None and final_intent != "repayment":
        # BNPL purchase
        keyboard = await _build_bnpl_keyboard(services, cid, parsed, bnpl_account_id)
    else:
        # Normal flow (regular or BNPL repayment)
        all_assets = await services.firefly.list_asset_accounts()
        compatible = [
            a for a in all_assets
            if (a.currency_code or "").upper() == parsed.currency.upper()
        ]
        if not compatible:
            await services.store.delete_pending_transaction(cid)
            await update.message.reply_text(
                format_error_message(
                    f"No asset account found in {parsed.currency}. "
                    f"Switch currency or create a {parsed.currency} account in Firefly first."
                ),
                parse_mode=ParseMode.HTML,
            )
            return
        keyboard = await _build_initial_keyboard(services, cid, parsed)
    preview = format_preview_message(parsed)

    try:
        sent = await update.message.reply_text(
            preview,
            parse_mode=ParseMode.HTML,
            reply_markup=keyboard,
        )
    except Exception as e:  # noqa: BLE001
        # If sending the preview fails, clean up the orphaned pending row.
        log.error("Failed to send preview: %s", e)
        await services.store.delete_pending_transaction(cid)
        raise

    # 4. Update the pending row with the real message_id so callbacks can edit it.
    await _update_message_id(services, cid, sent.message_id)

    log.info(
        "Preview sent for user %d, callback_id=%s, category=%s, confidence=%s",
        user_id, cid, parsed.category, parsed.confidence,
    )


async def _build_initial_keyboard(
    services: Services,
    callback_id: str,
    parsed: ParsedTransaction,
):
    """Build the keyboard with currency toggle + top-N accounts.

    Account selection logic:
        1. Fetch live asset accounts from Firefly.
        2. Prune account_usage of any IDs that no longer exist.
        3. Filter by currency — only show accounts denominated in
           parsed.currency (Firefly III can't post a foreign-currency
           transaction through an account it doesn't natively support).
        4. Take usage-ranked accounts (most-used first).
        5. Pad up to top_accounts_in_keyboard with unused accounts.
        6. Pad order: default account first, then by name.
    """
    asset_accounts = await services.firefly.list_asset_accounts()
    target_n = services.settings.toml.flow.top_accounts_in_keyboard
    default_id = services.default_asset_account_id

    # Prune stale usage rows pointing at deleted accounts.
    valid_ids = {a.id for a in asset_accounts}
    await services.store.prune_account_usage(valid_ids)

    # Filter by currency. An account's currency_code is its native currency
    # in Firefly; transactions in other currencies need a different account.
    currency = parsed.currency.upper()
    currency_compatible = [
        a for a in asset_accounts
        if (a.currency_code or "").upper() == currency
    ]

    # Step 1: usage-ranked (within currency-compatible set)
    top_usage = await services.store.get_top_accounts(limit=target_n)
    used_ids = [u.account_id for u in top_usage]
    by_id = {a.id: a for a in currency_compatible}

    ranked: list = []
    for uid in used_ids:
        if uid in by_id:
            ranked.append(by_id[uid])

    # Step 2: pad with unused currency-compatible accounts up to target_n
    if len(ranked) < target_n:
        used_set = {a.id for a in ranked}
        unused = [a for a in currency_compatible if a.id not in used_set]
        unused.sort(key=lambda a: (a.id != default_id, a.name.lower()))
        for acc in unused:
            if len(ranked) >= target_n:
                break
            ranked.append(acc)

    # Currency toggle
    primary = services.settings.toml.currencies.primary
    secondary = services.settings.toml.currencies.secondary
    selected = parsed.currency

    currencies = [
        CurrencyButton(code=primary, label=primary, selected=selected == primary),
        CurrencyButton(code=secondary, label=secondary, selected=selected == secondary),
    ]
    if selected not in (primary, secondary):
        currencies.append(CurrencyButton(code=selected, label=selected, selected=True))

    return build_awaiting_account_keyboard(
        callback_id=callback_id,
        currencies=currencies,
        accounts=ranked,
    )


async def _build_bnpl_keyboard(
    services: Services,
    callback_id: str,
    parsed: ParsedTransaction,
    bnpl_account_id: int,
):
    """Keyboard when BNPL is detected: liability source pre-selected, no
    asset-account picker. User just taps Confirm (or Edit/Cancel).

    Falls back to the normal keyboard if the liability account isn't
    found in Firefly (defensive — shouldn't happen if config is correct).
    """
    # Fetch asset + liability accounts to find the liability by ID.
    asset_accounts = await services.firefly.list_asset_accounts()
    liability_accounts = await services.firefly.list_liability_accounts()
    all_accounts = asset_accounts + liability_accounts
    liability = next((a for a in all_accounts if a.id == bnpl_account_id), None)
    if liability is None:
        log.warning(
            "BNPL liability account id %d not found in Firefly; falling back to normal keyboard",
            bnpl_account_id,
        )
        return await _build_initial_keyboard(services, callback_id, parsed)

    primary = services.settings.toml.currencies.primary
    secondary = services.settings.toml.currencies.secondary
    selected = parsed.currency

    currencies = [
        CurrencyButton(code=primary, label=primary, selected=selected == primary),
        CurrencyButton(code=secondary, label=secondary, selected=selected == secondary),
    ]
    if selected not in (primary, secondary):
        currencies.append(CurrencyButton(code=selected, label=selected, selected=True))

    return build_bnpl_awaiting_confirm_keyboard(
        callback_id=callback_id,
        currencies=currencies,
        liability_account=liability,
    )


async def _build_transfer_source_keyboard(
    services: Services,
    callback_id: str,
    parsed: ParsedTransaction,
):
    """Step 1 of the transfer flow: pick which account the money LEAVES.

    Same shape as the regular account picker but without the currency
    toggle (transfer currency is fixed by the LLM-extracted currency,
    can't toggle mid-flow).

    Filters to asset accounts in the transaction's currency.
    """
    asset_accounts = await services.firefly.list_asset_accounts()
    target_n = services.settings.toml.flow.top_accounts_in_keyboard
    default_id = services.default_asset_account_id

    valid_ids = {a.id for a in asset_accounts}
    await services.store.prune_account_usage(valid_ids)

    currency = parsed.currency.upper()
    compatible = [
        a for a in asset_accounts
        if (a.currency_code or "").upper() == currency
    ]

    top_usage = await services.store.get_top_accounts(limit=target_n)
    used_ids = [u.account_id for u in top_usage]
    by_id = {a.id: a for a in compatible}

    ranked: list = []
    for uid in used_ids:
        if uid in by_id:
            ranked.append(by_id[uid])

    if len(ranked) < target_n:
        used_set = {a.id for a in ranked}
        unused = [a for a in compatible if a.id not in used_set]
        unused.sort(key=lambda a: (a.id != default_id, a.name.lower()))
        for acc in unused:
            if len(ranked) >= target_n:
                break
            ranked.append(acc)

    # No currency toggle — transfers are locked to one currency in M2.3
    return build_awaiting_account_keyboard(
        callback_id=callback_id,
        currencies=[CurrencyButton(code=currency, label=currency, selected=True)],
        accounts=ranked,
    )


async def _update_message_id(services: Services, callback_id: str, message_id: int) -> None:
    """Direct UPDATE — not in StateStore's public API because it's only
    needed by this one site. Doing it inline keeps the StateStore
    interface narrower.
    """
    conn = services.store._require_conn()  # noqa: SLF001 — intentional
    await conn.execute(
        "UPDATE pending_transactions SET message_id = ? WHERE callback_id = ?",
        (message_id, callback_id),
    )
    await conn.commit()


async def _apply_field_edit(
    *,
    services: Services,
    update: Update,
    callback_id: str,
    field: str,
    new_value: str,
) -> None:
    """Update one field on a pending row and re-render the preview.

    Used when the user is in field-specific edit mode (e.g. tapped Tags
    sub-menu, then sent "electronics reimbursable"). Does NOT call the LLM.
    """
    import json as _json
    pending = await services.store.get_pending_transaction(callback_id)
    if pending is None:
        await update.message.reply_text(
            format_error_message("That pending transaction expired."),
            parse_mode="HTML",
        )
        return

    # Mutate the field on the JSON payload
    payload = _json.loads(pending.payload_json)
    if field == "tags":
        if new_value.lower() in ("none", "(none)", ""):
            payload["tags"] = []
        else:
            # Split on whitespace, dedupe while preserving order
            seen: set[str] = set()
            tags: list[str] = []
            for t in new_value.split():
                if t and t not in seen:
                    seen.add(t)
                    tags.append(t)
            payload["tags"] = tags
    elif field == "notes":
        if new_value.lower() in ("none", "(none)"):
            payload["notes"] = ""
        else:
            payload["notes"] = new_value[:1000]  # respect schema max
    elif field == "description":
        payload["description"] = new_value[:200]
    elif field == "merchant":
        payload["merchant"] = new_value[:120]
    else:
        log.warning("Unknown field for edit: %r", field)
        return

    # Validate the result (will raise if user provided something nonsensical)
    try:
        # Strip non-LLM keys before validation
        validation_input = {k: v for k, v in payload.items() if k != "bnpl_liability_id"}
        ParsedTransaction.model_validate(validation_input)
    except Exception as e:  # noqa: BLE001
        await update.message.reply_text(
            format_error_message(f"Couldn't apply edit: {e}"),
            parse_mode="HTML",
        )
        return

    # Persist the updated payload
    new_payload_json = _json.dumps(payload)
    conn = services.store._require_conn()  # noqa: SLF001
    await conn.execute(
        "UPDATE pending_transactions SET payload_json = ? WHERE callback_id = ?",
        (new_payload_json, callback_id),
    )
    await conn.commit()

    # Re-render the preview
    parsed_clean = {k: v for k, v in payload.items() if k != "bnpl_liability_id"}
    parsed = ParsedTransaction.model_validate(parsed_clean)
    bnpl_id = payload.get("bnpl_liability_id")

    if bnpl_id is not None and isinstance(bnpl_id, int):
        keyboard = await _build_bnpl_keyboard(services, callback_id, parsed, bnpl_id)
    else:
        keyboard = await _build_initial_keyboard(services, callback_id, parsed)
    preview = format_preview_message(parsed, selected_currency=pending.currency)

    await update.message.reply_text(
        preview,
        parse_mode="HTML",
        reply_markup=keyboard,
    )
    log.info(
        "Field edit applied: callback_id=%s, field=%s",
        callback_id, field,
    )
