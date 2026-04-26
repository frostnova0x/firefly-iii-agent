"""Pure formatting helpers — currency, preview message, inline keyboards.

Stateless. No I/O. Easy to unit test.

All rendered strings use Telegram's HTML parse mode. We escape any
dynamic content that could be misinterpreted as tags (merchant names
etc. via html.escape).
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

if TYPE_CHECKING:
    from firefly_agent.models import Account
    from firefly_agent.parsed import ParsedTransaction


# ============================================================
# Currency formatting
# ============================================================

_NO_DECIMAL_CURRENCIES = {"IDR", "JPY", "KRW", "VND"}

_CURRENCY_SYMBOL = {
    "IDR": "Rp",
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CNY": "¥",
    "KRW": "₩",
    "AUD": "A$",
    "SGD": "S$",
    "MYR": "RM",
    "THB": "฿",
    "VND": "₫",
}


def format_currency(amount: Decimal | int | float, currency: str) -> str:
    """Render a monetary amount for display.

    Examples:
        format_currency(45000, "IDR") -> "Rp 45,000"
        format_currency(Decimal("19.99"), "USD") -> "$19.99"
        format_currency(1500, "JPY") -> "¥1,500"
    """
    code = currency.upper()
    symbol = _CURRENCY_SYMBOL.get(code, code)
    amt = Decimal(str(amount))

    if code in _NO_DECIMAL_CURRENCIES:
        whole = int(amt.quantize(Decimal("1")))
        formatted = f"{whole:,}"
    else:
        formatted = f"{amt.quantize(Decimal('0.01')):,}"

    if symbol in {"Rp", "RM", "A$", "S$"} or len(symbol) == 3:
        return f"{symbol} {formatted}"
    return f"{symbol}{formatted}"


# ============================================================
# Preview / confirm messages
# ============================================================


def format_preview_message(
    parsed: ParsedTransaction,
    *,
    selected_currency: str | None = None,
) -> str:
    """Render the first-pass preview (before user picks anything)."""
    currency = (selected_currency or parsed.currency).upper()
    amount_str = format_currency(parsed.amount, currency)

    # Type label adapts to BNPL intent for clarity
    if parsed.intent == "repayment":
        type_label = "BNPL repayment"
    elif parsed.intent == "purchase":
        type_label = "BNPL purchase"
    else:
        type_label = "Withdrawal" if parsed.type == "withdrawal" else "Deposit"

    desc = html.escape(parsed.description)
    merchant = html.escape(parsed.merchant)
    cat = html.escape(parsed.category)

    # Combine description + merchant for readable preview, but they stay
    # as separate fields when posted to Firefly.
    if parsed.description.lower() == parsed.merchant.lower():
        # Avoid "Coffee at Coffee" type duplication
        what_line = f"<b>{desc}</b>"
    else:
        connector = "from" if parsed.type == "deposit" else "at"
        what_line = f"<b>{desc}</b> {connector} <b>{merchant}</b>"

    tags_part = ""
    if parsed.tags:
        tags_str = ", ".join(html.escape(t) for t in parsed.tags)
        tags_part = f" · <i>{tags_str}</i>"

    # Notes line — only shown when populated. For text inputs this is
    # always empty; for receipt photos the LLM may add brief context.
    notes_part = ""
    if parsed.notes:
        notes_part = f"\n📓 <i>{html.escape(parsed.notes)}</i>"

    confidence_part = ""
    if parsed.confidence == "low":
        confidence_part = "\n⚠️  <i>low confidence — please double-check</i>"

    return (
        "📝 <b>Preview</b>\n"
        f"<b>{type_label}:</b> {amount_str}\n"
        f"{what_line}\n"
        f"{cat}{tags_part}\n"
        f"<code>{parsed.date}</code>"
        f"{notes_part}"
        f"{confidence_part}"
    )


def format_confirm_message(
    parsed: ParsedTransaction,
    *,
    selected_currency: str,
    source_account: Account,
    foreign_amount: Decimal | None = None,
    foreign_currency: str | None = None,
) -> str:
    """Confirm-screen message. Foreign-amount line shown only if currencies differ."""
    currency = selected_currency.upper()
    amount_str = format_currency(parsed.amount, currency)

    if parsed.intent == "repayment":
        type_label = "BNPL repayment"
    elif parsed.intent == "purchase":
        type_label = "BNPL purchase"
    else:
        type_label = "Withdrawal" if parsed.type == "withdrawal" else "Deposit"

    desc = html.escape(parsed.description)
    merchant = html.escape(parsed.merchant)
    cat = html.escape(parsed.category)
    source = html.escape(source_account.name)

    if parsed.description.lower() == parsed.merchant.lower():
        what_line = f"<b>{desc}</b>"
    else:
        connector = "from" if parsed.type == "deposit" else "at"
        what_line = f"<b>{desc}</b> {connector} <b>{merchant}</b>"

    tags_part = ""
    if parsed.tags:
        tags_str = ", ".join(html.escape(t) for t in parsed.tags)
        tags_part = f" · <i>{tags_str}</i>"

    notes_part = ""
    if parsed.notes:
        notes_part = f"\n📓 <i>{html.escape(parsed.notes)}</i>"

    foreign_part = ""
    if foreign_amount is not None and foreign_currency:
        foreign_str = format_currency(foreign_amount, foreign_currency)
        foreign_part = f"\n<i>Booked as {foreign_str} on the account</i>"

    return (
        "<b>Confirm?</b>\n"
        f"<b>{type_label}:</b> {amount_str} from <b>{source}</b>\n"
        f"{what_line}\n"
        f"{cat}{tags_part}\n"
        f"<code>{parsed.date}</code>"
        f"{notes_part}"
        f"{foreign_part}"
    )


def format_logged_message(firefly_group_id: int, description: str) -> str:
    desc = html.escape(description)
    return f"✅ Logged: {desc}\n<code>Firefly ID: {firefly_group_id}</code>"


def format_cancelled_message() -> str:
    return "❌ Cancelled."


def format_expired_message() -> str:
    return "⏱ Expired. Send the transaction again to retry."


def format_error_message(human_reason: str) -> str:
    return f"⚠️ {html.escape(human_reason)}"


# ============================================================
# Inline keyboards
# ============================================================

# Callback data format:  "{action}:{callback_id}[:{arg}]"
CB_CURRENCY = "cur"
CB_ACCOUNT = "acc"
CB_CONFIRM = "conf"
CB_BACK = "back"
CB_CANCEL = "canc"
CB_EDIT = "edit"
# Edit sub-menu choice: edit:<cid>:<field> where field ∈
# {description, merchant, tags, notes, full}
CB_EDIT_FIELD = "ef"
# BNPL: a no-op button used as the "pre-selected liability" indicator
# on the BNPL preview. Tapping it does nothing user-visible (we just
# answer the callback query to dismiss the spinner).
CB_NOOP = "noop"


@dataclass
class CurrencyButton:
    code: str
    label: str
    selected: bool


def build_awaiting_account_keyboard(
    *,
    callback_id: str,
    currencies: list[CurrencyButton],
    accounts: list[Account],
) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []

    if len(currencies) > 1:
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{'✓ ' if cb.selected else ''}{cb.label}",
                    callback_data=f"{CB_CURRENCY}:{callback_id}:{cb.code}",
                )
                for cb in currencies
            ]
        )

    for acc in accounts:
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{_account_emoji(acc)} {acc.name}",
                    callback_data=f"{CB_ACCOUNT}:{callback_id}:{acc.id}",
                )
            ]
        )

    # Edit + Cancel on the bottom row
    rows.append(
        [
            InlineKeyboardButton(text="✏️ Edit", callback_data=f"{CB_EDIT}:{callback_id}"),
            InlineKeyboardButton(text="❌ Cancel", callback_data=f"{CB_CANCEL}:{callback_id}"),
        ]
    )
    return InlineKeyboardMarkup(rows)


def build_bnpl_awaiting_confirm_keyboard(
    *,
    callback_id: str,
    currencies: list[CurrencyButton],
    liability_account: Account,
) -> InlineKeyboardMarkup:
    """BNPL keyboard: liability source is fixed (visible-but-inactive button),
    user just confirms or edits or cancels.

    Currency toggle row still shown if multi-currency.
    """
    rows: list[list[InlineKeyboardButton]] = []

    if len(currencies) > 1:
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{'✓ ' if cb.selected else ''}{cb.label}",
                    callback_data=f"{CB_CURRENCY}:{callback_id}:{cb.code}",
                )
                for cb in currencies
            ]
        )

    # Pre-selected liability — visible but tapping is a no-op.
    rows.append(
        [
            InlineKeyboardButton(
                text=f"💳 {liability_account.name}  ✓",
                callback_data=f"{CB_NOOP}:{callback_id}",
            )
        ]
    )

    # Confirm / Edit / Cancel
    rows.append(
        [
            InlineKeyboardButton(text="✅ Confirm", callback_data=f"{CB_CONFIRM}:{callback_id}"),
            InlineKeyboardButton(text="✏️ Edit", callback_data=f"{CB_EDIT}:{callback_id}"),
        ]
    )
    rows.append(
        [InlineKeyboardButton(text="❌ Cancel", callback_data=f"{CB_CANCEL}:{callback_id}")]
    )
    return InlineKeyboardMarkup(rows)


def build_awaiting_confirm_keyboard(callback_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(text="✅ Confirm", callback_data=f"{CB_CONFIRM}:{callback_id}"),
                InlineKeyboardButton(text="⬅️ Back", callback_data=f"{CB_BACK}:{callback_id}"),
            ],
            [
                InlineKeyboardButton(text="✏️ Edit", callback_data=f"{CB_EDIT}:{callback_id}"),
                InlineKeyboardButton(text="❌ Cancel", callback_data=f"{CB_CANCEL}:{callback_id}"),
            ],
        ]
    )


def _account_emoji(account: Account) -> str:
    """Best-guess emoji based on account name. Purely cosmetic."""
    n = account.name.lower()
    if "credit" in n or "cc" in n:
        return "💳"
    if "cash" in n or "wallet" in n:
        return "💵"
    if "saving" in n:
        return "🏦"
    if "usd" in n or "dollar" in n:
        return "💵"
    return "🏦"


def build_edit_submenu_keyboard(callback_id: str) -> InlineKeyboardMarkup:
    """Sub-menu shown when user taps ✏️ Edit.

    Six options across three rows:
        [📝 Description] [🏪 Merchant]
        [🏷 Tags]        [📓 Notes]
        [🔄 Redo all]    [⬅️ Back]
    """
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    text="📝 Description",
                    callback_data=f"{CB_EDIT_FIELD}:{callback_id}:description",
                ),
                InlineKeyboardButton(
                    text="🏪 Merchant",
                    callback_data=f"{CB_EDIT_FIELD}:{callback_id}:merchant",
                ),
            ],
            [
                InlineKeyboardButton(
                    text="🏷 Tags",
                    callback_data=f"{CB_EDIT_FIELD}:{callback_id}:tags",
                ),
                InlineKeyboardButton(
                    text="📓 Notes",
                    callback_data=f"{CB_EDIT_FIELD}:{callback_id}:notes",
                ),
            ],
            [
                InlineKeyboardButton(
                    text="🔄 Redo all",
                    callback_data=f"{CB_EDIT_FIELD}:{callback_id}:full",
                ),
                InlineKeyboardButton(
                    text="⬅️ Back",
                    callback_data=f"{CB_BACK}:{callback_id}",
                ),
            ],
        ]
    )


# Per-field edit prompts shown to the user when they tap a sub-menu option.
EDIT_PROMPTS = {
    "description": (
        "📝 <b>Editing description</b>\n"
        "Current: <code>{current}</code>\n"
        "Send the new description (1–200 chars)."
    ),
    "merchant": (
        "🏪 <b>Editing merchant</b>\n"
        "Current: <code>{current}</code>\n"
        "Send the new merchant name (1–120 chars)."
    ),
    "tags": (
        "🏷 <b>Editing tags</b>\n"
        "Current: <code>{current}</code>\n"
        "Send the new tags space-separated, or <code>none</code> to clear."
    ),
    "notes": (
        "📓 <b>Editing notes</b>\n"
        "Current: <code>{current}</code>\n"
        "Send the new notes (or <code>none</code> to clear)."
    ),
    "full": (
        "🔄 <b>Redo all fields</b>\n"
        "Send the corrected text. Cancels in 5 min."
    ),
}
