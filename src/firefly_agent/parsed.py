"""Parsed-transaction schema.

This is what the LLM returns. A separate model from `models.NewTransaction`
because the LLM output is an intermediate representation:

- No source/destination account yet (user picks that via inline keyboard)
- No `foreign_amount`/`foreign_currency_code` yet (resolved later based
  on the chosen source account's native currency)
- Has a `confidence` signal the LLM emits about itself

The bot takes a `ParsedTransaction`, asks the user for currency + account,
then builds a `NewTransaction` from the combined info for the Firefly POST.

The JSON schema below is also what we pass to OpenRouter's
`response_format=json_schema` for strict-mode enforcement.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, field_validator


Confidence = Literal["high", "medium", "low"]

# Transaction intent — distinguishes a fresh BNPL purchase from a repayment
# of existing BNPL debt. "regular" is the default (everything that isn't BNPL).
# The LLM produces this field; the bot cross-checks it against keyword patterns
# (see bnpl.py).
Intent = Literal["purchase", "repayment", "regular", "transfer"]


class ParsedTransaction(BaseModel):
    """The LLM's structured interpretation of a user message or receipt."""

    type: Literal["withdrawal", "deposit"]
    amount: Decimal = Field(..., gt=0, description="Amount in the stated currency, positive")
    currency: str = Field(..., pattern=r"^[A-Z]{3}$")
    description: str = Field(..., min_length=1, max_length=200,
                             description="Short item/purpose, e.g. 'Coffee', 'Lunch', 'Domain renewal'")
    merchant: str = Field(..., min_length=1, max_length=120,
                          description="Where it happened: Starbucks, Excelso, Alfamart, Claude, etc.")
    category: str = Field(..., min_length=1)
    tags: list[str] = Field(default_factory=list)
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="ISO date YYYY-MM-DD")
    time: str = Field(
        "",
        description=(
            "Optional time portion in HH:MM:SS format (24h). Empty for "
            "text inputs. For receipt images, only filled if the receipt "
            "actually shows a clock time. The bot uses current local time "
            "when this is empty."
        ),
    )
    confidence: Confidence = "medium"
    intent: Intent = Field(
        "regular",
        description=(
            "Distinguishes BNPL purchases (creates debt) from BNPL repayments "
            "(transfers from asset to liability) from regular transactions."
        ),
    )
    notes: str = Field(
        "",
        max_length=1000,
        description=(
            "Free-form notes. EMPTY for text inputs; only filled by the LLM "
            "for receipt-image inputs to capture context that the structured "
            "fields don't (e.g. 'Subtotal Rp 45000 + service Rp 2500')."
        ),
    )

    @field_validator("amount", mode="before")
    @classmethod
    def _coerce_amount(cls, v: object) -> Decimal:
        """Accept int/float/str from the LLM and normalize to Decimal."""
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))

    @field_validator("currency", mode="before")
    @classmethod
    def _uppercase_currency(cls, v: object) -> str:
        if isinstance(v, str):
            return v.upper()
        return str(v)

    @field_validator("date")
    @classmethod
    def _validate_date_real(cls, v: str) -> str:
        """Pattern only checks shape; this parses to confirm it's a real date."""
        try:
            datetime.strptime(v, "%Y-%m-%d")  # noqa: DTZ007
        except ValueError as e:
            raise ValueError(f"Invalid date: {v}") from e
        return v

    @property
    def parsed_date(self) -> date:
        return datetime.strptime(self.date, "%Y-%m-%d").date()  # noqa: DTZ007

    def to_iso_datetime(self, tz_name: str) -> str:
        """Return a full ISO 8601 datetime string with timezone offset,
        suitable for Firefly III's transaction `date` field.

        - Date portion: from self.date (LLM-extracted)
        - Time portion: from self.time if set (e.g. receipt timestamp),
          otherwise the bot's current local wall-clock time in tz_name
        - Timezone: tz_name (an IANA name like "Asia/Jakarta")

        Examples:
            date="2026-04-25", time="",         tz="Asia/Jakarta"
              → "2026-04-25T<now>:00+07:00"
            date="2026-04-24", time="14:30:00", tz="Asia/Jakarta"
              → "2026-04-24T14:30:00+07:00"
        """
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
        if self.time:
            # LLM extracted a time from a receipt — use it
            try:
                t = datetime.strptime(self.time, "%H:%M:%S").time()
            except ValueError:
                # Defensive: malformed time → fall back to "now"
                t = datetime.now(tz).time().replace(microsecond=0)
        else:
            # Use current local wall-clock time in user's timezone
            t = datetime.now(tz).time().replace(microsecond=0)
        local_dt = datetime.combine(self.parsed_date, t).replace(tzinfo=tz)
        # Firefly accepts standard ISO 8601, e.g. 2026-04-25T14:30:00+07:00
        return local_dt.isoformat(timespec="seconds")


# ------------------------------------------------------------
# JSON Schema for OpenRouter structured outputs
# ------------------------------------------------------------
# OpenRouter's strict mode is picky:
# - `additionalProperties: false` is required
# - All `required` fields must be listed
# - No pydantic-specific keywords that OpenAI JSON Schema doesn't support
#
# We build this by hand rather than using model_json_schema() because
# pydantic emits some keys OpenAI strict mode rejects.


def parsed_transaction_json_schema(
    allowed_categories: list[str],
    allowed_tags: list[str],
    allowed_currencies: list[str],
) -> dict:
    """Construct the strict JSON schema to send as response_format.

    Categories and currencies are enumerated (closed set). Tags are left
    as a plain string array — we can't enum them cleanly because of
    free-form `trip:*` tags. We validate tags in Python after parsing.
    """
    return {
        "name": "parsed_transaction",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "type",
                "amount",
                "currency",
                "description",
                "merchant",
                "category",
                "tags",
                "date",
                "time",
                "confidence",
                "intent",
                "notes",
            ],
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["withdrawal", "deposit"],
                    "description": "withdrawal for spending, deposit for incoming money",
                },
                "amount": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Positive amount in the stated currency",
                },
                "currency": {
                    "type": "string",
                    "enum": allowed_currencies,
                    "description": "ISO 4217 currency code",
                },
                "description": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200,
                    "description": (
                        "WHAT the transaction was for. Short item or purpose. "
                        "Examples: 'Coffee', 'Lunch', 'Groceries', 'Domain renewal', "
                        "'API credits'. NOT the merchant — that goes in 'merchant'."
                    ),
                },
                "merchant": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 120,
                    "description": (
                        "WHERE the transaction happened — the merchant or counterparty. "
                        "Examples: 'Starbucks', 'Excelso', 'Alfamart', 'Gojek', 'Claude', "
                        "'Anthropic'. If unknown, use a category-based fallback like "
                        "'Generic Restaurant', 'Generic Grocery'."
                    ),
                },
                "category": {
                    "type": "string",
                    "enum": allowed_categories,
                    "description": "Must be one of the allowed categories, exactly as written",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Zero or more tags from the taxonomy; trip:* is free-form",
                },
                "date": {
                    "type": "string",
                    "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
                    "description": "YYYY-MM-DD",
                },
                "time": {
                    "type": "string",
                    "description": (
                        "MUST BE EMPTY STRING for text-only inputs. For receipt "
                        "images, fill ONLY if the receipt clearly shows a clock "
                        "time (printed on the receipt). Format: HH:MM:SS in 24-hour. "
                        "If the receipt only shows a date, leave empty."
                    ),
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
                "intent": {
                    "type": "string",
                    "enum": ["purchase", "repayment", "regular", "transfer"],
                    "description": (
                        "purchase = new BNPL purchase (creates debt). "
                        "repayment = paying down existing BNPL debt. "
                        "transfer = moving money between two of YOUR OWN accounts "
                        "(e.g. 'transfer 500k from BCA to cash', 'top up wallet 200k'). "
                        "regular = anything else (default; majority of cases)."
                    ),
                },
                "notes": {
                    "type": "string",
                    "maxLength": 1000,
                    "description": (
                        "Free-form notes. MUST BE EMPTY STRING for text-only inputs. "
                        "Only fill for receipt-image inputs with context the structured "
                        "fields don't capture (e.g. 'Includes Rp 2500 service charge')."
                    ),
                },
            },
        },
    }
