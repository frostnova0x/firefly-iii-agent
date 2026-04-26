"""Pydantic models for Firefly III API I/O.

Firefly's API is JSON:API-style: responses wrap the real payload inside
`{"data": {...}, "meta": ..., "links": ...}` (or `"data": [...]` for
collections). These models unwrap to the pieces we actually care about.

Only fields we use are declared. Firefly's full schema has many more
fields; pydantic ignores unknown fields by default, so forward-compat
with new Firefly versions is fine as long as they don't rename the
fields we use.

Reference: https://api-docs.firefly-iii.org/
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ============================================================
# About / health
# ============================================================


class AboutAttributes(BaseModel):
    version: str
    api_version: str
    os: str | None = None
    php_version: str | None = None
    driver: str | None = None


# ============================================================
# Accounts
# ============================================================

AccountType = Literal[
    "asset",
    "expense",
    "revenue",
    "liabilities",
    "cash",
    "initial-balance",
    "reconciliation",
    "loan",
    "debt",
    "mortgage",
]


class AccountAttributes(BaseModel):
    name: str
    type: AccountType
    currency_code: str | None = None
    current_balance: Decimal | None = None
    liability_type: str | None = None  # "debt", "loan", "mortgage" for liabilities


class Account(BaseModel):
    """Single account record (normalized from JSON:API wrapper).

    Firefly wraps each in {"type": "accounts", "id": "6", "attributes": {...}}.
    We flatten the id up and keep the rest of the attrs at the top level.
    """

    id: int
    name: str
    type: AccountType
    currency_code: str | None = None
    current_balance: Decimal | None = None
    liability_type: str | None = None

    @classmethod
    def from_jsonapi(cls, payload: dict) -> Account:
        attrs = AccountAttributes.model_validate(payload["attributes"])
        return cls(
            id=int(payload["id"]),
            name=attrs.name,
            type=attrs.type,
            currency_code=attrs.currency_code,
            current_balance=attrs.current_balance,
            liability_type=attrs.liability_type,
        )


# ============================================================
# Categories
# ============================================================


class CategoryAttributes(BaseModel):
    name: str


class Category(BaseModel):
    id: int
    name: str

    @classmethod
    def from_jsonapi(cls, payload: dict) -> Category:
        attrs = CategoryAttributes.model_validate(payload["attributes"])
        return cls(id=int(payload["id"]), name=attrs.name)


# ============================================================
# Transactions
# ============================================================

TransactionType = Literal["withdrawal", "deposit", "transfer"]


class TransactionSplit(BaseModel):
    """A single line in a transaction. Most transactions have exactly one split.

    For foreign-currency bookings:
    - `currency_code` = the currency charged in the transaction
      (e.g., USD for an OpenAI sub)
    - `amount` = the amount in that currency
    - `foreign_currency_code` = the source account's native currency
      (e.g., IDR for your credit card)
    - `foreign_amount` = the amount in that native currency
    When they match, the foreign fields are omitted.
    """

    model_config = ConfigDict(populate_by_name=True)

    type: TransactionType
    date: datetime | date | str  # ISO 8601 string preferred; datetime/date for back-compat
    amount: Decimal
    description: str
    currency_code: str

    # Asset/liability/expense/revenue account IDs. Which fields are required
    # depends on `type`:
    #   withdrawal → source_id (asset), destination by name or id (expense)
    #   deposit    → source by name or id (revenue), destination_id (asset)
    #   transfer   → source_id (asset), destination_id (asset)
    source_id: int | None = None
    source_name: str | None = None
    destination_id: int | None = None
    destination_name: str | None = None

    category_name: str | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None

    foreign_amount: Decimal | None = None
    foreign_currency_code: str | None = None

    @field_validator("amount", "foreign_amount")
    @classmethod
    def _quantize_amount(cls, v: Decimal | None) -> Decimal | None:
        """Firefly doesn't accept scientific notation; normalize."""
        if v is None:
            return None
        return Decimal(str(v))

    def to_firefly_json(self) -> dict:
        """Serialize to the exact shape Firefly POST /transactions expects."""
        # Date can be a pre-formatted ISO string, a datetime, or a date.
        if isinstance(self.date, str):
            date_str = self.date
        elif isinstance(self.date, datetime):
            date_str = self.date.isoformat()
        else:  # date
            date_str = self.date.isoformat()
        payload: dict = {
            "type": self.type,
            "date": date_str,
            "amount": str(self.amount),
            "description": self.description,
            "currency_code": self.currency_code,
        }
        if self.source_id is not None:
            payload["source_id"] = str(self.source_id)
        if self.source_name is not None:
            payload["source_name"] = self.source_name
        if self.destination_id is not None:
            payload["destination_id"] = str(self.destination_id)
        if self.destination_name is not None:
            payload["destination_name"] = self.destination_name
        if self.category_name:
            payload["category_name"] = self.category_name
        if self.tags:
            payload["tags"] = list(self.tags)
        if self.notes:
            payload["notes"] = self.notes
        if self.foreign_amount is not None and self.foreign_currency_code is not None:
            payload["foreign_amount"] = str(self.foreign_amount)
            payload["foreign_currency_code"] = self.foreign_currency_code
        return payload


class NewTransaction(BaseModel):
    """Top-level envelope for POST /transactions.

    Firefly supports split transactions (one logical transaction with
    multiple lines). We only use single-split for now; the list wrapper
    is required by the API shape.
    """

    error_if_duplicate_hash: bool = False
    apply_rules: bool = True
    group_title: str | None = None
    transactions: list[TransactionSplit]

    def to_firefly_json(self) -> dict:
        return {
            "error_if_duplicate_hash": self.error_if_duplicate_hash,
            "apply_rules": self.apply_rules,
            "group_title": self.group_title,
            "transactions": [t.to_firefly_json() for t in self.transactions],
        }


class CreatedTransaction(BaseModel):
    """Response payload after POST /transactions.

    Firefly returns the full created transaction group; we just need the
    IDs for /undo.
    """

    group_id: int  # the transaction group ID (what DELETE needs)
    split_ids: list[int] = Field(default_factory=list)
    description: str  # for log readability

    @classmethod
    def from_jsonapi(cls, payload: dict) -> CreatedTransaction:
        data = payload["data"]
        attrs = data["attributes"]
        splits = attrs.get("transactions", [])
        return cls(
            group_id=int(data["id"]),
            split_ids=[int(s.get("transaction_journal_id", 0)) for s in splits],
            description=splits[0]["description"] if splits else "",
        )
