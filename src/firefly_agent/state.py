"""SQLite state layer.

Three tables:

- `pending_transactions` — in-flight transactions awaiting user
  confirmation. Keyed by short callback_id used in Telegram button
  callback_data.
- `account_usage` — usage counter per Firefly account ID. Drives the
  top-N ranking of account buttons in the inline keyboard.
- `last_transaction` — most recent Firefly transaction group ID per
  Telegram user, for /undo.

Plus a `schema_version` table for future migrations.

Design:

- **aiosqlite** for native async. Single connection, shared across the
  service lifetime. SQLite handles serialization internally; no pool
  needed for our workload.
- **WAL mode** — readers don't block writers, writers don't block
  readers. Essential if we ever run concurrent coroutines touching
  the DB.
- **foreign_keys=ON** — referential integrity enforced at DB level.
  (We don't have FKs yet, but enabling by default prevents surprises
  if we add them later.)
- **busy_timeout=5000ms** — retries on lock contention before raising.
- **Parameterized queries only** — no string interpolation, no SQL
  injection risk. (We don't accept SQL from users anyway, but good
  habit.)
- **Timestamps as ISO 8601 UTC strings** — human-readable in sqlite3
  CLI, trivially parseable, no timezone ambiguity.

Cleanup strategy:

- **Periodic**: `StateStore.reap_expired()` called by a background task
  (bot.py) every 5 minutes.
- **On-demand**: called automatically at the end of every
  `insert_pending_transaction()` call.
"""

from __future__ import annotations

import json
import logging
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

log = logging.getLogger(__name__)

# Bumped whenever the schema changes. apply_migrations() runs whatever
# is needed to bring an existing DB up to this version.
CURRENT_SCHEMA_VERSION = 4


# ============================================================
# Data classes
# ============================================================


@dataclass
class PendingTransaction:
    """A transaction being confirmed by the user.

    `payload_json` is the serialized ParsedTransaction from the LLM plus
    any overrides the user has applied (currency switch, etc.). We store
    it as opaque JSON so this layer doesn't need to depend on parsed.py.
    """

    callback_id: str
    user_id: int
    chat_id: int
    message_id: int
    payload_json: str
    state: str  # "awaiting_account" | "awaiting_destination" | "awaiting_confirm"
    source_account_id: int | None
    destination_account_id: int | None  # only set for transfers
    currency: str  # current selected currency (may change before confirm)
    created_at: str  # ISO 8601 UTC
    expires_at: str  # ISO 8601 UTC


@dataclass
class AccountUsage:
    """Usage counter for ranking accounts in the inline keyboard."""

    account_id: int
    account_name: str
    use_count: int
    last_used: str  # ISO 8601 UTC


@dataclass
class LastTransaction:
    """Most recent Firefly transaction group ID per user (for /undo)."""

    user_id: int
    firefly_transaction_group_id: int
    description: str
    logged_at: str  # ISO 8601 UTC


# ============================================================
# Helpers
# ============================================================


def _now_iso() -> str:
    """UTC timestamp as ISO 8601 with +00:00 suffix."""
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _iso_delta(minutes: int) -> str:
    return (datetime.now(tz=UTC) + timedelta(minutes=minutes)).isoformat(timespec="seconds")


def generate_callback_id() -> str:
    """8-char URL-safe ID for Telegram callback_data.

    Telegram limits callback_data to 64 bytes. We use 8 chars so we have
    room for the action prefix (e.g. "acct:CALLBACKID").

    secrets.token_urlsafe(6) gives ~8 chars of ~48 bits of entropy —
    collision probability is negligible for our scale (hundreds of
    pending tx/day max, 30-min TTL → maybe 10 live at peak).
    """
    return secrets.token_urlsafe(6)[:8]


# ============================================================
# StateStore
# ============================================================


class StateStore:
    """Async SQLite-backed state layer.

    Usage:
        store = StateStore("/path/to/state.db")
        await store.initialize()
        try:
            ...
        finally:
            await store.close()

    Or via context manager:
        async with StateStore("/path/to/state.db") as store:
            ...
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn: aiosqlite.Connection | None = None

    # ----- Lifecycle -----

    async def initialize(self) -> None:
        """Open the connection, apply pragmas, run migrations.

        Idempotent. Safe to call again if already initialized.
        """
        if self._conn is not None:
            return

        # Ensure parent directory exists before opening
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row

        # Production pragmas — apply BEFORE any other queries
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute("PRAGMA busy_timeout=5000")
        await self._conn.execute("PRAGMA synchronous=NORMAL")  # WAL makes FULL unnecessary
        await self._conn.commit()

        await self._apply_migrations()
        log.info("StateStore initialized at %s (schema v%d)", self._db_path, CURRENT_SCHEMA_VERSION)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> StateStore:
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()

    # ----- Migrations -----

    async def _apply_migrations(self) -> None:
        """Run migrations from the DB's current schema_version up to CURRENT_SCHEMA_VERSION."""
        assert self._conn is not None

        # schema_version table must always exist
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
            """
        )
        await self._conn.commit()

        cur = await self._conn.execute("SELECT MAX(version) FROM schema_version")
        row = await cur.fetchone()
        current = row[0] if row and row[0] is not None else 0

        if current >= CURRENT_SCHEMA_VERSION:
            return

        # Migration v0 → v1: initial schema
        if current < 1:
            log.info("Applying migration v1: initial schema")
            await self._conn.executescript(_V1_MIGRATION_SQL)
            await self._conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (1, _now_iso()),
            )
            await self._conn.commit()

        # Migration v1 → v2: edit_mode table for "send replacement" feature
        if current < 2:
            log.info("Applying migration v2: edit_mode table")
            await self._conn.executescript(_V2_MIGRATION_SQL)
            await self._conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (2, _now_iso()),
            )
            await self._conn.commit()

        # Migration v2 → v3: edit_mode gains a `field` column so we know
        # which field the user is editing (description/merchant/tags/notes/full).
        if current < 3:
            log.info("Applying migration v3: edit_mode.field column")
            await self._conn.executescript(_V3_MIGRATION_SQL)
            await self._conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (3, _now_iso()),
            )
            await self._conn.commit()

        # Migration v3 → v4: pending_transactions gains destination_account_id
        # for transfers (account-to-account moves need TWO accounts, not one).
        if current < 4:
            log.info("Applying migration v4: pending_transactions.destination_account_id")
            await self._conn.executescript(_V4_MIGRATION_SQL)
            await self._conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (4, _now_iso()),
            )
            await self._conn.commit()

    # ----- Internal helpers -----

    def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("StateStore not initialized; call initialize() or use async with.")
        return self._conn

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Explicit transaction context. Auto-commits on success, rolls back on error."""
        conn = self._require_conn()
        try:
            yield conn
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise

    # ============================================================
    # Pending transactions
    # ============================================================

    async def insert_pending_transaction(
        self,
        *,
        user_id: int,
        chat_id: int,
        message_id: int,
        payload_json: str,
        currency: str,
        ttl_minutes: int = 30,
    ) -> str:
        """Insert a new pending transaction; returns the generated callback_id.

        Also opportunistically reaps any expired rows so the table doesn't
        grow unboundedly if the background task dies.
        """
        callback_id = generate_callback_id()
        now = _now_iso()
        expires = _iso_delta(ttl_minutes)

        async with self._transaction() as conn:
            # Opportunistic cleanup
            await conn.execute(
                "DELETE FROM pending_transactions WHERE expires_at < ?",
                (now,),
            )
            # Insert. destination_account_id starts NULL; transfers fill it
            # via update_pending_destination after user picks the dest.
            await conn.execute(
                """
                INSERT INTO pending_transactions
                    (callback_id, user_id, chat_id, message_id, payload_json,
                     state, source_account_id, destination_account_id,
                     currency, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, 'awaiting_account', NULL, NULL, ?, ?, ?)
                """,
                (
                    callback_id,
                    user_id,
                    chat_id,
                    message_id,
                    payload_json,
                    currency,
                    now,
                    expires,
                ),
            )

        log.debug("Pending transaction %s created for user %d", callback_id, user_id)
        return callback_id

    async def get_pending_transaction(self, callback_id: str) -> PendingTransaction | None:
        conn = self._require_conn()
        cur = await conn.execute(
            """
            SELECT callback_id, user_id, chat_id, message_id, payload_json,
                   state, source_account_id, destination_account_id,
                   currency, created_at, expires_at
            FROM pending_transactions
            WHERE callback_id = ?
              AND expires_at >= ?
            """,
            (callback_id, _now_iso()),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return PendingTransaction(**dict(row))

    async def update_pending_currency(self, callback_id: str, currency: str) -> bool:
        """User tapped a currency button in the preview. Returns True on success."""
        async with self._transaction() as conn:
            cur = await conn.execute(
                """
                UPDATE pending_transactions
                SET currency = ?
                WHERE callback_id = ?
                  AND state = 'awaiting_account'
                  AND expires_at >= ?
                """,
                (currency, callback_id, _now_iso()),
            )
            return cur.rowcount > 0

    async def update_pending_to_awaiting_confirm(
        self,
        callback_id: str,
        source_account_id: int,
    ) -> bool:
        """User picked an account. Advance state. Returns True on success."""
        async with self._transaction() as conn:
            cur = await conn.execute(
                """
                UPDATE pending_transactions
                SET state = 'awaiting_confirm',
                    source_account_id = ?
                WHERE callback_id = ?
                  AND state = 'awaiting_account'
                  AND expires_at >= ?
                """,
                (source_account_id, callback_id, _now_iso()),
            )
            return cur.rowcount > 0

    async def advance_to_awaiting_destination(
        self,
        callback_id: str,
        source_account_id: int,
    ) -> bool:
        """Transfer flow: user picked SOURCE; show destination picker next.

        Differs from advance_to_confirm in that it sets state to
        'awaiting_destination' (a transfer-only intermediate state) and
        leaves destination_account_id NULL until step 3.
        """
        async with self._transaction() as conn:
            cur = await conn.execute(
                """
                UPDATE pending_transactions
                SET state = 'awaiting_destination',
                    source_account_id = ?
                WHERE callback_id = ?
                  AND state = 'awaiting_account'
                  AND expires_at >= ?
                """,
                (source_account_id, callback_id, _now_iso()),
            )
            return cur.rowcount > 0

    async def advance_destination_to_confirm(
        self,
        callback_id: str,
        destination_account_id: int,
    ) -> bool:
        """Transfer flow: user picked DESTINATION; advance to confirm."""
        async with self._transaction() as conn:
            cur = await conn.execute(
                """
                UPDATE pending_transactions
                SET state = 'awaiting_confirm',
                    destination_account_id = ?
                WHERE callback_id = ?
                  AND state = 'awaiting_destination'
                  AND expires_at >= ?
                """,
                (destination_account_id, callback_id, _now_iso()),
            )
            return cur.rowcount > 0

    async def update_pending_back_to_awaiting_account(self, callback_id: str) -> bool:
        """User tapped 'Back' from the confirm screen. Returns True on success.

        For transfers, also clears destination_account_id and reverts
        through the three-state flow.
        """
        async with self._transaction() as conn:
            cur = await conn.execute(
                """
                UPDATE pending_transactions
                SET state = 'awaiting_account',
                    source_account_id = NULL,
                    destination_account_id = NULL
                WHERE callback_id = ?
                  AND state IN ('awaiting_confirm', 'awaiting_destination')
                  AND expires_at >= ?
                """,
                (callback_id, _now_iso()),
            )
            return cur.rowcount > 0

    async def delete_pending_transaction(self, callback_id: str) -> bool:
        """Remove a pending (confirmed, cancelled, or stale). Returns True if a row was deleted."""
        async with self._transaction() as conn:
            cur = await conn.execute(
                "DELETE FROM pending_transactions WHERE callback_id = ?",
                (callback_id,),
            )
            return cur.rowcount > 0

    async def reap_expired(self) -> int:
        """Delete all pending rows whose expires_at < now. Returns count deleted.

        Called by both the background task (every 5 min) and opportunistically
        on insert_pending_transaction().
        """
        async with self._transaction() as conn:
            cur = await conn.execute(
                "DELETE FROM pending_transactions WHERE expires_at < ?",
                (_now_iso(),),
            )
            count = cur.rowcount or 0
        if count > 0:
            log.info("Reaped %d expired pending transaction(s)", count)
        return count

    # ============================================================
    # Account usage
    # ============================================================

    async def record_account_use(self, account_id: int, account_name: str) -> None:
        """Increment the usage counter. Inserts if not present."""
        now = _now_iso()
        async with self._transaction() as conn:
            await conn.execute(
                """
                INSERT INTO account_usage (account_id, account_name, use_count, last_used)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(account_id) DO UPDATE SET
                    use_count = use_count + 1,
                    account_name = excluded.account_name,
                    last_used = excluded.last_used
                """,
                (account_id, account_name, now),
            )

    async def get_top_accounts(self, limit: int = 3) -> list[AccountUsage]:
        """Top N accounts by use_count, tie-broken by last_used (most recent first)."""
        conn = self._require_conn()
        cur = await conn.execute(
            """
            SELECT account_id, account_name, use_count, last_used
            FROM account_usage
            ORDER BY use_count DESC, last_used DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cur.fetchall()
        return [AccountUsage(**dict(r)) for r in rows]

    async def get_all_account_usage(self) -> list[AccountUsage]:
        conn = self._require_conn()
        cur = await conn.execute(
            """
            SELECT account_id, account_name, use_count, last_used
            FROM account_usage
            ORDER BY account_name
            """
        )
        rows = await cur.fetchall()
        return [AccountUsage(**dict(r)) for r in rows]

    # ============================================================
    # Last transaction (/undo)
    # ============================================================

    async def set_last_transaction(
        self,
        *,
        user_id: int,
        firefly_transaction_group_id: int,
        description: str,
    ) -> None:
        """Upsert the last transaction for /undo."""
        async with self._transaction() as conn:
            await conn.execute(
                """
                INSERT INTO last_transaction
                    (user_id, firefly_transaction_group_id, description, logged_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    firefly_transaction_group_id = excluded.firefly_transaction_group_id,
                    description = excluded.description,
                    logged_at = excluded.logged_at
                """,
                (user_id, firefly_transaction_group_id, description, _now_iso()),
            )

    async def get_last_transaction(self, user_id: int) -> LastTransaction | None:
        conn = self._require_conn()
        cur = await conn.execute(
            """
            SELECT user_id, firefly_transaction_group_id, description, logged_at
            FROM last_transaction
            WHERE user_id = ?
            """,
            (user_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return LastTransaction(**dict(row))

    async def clear_last_transaction(self, user_id: int) -> bool:
        """Called after a successful /undo. Returns True if a row existed."""
        async with self._transaction() as conn:
            cur = await conn.execute(
                "DELETE FROM last_transaction WHERE user_id = ?",
                (user_id,),
            )
            return cur.rowcount > 0

    # ============================================================
    # Edit mode (✏️ Edit button → "send replacement" flow)
    # ============================================================

    async def set_edit_mode(
        self,
        *,
        user_id: int,
        callback_id: str,
        field: str = "full",
        ttl_minutes: int = 5,
    ) -> None:
        """Mark user as in edit mode for a specific pending transaction.

        `field` says WHICH field is being edited:
          - "full"        — legacy behavior; replace the whole transaction
          - "description" — replace just description
          - "merchant"    — replace just merchant
          - "tags"        — replace tags (space-separated, or "none" to clear)
          - "notes"       — replace notes
        """
        if field not in ("full", "description", "merchant", "tags", "notes"):
            raise ValueError(f"Invalid edit field: {field!r}")
        async with self._transaction() as conn:
            await conn.execute(
                """
                INSERT INTO edit_mode (user_id, callback_id, field, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    callback_id = excluded.callback_id,
                    field = excluded.field,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                (user_id, callback_id, field, _now_iso(), _iso_delta(ttl_minutes)),
            )

    async def get_edit_mode(self, user_id: int) -> tuple[str, str] | None:
        """Returns (callback_id, field) the user is editing, or None.

        Auto-expires: rows past expires_at return None.
        """
        conn = self._require_conn()
        cur = await conn.execute(
            """
            SELECT callback_id, field FROM edit_mode
            WHERE user_id = ? AND expires_at >= ?
            """,
            (user_id, _now_iso()),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return (row[0], row[1])

    async def clear_edit_mode(self, user_id: int) -> bool:
        async with self._transaction() as conn:
            cur = await conn.execute(
                "DELETE FROM edit_mode WHERE user_id = ?",
                (user_id,),
            )
            return cur.rowcount > 0

    async def reap_expired_edit_modes(self) -> int:
        """Cleanup expired edit-mode rows. Called by the same reap job
        that handles pending transactions.
        """
        async with self._transaction() as conn:
            cur = await conn.execute(
                "DELETE FROM edit_mode WHERE expires_at < ?",
                (_now_iso(),),
            )
            return cur.rowcount or 0

    # ============================================================
    # Account usage cleanup
    # ============================================================

    async def prune_account_usage(self, valid_account_ids: set[int]) -> int:
        """Delete account_usage rows whose account_id isn't in the live set.

        Called when we've fetched fresh accounts from Firefly and noticed
        usage rows pointing at IDs that no longer exist (deleted/recreated
        with new ID). Returns count pruned.
        """
        if not valid_account_ids:
            # Defensive: don't wipe everything if we somehow get an empty set
            return 0
        placeholders = ",".join("?" for _ in valid_account_ids)
        async with self._transaction() as conn:
            cur = await conn.execute(
                f"DELETE FROM account_usage WHERE account_id NOT IN ({placeholders})",  # noqa: S608
                tuple(valid_account_ids),
            )
            count = cur.rowcount or 0
        if count > 0:
            log.info("Pruned %d stale account_usage row(s)", count)
        return count

    # ============================================================
    # Diagnostics
    # ============================================================

    async def counts(self) -> dict[str, int]:
        """Simple row counts for each table, useful for /stats and tests."""
        conn = self._require_conn()
        out: dict[str, int] = {}
        for table in ("pending_transactions", "account_usage", "last_transaction"):
            cur = await conn.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608 — table names are hardcoded
            row = await cur.fetchone()
            out[table] = row[0] if row else 0
        return out


# ============================================================
# Migration SQL
# ============================================================

_V1_MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS pending_transactions (
    callback_id         TEXT PRIMARY KEY,
    user_id             INTEGER NOT NULL,
    chat_id             INTEGER NOT NULL,
    message_id          INTEGER NOT NULL,
    payload_json        TEXT NOT NULL,
    state               TEXT NOT NULL
        CHECK (state IN ('awaiting_account', 'awaiting_destination', 'awaiting_confirm')),
    source_account_id   INTEGER,
    currency            TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    expires_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pending_expires ON pending_transactions(expires_at);
CREATE INDEX IF NOT EXISTS idx_pending_user_id ON pending_transactions(user_id);

CREATE TABLE IF NOT EXISTS account_usage (
    account_id      INTEGER PRIMARY KEY,
    account_name    TEXT NOT NULL,
    use_count       INTEGER NOT NULL DEFAULT 0,
    last_used       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_account_usage_rank
    ON account_usage(use_count DESC, last_used DESC);

CREATE TABLE IF NOT EXISTS last_transaction (
    user_id                         INTEGER PRIMARY KEY,
    firefly_transaction_group_id    INTEGER NOT NULL,
    description                     TEXT NOT NULL,
    logged_at                       TEXT NOT NULL
);
"""

_V2_MIGRATION_SQL = """
-- Tracks "user X tapped Edit on pending Y; their next message should
-- replace pending Y instead of starting a new transaction".
-- One row per user. TTL'd via expires_at.
CREATE TABLE IF NOT EXISTS edit_mode (
    user_id     INTEGER PRIMARY KEY,
    callback_id TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_edit_mode_expires ON edit_mode(expires_at);
"""

_V3_MIGRATION_SQL = """
-- Edit-mode gains a `field` column. Values: 'full' (legacy), 'description',
-- 'merchant', 'tags', 'notes'. Existing rows get 'full' to preserve
-- backwards-compatible behavior for anything in flight at upgrade time.
ALTER TABLE edit_mode ADD COLUMN field TEXT NOT NULL DEFAULT 'full'
    CHECK (field IN ('full', 'description', 'merchant', 'tags', 'notes'));
"""

_V4_MIGRATION_SQL = """
-- pending_transactions gains:
--   1. destination_account_id  (transfers need a SECOND account)
--   2. relaxed CHECK constraint (new state 'awaiting_destination')
--
-- SQLite can't modify a CHECK constraint in-place, so we rebuild the table.
-- Pending rows in flight at upgrade time are preserved verbatim.

CREATE TABLE pending_transactions_new (
    callback_id              TEXT PRIMARY KEY,
    user_id                  INTEGER NOT NULL,
    chat_id                  INTEGER NOT NULL,
    message_id               INTEGER NOT NULL,
    payload_json             TEXT NOT NULL,
    state                    TEXT NOT NULL
        CHECK (state IN ('awaiting_account', 'awaiting_destination', 'awaiting_confirm')),
    source_account_id        INTEGER,
    destination_account_id   INTEGER,
    currency                 TEXT NOT NULL,
    created_at               TEXT NOT NULL,
    expires_at               TEXT NOT NULL
);

INSERT INTO pending_transactions_new
    (callback_id, user_id, chat_id, message_id, payload_json, state,
     source_account_id, destination_account_id, currency, created_at, expires_at)
SELECT
    callback_id, user_id, chat_id, message_id, payload_json, state,
    source_account_id, NULL, currency, created_at, expires_at
FROM pending_transactions;

DROP TABLE pending_transactions;
ALTER TABLE pending_transactions_new RENAME TO pending_transactions;

CREATE INDEX IF NOT EXISTS idx_pending_expires ON pending_transactions(expires_at);
CREATE INDEX IF NOT EXISTS idx_pending_user_id ON pending_transactions(user_id);
"""


__all__ = [
    "StateStore",
    "PendingTransaction",
    "AccountUsage",
    "LastTransaction",
    "generate_callback_id",
    "CURRENT_SCHEMA_VERSION",
]
