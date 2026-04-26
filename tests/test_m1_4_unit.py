"""Unit tests for M1.4 — SQLite state layer.

No network. Each test gets an isolated DB via tmp_path fixture.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from firefly_agent.state import (
    CURRENT_SCHEMA_VERSION,
    AccountUsage,
    LastTransaction,
    PendingTransaction,
    StateStore,
    generate_callback_id,
)


@pytest.fixture
async def store(tmp_path: Path):
    """Fresh, isolated StateStore per test."""
    db = tmp_path / "state.db"
    async with StateStore(db) as s:
        yield s


# ============================================================
# Callback ID generation
# ============================================================


class TestGenerateCallbackId:
    def test_length(self) -> None:
        assert len(generate_callback_id()) == 8

    def test_url_safe(self) -> None:
        """Must be safe in Telegram callback_data (no whitespace, no /\\'\")."""
        for _ in range(100):
            cid = generate_callback_id()
            for bad in (" ", "/", "\\", "'", '"', "\n", "\t"):
                assert bad not in cid, f"unsafe char {bad!r} in {cid!r}"

    def test_uniqueness(self) -> None:
        """A thousand IDs without collision (probability is astronomically low)."""
        ids = {generate_callback_id() for _ in range(1000)}
        assert len(ids) == 1000


# ============================================================
# Initialization & migrations
# ============================================================


class TestInitialization:
    @pytest.mark.asyncio
    async def test_creates_db_file(self, tmp_path: Path) -> None:
        db = tmp_path / "subdir" / "state.db"
        async with StateStore(db) as s:
            await s.counts()
        assert db.is_file()

    @pytest.mark.asyncio
    async def test_records_schema_version(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        async with StateStore(db) as s:
            conn = s._require_conn()
            cur = await conn.execute("SELECT MAX(version) FROM schema_version")
            row = await cur.fetchone()
            assert row[0] == CURRENT_SCHEMA_VERSION

    @pytest.mark.asyncio
    async def test_idempotent_initialize(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        s = StateStore(db)
        await s.initialize()
        await s.initialize()  # should not raise or duplicate tables
        counts = await s.counts()
        assert counts["pending_transactions"] == 0
        await s.close()

    @pytest.mark.asyncio
    async def test_reopen_preserves_data(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        async with StateStore(db) as s:
            await s.record_account_use(account_id=6, account_name="BCA")

        # Reopen — data should survive
        async with StateStore(db) as s:
            top = await s.get_top_accounts()
            assert len(top) == 1
            assert top[0].account_id == 6

    @pytest.mark.asyncio
    async def test_uninitialized_raises(self, tmp_path: Path) -> None:
        s = StateStore(tmp_path / "state.db")
        with pytest.raises(RuntimeError, match="not initialized"):
            await s.get_pending_transaction("anything")

    @pytest.mark.asyncio
    async def test_pragmas_applied(self, tmp_path: Path) -> None:
        """WAL, foreign_keys, busy_timeout should all be active."""
        async with StateStore(tmp_path / "state.db") as s:
            conn = s._require_conn()

            cur = await conn.execute("PRAGMA journal_mode")
            row = await cur.fetchone()
            assert row[0].lower() == "wal"

            cur = await conn.execute("PRAGMA foreign_keys")
            row = await cur.fetchone()
            assert row[0] == 1

            cur = await conn.execute("PRAGMA busy_timeout")
            row = await cur.fetchone()
            assert row[0] == 5000


# ============================================================
# Pending transactions — full lifecycle
# ============================================================


class TestPendingTransactionLifecycle:
    @pytest.mark.asyncio
    async def test_insert_and_get(self, store: StateStore) -> None:
        cid = await store.insert_pending_transaction(
            user_id=1839631182,
            chat_id=1839631182,
            message_id=42,
            payload_json='{"amount": "45000", "currency": "IDR"}',
            currency="IDR",
        )
        assert len(cid) == 8

        p = await store.get_pending_transaction(cid)
        assert p is not None
        assert p.callback_id == cid
        assert p.user_id == 1839631182
        assert p.state == "awaiting_account"
        assert p.currency == "IDR"
        assert p.source_account_id is None

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, store: StateStore) -> None:
        assert await store.get_pending_transaction("does-not") is None

    @pytest.mark.asyncio
    async def test_update_currency(self, store: StateStore) -> None:
        cid = await store.insert_pending_transaction(
            user_id=1, chat_id=1, message_id=1,
            payload_json="{}", currency="IDR",
        )
        assert await store.update_pending_currency(cid, "USD") is True

        p = await store.get_pending_transaction(cid)
        assert p is not None
        assert p.currency == "USD"

    @pytest.mark.asyncio
    async def test_update_currency_nonexistent_returns_false(
        self, store: StateStore
    ) -> None:
        assert await store.update_pending_currency("fake-id", "USD") is False

    @pytest.mark.asyncio
    async def test_advance_to_awaiting_confirm(self, store: StateStore) -> None:
        cid = await store.insert_pending_transaction(
            user_id=1, chat_id=1, message_id=1,
            payload_json="{}", currency="IDR",
        )

        ok = await store.update_pending_to_awaiting_confirm(cid, source_account_id=6)
        assert ok is True

        p = await store.get_pending_transaction(cid)
        assert p is not None
        assert p.state == "awaiting_confirm"
        assert p.source_account_id == 6

    @pytest.mark.asyncio
    async def test_cannot_advance_twice(self, store: StateStore) -> None:
        """State transition is idempotent-ish: calling it from the wrong
        state returns False (no matching row WHERE clause)."""
        cid = await store.insert_pending_transaction(
            user_id=1, chat_id=1, message_id=1,
            payload_json="{}", currency="IDR",
        )
        assert await store.update_pending_to_awaiting_confirm(cid, 6) is True
        # Now state is awaiting_confirm; calling again should no-op
        assert await store.update_pending_to_awaiting_confirm(cid, 7) is False

    @pytest.mark.asyncio
    async def test_back_to_awaiting_account(self, store: StateStore) -> None:
        """User taps 'Back' on the confirm screen."""
        cid = await store.insert_pending_transaction(
            user_id=1, chat_id=1, message_id=1,
            payload_json="{}", currency="IDR",
        )
        await store.update_pending_to_awaiting_confirm(cid, 6)

        assert await store.update_pending_back_to_awaiting_account(cid) is True

        p = await store.get_pending_transaction(cid)
        assert p is not None
        assert p.state == "awaiting_account"
        assert p.source_account_id is None  # cleared

    @pytest.mark.asyncio
    async def test_delete(self, store: StateStore) -> None:
        cid = await store.insert_pending_transaction(
            user_id=1, chat_id=1, message_id=1,
            payload_json="{}", currency="IDR",
        )
        assert await store.delete_pending_transaction(cid) is True
        assert await store.get_pending_transaction(cid) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(self, store: StateStore) -> None:
        assert await store.delete_pending_transaction("nothere") is False


# ============================================================
# Expiry / reaping
# ============================================================


class TestExpiry:
    @pytest.mark.asyncio
    async def test_expired_pending_not_returned(self, store: StateStore) -> None:
        """TTL=0 means immediate expiry on any future read."""
        cid = await store.insert_pending_transaction(
            user_id=1, chat_id=1, message_id=1,
            payload_json="{}", currency="IDR",
            ttl_minutes=0,
        )
        # Give the clock a beat so `now > expires_at` strictly
        await asyncio.sleep(1.1)
        assert await store.get_pending_transaction(cid) is None

    @pytest.mark.asyncio
    async def test_reap_deletes_only_expired(self, store: StateStore) -> None:
        cid_fresh = await store.insert_pending_transaction(
            user_id=1, chat_id=1, message_id=1,
            payload_json="{}", currency="IDR",
            ttl_minutes=30,
        )
        cid_dead = await store.insert_pending_transaction(
            user_id=1, chat_id=1, message_id=2,
            payload_json="{}", currency="IDR",
            ttl_minutes=0,
        )
        await asyncio.sleep(1.1)

        reaped = await store.reap_expired()
        assert reaped == 1

        # Fresh survived; dead is gone
        assert await store.get_pending_transaction(cid_fresh) is not None
        assert await store.get_pending_transaction(cid_dead) is None

    @pytest.mark.asyncio
    async def test_reap_none_to_reap(self, store: StateStore) -> None:
        await store.insert_pending_transaction(
            user_id=1, chat_id=1, message_id=1,
            payload_json="{}", currency="IDR",
            ttl_minutes=30,
        )
        assert await store.reap_expired() == 0

    @pytest.mark.asyncio
    async def test_insert_opportunistically_cleans_up(self, store: StateStore) -> None:
        """Inserting a new pending reaps expired ones as a side effect."""
        await store.insert_pending_transaction(
            user_id=1, chat_id=1, message_id=1,
            payload_json="{}", currency="IDR",
            ttl_minutes=0,
        )
        await asyncio.sleep(1.1)

        # Row exists (reap not run yet)
        counts_before = await store.counts()
        assert counts_before["pending_transactions"] == 1

        # New insert should reap the expired row AND add a new one
        await store.insert_pending_transaction(
            user_id=2, chat_id=2, message_id=2,
            payload_json="{}", currency="IDR",
            ttl_minutes=30,
        )

        counts_after = await store.counts()
        assert counts_after["pending_transactions"] == 1  # new one only


# ============================================================
# Account usage
# ============================================================


class TestAccountUsage:
    @pytest.mark.asyncio
    async def test_first_use_inserts(self, store: StateStore) -> None:
        await store.record_account_use(account_id=6, account_name="BCA")
        top = await store.get_top_accounts()
        assert len(top) == 1
        assert top[0].account_id == 6
        assert top[0].use_count == 1

    @pytest.mark.asyncio
    async def test_subsequent_use_increments(self, store: StateStore) -> None:
        await store.record_account_use(6, "BCA")
        await store.record_account_use(6, "BCA")
        await store.record_account_use(6, "BCA")
        top = await store.get_top_accounts()
        assert top[0].use_count == 3

    @pytest.mark.asyncio
    async def test_name_updated_on_conflict(self, store: StateStore) -> None:
        """Account names can be renamed in Firefly; we should update ours."""
        await store.record_account_use(6, "BCA")
        await store.record_account_use(6, "BCA Savings")
        top = await store.get_top_accounts()
        assert top[0].account_name == "BCA Savings"
        assert top[0].use_count == 2

    @pytest.mark.asyncio
    async def test_top_n_ordering(self, store: StateStore) -> None:
        """Sorted by use_count DESC, then last_used DESC."""
        await store.record_account_use(6, "BCA")
        await store.record_account_use(7, "Cash")
        await store.record_account_use(7, "Cash")
        await store.record_account_use(9, "CC")
        await store.record_account_use(9, "CC")
        await store.record_account_use(9, "CC")

        top3 = await store.get_top_accounts(limit=3)
        assert [t.account_id for t in top3] == [9, 7, 6]

    @pytest.mark.asyncio
    async def test_empty_returns_empty(self, store: StateStore) -> None:
        assert await store.get_top_accounts() == []


# ============================================================
# Last transaction / undo
# ============================================================


class TestLastTransaction:
    @pytest.mark.asyncio
    async def test_set_and_get(self, store: StateStore) -> None:
        await store.set_last_transaction(
            user_id=1839631182,
            firefly_transaction_group_id=42,
            description="Coffee at Starbucks",
        )
        lt = await store.get_last_transaction(1839631182)
        assert lt is not None
        assert lt.firefly_transaction_group_id == 42
        assert lt.description == "Coffee at Starbucks"

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, store: StateStore) -> None:
        assert await store.get_last_transaction(9999) is None

    @pytest.mark.asyncio
    async def test_upsert_overwrites(self, store: StateStore) -> None:
        """Logging a second transaction replaces the 'last' record."""
        await store.set_last_transaction(
            user_id=1, firefly_transaction_group_id=10, description="First"
        )
        await store.set_last_transaction(
            user_id=1, firefly_transaction_group_id=20, description="Second"
        )
        lt = await store.get_last_transaction(1)
        assert lt is not None
        assert lt.firefly_transaction_group_id == 20
        assert lt.description == "Second"

    @pytest.mark.asyncio
    async def test_per_user_isolation(self, store: StateStore) -> None:
        await store.set_last_transaction(user_id=1, firefly_transaction_group_id=10, description="A")
        await store.set_last_transaction(user_id=2, firefly_transaction_group_id=20, description="B")
        a = await store.get_last_transaction(1)
        b = await store.get_last_transaction(2)
        assert a is not None and a.firefly_transaction_group_id == 10
        assert b is not None and b.firefly_transaction_group_id == 20

    @pytest.mark.asyncio
    async def test_clear(self, store: StateStore) -> None:
        await store.set_last_transaction(
            user_id=1, firefly_transaction_group_id=10, description="x"
        )
        assert await store.clear_last_transaction(1) is True
        assert await store.get_last_transaction(1) is None
        # Clear again → False
        assert await store.clear_last_transaction(1) is False


# ============================================================
# Counts & diagnostics
# ============================================================


class TestCounts:
    @pytest.mark.asyncio
    async def test_empty(self, store: StateStore) -> None:
        assert await store.counts() == {
            "pending_transactions": 0,
            "account_usage": 0,
            "last_transaction": 0,
        }

    @pytest.mark.asyncio
    async def test_after_inserts(self, store: StateStore) -> None:
        await store.insert_pending_transaction(
            user_id=1, chat_id=1, message_id=1, payload_json="{}", currency="IDR"
        )
        await store.record_account_use(6, "BCA")
        await store.set_last_transaction(
            user_id=1, firefly_transaction_group_id=10, description="x"
        )

        counts = await store.counts()
        assert counts == {
            "pending_transactions": 1,
            "account_usage": 1,
            "last_transaction": 1,
        }


# ============================================================
# Concurrency — multiple coroutines hitting the same DB
# ============================================================


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_parallel_inserts(self, store: StateStore) -> None:
        """10 concurrent inserts should all succeed (WAL mode + busy_timeout)."""
        async def insert(i: int) -> str:
            return await store.insert_pending_transaction(
                user_id=i, chat_id=i, message_id=i,
                payload_json="{}", currency="IDR",
            )

        callback_ids = await asyncio.gather(*(insert(i) for i in range(10)))
        assert len(set(callback_ids)) == 10  # all unique
        counts = await store.counts()
        assert counts["pending_transactions"] == 10

    @pytest.mark.asyncio
    async def test_parallel_account_use_increments(self, store: StateStore) -> None:
        """Concurrent increments to the same account must all land."""
        await asyncio.gather(
            *(store.record_account_use(6, "BCA") for _ in range(20))
        )
        top = await store.get_top_accounts()
        assert top[0].use_count == 20
