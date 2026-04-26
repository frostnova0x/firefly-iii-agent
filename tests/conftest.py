"""Shared pytest fixtures.

Integration tests require the user to explicitly opt in. Rationale:

1. Integration tests can create side-effects in Firefly if cleanup is
   imperfect (should never happen, but belt-and-braces).
2. Running the integration suite by accident against someone's real
   instance is worse than just skipping them.

So they're gated on FIREFLY_INTEGRATION_OK=1 AND real creds being
present. Without the gate, the suite skips cleanly — useful in CI.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Iterator

import pytest


def _integration_enabled() -> bool:
    return (
        os.environ.get("FIREFLY_INTEGRATION_OK") == "1"
        and bool(os.environ.get("FIREFLY_URL"))
        and bool(os.environ.get("FIREFLY_PAT"))
    )


def _openrouter_integration_enabled() -> bool:
    return (
        os.environ.get("OPENROUTER_INTEGRATION_OK") == "1"
        and bool(os.environ.get("OPENROUTER_API_KEY"))
    )


@pytest.fixture(scope="session")
def integration_enabled() -> bool:
    return _integration_enabled()


@pytest.fixture(scope="session")
def openrouter_integration_enabled() -> bool:
    return _openrouter_integration_enabled()


@pytest.fixture(scope="session")
def openrouter_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        pytest.skip("OPENROUTER_API_KEY not set.")
    return key


@pytest.fixture(scope="session")
def firefly_url() -> str:
    url = os.environ.get("FIREFLY_URL")
    if not url:
        pytest.skip("FIREFLY_URL not set; skipping integration tests.")
    return url


@pytest.fixture(scope="session")
def firefly_pat() -> str:
    pat = os.environ.get("FIREFLY_PAT")
    if not pat:
        pytest.skip("FIREFLY_PAT not set; skipping integration tests.")
    return pat


@pytest.fixture(scope="session")
def default_asset_account_id() -> int:
    raw = os.environ.get("DEFAULT_ASSET_ACCOUNT_ID")
    if not raw:
        pytest.skip("DEFAULT_ASSET_ACCOUNT_ID not set.")
    return int(raw)


@pytest.fixture(autouse=True)
def _skip_integration_if_not_enabled(request: pytest.FixtureRequest) -> None:
    """Auto-skip based on test markers:
    - `integration`            → needs FIREFLY_INTEGRATION_OK + creds
    - `openrouter_integration` → needs OPENROUTER_INTEGRATION_OK + key
    """
    if request.node.get_closest_marker("integration") is not None:
        if not _integration_enabled():
            pytest.skip(
                "Needs FIREFLY_INTEGRATION_OK=1 + FIREFLY_URL + FIREFLY_PAT."
            )
    if request.node.get_closest_marker("openrouter_integration") is not None:
        if not _openrouter_integration_enabled():
            pytest.skip(
                "Needs OPENROUTER_INTEGRATION_OK=1 + OPENROUTER_API_KEY."
            )


@pytest.fixture
def test_tag_prefix() -> str:
    """Prefix every test transaction description so orphans are findable.

    If cleanup ever fails (bug, network flake, killed mid-test),
    searching Firefly for this prefix lists every test residue.
    """
    return "TEST M1.2"


class TransactionTracker:
    """Records transaction group IDs created during a test so teardown
    can delete them even if the test body raises.
    """

    def __init__(self) -> None:
        self._ids: list[int] = []

    def track(self, group_id: int) -> None:
        self._ids.append(group_id)

    @property
    def tracked_ids(self) -> list[int]:
        return list(self._ids)


@pytest.fixture
async def transaction_tracker(
    firefly_url: str, firefly_pat: str
) -> AsyncGenerator[TransactionTracker, None]:
    """Yields a tracker; on teardown, deletes every tracked transaction.

    Uses the client directly (not a fixture) to avoid circularity and
    to survive any failures in the test body.
    """
    # Late import keeps the rest of the test module importable without
    # httpx present.
    from firefly_agent.firefly import FireflyClient, FireflyNotFoundError

    tracker = TransactionTracker()
    yield tracker

    # Teardown
    if not tracker.tracked_ids:
        return

    async with FireflyClient(firefly_url, firefly_pat) as client:
        for tx_id in tracker.tracked_ids:
            try:
                await client.delete_transaction(tx_id)
            except FireflyNotFoundError:
                pass  # already gone — fine
            except Exception as e:  # noqa: BLE001
                # Don't mask test failures with teardown errors, but log loudly
                import warnings
                warnings.warn(
                    f"Teardown failed to delete tx {tx_id}: {e}", stacklevel=1
                )
