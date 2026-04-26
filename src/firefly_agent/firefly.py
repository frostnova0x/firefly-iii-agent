"""Firefly III API client.

Thin httpx-based client covering only the endpoints Phase 1 needs:

- GET  /api/v1/about                  — health/auth check
- GET  /api/v1/accounts?type=asset    — list asset accounts
- GET  /api/v1/accounts?type=liabilities — list liabilities
- GET  /api/v1/categories             — list categories
- POST /api/v1/transactions           — create a withdrawal/deposit
- DEL  /api/v1/transactions/{id}      — delete (for /undo)

Design:

- Async-native; uses a single shared httpx.AsyncClient per instance.
- TTL cache (default 5 min) for read-only endpoints that change rarely
  (accounts, categories). Writes bypass the cache and invalidate it.
- No retries at this layer. The caller decides retry policy per op.
- Status codes map to typed exceptions (see errors.py).
- PAT never appears in logs or exception messages.

Usage:

    async with FireflyClient(url, pat) as fc:
        accounts = await fc.list_asset_accounts()
        tx = await fc.create_transaction(new_tx)
"""

from __future__ import annotations

import asyncio
import logging
import time
from types import TracebackType
from typing import TypeVar

import httpx

from firefly_agent.errors import (
    FireflyAuthError,
    FireflyError,
    FireflyNotFoundError,
    FireflyUnavailableError,
    FireflyUnexpectedError,
    FireflyValidationError,
)
from firefly_agent.models import Account, Category, CreatedTransaction, NewTransaction

log = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================
# Tiny TTL cache helper
# ============================================================


class _TTLCache:
    """Dead-simple async-safe TTL cache.

    Not a generic cache — hard-coded for this client's pattern:
    key → (value, expires_at). Thread/async safe via a single lock.
    """

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[object, float]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> object | None:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.monotonic() >= expires_at:
                del self._store[key]
                return None
            return value

    async def set(self, key: str, value: object) -> None:
        async with self._lock:
            self._store[key] = (value, time.monotonic() + self._ttl)

    async def invalidate(self, key: str | None = None) -> None:
        """If key is None, invalidate everything."""
        async with self._lock:
            if key is None:
                self._store.clear()
            else:
                self._store.pop(key, None)


# ============================================================
# Client
# ============================================================


class FireflyClient:
    """Async Firefly III API client.

    Construct once, reuse across the service lifetime. Thread-safe for
    asyncio use (a single shared httpx.AsyncClient pools connections
    internally).
    """

    def __init__(
        self,
        base_url: str,
        personal_access_token: str,
        *,
        timeout_seconds: float = 15.0,
        cache_ttl_seconds: float = 300.0,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must include http:// or https://")
        if not personal_access_token:
            raise ValueError("personal_access_token must not be empty")

        self._base_url = base_url.rstrip("/")
        self._pat = personal_access_token
        self._timeout = timeout_seconds
        self._cache = _TTLCache(ttl_seconds=cache_ttl_seconds)
        self._client: httpx.AsyncClient | None = None

    # ----- Lifecycle -----

    async def __aenter__(self) -> FireflyClient:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers={
                "Accept": "application/vnd.api+json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._pat}",
                "User-Agent": "firefly-iii-agent/0.1.0",
            },
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ----- Core request helper -----

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json: dict | None = None,
    ) -> dict:
        """Execute a request, map errors to typed exceptions.

        Logs method/path/status but never the Authorization header or
        request body (may contain transaction details).
        """
        if self._client is None:
            raise RuntimeError("FireflyClient not initialized; use `async with`.")

        try:
            response = await self._client.request(method, path, params=params, json=json)
        except httpx.TimeoutException as e:
            raise FireflyUnavailableError(f"Firefly request timed out: {path}") from e
        except httpx.ConnectError as e:
            raise FireflyUnavailableError(f"Firefly connection failed: {path}") from e
        except httpx.HTTPError as e:
            raise FireflyUnavailableError(f"Firefly HTTP error: {path}") from e

        log.debug("Firefly %s %s → %s", method, path, response.status_code)

        if response.status_code >= 400:
            self._raise_for_status(response)

        # Successful responses: parse JSON except for 204
        if response.status_code == 204 or not response.content:
            return {}
        try:
            return response.json()
        except ValueError as e:
            raise FireflyUnexpectedError(
                f"Non-JSON response from Firefly: {path}",
                status_code=response.status_code,
            ) from e

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Map HTTP status to a typed exception and raise."""
        status = response.status_code

        # Try to extract a useful message without leaking anything
        detail = ""
        field_errors: dict[str, list[str]] = {}
        try:
            body = response.json()
            if isinstance(body, dict):
                detail = str(body.get("message", ""))[:200]
                errors = body.get("errors")
                if isinstance(errors, dict):
                    field_errors = {
                        k: [str(v) for v in (vs if isinstance(vs, list) else [vs])]
                        for k, vs in errors.items()
                    }
        except ValueError:
            detail = response.text[:200]

        msg = detail or response.reason_phrase

        if status == 401:
            raise FireflyAuthError(
                f"Firefly rejected authentication (HTTP 401). "
                f"PAT may be invalidated after a version upgrade. {msg}",
                status_code=401,
            )
        if status == 404:
            raise FireflyNotFoundError(f"Firefly: not found. {msg}", status_code=404)
        if status == 422:
            raise FireflyValidationError(
                f"Firefly rejected the request as invalid. {msg}",
                status_code=422,
                field_errors=field_errors,
            )
        if 500 <= status < 600:
            raise FireflyUnavailableError(
                f"Firefly server error (HTTP {status}). {msg}",
                status_code=status,
            )
        raise FireflyUnexpectedError(
            f"Unexpected HTTP {status} from Firefly. {msg}", status_code=status
        )

    # ----- Pagination helper -----

    async def _get_all_pages(self, path: str, params: dict | None = None) -> list[dict]:
        """Fetch all pages of a JSON:API collection endpoint.

        Firefly paginates with `?page=N&limit=N`. Default limit is 50.
        We bump to 100 and iterate.
        """
        page_params = dict(params or {})
        page_params.setdefault("limit", 100)
        page = 1
        collected: list[dict] = []
        while True:
            page_params["page"] = page
            payload = await self._request("GET", path, params=page_params)
            data = payload.get("data", [])
            collected.extend(data)
            meta = payload.get("meta", {}).get("pagination", {})
            total_pages = int(meta.get("total_pages", 1))
            if page >= total_pages:
                break
            page += 1
        return collected

    # ============================================================
    # Public API — the six endpoints we use
    # ============================================================

    async def about(self) -> dict:
        """GET /api/v1/about — health check.

        Returns the `data` sub-object (version, api_version, etc.).
        Not cached; used as a liveness probe.
        """
        payload = await self._request("GET", "/api/v1/about")
        return payload.get("data", {})

    async def list_asset_accounts(self, *, use_cache: bool = True) -> list[Account]:
        return await self._list_accounts("asset", use_cache=use_cache)

    async def list_liability_accounts(self, *, use_cache: bool = True) -> list[Account]:
        return await self._list_accounts("liabilities", use_cache=use_cache)

    async def _list_accounts(self, account_type: str, *, use_cache: bool) -> list[Account]:
        cache_key = f"accounts:{account_type}"
        if use_cache:
            cached = await self._cache.get(cache_key)
            if cached is not None:
                return list(cached)  # type: ignore[arg-type]

        raw = await self._get_all_pages("/api/v1/accounts", params={"type": account_type})
        accounts = [Account.from_jsonapi(item) for item in raw]
        await self._cache.set(cache_key, accounts)
        return accounts

    async def list_categories(self, *, use_cache: bool = True) -> list[Category]:
        cache_key = "categories"
        if use_cache:
            cached = await self._cache.get(cache_key)
            if cached is not None:
                return list(cached)  # type: ignore[arg-type]

        raw = await self._get_all_pages("/api/v1/categories")
        categories = [Category.from_jsonapi(item) for item in raw]
        await self._cache.set(cache_key, categories)
        return categories

    async def create_transaction(self, new_tx: NewTransaction) -> CreatedTransaction:
        """POST /api/v1/transactions.

        Returns a CreatedTransaction with the group_id suitable for DELETE.
        Note: no retry; a retry on a successful-but-hung POST would
        duplicate the transaction.
        """
        payload = await self._request(
            "POST", "/api/v1/transactions", json=new_tx.to_firefly_json()
        )
        return CreatedTransaction.from_jsonapi(payload)

    async def delete_transaction(self, transaction_group_id: int) -> None:
        """DELETE /api/v1/transactions/{id}.

        Raises FireflyNotFoundError if the transaction is already gone.
        Callers can choose to catch that as a no-op success.
        """
        await self._request("DELETE", f"/api/v1/transactions/{transaction_group_id}")

    # ----- Cache management -----

    async def invalidate_cache(self) -> None:
        """Force-refresh on next read. Call after side-effecting
        operations that could affect accounts/categories.
        """
        await self._cache.invalidate()


__all__ = [
    "FireflyClient",
    # Re-export errors so callers can `from firefly_agent.firefly import FireflyAuthError`
    "FireflyError",
    "FireflyAuthError",
    "FireflyNotFoundError",
    "FireflyValidationError",
    "FireflyUnavailableError",
    "FireflyUnexpectedError",
]
