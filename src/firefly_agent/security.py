"""Security primitives — owner whitelist and secret redaction.

Design principles:

- **Silent drop for unauthorized users.** No response, no error message,
  no LLM call. The bot should appear invisible to anyone not on the
  whitelist. An error reply would (a) waste OpenRouter quota, and
  (b) confirm the bot exists to reconnaissance scans.

- **Optional low-noise logging of unauthorized attempts.** Logged at
  INFO level with the user ID so legitimate new users can be
  whitelisted manually, but never to ERROR (which could page alerts).
  Controlled by the `log_unauthorized_attempts` config flag.

- **Secret redaction for logs.** If we ever log a dict that contains
  token-like keys, `redact()` ensures values get replaced with a
  placeholder. This is belt-and-braces; primary defense is not
  logging sensitive data in the first place.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeVar

log = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


# ============================================================
# Secret-redaction for log output
# ============================================================

_REDACTED = "[REDACTED]"

# Any key matching one of these substrings (case-insensitive)
# gets its value redacted when we log dicts.
_SECRET_KEY_PARTS = (
    "token",
    "secret",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "pat",
    "bearer",
    "authorization",
    "cookie",
)


def redact(obj: Any, depth: int = 0) -> Any:
    """Recursively redact secret-looking values from a dict/list for safe logging.

    Best-effort: based on key names. Does not inspect string contents.
    """
    if depth > 10:  # arbitrary recursion guard
        return "[TOO_DEEP]"

    if isinstance(obj, dict):
        result: dict[str, Any] = {}
        for k, v in obj.items():
            key_lower = str(k).lower()
            if any(part in key_lower for part in _SECRET_KEY_PARTS):
                result[str(k)] = _REDACTED
            else:
                result[str(k)] = redact(v, depth + 1)
        return result

    if isinstance(obj, list):
        return [redact(item, depth + 1) for item in obj]

    if isinstance(obj, tuple):
        return tuple(redact(item, depth + 1) for item in obj)

    return obj


# ============================================================
# Owner whitelist
# ============================================================


class OwnerGuard:
    """Single source of truth for who's allowed to talk to the bot.

    Initialized once with the whitelist from config. Every inbound
    Telegram update goes through `is_owner()` before any further
    processing — including before any LLM call, DB write, or
    Firefly API request.
    """

    def __init__(self, owner_ids: list[int], log_unauthorized: bool = True) -> None:
        if not owner_ids:
            raise ValueError("OwnerGuard must have at least one owner ID")
        self._owners: frozenset[int] = frozenset(owner_ids)
        self._log_unauthorized = log_unauthorized

    def is_owner(self, user_id: int | None) -> bool:
        """True iff the user ID is whitelisted.

        None (e.g., from a channel post with no user) is never an owner.
        """
        return user_id is not None and user_id in self._owners

    def log_unauthorized(self, user_id: int | None, context: str = "") -> None:
        """Log a denied access attempt, if logging is enabled.

        Kept at INFO — we want visibility for whitelisting decisions
        but don't want to page. Does not log usernames, message content,
        or anything else that could be sensitive; just the numeric ID.
        """
        if not self._log_unauthorized:
            return
        ctx = f" ({context})" if context else ""
        log.info("Unauthorized access attempt from user_id=%s%s", user_id, ctx)

    @property
    def owner_count(self) -> int:
        return len(self._owners)


# ============================================================
# Decorator for async Telegram handlers
# ============================================================


def require_owner(guard: OwnerGuard) -> Callable[[F], F]:
    """Decorator: silently drops handler calls from non-owners.

    Assumes a python-telegram-bot handler signature:
        async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None

    If the update has no effective_user, it's also dropped (channel
    posts, etc.).

    Usage:
        guard = OwnerGuard([1839631182])

        @require_owner(guard)
        async def on_text(update, context):
            ...
    """

    def decorator(handler: F) -> F:
        @wraps(handler)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # python-telegram-bot passes (update, context) positionally
            update = args[0] if args else kwargs.get("update")
            user_id = None
            if update is not None:
                eu = getattr(update, "effective_user", None)
                if eu is not None:
                    user_id = getattr(eu, "id", None)

            if not guard.is_owner(user_id):
                guard.log_unauthorized(user_id, context=handler.__name__)
                return None  # silent drop

            return await handler(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
