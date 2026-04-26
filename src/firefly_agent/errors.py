"""Typed exceptions for API clients.

Firefly III client errors:

    FireflyError                            (base)
    ├── FireflyAuthError                    (401 — PAT rejected)
    ├── FireflyValidationError              (422 — bad request body)
    ├── FireflyNotFoundError                (404 — id doesn't exist)
    ├── FireflyUnavailableError             (5xx, timeout, connection)
    └── FireflyUnexpectedError              (everything else)

OpenRouter client errors:

    OpenRouterError                         (base)
    ├── OpenRouterAuthError                 (401 — API key rejected)
    ├── OpenRouterRateLimitError            (429 — daily/burst quota)
    ├── OpenRouterUnavailableError          (5xx, timeout, connection)
    ├── OpenRouterSchemaError               (model output failed validation)
    ├── OpenRouterAllModelsFailedError      (every model in the chain errored)
    └── OpenRouterUnexpectedError           (everything else)

Design rules:

- No exception ever carries the PAT/API key or the raw `Authorization` header.
- Exception messages are safe to log; they're NOT meant to be surfaced to
  end users verbatim. Handlers format friendly Telegram copy.
- HTTP status is recorded where available to help with logging.
"""

from __future__ import annotations


# ============================================================
# Firefly III client errors
# ============================================================


class FireflyError(Exception):
    """Base for all Firefly III client errors."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class FireflyAuthError(FireflyError):
    """401 — PAT is invalid, revoked, or expired.

    Most commonly seen after a Firefly III major-version upgrade
    invalidates all OAuth tokens. Operator action required:
    regenerate PAT, update your env file (.env or systemd EnvironmentFile),
    restart the bot.
    """


class FireflyValidationError(FireflyError):
    """422 — Firefly rejected the payload as malformed.

    Carries the list of field-level errors for debugging. These are
    safe to log (no secrets) but not meant to be shown verbatim to
    end users.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = 422,
        field_errors: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code)
        self.field_errors = field_errors or {}


class FireflyNotFoundError(FireflyError):
    """404 — Requested resource does not exist.

    Example: DELETE on an already-deleted transaction.
    """


class FireflyUnavailableError(FireflyError):
    """5xx, connection refused, or timeout.

    Retry-safe from the caller's perspective ONLY for read operations.
    For writes (POST), the caller must be careful — the request may
    have been partially processed before the failure.
    """


class FireflyUnexpectedError(FireflyError):
    """Any other non-success status code we didn't model explicitly."""


# ============================================================
# OpenRouter client errors
# ============================================================


class OpenRouterError(Exception):
    """Base for all OpenRouter client errors."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OpenRouterAuthError(OpenRouterError):
    """401 — OpenRouter API key rejected.

    Operator action: regenerate key, update your env file (.env or
    systemd EnvironmentFile), restart the bot.
    """


class OpenRouterRateLimitError(OpenRouterError):
    """429 — hit daily or burst quota on this model.

    Usually transient. Callers should try the next model in the chain.
    """


class OpenRouterUnavailableError(OpenRouterError):
    """5xx, timeout, connection refused."""


class OpenRouterSchemaError(OpenRouterError):
    """The model's output failed JSON or schema validation locally.

    Carries the raw response snippet (truncated) for debugging.
    Does NOT contain the user's original prompt text.
    """

    def __init__(
        self,
        message: str,
        *,
        raw_snippet: str = "",
    ) -> None:
        super().__init__(message)
        # Truncate to prevent logging excessive data
        self.raw_snippet = raw_snippet[:500]


class OpenRouterAllModelsFailedError(OpenRouterError):
    """Every model in the fallback chain errored.

    Carries the list of (model, exception) pairs for diagnosis.
    """

    def __init__(
        self,
        message: str,
        *,
        failures: list[tuple[str, Exception]] | None = None,
    ) -> None:
        super().__init__(message)
        self.failures = failures or []


class OpenRouterUnexpectedError(OpenRouterError):
    """Any other non-success condition we didn't model explicitly."""
