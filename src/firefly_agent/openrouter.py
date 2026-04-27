"""OpenRouter API client for LLM-backed transaction parsing.

Scope (Phase 1):
    - parse_transaction_text(text) → ParsedTransaction

Design:

- **Strict JSON schema** enforced server-side via
  `response_format={"type": "json_schema", "strict": true, ...}`.
  The models we chose (gpt-oss-120b, nemotron-3-super) both support
  this. If the model tries to emit malformed JSON or omit fields,
  OpenRouter rejects it before we see it.

- **Manual fallback chain** — we do NOT use OpenRouter's built-in
  `models` array because it has poor observability (we can't see
  which model was used or which failed, and retry/backoff is opaque).
  Instead we iterate the chain ourselves: try model 1, on error try
  model 2, etc. Full visibility, easy to log, easy to tune.

- **Retry on 429 only** — with exponential backoff. Other errors
  (5xx, timeout, schema failure) fall through to the next model.

- **Per-model stats** — tracks call count, success count, and
  latency distribution per model. Exposed via `stats()` for future
  `/stats` Telegram command.

- **Privacy** — user text is logged at DEBUG only. INFO logs contain
  metadata only (model used, latency, char count).

Usage:
    async with OpenRouterClient(api_key, config) as llm:
        parsed = await llm.parse_transaction_text(
            text="coffee 45k at starbucks",
            now=datetime.now(),
        )
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from types import TracebackType

import httpx
from pydantic import ValidationError

from firefly_agent.errors import (
    OpenRouterAllModelsFailedError,
    OpenRouterAuthError,
    OpenRouterError,
    OpenRouterRateLimitError,
    OpenRouterSchemaError,
    OpenRouterUnavailableError,
    OpenRouterUnexpectedError,
)
from firefly_agent.parsed import ParsedTransaction, parsed_transaction_json_schema
from firefly_agent.prompts.transaction_text import build_system_prompt, build_user_message

log = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# ============================================================
# Per-model stats
# ============================================================


@dataclass
class ModelStats:
    """Running counters for a single model ID."""

    calls: int = 0
    successes: int = 0
    rate_limits: int = 0
    errors: int = 0
    total_latency_seconds: float = 0.0
    last_error: str | None = None

    @property
    def avg_latency(self) -> float:
        return self.total_latency_seconds / self.calls if self.calls else 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.calls if self.calls else 0.0


@dataclass
class ClientStats:
    """Client-wide stats; indexed by model ID."""

    per_model: dict[str, ModelStats] = field(default_factory=dict)

    def get_or_create(self, model: str) -> ModelStats:
        if model not in self.per_model:
            self.per_model[model] = ModelStats()
        return self.per_model[model]

    def summary(self) -> str:
        if not self.per_model:
            return "No LLM calls yet."
        lines = []
        for name, s in self.per_model.items():
            lines.append(
                f"  {name}: {s.calls} calls, "
                f"{s.successes} ok, "
                f"{s.rate_limits} 429s, "
                f"{s.errors} errs, "
                f"avg {s.avg_latency:.2f}s"
            )
        return "\n".join(lines)


# ============================================================
# Client
# ============================================================


class OpenRouterClient:
    """Async OpenRouter client for transaction parsing.

    Construct once at startup, reuse across the service lifetime.
    """

    def __init__(
        self,
        api_key: str,
        *,
        models: list[str],
        allowed_categories: list[str],
        allowed_tags: list[str],
        allowed_currencies: list[str],
        default_currency: str,
        tag_groups: dict[str, list[str]],
        timezone: str = "UTC",
        http_referer: str = "https://github.com/firefly-iii/firefly-iii-agent",
        x_title: str = "firefly-bot",
        timeout_seconds: float = 30.0,
        temperature: float = 0.2,
        max_tokens: int = 800,
        max_retries_on_429: int = 2,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        if not models:
            raise ValueError("models must have at least one entry")

        self._api_key = api_key
        self._models = list(models)
        self._allowed_categories = list(allowed_categories)
        self._allowed_tags = list(allowed_tags)
        self._allowed_currencies = list(allowed_currencies)
        self._default_currency = default_currency
        self._tag_groups = dict(tag_groups)
        self._timezone = timezone

        self._http_referer = http_referer
        self._x_title = x_title
        self._timeout = timeout_seconds
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries_on_429

        self._client: httpx.AsyncClient | None = None
        self._stats = ClientStats()

        # Pre-build the JSON schema (doesn't change across calls)
        self._schema = parsed_transaction_json_schema(
            allowed_categories=self._allowed_categories,
            allowed_tags=self._allowed_tags,
            allowed_currencies=self._allowed_currencies,
        )

    # ----- Lifecycle -----

    async def __aenter__(self) -> OpenRouterClient:
        self._client = httpx.AsyncClient(
            base_url=OPENROUTER_BASE_URL,
            timeout=self._timeout,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self._http_referer,
                "X-Title": self._x_title,
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

    # ----- Stats -----

    @property
    def stats(self) -> ClientStats:
        return self._stats

    # ============================================================
    # Public: unified parsing (text + optional image)
    # ============================================================

    async def parse_transaction(
        self,
        text: str | None = None,
        *,
        image_bytes: bytes | None = None,
        image_mime: str = "image/jpeg",
        now: datetime | None = None,
    ) -> ParsedTransaction:
        """Parse free-form text and/or a receipt image into a ParsedTransaction.

        Exactly one of `text` or `image_bytes` must be provided (or both:
        a text hint alongside a receipt photo is valid).

        Tries each model in `models` in order. On success, returns
        immediately. On failure, falls through to the next model. If
        every model fails, raises OpenRouterAllModelsFailedError.

        Phase 1 passes `text` only; Phase 2 will pass `image_bytes` and
        optionally a text hint like "receipt from yesterday".
        """
        if self._client is None:
            raise RuntimeError("OpenRouterClient not initialized; use `async with`.")
        if not text and not image_bytes:
            raise ValueError("Provide text, image_bytes, or both")
        if text is not None and not text.strip() and not image_bytes:
            raise ValueError("text must not be empty if provided without image")

        # Compute "today" in the user's configured timezone — NOT the
        # container's system timezone (which is usually UTC and would
        # cause date-shift bugs at the local-midnight boundary).
        if now is not None:
            today = now.date()
        else:
            from zoneinfo import ZoneInfo
            today = datetime.now(ZoneInfo(self._timezone)).date()
        system_prompt = build_system_prompt(
            today=today,
            default_currency=self._default_currency,
            allowed_categories=self._allowed_categories,
            tag_groups=self._tag_groups,
        )

        # Build user message depending on whether we have an image.
        user_message = self._build_user_message_with_optional_image(
            text=text, image_bytes=image_bytes, image_mime=image_mime
        )

        # Metadata for logging (never log text/image content at INFO)
        has_image = image_bytes is not None
        meta = (
            f"text_len={len(text) if text else 0} chars, "
            f"image={'yes' if has_image else 'no'}"
        )
        if text:
            log.debug("parse_transaction: text = %r", text)
        if has_image:
            log.debug("parse_transaction: image = %d bytes %s", len(image_bytes), image_mime)
        log.info("parse_transaction: starting (%s)", meta)

        failures: list[tuple[str, Exception]] = []

        for model in self._models:
            try:
                parsed = await self._call_with_retry(
                    model=model,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    today=today,
                )
                log.info(
                    "parse_transaction: ok via %s (category=%s, confidence=%s)",
                    model,
                    parsed.category,
                    parsed.confidence,
                )
                return parsed
            except (
                OpenRouterRateLimitError,
                OpenRouterUnavailableError,
                OpenRouterSchemaError,
                OpenRouterUnexpectedError,
            ) as e:
                failures.append((model, e))
                log.warning("parse_transaction: %s failed (%s), trying next", model, e)
                continue
            except OpenRouterAuthError:
                # Auth errors are the same across all models; don't bother
                # trying the rest of the chain.
                raise

        raise OpenRouterAllModelsFailedError(
            f"All {len(self._models)} models failed",
            failures=failures,
        )

    # Keep the old name around temporarily as a thin alias so tests
    # don't all break at once. Phase 2 can remove this once callers
    # are migrated.
    async def parse_transaction_text(
        self, text: str, *, now: datetime | None = None
    ) -> ParsedTransaction:
        """Deprecated: use parse_transaction() instead."""
        return await self.parse_transaction(text=text, now=now)

    def _build_user_message_with_optional_image(
        self,
        *,
        text: str | None,
        image_bytes: bytes | None,
        image_mime: str,
    ) -> str | list[dict]:
        """Build the content field for the user-role message.

        - Pure text → a plain string (OpenAI / Anthropic / Gemini
          all accept this shape for text-only).
        - Image (with or without text hint) → a list of content blocks
          in OpenAI's multimodal format. OpenRouter normalizes this
          to the right shape for each provider.
        """
        if image_bytes is None:
            assert text is not None
            return build_user_message(text)

        import base64
        b64 = base64.b64encode(image_bytes).decode("ascii")
        image_url = f"data:{image_mime};base64,{b64}"

        content_blocks: list[dict] = []
        if text and text.strip():
            content_blocks.append({"type": "text", "text": text})
        else:
            content_blocks.append(
                {"type": "text", "text": "Parse this receipt into the schema."}
            )
        content_blocks.append({"type": "image_url", "image_url": {"url": image_url}})
        return content_blocks

    # ============================================================
    # Internal: single model call with 429 retry
    # ============================================================

    async def _call_with_retry(
        self,
        *,
        model: str,
        system_prompt: str,
        user_message: str | list[dict],
        today: date,
    ) -> ParsedTransaction:
        """Call one model; retry on 429 with backoff up to max_retries."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await self._call_once(
                    model=model,
                    system_prompt=system_prompt,
                    user_message=user_message,
                )
            except OpenRouterRateLimitError as e:
                last_exc = e
                if attempt == self._max_retries:
                    break
                delay = self._backoff_delay(attempt)
                log.info(
                    "429 on %s (attempt %d), retrying in %.1fs",
                    model, attempt + 1, delay,
                )
                await asyncio.sleep(delay)

        # All retries exhausted; re-raise the last 429 for caller to see
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        """Exponential backoff: 1s, 2s, 4s... with jitter."""
        base = 2.0**attempt
        jitter = random.uniform(0.0, 0.5)  # noqa: S311 — jitter, not security
        return base + jitter

    # ============================================================
    # Internal: a single POST /chat/completions
    # ============================================================

    async def _call_once(
        self,
        *,
        model: str,
        system_prompt: str,
        user_message: str | list[dict],
    ) -> ParsedTransaction:
        """One HTTP call, one model. Returns a parsed/validated transaction
        or raises a typed exception.

        `user_message` is either a plain string (text-only) or a list of
        content blocks (multimodal with image_url blocks). Either shape
        is valid in OpenRouter/OpenAI-compatible chat completions.
        """
        assert self._client is not None

        stats = self._stats.get_or_create(model)
        stats.calls += 1

        payload = {
            "model": model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": self._schema,
            },
        }

        start = time.monotonic()
        try:
            response = await self._client.post("/chat/completions", json=payload)
        except httpx.TimeoutException as e:
            stats.errors += 1
            stats.last_error = "timeout"
            raise OpenRouterUnavailableError(f"Timeout calling {model}") from e
        except httpx.ConnectError as e:
            stats.errors += 1
            stats.last_error = "connect"
            raise OpenRouterUnavailableError(f"Connection failed to OpenRouter: {e}") from e
        except httpx.HTTPError as e:
            stats.errors += 1
            stats.last_error = "http"
            raise OpenRouterUnavailableError(f"HTTP error calling {model}: {e}") from e

        latency = time.monotonic() - start
        stats.total_latency_seconds += latency
        log.debug("OpenRouter %s → HTTP %d in %.2fs", model, response.status_code, latency)

        self._raise_for_status(response, model=model, stats=stats)

        try:
            body = response.json()
        except ValueError as e:
            stats.errors += 1
            stats.last_error = "non_json_body"
            raise OpenRouterUnexpectedError(
                f"Non-JSON response from OpenRouter ({model})"
            ) from e

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            stats.errors += 1
            stats.last_error = "no_content"
            raise OpenRouterUnexpectedError(
                f"Unexpected response shape from {model}"
            ) from e

        # Defensive: content can be None or empty if the provider hiccups
        # mid-generation but still returns HTTP 200. Treat as a schema
        # failure so the fallback chain takes over cleanly.
        if not content or not isinstance(content, str):
            stats.errors += 1
            stats.last_error = "empty_content"
            raise OpenRouterSchemaError(
                f"Model {model} returned empty/null content despite HTTP 200",
                raw_snippet=str(content)[:200],
            )

        # Parse and validate the JSON content.
        # With strict schema, we expect content to be a clean JSON string.
        try:
            import json
            raw = json.loads(content)
        except ValueError as e:
            stats.errors += 1
            stats.last_error = "invalid_json"
            raise OpenRouterSchemaError(
                f"Model {model} returned invalid JSON despite strict schema",
                raw_snippet=str(content),
            ) from e

        try:
            parsed = ParsedTransaction.model_validate(raw)
        except ValidationError as e:
            stats.errors += 1
            stats.last_error = "schema_validation"
            raise OpenRouterSchemaError(
                f"Model {model} output failed validation: {e}",
                raw_snippet=content,
            ) from e

        stats.successes += 1
        return parsed

    def _raise_for_status(
        self,
        response: httpx.Response,
        *,
        model: str,
        stats: ModelStats,
    ) -> None:
        """Map HTTP status to a typed exception and raise."""
        status = response.status_code
        if status < 400:
            return

        # Extract message safely
        detail = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                err = body.get("error") or {}
                if isinstance(err, dict):
                    detail = str(err.get("message", ""))[:200]
                elif isinstance(err, str):
                    detail = err[:200]
                if not detail:
                    detail = str(body.get("message", ""))[:200]
        except ValueError:
            detail = response.text[:200]

        msg = detail or response.reason_phrase

        if status == 401:
            stats.errors += 1
            stats.last_error = "401 auth"
            raise OpenRouterAuthError(
                f"OpenRouter rejected API key (HTTP 401). {msg}",
                status_code=401,
            )
        if status == 429:
            stats.rate_limits += 1
            stats.last_error = "429 rate limit"
            raise OpenRouterRateLimitError(
                f"Rate limited on {model}. {msg}",
                status_code=429,
            )
        if 500 <= status < 600:
            stats.errors += 1
            stats.last_error = f"{status}"
            raise OpenRouterUnavailableError(
                f"Server error from OpenRouter {model} (HTTP {status}). {msg}",
                status_code=status,
            )

        stats.errors += 1
        stats.last_error = f"{status}"
        raise OpenRouterUnexpectedError(
            f"Unexpected HTTP {status} from OpenRouter {model}. {msg}",
            status_code=status,
        )


__all__ = [
    "OpenRouterClient",
    "ClientStats",
    "ModelStats",
    "OpenRouterError",
    "OpenRouterAuthError",
    "OpenRouterRateLimitError",
    "OpenRouterUnavailableError",
    "OpenRouterSchemaError",
    "OpenRouterAllModelsFailedError",
    "OpenRouterUnexpectedError",
]
