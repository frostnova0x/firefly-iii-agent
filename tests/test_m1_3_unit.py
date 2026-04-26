"""Unit tests for M1.3 — OpenRouter client.

No network. Validates:
- ParsedTransaction pydantic model
- JSON schema shape for OpenRouter strict mode
- Prompt building
- Stats tracking
- Error hierarchy

Integration tests in test_m1_3_integration.py hit real OpenRouter.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from firefly_agent.errors import (
    OpenRouterAllModelsFailedError,
    OpenRouterAuthError,
    OpenRouterError,
    OpenRouterRateLimitError,
    OpenRouterSchemaError,
    OpenRouterUnavailableError,
)
from firefly_agent.openrouter import ClientStats, ModelStats, OpenRouterClient
from firefly_agent.parsed import ParsedTransaction, parsed_transaction_json_schema
from firefly_agent.prompts.transaction_text import (
    build_system_prompt,
    build_user_message,
)


# ============================================================
# ParsedTransaction
# ============================================================


class TestParsedTransaction:
    def _valid(self, **overrides: object) -> dict:
        base: dict[str, object] = {
            "type": "withdrawal",
            "amount": 45000,
            "currency": "IDR",
            "description": "Coffee",
            "merchant": "Starbucks",
            "category": "Food & Beverages",
            "tags": ["coffee"],
            "date": "2026-04-24",
            "confidence": "high",
        }
        base.update(overrides)
        return base

    def test_parses_valid_payload(self) -> None:
        p = ParsedTransaction.model_validate(self._valid())
        assert p.type == "withdrawal"
        assert p.amount == Decimal("45000")
        assert p.currency == "IDR"
        assert p.tags == ["coffee"]

    def test_amount_coerced_from_string(self) -> None:
        p = ParsedTransaction.model_validate(self._valid(amount="45000.50"))
        assert p.amount == Decimal("45000.50")

    def test_amount_coerced_from_float(self) -> None:
        p = ParsedTransaction.model_validate(self._valid(amount=45000.50))
        assert p.amount == Decimal("45000.50")

    def test_negative_amount_rejected(self) -> None:
        with pytest.raises(Exception):
            ParsedTransaction.model_validate(self._valid(amount=-100))

    def test_zero_amount_rejected(self) -> None:
        with pytest.raises(Exception):
            ParsedTransaction.model_validate(self._valid(amount=0))

    def test_lowercase_currency_uppercased(self) -> None:
        p = ParsedTransaction.model_validate(self._valid(currency="idr"))
        assert p.currency == "IDR"

    def test_invalid_currency_length_rejected(self) -> None:
        with pytest.raises(Exception):
            ParsedTransaction.model_validate(self._valid(currency="IDRA"))

    def test_invalid_date_format_rejected(self) -> None:
        with pytest.raises(Exception):
            ParsedTransaction.model_validate(self._valid(date="2026/04/24"))

    def test_non_existent_date_rejected(self) -> None:
        # Shape matches but Feb 30 isn't a real date
        with pytest.raises(Exception):
            ParsedTransaction.model_validate(self._valid(date="2026-02-30"))

    def test_parsed_date_property(self) -> None:
        p = ParsedTransaction.model_validate(self._valid())
        assert p.parsed_date == date(2026, 4, 24)

    def test_description_too_long_rejected(self) -> None:
        with pytest.raises(Exception):
            ParsedTransaction.model_validate(self._valid(description="x" * 201))

    def test_empty_description_rejected(self) -> None:
        with pytest.raises(Exception):
            ParsedTransaction.model_validate(self._valid(description=""))

    def test_empty_tags_ok(self) -> None:
        p = ParsedTransaction.model_validate(self._valid(tags=[]))
        assert p.tags == []


# ============================================================
# JSON schema for OpenRouter
# ============================================================


class TestJsonSchema:
    def test_required_keys_present(self) -> None:
        schema = parsed_transaction_json_schema(
            allowed_categories=["Food & Beverages"],
            allowed_tags=["coffee"],
            allowed_currencies=["IDR", "USD"],
        )
        inner = schema["schema"]
        assert inner["additionalProperties"] is False
        assert set(inner["required"]) == {
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
        }

    def test_category_enum_matches_input(self) -> None:
        schema = parsed_transaction_json_schema(
            allowed_categories=["A", "B", "C"],
            allowed_tags=[],
            allowed_currencies=["IDR"],
        )
        assert schema["schema"]["properties"]["category"]["enum"] == ["A", "B", "C"]

    def test_currency_enum_matches_input(self) -> None:
        schema = parsed_transaction_json_schema(
            allowed_categories=["X"],
            allowed_tags=[],
            allowed_currencies=["IDR", "USD", "EUR"],
        )
        assert schema["schema"]["properties"]["currency"]["enum"] == ["IDR", "USD", "EUR"]

    def test_strict_mode_enabled(self) -> None:
        schema = parsed_transaction_json_schema([], [], ["IDR"])
        assert schema["strict"] is True


# ============================================================
# Prompt building
# ============================================================


class TestBuildSystemPrompt:
    def test_includes_today_date(self) -> None:
        prompt = build_system_prompt(
            today=date(2026, 4, 24),
            default_currency="IDR",
            allowed_categories=["Tech"],
            tag_groups={"tech": ["llm-api"]},
        )
        assert "2026-04-24" in prompt

    def test_includes_default_currency(self) -> None:
        prompt = build_system_prompt(
            today=date(2026, 4, 24),
            default_currency="IDR",
            allowed_categories=["Tech"],
            tag_groups={"tech": ["llm-api"]},
        )
        assert "IDR" in prompt

    def test_includes_category_list(self) -> None:
        prompt = build_system_prompt(
            today=date(2026, 4, 24),
            default_currency="IDR",
            allowed_categories=["Food & Beverages", "Tech"],
            tag_groups={},
        )
        assert "Food & Beverages" in prompt
        assert "Tech" in prompt

    def test_includes_indonesian_context(self) -> None:
        """Sanity: we want the prompt aware of IDR/Bahasa/local merchants."""
        prompt = build_system_prompt(
            today=date(2026, 4, 24),
            default_currency="IDR",
            allowed_categories=["Groceries"],
            tag_groups={},
        )
        # These Indonesian-specific cues should be in the prompt
        for cue in ["45k", "jt", "Alfamart", "warung"]:
            assert cue in prompt, f"prompt missing Indonesian cue: {cue}"

    def test_user_message_is_raw(self) -> None:
        assert build_user_message("coffee 45k") == "coffee 45k"


# ============================================================
# Stats tracking
# ============================================================


class TestModelStats:
    def test_avg_latency_empty_is_zero(self) -> None:
        s = ModelStats()
        assert s.avg_latency == 0.0
        assert s.success_rate == 0.0

    def test_avg_latency_computes(self) -> None:
        s = ModelStats(calls=4, total_latency_seconds=8.0)
        assert s.avg_latency == 2.0

    def test_success_rate(self) -> None:
        s = ModelStats(calls=10, successes=7)
        assert s.success_rate == 0.7


class TestClientStats:
    def test_get_or_create_new(self) -> None:
        cs = ClientStats()
        m = cs.get_or_create("model-a")
        assert isinstance(m, ModelStats)
        assert "model-a" in cs.per_model

    def test_get_or_create_existing(self) -> None:
        cs = ClientStats()
        a = cs.get_or_create("model-a")
        b = cs.get_or_create("model-a")
        assert a is b

    def test_summary_empty(self) -> None:
        cs = ClientStats()
        assert "No LLM calls" in cs.summary()

    def test_summary_with_data(self) -> None:
        cs = ClientStats()
        s = cs.get_or_create("model-a")
        s.calls = 5
        s.successes = 4
        assert "model-a" in cs.summary()
        assert "5 calls" in cs.summary()


# ============================================================
# Client construction
# ============================================================


class TestClientConstruction:
    def _base_kwargs(self) -> dict:
        return {
            "models": ["openai/gpt-5.4-nano"],
            "allowed_categories": ["Food & Beverages"],
            "allowed_tags": ["coffee"],
            "allowed_currencies": ["IDR", "USD"],
            "default_currency": "IDR",
            "tag_groups": {"food": ["coffee"]},
        }

    def test_rejects_empty_api_key(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            OpenRouterClient(api_key="", **self._base_kwargs())

    def test_rejects_empty_model_list(self) -> None:
        kwargs = self._base_kwargs() | {"models": []}
        with pytest.raises(ValueError, match="models"):
            OpenRouterClient(api_key="k", **kwargs)

    def test_construction_builds_schema_once(self) -> None:
        c = OpenRouterClient(api_key="k", **self._base_kwargs())
        assert c._schema["schema"]["properties"]["category"]["enum"] == ["Food & Beverages"]


# ============================================================
# Multimodal content builder
# ============================================================


class TestMultimodalMessageBuilder:
    """Tests for the text/image content builder — no network needed."""

    def _client(self) -> OpenRouterClient:
        return OpenRouterClient(
            api_key="k",
            models=["openai/gpt-5.4-nano"],
            allowed_categories=["Food & Beverages"],
            allowed_tags=["coffee"],
            allowed_currencies=["IDR"],
            default_currency="IDR",
            tag_groups={"food": ["coffee"]},
        )

    def test_text_only_returns_string(self) -> None:
        c = self._client()
        result = c._build_user_message_with_optional_image(
            text="coffee 45k", image_bytes=None, image_mime="image/jpeg"
        )
        assert isinstance(result, str)
        assert "coffee 45k" in result

    def test_image_only_returns_list_with_blocks(self) -> None:
        c = self._client()
        result = c._build_user_message_with_optional_image(
            text=None, image_bytes=b"\xff\xd8\xff\xe0fakejpeg", image_mime="image/jpeg"
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image_url"
        assert result[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_text_plus_image_includes_both(self) -> None:
        c = self._client()
        result = c._build_user_message_with_optional_image(
            text="from yesterday", image_bytes=b"\x89PNG\r\n", image_mime="image/png"
        )
        assert isinstance(result, list)
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "from yesterday"
        assert result[1]["image_url"]["url"].startswith("data:image/png;base64,")


# ============================================================
# Exception hierarchy
# ============================================================


class TestExceptionHierarchy:
    def test_all_inherit_base(self) -> None:
        for exc_cls in (
            OpenRouterAuthError,
            OpenRouterRateLimitError,
            OpenRouterUnavailableError,
            OpenRouterSchemaError,
            OpenRouterAllModelsFailedError,
        ):
            assert issubclass(exc_cls, OpenRouterError)

    def test_schema_error_truncates_raw_snippet(self) -> None:
        big = "x" * 1000
        err = OpenRouterSchemaError("bad", raw_snippet=big)
        assert len(err.raw_snippet) == 500

    def test_all_models_failed_carries_failures(self) -> None:
        failures = [("model-a", ValueError("x")), ("model-b", RuntimeError("y"))]
        err = OpenRouterAllModelsFailedError("all failed", failures=failures)
        assert len(err.failures) == 2

    def test_backoff_delay_increases(self) -> None:
        """Exponential backoff monotonic before jitter effect overwhelms it."""
        # delay = 2**attempt + jitter (0..0.5)
        # attempt 0 -> 1.0 to 1.5
        # attempt 2 -> 4.0 to 4.5
        # so attempt 2 must exceed attempt 0
        c = OpenRouterClient(
            api_key="k",
            models=["x"],
            allowed_categories=["X"],
            allowed_tags=[],
            allowed_currencies=["IDR"],
            default_currency="IDR",
            tag_groups={},
        )
        assert c._backoff_delay(0) < c._backoff_delay(2)
